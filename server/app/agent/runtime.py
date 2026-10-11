from __future__ import annotations

from livekit.agents import AgentSession, TurnHandlingOptions, inference

from app.agent.agent import ConversationAgent
from app.agent.handoff.briefing import BriefingBuilder
from app.agent.handoff.coordinator import HandoffCoordinator
from app.agent.handoff.transfer import (
    HumanAvailability,
    NoStaffSystem,
    NoTransport,
    TransferTransport,
)
from app.agent.prompt.compiler import PromptCompiler
from app.agent.prompt.greeting import Greeting
from app.agent.prompt.section import PromptInputs
from app.agent.skills.loader import SkillLoader, SkillToolset
from app.agent.skills.selector import SkillSelector
from app.agent.state import CallState
from app.agent.supervisor import TurnSupervisor
from app.agent.tools.adapter import FrameworkToolAdapter
from app.agent.tools.exposure import ToolExposure
from app.agent.tools.gateway import ToolGateway
from app.analysis.model import AnalysisModel
from app.core.config import Settings
from app.core.database import Database
from app.core.enums import Eagerness, Optimisation
from app.modules.config.schemas import AgentConfig
from app.modules.config.utils import OptimisationFlags
from app.modules.context.schemas import ContextPack
from app.modules.detection.classifiers import HumanRequestDetector, TurnClassifier
from app.modules.detection.guard import OutputGuard
from app.modules.detection.pipeline import DetectionPipeline
from app.modules.detection.policy import EscalationPolicy
from app.voice.fillers import FillerLibrary
from app.voice.greeting import GreetingPreparer


class EndpointingProfile:
    SETTINGS = {
        Eagerness.LOW: {"mode": "dynamic", "min_delay": 0.8, "max_delay": 6.0},
        Eagerness.MEDIUM: {"mode": "dynamic"},
        Eagerness.HIGH: {"mode": "dynamic", "min_delay": 0.3, "max_delay": 2.0},
    }

    @classmethod
    def for_eagerness(cls, eagerness: Eagerness):
        return dict(
            cls.SETTINGS[eagerness],
        )


class AgentRuntime:
    """Assembles the speech pipeline and the agent around a context pack."""

    def __init__(
        self,
        settings: Settings,
        fillers: FillerLibrary,
        availability: HumanAvailability | None = None,
        transport: TransferTransport | None = None,
    ) -> None:
        self._settings = settings
        self._fillers = fillers
        self._compiler = PromptCompiler(settings.prompt)
        self._selector = SkillSelector()
        self._loader = SkillLoader()
        self._exposure = ToolExposure()
        self._guard = OutputGuard()
        self._greetings = GreetingPreparer(self._guard)
        self._policy = EscalationPolicy(settings.detection)
        detection = settings.detection
        self._pipeline = DetectionPipeline(
            TurnClassifier(AnalysisModel(detection.model, detection.timeout_s)),
            HumanRequestDetector(),
            detection.timeout_s,
        )
        self._briefing = BriefingBuilder(
            AnalysisModel(settings.speech.llm_model, settings.memory.llm_timeout_s), detection
        )
        self._availability = availability or NoStaffSystem()
        self._transport = transport or NoTransport()

    def state(self, pack: ContextPack, config: AgentConfig) -> CallState:
        # Create the call state from the caller/organizati
        # on context
        # and compile the prompt that the agent will use.
        skills = self._selector.initial(pack, config)
        compiled = self._compiler.compile(
            PromptInputs(pack=pack, skills=skills, collections=config.collections)
        )
        state = CallState(
            pack=pack,
            compiled=compiled,
            config=config,
            skills=skills,
            active_skill=skills.active_key,
            optimisations=OptimisationFlags.for_call(config.optimisations, pack.call_id),
        )
        return state

    def voice(self, state: CallState) -> str:
        profile = state.config.profile
        return profile.voice or state.pack.organization.voice or self._settings.speech.tts_voice

    def session(self, state: CallState) -> AgentSession[CallState]:
        # Build the voice pipeline: STT listens to the caller
        # LLM generates the response, and TTS speaks it back.
        speech, profile = self._settings.speech, state.config.profile
        return AgentSession[CallState](
            userdata=state,
            stt=inference.STT(
                model=profile.stt_model or speech.stt_model,
                language=profile.stt_language or speech.stt_language,
            ),
            llm=inference.LLM(model=profile.llm_model or speech.llm_model),
            tts=inference.TTS(model=profile.tts_model or speech.tts_model, voice=self.voice(state)),
            turn_handling=TurnHandlingOptions(
                turn_detection=inference.TurnDetector(),
                endpointing=EndpointingProfile.for_eagerness(profile.eagerness),
                preemptive_generation={
                    "enabled": Optimisation.PREEMPTIVE_GENERATION in state.optimisations
                },
            ),
        )

    def supervisor(
        self, state: CallState, gateway: ToolGateway, database: Database
    ) -> TurnSupervisor:
        coordinator = HandoffCoordinator(
            database, self._availability, self._transport, self._briefing, gateway
        )
        return TurnSupervisor(
            state,
            self._pipeline,
            self._policy,
            coordinator,
            self._briefing,
            self._settings.detection,
        )

    def agent(
        self, state: CallState, gateway: ToolGateway, supervisor: TurnSupervisor
    ) -> ConversationAgent:
        # Create the conversation agent with its state, greeting,
        #  prompt compiler, and tools it can use during the call.
        return ConversationAgent(
            compiler=self._compiler,
            state=state,
            greeting=Greeting(state.pack).instruction(),
            toolsets=[SkillToolset(self._loader)],
            adapter=FrameworkToolAdapter(gateway, self._fillers, self.voice(state), supervisor),
            exposure=self._exposure,
            supervisor=supervisor,
            guard=self._guard,
            greeting_timeout_s=self._settings.latency.greeting_timeout_s,
        )

    def prepare_greeting(self, state: CallState, synthesiser) -> None:
        """Starts the greeting during setup, when the switch is on for this call."""
        if Optimisation.PREPARED_GREETING not in state.optimisations:
            return
        name = state.config.profile.llm_model or self._settings.speech.llm_model
        model = AnalysisModel(name, self._settings.latency.greeting_timeout_s)
        instruction = Greeting(state.pack).instruction()
        state.greeting = self._greetings.start(state, synthesiser, model, instruction)
