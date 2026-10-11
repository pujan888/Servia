from __future__ import annotations

import os
import uuid

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import FlagType, Optimisation


class DatabaseSettings(BaseModel):
    url: str
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 5
    pool_recycle: int = 1800


class LiveKitSettings(BaseModel):
    url: str
    api_key: str
    api_secret: str
    agent_name: str = "servia"
    token_ttl_seconds: int = 900
    # Warm job processes kept ready. Each holds the models in memory, so keep it low locally.
    idle_processes: int = 1

    def export_env(self) -> None:
        """The LiveKit CLI and inference clients read these from the environment."""
        for name, value in (
            ("LIVEKIT_URL", self.url),
            ("LIVEKIT_API_KEY", self.api_key),
            ("LIVEKIT_API_SECRET", self.api_secret),
        ):
            os.environ.setdefault(name, value)


class SpeechSettings(BaseModel):
    stt_model: str = "deepgram/nova-3"
    stt_language: str = "multi"
    llm_model: str = "google/gemma-4-31b-it"
    tts_model: str = "inworld/inworld-tts-2"
    tts_voice: str = "Ashley"


class ContextSettings(BaseModel):
    max_tasks: int = 8
    deadline_ms: int = 250
    required_deadline_ms: int = 1000
    memory_pending_s: int = 600


class PromptSettings(BaseModel):
    speech_tokens: int = 150
    identity_tokens: int = 150
    guardrail_tokens: int = 300
    skill_menu_tokens: int = 400
    pinned_skill_tokens: int = 500
    context_tokens: int = 600
    knowledge_tokens: int = 800
    turn_tokens: int = 150
    total_tokens: int = 3050
    token_margin: float = 1.1


class KnowledgeSettings(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_tokens: int = 512
    faq_chunk_tokens: int = 300
    overlap_ratio: float = 0.12
    chunk_threshold: float = 0.60
    tenant_threshold_margin: float = 0.05
    card_threshold: float = 0.80
    top_k: int = 3
    candidate_k: int = 20
    code_bonus: float = 0.1
    injection_tokens: int = 800
    min_tests: int = 10
    gate_pass_rate: float = 0.9
    search_cache_ttl_s: int = 300
    llm_enrichment: bool = False
    enrichment_concurrency: int = 4
    keep_versions: int = 2
    max_url_bytes: int = 2_000_000


class MemorySettings(BaseModel):
    summary_sentences: int = 3
    min_confidence: float = 0.6
    digest_tokens: int = 120
    digest_slots: int = 4
    history_limit: int = 3
    forget_after_days: int = 180
    forget_below_confidence: float = 0.7
    retention_interval_s: int = 86_400
    llm_timeout_s: float = 30.0


class DetectionSettings(BaseModel):
    model: str = "google/gemini-3.1-flash-lite"
    timeout_s: float = 6.0
    thresholds: dict[FlagType, float] = Field(
        default_factory=lambda: {
            FlagType.ABUSE: 0.7,
            FlagType.DISTRESS: 0.7,
            FlagType.MANIPULATION: 0.7,
            FlagType.COMPLIANCE: 0.7,
            FlagType.HUMAN_REQUEST: 0.7,
            FlagType.OFF_TOPIC: 1.0,
            FlagType.NONSENSE: 1.0,
        }
    )
    severe_abuse: float = 0.95
    frustration_note: float = 0.5
    frustration_offer: float = 0.75
    streak_limit: int = 3
    tool_failure_limit: int = 3
    unanswered_limit: int = 2
    briefing_every: int = 3
    briefing_words: int = 60


class WorkSettings(BaseModel):
    summary_sentences: int = 2
    duplicate_similarity: float = 0.80
    duplicate_days: int = 14


class LatencySettings(BaseModel):
    """Each optimisation is on for this percentage of calls unless a tenant
    sets its own. The budget is the plan's first-audio target.

    Measured on web calls: a prepared greeting gave first audio after 3.6 s,
    against 2.4 s without it, because a web caller is already in the room when
    the agent starts, so there is no setup time to use. It is off by default.
    Preemptive generation is LiveKit's own default, kept on."""

    optimisations: dict[Optimisation, int] = Field(
        default_factory=lambda: {
            Optimisation.PREEMPTIVE_GENERATION: 100,
            Optimisation.PREPARED_GREETING: 0,
        }
    )
    greeting_timeout_s: float = 4.0
    first_audio_p50_ms: int = 800
    first_audio_p95_ms: int = 1500
    monitor_interval_s: int = 300
    monitor_window_h: int = 1
    monitor_min_samples: int = 20


class ReleaseSettings(BaseModel):
    """A new release goes to `canary_percent` of calls first. It goes to all
    calls after `canary_calls` calls or `canary_hours`, if no measurement is
    worse than the stable release by more than its tolerance."""

    canary_percent: int = 10
    canary_calls: int = 50
    canary_hours: int = 24
    tolerance_escalation: float = 0.05
    tolerance_refusal: float = 0.05
    tolerance_completion: float = 0.05
    tolerance_p95_ms: float = 0.10
    monitor_interval_s: int = 600


class EvaluationSettings(BaseModel):
    """`concurrency` is how many scenarios replay at once. Each replayed turn
    makes two or three model requests, so raise it only as far as the model
    account's request limit allows."""

    runs: int = 3
    runs_to_pass: int = 2
    concurrency: int = 1
    judge_enabled: bool = False
    judge_model: str = "google/gemini-3.1-flash-lite"
    judge_agreement: float = 0.9
    warn_pass_rate: float = 0.01
    block_pass_rate: float = 0.02
    warn_completion: float = 0.02
    block_completion: float = 0.03
    warn_faithfulness: float = 0.1
    block_faithfulness: float = 0.2
    warn_turn_p95: float = 0.05
    block_turn_p95: float = 0.10
    nightly_interval_s: int = 86_400
    run_days: int = 90


class AnalyticsSettings(BaseModel):
    """Prices are per unit, in the tenant's billing currency. Zero means the
    report gives the units only."""

    gap_distance: float = 0.25
    gap_max_texts: int = 1000
    gap_groups: int = 20
    gap_examples: int = 3
    gap_days: int = 7
    fcr_days: int = 7
    price_prompt_mtok: float = 0.0
    price_cached_mtok: float = 0.0
    price_completion_mtok: float = 0.0
    price_speech_minute: float = 0.0
    price_synthesis_mchar: float = 0.0
    daily_interval_s: int = 86_400
    weekly_interval_s: int = 604_800
    measurement_days: int = 396
    gap_group_days: int = 183


class CacheSettings(BaseModel):
    url: str = "redis://localhost:6379/0"
    memory_entries: int = 10_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    env: str = "local"
    log_level: str = "INFO"
    dev_call_id: uuid.UUID | None = None
    worker_region: str = "unknown"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])

    database: DatabaseSettings
    livekit: LiveKitSettings
    speech: SpeechSettings = SpeechSettings()
    context: ContextSettings = ContextSettings()
    prompt: PromptSettings = PromptSettings()
    knowledge: KnowledgeSettings = KnowledgeSettings()
    memory: MemorySettings = MemorySettings()
    detection: DetectionSettings = DetectionSettings()
    work: WorkSettings = WorkSettings()
    cache: CacheSettings = CacheSettings()
    latency: LatencySettings = LatencySettings()
    releases: ReleaseSettings = ReleaseSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    analytics: AnalyticsSettings = AnalyticsSettings()


class SettingsFactory:
    _instance: Settings | None = None

    @classmethod
    def load(cls) -> Settings:
        if cls._instance is None:
            cls._instance = Settings()
        return cls._instance
