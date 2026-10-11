from __future__ import annotations

from enum import IntEnum, StrEnum


class EnrollmentStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class TaskStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"

    @property
    def is_open(self) -> bool:
        return self in (TaskStatus.OPEN, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED)


class CallStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class CallChannel(StrEnum):
    WEB = "web"
    PHONE = "phone"
    REPLAY = "replay"


class IdentityLevel(IntEnum):
    UNKNOWN = 0
    CLAIMED = 1
    RECOGNISED = 2
    VERIFIED = 3


class Eagerness(StrEnum):
    """How eager the agent should be to respond to a user prompt."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LatencyClass(StrEnum):
    """How long the agent should wait before responding to a user prompt."""

    A = "A"
    B = "B"
    C = "C"

    @property
    def p95_limit_ms(self) -> int | None:
        return {LatencyClass.A: 150, LatencyClass.B: 2500, LatencyClass.C: None}.get(self)


class SideEffect(StrEnum):
    """Side effects that can be triggered by the agent."""

    NONE = "none"
    WRITE = "write"


class ToolStatus(StrEnum):
    """Status of a tool."""

    OK = "ok"
    EMPTY = "empty"
    REFUSED = "refused"
    ACCEPTED = "accepted"
    FAILED = "failed"


class JobStatus(StrEnum):
    """Status of a job."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class CallOutcome(StrEnum):
    ESCALATED = "escalated"
    CALLBACK_PROMISED = "callback_promised"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"
    UNRESOLVED = "unresolved"


class Speaker(StrEnum):
    CALLER = "caller"
    AGENT = "agent"
    SYSTEM = "system"


class SlotSource(StrEnum):
    SYSTEM = "system"
    STATED = "stated"
    INFERRED = "inferred"

    @property
    def rank(self) -> int:
        return {SlotSource.SYSTEM: 3, SlotSource.STATED: 2, SlotSource.INFERRED: 1}[self]


class SlotValueType(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"


class FlagType(StrEnum):
    ABUSE = "abuse"
    DISTRESS = "distress"
    OFF_TOPIC = "off_topic"
    NONSENSE = "nonsense"
    MANIPULATION = "manipulation"
    COMPLIANCE = "compliance"
    HUMAN_REQUEST = "human_request"
    GUARD = "guard"


class Severity(StrEnum):
    INFO = "info"
    NEXT_TURN = "next_turn"
    NOW = "now"


class Relevance(StrEnum):
    ON_TASK = "on_task"
    SMALL_TALK = "small_talk"
    OFF_TOPIC = "off_topic"
    NONSENSE = "nonsense"


class Escalation(StrEnum):
    NONE = "none"
    OFFERED = "offered"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class Priority(StrEnum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"


class ReviewStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class Optimisation(StrEnum):
    PREEMPTIVE_GENERATION = "preemptive_generation"
    PREPARED_GREETING = "prepared_greeting"


class ReleaseStatus(StrEnum):
    DRAFT = "draft"
    STAGED = "staged"
    LIVE = "live"
    RETIRED = "retired"


class GateDecision(StrEnum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class Mark(StrEnum):
    GOOD = "good"
    BAD = "bad"
