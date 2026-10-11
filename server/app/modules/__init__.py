"""Importing this package registers every table on the shared metadata."""

from app.modules.accounts import models as accounts
from app.modules.calls import models as calls
from app.modules.catalog import models as catalog
from app.modules.config import models as config
from app.modules.detection import models as detection
from app.modules.evaluation import models as evaluation
from app.modules.jobs import models as jobs
from app.modules.knowledge import models as knowledge
from app.modules.memory import models as memory
from app.modules.releases import models as releases
from app.modules.work import models as work

__all__ = [
    "accounts",
    "calls",
    "catalog",
    "config",
    "detection",
    "evaluation",
    "jobs",
    "knowledge",
    "memory",
    "releases",
    "work",
]
