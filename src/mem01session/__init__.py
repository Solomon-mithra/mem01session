"""OpenAI Agents SDK embedded session adapter for mem01."""

from .metrics import ContextEstimate, estimate_prompt_context
from .runtime import (
    EmbeddedMem01Runtime,
    SharedRuntimeLease,
    acquire_shared_runtime,
    close_shared_runtimes,
)
from .session import Mem01MemoryError, Mem01Session

memSession = Mem01Session

__all__ = [
    "ContextEstimate",
    "EmbeddedMem01Runtime",
    "Mem01Session",
    "Mem01MemoryError",
    "memSession",
    "SharedRuntimeLease",
    "acquire_shared_runtime",
    "close_shared_runtimes",
    "estimate_prompt_context",
]
__version__ = "0.1.0"
