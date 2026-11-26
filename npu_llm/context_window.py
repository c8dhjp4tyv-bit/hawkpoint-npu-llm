"""Context-window geometry shared by the decoder graphs and their host runtime.

Attention does not hold a whole head's K/V cache in a tile's data memory. The
cache is streamed from DDR in fixed position tiles, and the hub tile keeps only
one tile of keys and one tile of values at a time plus the per-head softmax
bookkeeping. ``KV_TILE`` is therefore the streaming granularity, not the
context length; a 64-token model simply uses one partially filled tile.

Dependency-free so both the designs and the model catalog can import it.
"""

import os

DEFAULT_CONTEXT_LENGTH = 8192
MAX_CONTEXT_LENGTH = 8192
KV_TILE = 128
# Streaming granularity: the hub's tile loop and the DMA prefix a dispatch
# streams are fixed when the graph is compiled, so the host grows the window
# in these steps instead of compiling one graph per position.
CONTEXT_BUCKETS = (1, 4, 16, 64)


def validate_context_length(value):
    """Return ``value`` as a context length, or raise if the graphs cannot."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("context_length must be an integer")
    if not 1 <= value <= MAX_CONTEXT_LENGTH:
        raise ValueError(
            f"context_length must be between 1 and {MAX_CONTEXT_LENGTH}"
        )
    return value


def resolve_context_length(declared=None):
    """Context a decoder runs with: ``HAWKPOINT_CONTEXT``, then the model.

    Callers that need a number before a decoder exists (the API's per-request
    token limit) use this too, so the bound they advertise is the one the
    decoder will actually apply. An unusable override raises, exactly where the
    decoder would; the API turns that into the model's own declaration.
    """
    override = os.environ.get("HAWKPOINT_CONTEXT")
    if override:
        return validate_context_length(int(override))
    return validate_context_length(int(declared or DEFAULT_CONTEXT_LENGTH))


def context_tiles(context_length):
    """Number of streamed K/V tiles the cache reserves for ``context_length``."""
    return max(1, -(-int(context_length) // KV_TILE))
