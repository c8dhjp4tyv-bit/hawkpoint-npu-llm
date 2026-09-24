"""Host-side stop sequences and log-probability options for completions.

Like :mod:`runtime.sampling`, this is pure Python/NumPy with no ``aie``
import, so it is validated at the HTTP boundary, revalidated inside the
inference worker, and tested without hardware.
"""

import numpy as np


# OpenAI accepts at most four stop sequences and at most 20 top log
# probabilities per token.
MAX_STOP_SEQUENCES = 4
MAX_STOP_LENGTH = 256
MAX_TOP_LOGPROBS = 20


def normalize_stop(value):
    """Validate an OpenAI ``stop`` value and return a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("stop must be a string or an array of strings")
    if len(value) > MAX_STOP_SEQUENCES:
        raise ValueError(
            f"stop accepts at most {MAX_STOP_SEQUENCES} sequences"
        )
    stops = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError("each stop sequence must be a non-empty string")
        if len(item) > MAX_STOP_LENGTH:
            raise ValueError(
                f"stop sequences cannot exceed {MAX_STOP_LENGTH} characters"
            )
        if item not in stops:
            stops.append(item)
    return tuple(stops)


def normalize_logprobs(value):
    """Validate a top-logprob count; ``None`` disables log probabilities."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("top_logprobs must be an integer")
    if not 0 <= value <= MAX_TOP_LOGPROBS:
        raise ValueError(
            f"top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}"
        )
    return value


class StopMatcher:
    """Find stop sequences in streamed text without emitting any part of one.

    Text that could still be the start of a stop sequence is held back until
    the next piece of text either completes the sequence or rules it out. When
    a stop sequence is found, everything from its first character onward is
    discarded, matching the OpenAI API.
    """

    def __init__(self, stops=()):
        self.stops = tuple(stops)
        self.matched = None
        self._pending = ""

    def feed(self, text):
        """Consume ``text`` and return the part that can be emitted now."""
        if self.matched is not None:
            return ""
        if not self.stops:
            return text
        buffer = self._pending + text
        found = None
        for stop in self.stops:
            index = buffer.find(stop)
            if index >= 0 and (found is None or index < found[0]):
                found = (index, stop)
        if found is not None:
            self.matched = found[1]
            self._pending = ""
            return buffer[: found[0]]
        held = 0
        for stop in self.stops:
            for length in range(min(len(stop) - 1, len(buffer)), held, -1):
                if buffer.endswith(stop[:length]):
                    held = length
                    break
        self._pending = buffer[len(buffer) - held :] if held else ""
        return buffer[: len(buffer) - held]

    def flush(self):
        """Release held text once generation ends without a match."""
        pending, self._pending = self._pending, ""
        return "" if self.matched is not None else pending


def log_softmax(logits):
    """Return float64 log probabilities for one logit vector."""
    scores = np.asarray(logits, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("logits contain no finite value")
    peak = scores[finite].max()
    shifted = np.where(finite, scores - peak, -np.inf)
    return shifted - np.log(np.exp(shifted).sum())
