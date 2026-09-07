"""Host-side token sampling for the XDNA1 decode loop.

The NPU produces one logit vector per position; token selection is a host
operation in every supported placement (all-NPU, hybrid, and CPU-only). This
module is therefore pure NumPy and imports nothing from ``aie``, so it can be
tested without hardware.

Selection order matches the convention used by llama.cpp and vLLM:

1. repetition, presence, and frequency penalties over the token history,
2. temperature scaling,
3. top-k truncation,
4. top-p (nucleus) truncation,
5. a multinomial draw from the remaining renormalized distribution.

``temperature == 0`` means greedy decoding and is the default, so the existing
exact-token release gates keep selecting ``argmax`` unless a request opts in.
"""

from dataclasses import asdict, dataclass
import secrets

import numpy as np


# Bounds are enforced at the API boundary and again here so a decoder used
# directly from Python cannot be driven with values that make the distribution
# meaningless (a negative temperature flips the ranking, top_p == 0 empties the
# candidate set).
TEMPERATURE_RANGE = (0.0, 2.0)
TOP_P_RANGE = (0.0, 1.0)
REPETITION_PENALTY_RANGE = (0.1, 2.0)
PENALTY_RANGE = (-2.0, 2.0)
SEED_RANGE = (0, 2**63 - 1)


class SamplingError(ValueError):
    """An invalid sampling parameter was supplied."""


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplingError(f"{name} must be a number")
    value = float(value)
    if not np.isfinite(value):
        raise SamplingError(f"{name} must be finite")
    return value


def _bounded(value, name, bounds, *, exclusive_low=False):
    value = _number(value, name)
    low, high = bounds
    if value < low or value > high or (exclusive_low and value == low):
        edge = "exclusive" if exclusive_low else "inclusive"
        raise SamplingError(
            f"{name} must be between {low} ({edge}) and {high} (inclusive)"
        )
    return value


@dataclass(frozen=True)
class SamplingParams:
    """Validated sampling configuration for one completion request."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None

    @property
    def greedy(self):
        """True when the parameters cannot change the argmax selection.

        ``top_k == 1`` collapses the candidate set to the highest-scoring token,
        but only when no penalty can reorder the logits first.
        """
        penalized = (
            self.repetition_penalty != 1.0
            or self.presence_penalty != 0.0
            or self.frequency_penalty != 0.0
        )
        return not penalized and (self.temperature == 0.0 or self.top_k == 1)

    @property
    def deterministic(self):
        """True when repeated runs of the same request select the same tokens."""
        return self.greedy or self.temperature == 0.0 or self.seed is not None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        """Rebuild from a transport dict, re-validating every field.

        Sampling parameters cross a process boundary to reach the inference
        worker; they are revalidated on arrival rather than trusted.
        """
        if values is None:
            return cls()
        if not isinstance(values, dict):
            raise SamplingError("sampling parameters must be an object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise SamplingError(
                "unsupported sampling parameters: "
                + ", ".join(sorted(unknown))
            )
        return cls.build(**values)

    @classmethod
    def build(
        cls,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        seed=None,
    ):
        """Validate raw request values, treating ``None`` as "use the default"."""
        if temperature is None:
            temperature = 0.0
        else:
            temperature = _bounded(temperature, "temperature", TEMPERATURE_RANGE)
        if top_p is None:
            top_p = 1.0
        else:
            top_p = _bounded(top_p, "top_p", TOP_P_RANGE, exclusive_low=True)
        if top_k is None:
            top_k = 0
        elif isinstance(top_k, bool) or not isinstance(top_k, int):
            raise SamplingError("top_k must be an integer")
        elif top_k < 0:
            raise SamplingError("top_k cannot be negative")
        if repetition_penalty is None:
            repetition_penalty = 1.0
        else:
            repetition_penalty = _bounded(
                repetition_penalty,
                "repetition_penalty",
                REPETITION_PENALTY_RANGE,
            )
        presence_penalty = (
            0.0
            if presence_penalty is None
            else _bounded(presence_penalty, "presence_penalty", PENALTY_RANGE)
        )
        frequency_penalty = (
            0.0
            if frequency_penalty is None
            else _bounded(frequency_penalty, "frequency_penalty", PENALTY_RANGE)
        )
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise SamplingError("seed must be an integer")
            if not SEED_RANGE[0] <= seed <= SEED_RANGE[1]:
                raise SamplingError(
                    f"seed must be between {SEED_RANGE[0]} and {SEED_RANGE[1]}"
                )
        return cls(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )

    @classmethod
    def from_request(cls, request):
        """Build from an OpenAI-style chat completion request body."""
        return cls.build(
            temperature=request.get("temperature"),
            top_p=request.get("top_p"),
            top_k=request.get("top_k"),
            repetition_penalty=request.get("repetition_penalty"),
            presence_penalty=request.get("presence_penalty"),
            frequency_penalty=request.get("frequency_penalty"),
            seed=request.get("seed"),
        )


GREEDY = SamplingParams()


def apply_penalties(logits, history, params):
    """Penalize previously seen tokens in place-safe fashion.

    ``repetition_penalty`` follows the CTRL/Hugging Face convention: it divides
    positive logits and multiplies negative ones, so a penalty above 1 always
    moves a score toward -inf regardless of sign. ``presence_penalty`` and
    ``frequency_penalty`` follow the OpenAI convention and subtract a constant
    or a count-scaled term.
    """
    if not len(history):
        return logits
    if (
        params.repetition_penalty == 1.0
        and params.presence_penalty == 0.0
        and params.frequency_penalty == 0.0
    ):
        return logits
    tokens, counts = np.unique(
        np.asarray(history, dtype=np.int64), return_counts=True
    )
    # A history token outside the vocabulary would silently wrap with negative
    # indexing, penalizing an unrelated token.
    in_range = (tokens >= 0) & (tokens < logits.shape[0])
    tokens = tokens[in_range]
    counts = counts[in_range].astype(np.float32)
    if not tokens.shape[0]:
        return logits
    selected = logits[tokens]
    if params.repetition_penalty != 1.0:
        selected = np.where(
            selected > 0,
            selected / params.repetition_penalty,
            selected * params.repetition_penalty,
        )
    selected = selected - params.frequency_penalty * counts
    selected = selected - params.presence_penalty
    logits[tokens] = selected
    return logits


def filter_top_k(logits, top_k):
    """Mask every logit outside the ``top_k`` highest scores."""
    if top_k <= 0 or top_k >= logits.shape[0]:
        return logits
    threshold = np.partition(logits, -top_k)[-top_k]
    logits[logits < threshold] = -np.inf
    return logits


def filter_top_p(logits, top_p):
    """Mask the tail of the distribution outside the ``top_p`` nucleus.

    The highest-probability token is always retained, so a very small ``top_p``
    degenerates to greedy selection instead of an empty candidate set.
    """
    if top_p >= 1.0:
        return logits
    order = np.argsort(logits)[::-1]
    probabilities = _softmax(logits[order])
    cumulative = np.cumsum(probabilities)
    # Keep every token up to and including the one that crosses the threshold.
    keep = cumulative - probabilities < top_p
    keep[0] = True
    logits[order[~keep]] = -np.inf
    return logits


def _softmax(scores):
    finite = scores[np.isfinite(scores)]
    peak = finite.max() if finite.size else 0.0
    shifted = np.exp(
        scores - peak, where=np.isfinite(scores), out=np.zeros_like(scores)
    )
    total = shifted.sum()
    if not np.isfinite(total) or total <= 0:
        # Every candidate underflowed; fall back to the single best score.
        result = np.zeros_like(scores)
        result[int(np.argmax(scores))] = 1.0
        return result
    return shifted / total


class Sampler:
    """Stateful token selector for one completion.

    One instance owns the request's random stream, so a request that supplies a
    ``seed`` reproduces its output exactly while an unseeded request still
    reports the seed it actually used.
    """

    def __init__(self, params=GREEDY):
        self.params = params
        self.seed = params.seed
        if not params.greedy and self.seed is None:
            self.seed = secrets.randbelow(SEED_RANGE[1] + 1)
        self._rng = None if params.greedy else np.random.default_rng(self.seed)

    @property
    def greedy(self):
        return self.params.greedy

    def select(self, logits, history=()):
        """Return the next token id for ``logits`` given the tokens so far."""
        scores = np.asarray(logits, dtype=np.float32)
        if self.params.greedy:
            return int(np.argmax(scores))
        scores = apply_penalties(scores.copy(), history, self.params)
        if self.params.temperature == 0.0:
            return int(np.argmax(scores))
        scores = scores / self.params.temperature
        scores = filter_top_k(scores, self.params.top_k)
        scores = filter_top_p(scores, self.params.top_p)
        probabilities = _softmax(scores).astype(np.float64)
        # np.random.Generator.choice rejects a probability vector that does not
        # sum to one within tolerance; float32 accumulation can drift past it.
        probabilities /= probabilities.sum()
        return int(self._rng.choice(probabilities.shape[0], p=probabilities))

    def stats(self):
        """Sampling metadata reported alongside generation statistics."""
        return {
            "mode": "greedy" if self.params.greedy else "sampled",
            "seed": self.seed,
            **self.params.to_dict(),
        }
