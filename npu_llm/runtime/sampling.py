"""Host-side token sampling for the XDNA1 decode loop.

The NPU produces one logit vector per position; token selection is a host
operation in every supported placement (all-NPU, hybrid, and CPU-only). This
module is therefore pure NumPy and imports nothing from ``aie``, so it can be
tested without hardware.

Selection order matches the convention used by llama.cpp and vLLM:

1. repetition, presence, and frequency penalties over the token history,
2. the request's ``logit_bias`` adjustments,
3. temperature scaling,
4. top-k truncation,
5. top-p (nucleus) truncation,
6. a multinomial draw from the remaining renormalized distribution.

``temperature == 0`` means greedy decoding and is the default, so the existing
exact-token release gates keep selecting ``argmax`` unless a request opts in.
"""

from dataclasses import asdict, dataclass, field
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
LOGIT_BIAS_RANGE = (-100.0, 100.0)
SEED_RANGE = (0, 2**63 - 1)
# A request body is already size-capped, but a per-token bias loop runs once
# per decode step; keep the map small enough that this cannot become a
# denial-of-service vector.
MAX_LOGIT_BIAS_ENTRIES = 1024


class SamplingError(ValueError):
    """An invalid sampling parameter was supplied."""


class _FrozenBias(dict):
    """A validated ``logit_bias`` map that refuses further mutation.

    ``dict`` stays the base class so ``dataclasses.asdict`` and ``json.dumps``
    can serialize the field without a custom encoder. The immutable view keeps
    a caller from writing a NaN or an out-of-range value after validation.
    """

    def _read_only(self, *args, **kwargs):
        raise TypeError("logit_bias is immutable")

    __setitem__ = _read_only
    __delitem__ = _read_only
    clear = _read_only
    pop = _read_only
    popitem = _read_only
    setdefault = _read_only
    update = _read_only

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplingError(f"{name} must be a number")
    try:
        value = float(value)
    except OverflowError:
        raise SamplingError(f"{name} must be finite") from None
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


def _normalize_logit_bias(value):
    """Validate an OpenAI ``logit_bias`` map and return ``{token_id: bias}``.

    JSON object keys are always strings, while callers that build parameters
    directly use integers. Both forms are accepted, normalized to integers, and
    a bias of exactly zero is dropped so an all-zero map still compares equal to
    the greedy defaults.
    """
    if value is None:
        return _FrozenBias()
    if not isinstance(value, dict):
        raise SamplingError("logit_bias must be an object")
    if len(value) > MAX_LOGIT_BIAS_ENTRIES:
        raise SamplingError(
            f"logit_bias accepts at most {MAX_LOGIT_BIAS_ENTRIES} entries"
        )
    normalized = {}
    for key, raw_bias in value.items():
        if isinstance(key, bool):
            raise SamplingError("logit_bias keys must be token ids")
        if isinstance(key, int):
            token_id = key
        elif isinstance(key, str):
            if not key.isascii() or not key.isdigit():
                raise SamplingError(
                    f"logit_bias key {key!r} is not a token id"
                )
            token_id = int(key, 10)
        else:
            raise SamplingError("logit_bias keys must be token ids")
        if token_id < 0:
            raise SamplingError("logit_bias token ids cannot be negative")
        bias = _bounded(
            raw_bias, f"logit_bias[{token_id}]", LOGIT_BIAS_RANGE
        )
        if bias != 0.0:
            normalized[token_id] = bias
        else:
            # A later duplicate key that normalizes to the same token id wins
            # even when its value is zero, so an explicit no-op removes the
            # earlier entry instead of silently keeping it.
            normalized.pop(token_id, None)
    return _FrozenBias(normalized)


@dataclass(frozen=True)
class SamplingParams:
    """Validated sampling configuration for one completion request."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] = field(default_factory=_FrozenBias)
    seed: int | None = None

    def __post_init__(self):
        # Direct construction bypasses ``build``; normalize here as well so a
        # ``SamplingParams`` instance is always validated and immutable.
        object.__setattr__(
            self, "logit_bias", _normalize_logit_bias(self.logit_bias)
        )

    def _reorder_free(self):
        """True when nothing before top-k/top-p can change the logit order."""
        return (
            self.repetition_penalty == 1.0
            and self.presence_penalty == 0.0
            and self.frequency_penalty == 0.0
            and (self.temperature == 0.0 or self.top_k == 1)
        )

    @property
    def greedy(self):
        """True when the parameters cannot change the argmax selection.

        ``top_k == 1`` collapses the candidate set to the highest-scoring token,
        but only when no penalty or logit bias can reorder the logits first.
        """
        return self._reorder_free() and not self.logit_bias

    def greedy_for(self, vocab_size):
        """Like :attr:`greedy`, ignoring bias entries outside the vocabulary.

        ``apply_logit_bias`` skips out-of-range token ids, so a map that only
        names such ids cannot reorder anything and must keep the argmax path.
        """
        if not self._reorder_free():
            return False
        return not any(
            0 <= token_id < vocab_size for token_id in self.logit_bias
        )

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
        logit_bias=None,
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
            logit_bias=logit_bias,
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
            logit_bias=request.get("logit_bias"),
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


def apply_logit_bias(logits, logit_bias):
    """Add the request's per-token bias to ``logits`` in place.

    OpenAI semantics are a plain addition to the logit, so a bias of ``-100``
    effectively bans a token and ``+100`` forces it. Token ids outside the
    model vocabulary are ignored rather than wrapping through negative indexing
    or raising, matching how penalties treat out-of-range history.
    """
    if not logit_bias:
        return logits
    for token_id, bias in logit_bias.items():
        token_id = int(token_id)
        if 0 <= token_id < logits.shape[0]:
            logits[token_id] += bias
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

    def __init__(self, params=GREEDY, vocab_size=None):
        self.params = params
        self.vocab_size = vocab_size
        self.seed = params.seed
        if not self.greedy and self.seed is None:
            self.seed = secrets.randbelow(SEED_RANGE[1] + 1)
        self._rng = None if self.greedy else np.random.default_rng(self.seed)

    @property
    def greedy(self):
        if self.vocab_size is None:
            return self.params.greedy
        return self.params.greedy_for(self.vocab_size)

    def select(self, logits, history=()):
        """Return the next token id for ``logits`` given the tokens so far."""
        scores = np.asarray(logits, dtype=np.float32)
        if self.greedy:
            return int(np.argmax(scores))
        scores = apply_penalties(scores.copy(), history, self.params)
        scores = apply_logit_bias(scores, self.params.logit_bias)
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
            "mode": "greedy" if self.greedy else "sampled",
            "seed": self.seed,
            **self.params.to_dict(),
        }
