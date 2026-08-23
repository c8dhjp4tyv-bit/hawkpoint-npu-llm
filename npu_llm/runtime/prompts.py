"""Default system prompts for the supported checkpoint families.

Kept free of heavy imports so both the runtime and the terminal CLI can share
one definition of "which identity does this checkpoint expect", and so the
choice stays unit-testable without NumPy or an MLIR-AIE environment.
"""


DEFAULT_SYSTEM_PROMPTS = {
    "qwen2": (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    ),
    "llama": "You are a helpful AI assistant named SmolLM.",
}

FALLBACK_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPTS["llama"]


def default_system_prompt(model_family):
    """Return the system prompt the given checkpoint family was trained with."""
    return DEFAULT_SYSTEM_PROMPTS.get(model_family, FALLBACK_SYSTEM_PROMPT)


def seed_messages(system_prompt=None):
    """Return the opening conversation for a CLI session.

    An explicit ``system_prompt`` is used verbatim. When it is omitted the
    conversation starts empty so the tokenizer can insert the family default
    for the checkpoint that is actually loaded.
    """
    if system_prompt:
        return [{"role": "system", "content": system_prompt}]
    return []
