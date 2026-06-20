"""Per-model-family instruction templating.

The agent/reasoning-loop code wraps every (system, user, assistant) turn in
whatever special tokens the loaded GGUF was actually instruction-tuned with.
Previously only Qwen got real ChatML wrapping; every other model fell back to
a bare "Task: {task}" suffix with no chat structure at all, which reliably
produced rambling, unstructured completions on non-Qwen models.

Template selection, in order of preference:

1. Sniff the GGUF's own embedded `tokenizer.chat_template` (a jinja template
   string exposed by llama-cpp-python as `Llama.metadata`) for tokens
   distinctive of a known family. This is the most reliable signal because it
   comes from the model file itself, not its name.
2. Fall back to matching the model_name/path string against the same
   families.
3. Fall back to ChatML for everything else — simple, widely tolerated by
   instruct models even when they weren't literally trained on it, and a
   large improvement over no templating at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ChatTemplate:
    """A model family's native turn-wrapping format.

    `supports_system` is False for families (Mistral, Gemma) whose template
    has no dedicated system role — callers should fold the system prompt into
    the first user turn instead of emitting `wrap_system`.
    """

    name: str
    supports_system: bool
    system_fmt: str
    user_fmt: str
    assistant_open: str
    assistant_close: str   # appended when closing an assistant turn in a multi-turn convo
    stop_token: str        # bare marker (no surrounding whitespace) to stop generation on

    def wrap_system(self, content: str) -> str:
        return self.system_fmt.format(content=content)

    def wrap_user(self, content: str) -> str:
        return self.user_fmt.format(content=content)


CHATML = ChatTemplate(
    name="chatml",
    supports_system=True,
    system_fmt="<|im_start|>system\n{content}<|im_end|>\n",
    user_fmt="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_open="<|im_start|>assistant\n",
    assistant_close="<|im_end|>\n",
    stop_token="<|im_end|>",
)

LLAMA3 = ChatTemplate(
    name="llama3",
    supports_system=True,
    system_fmt="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user_fmt="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
    assistant_open="<|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_close="<|eot_id|>",
    stop_token="<|eot_id|>",
)

GEMMA = ChatTemplate(
    name="gemma",
    supports_system=False,  # no system role; caller folds system text into the first user turn
    system_fmt="{content}",
    user_fmt="<start_of_turn>user\n{content}<end_of_turn>\n",
    assistant_open="<start_of_turn>model\n",
    assistant_close="<end_of_turn>\n",
    stop_token="<end_of_turn>",
)

MISTRAL = ChatTemplate(
    name="mistral",
    supports_system=False,  # no system role; folded into the first [INST] block
    system_fmt="{content}",
    user_fmt="[INST] {content} [/INST]",
    assistant_open="",
    assistant_close="</s>",
    stop_token="</s>",
)

PHI3 = ChatTemplate(
    name="phi3",
    supports_system=True,
    system_fmt="<|system|>\n{content}<|end|>\n",
    user_fmt="<|user|>\n{content}<|end|>\n",
    assistant_open="<|assistant|>\n",
    assistant_close="<|end|>\n",
    stop_token="<|end|>",
)

_FAMILIES = [LLAMA3, MISTRAL, GEMMA, PHI3, CHATML]
_BY_NAME = {t.name: t for t in _FAMILIES}

# Distinctive substrings to sniff out of the GGUF's own embedded jinja
# chat_template -- checked in this order so more-specific markers
# (e.g. llama3's <|eot_id|>) are matched before falling through to chatml.
_TEMPLATE_SNIFFS = [
    ("llama3", ("<|start_header_id|>", "<|eot_id|>")),
    ("mistral", ("[INST]", "[/INST]")),
    ("gemma", ("<start_of_turn>",)),
    ("phi3", ("<|system|>", "<|end|>")),
    ("chatml", ("<|im_start|>",)),
]

# Same families, sniffed from the model name/path when GGUF metadata isn't
# available or its chat_template doesn't match anything known.
_NAME_SNIFFS = [
    ("llama3", ("llama-3", "llama3")),
    ("mistral", ("mistral", "mixtral")),
    ("gemma", ("gemma",)),
    ("phi3", ("phi-3", "phi3")),
    ("chatml", ("qwen", "yi-", "deepseek", "chatml")),
]


def detect_template(model_name: str, metadata: Optional[Dict[str, str]] = None) -> ChatTemplate:
    """Pick the wrapping format for the active model.

    Prefers sniffing the GGUF's own embedded chat_template (a model-file-level
    signal that doesn't depend on how the file happens to be named); falls
    back to the model name; falls back to ChatML.
    """
    jinja = (metadata or {}).get("tokenizer.chat_template", "") or ""
    if jinja:
        for family, needles in _TEMPLATE_SNIFFS:
            if any(n in jinja for n in needles):
                return _BY_NAME[family]

    lowered = model_name.lower()
    for family, needles in _NAME_SNIFFS:
        if any(n in lowered for n in needles):
            return _BY_NAME[family]

    return CHATML
