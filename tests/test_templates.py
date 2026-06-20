"""Tests for per-model-family instruction templating."""

from cacheflow.templates import (
    CHATML, GEMMA, LLAMA3, MISTRAL, PHI3, detect_template,
)


def test_detect_by_name_qwen():
    assert detect_template("qwen2.5-coder:7b") is CHATML


def test_detect_by_name_llama3():
    assert detect_template("llama-3.1-8b-instruct") is LLAMA3


def test_detect_by_name_mistral():
    assert detect_template("mistral-7b-instruct-v0.3") is MISTRAL


def test_detect_by_name_gemma():
    assert detect_template("gemma-2-9b-it") is GEMMA


def test_detect_by_name_phi3():
    assert detect_template("phi-3-mini-4k-instruct") is PHI3


def test_detect_unknown_falls_back_to_chatml():
    assert detect_template("some-custom-finetune-v2") is CHATML


def test_metadata_chat_template_takes_priority_over_name():
    # Filename says mistral, but the GGUF's own embedded template is llama3-style --
    # the embedded template should win since it's the authoritative, model-file-level signal.
    metadata = {"tokenizer.chat_template": "...{{ '<|start_header_id|>' }}...{{ '<|eot_id|>' }}..."}
    assert detect_template("my-mistral-rename.gguf", metadata) is LLAMA3


def test_metadata_chatml_sniff():
    metadata = {"tokenizer.chat_template": "{{ '<|im_start|>' + message['role'] }}"}
    assert detect_template("unnamed-model", metadata) is CHATML


def test_metadata_empty_falls_back_to_name():
    assert detect_template("gemma-2b", metadata={}) is GEMMA


def test_chatml_wrap_system_and_user():
    assert CHATML.wrap_system("sys") == "<|im_start|>system\nsys<|im_end|>\n"
    assert CHATML.wrap_user("hi") == "<|im_start|>user\nhi<|im_end|>\n"
    assert CHATML.supports_system is True


def test_llama3_wrap():
    assert LLAMA3.wrap_user("hi") == "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
    assert LLAMA3.supports_system is True


def test_mistral_has_no_system_role():
    assert MISTRAL.supports_system is False
    assert MISTRAL.wrap_user("hi") == "[INST] hi [/INST]"


def test_gemma_has_no_system_role():
    assert GEMMA.supports_system is False
    assert GEMMA.wrap_user("hi") == "<start_of_turn>user\nhi<end_of_turn>\n"
