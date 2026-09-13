import pytest

from personas.prompt import (
    PROMPT_VERSION,
    parse_sections,
    render_attributes,
    render_prompt,
    render_target,
    render_training_text,
)
from personas.schema import SECTIONS, PersonaAttributes

ATTRS = PersonaAttributes(
    name="Marcos Antunes",
    sex="Masculino",
    age=22,
    marital_status="Solteiro",
    education_level="Fundamental completo e médio incompleto",
    occupation="Operador de instalação ou máquina ou montador",
    municipality="São Pedro de Alcântara",
    state="Santa Catarina",
    country="Brasil",
)

ROW = {column: f"texto de {column}" for _, column in SECTIONS}


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained("unsloth/Qwen3-8B")


def test_prompt_version_is_set():
    assert PROMPT_VERSION


def test_attributes_block_contains_every_attribute():
    rendered = render_attributes(ATTRS)
    for value in ("Marcos Antunes", "Masculino", "22", "Solteiro",
                  "São Pedro de Alcântara", "Santa Catarina", "Brasil"):
        assert value in rendered


def test_target_has_six_sections_in_order():
    parsed = parse_sections(render_target(ROW))
    assert list(parsed) == [heading for heading, _ in SECTIONS]


@pytest.mark.slow
def test_training_text_starts_with_the_inference_prompt(tokenizer):
    """The model must see at inference exactly the prefix it was trained on."""
    prompt = render_prompt(ATTRS, tokenizer)
    training_text = render_training_text(ATTRS, ROW, tokenizer)
    assert training_text.startswith(prompt), (
        "Train/inference prompt drift.\n"
        f"prompt tail:   {prompt[-120:]!r}\n"
        f"training text: {training_text[len(prompt) - 120:len(prompt) + 60]!r}"
    )


@pytest.mark.slow
def test_no_thinking_block_is_requested(tokenizer):
    prompt = render_prompt(ATTRS, tokenizer)
    # enable_thinking=False may emit an EMPTY think block; it must never be an open one.
    assert "<think>" not in prompt or "<think>\n\n</think>" in prompt
