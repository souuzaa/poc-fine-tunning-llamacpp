from personas.grounding import check_format, is_portuguese, score_grounding
from personas.schema import SECTIONS

VALID = "\n\n".join(f"## {heading}\nconteúdo de {heading}." for heading, _ in SECTIONS)

ATTRS = {
    "name": "Marcos Antunes",
    "sex": "Masculino",
    "age": 22,
    "marital_status": "Solteiro",
    "education_level": "Fundamental completo",
    "occupation": "Operador de máquina",
    "municipality": "São Pedro de Alcântara",
    "state": "Santa Catarina",
    "country": "Brasil",
}


def test_valid_output_passes_every_format_check():
    result = check_format(VALID)
    assert result["valid"] is True
    assert result["has_all_sections"] is True
    assert result["order_correct"] is True
    assert result["think_leak"] is False


def test_missing_section_fails():
    truncated = VALID.split("## Habilidades")[0]
    result = check_format(truncated)
    assert result["valid"] is False
    assert result["has_all_sections"] is False


def test_out_of_order_sections_fail():
    headings = [h for h, _ in SECTIONS]
    swapped = [headings[1], headings[0], *headings[2:]]
    text = "\n\n".join(f"## {h}\ncorpo." for h in swapped)
    assert check_format(text)["order_correct"] is False


def test_think_leak_is_detected_and_fails_validity():
    result = check_format("<think>hmm</think>\n\n" + VALID)
    assert result["think_leak"] is True
    assert result["valid"] is False


def test_grounding_finds_mentioned_attributes():
    text = ("## Síntese\nMarcos Antunes, 22 anos, mora em São Pedro de Alcântara, "
            "Santa Catarina, e trabalha como operador de máquina.")
    scores = score_grounding(text, ATTRS)
    assert scores["name"] is True
    assert scores["municipality"] is True
    assert scores["state"] is True
    assert scores["age"] is True
    assert scores["occupation"] is True


def test_grounding_misses_absent_attributes():
    scores = score_grounding("## Síntese\nUma pessoa genérica qualquer.", ATTRS)
    assert scores["name"] is False
    assert scores["municipality"] is False
    assert scores["recall"] == 0.0


def test_portuguese_detection():
    assert is_portuguese("Ele trabalha na fábrica e mora com a família dele.") is True
    assert is_portuguese("He works at the factory and lives with his family.") is False
