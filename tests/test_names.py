import pytest

from personas.names import extract_name


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Marcos Antunes é um operador técnico organizado.", "Marcos Antunes"),
        ("Ana Paula dos Santos trabalha como enfermeira.", "Ana Paula dos Santos"),
        ("João da Silva mora em Recife.", "João da Silva"),
        ("Luíza Gonçalves, 34 anos, é professora.", "Luíza Gonçalves"),
    ],
)
def test_extracts_leading_proper_name(text, expected):
    assert extract_name(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "uma pessoa comum que trabalha muito",
        "Ele é um operador técnico.",
    ],
)
def test_returns_none_when_no_name_present(text):
    assert extract_name(text) is None


def test_does_not_swallow_the_rest_of_the_sentence():
    name = extract_name("Carlos Eduardo Ferreira Lima Souza Pereira é engenheiro.")
    assert name is not None
    assert len(name.split()) <= 4
