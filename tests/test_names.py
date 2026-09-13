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


# --- mononyms and cross-field fallback -------------------------------------------------
# Roughly 5-6% of rows carry a single-token name, bury the name mid-sentence, or omit it
# from `persona` entirely. A real name repeats across the narrative columns; an adjective
# that happens to open a sentence does not. That is the signal used to confirm mononyms.


def test_single_token_name_confirmed_by_another_column():
    name = extract_name(
        "Irani é um operador logístico de 40 anos.",
        ["Irani trabalha no centro de distribuição.", "Irani cresceu em Maringá."],
    )
    assert name == "Irani"


def test_single_token_name_rejected_without_confirmation():
    assert extract_name("Irani é um operador logístico.") is None


def test_adjective_opener_is_not_mistaken_for_a_mononym():
    name = extract_name(
        "Energético, curioso e sociável, Luís Ricardo Aguiar combina disciplina e fé.",
        ["Luís Ricardo Aguiar atua como fotógrafo.", "Luís Ricardo Aguiar nasceu no Recife."],
    )
    assert name == "Luís Ricardo Aguiar"


def test_falls_back_to_another_column_when_persona_has_no_name():
    name = extract_name(
        "Um bombeiro militar metódico e secular, que alia a ordem das emergências.",
        ["Rogério Salles atua no corpo de bombeiros.", "Rogério Salles cresceu em Santos."],
    )
    assert name == "Rogério Salles"


def test_mid_sentence_name_recovered_from_another_column():
    name = extract_name(
        "Mestre de obras calmo, competitivo e comunitário, Carlos une fé e tradição.",
        ["Carlos Henrique Prado lidera equipes na construção civil."],
    )
    assert name == "Carlos Henrique Prado"


def test_still_returns_none_when_no_column_has_a_name():
    assert extract_name(
        "Um artesão de 64 anos, católico dedicado.",
        ["Trabalha com madeira desde jovem.", "Gosta de futebol aos domingos."],
    ) is None
