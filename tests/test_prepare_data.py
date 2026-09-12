import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "01_prepare_data.py"
spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)

ROW = {
    "uuid": "abc123",
    "persona": "Marcos Antunes é um operador técnico organizado.",
    "professional_persona": "Marcos Antunes opera máquinas CNC.",
    "cultural_background": "Marcos cresceu em São Pedro de Alcântara.",
    "skills_and_expertise": "Marcos possui experiência com máquinas.",
    "hobbies_and_interests": "Marcos gosta de futebol.",
    "career_goals_and_ambitions": "Marcos pretende concluir o ensino médio.",
    "sex": "Masculino",
    "age": 22,
    "marital_status": "Solteiro",
    "education_level": "Fundamental completo e médio incompleto",
    "occupation": "Operador de instalação ou máquina ou montador",
    "municipality": "São Pedro de Alcântara",
    "state": "Santa Catarina",
    "country": "Brasil",
}


class FakeTokenizer:
    """Stands in for the Qwen3 tokenizer: same interface, trivial template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True, **kwargs):
        assert enable_thinking is False, "thinking mode must always be disabled"
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


def test_build_record_produces_a_prompt_that_prefixes_the_training_text():
    record = prepare_data.build_record(ROW, FakeTokenizer())
    assert record is not None
    assert record["text"].startswith(record["prompt"])


def test_build_record_injects_the_extracted_name():
    record = prepare_data.build_record(ROW, FakeTokenizer())
    assert record["attributes"]["name"] == "Marcos Antunes"
    assert "Marcos Antunes" in record["prompt"]


def test_build_record_returns_none_when_the_name_cannot_be_extracted():
    row = dict(ROW, persona="uma pessoa comum que trabalha muito")
    assert prepare_data.build_record(row, FakeTokenizer()) is None


def test_build_record_returns_none_on_a_missing_section():
    row = dict(ROW, hobbies_and_interests="")
    assert prepare_data.build_record(row, FakeTokenizer()) is None


def test_split_indices_are_disjoint():
    train, val, test = prepare_data.split_indices(total=100, n_train=60, n_val=20,
                                                  n_test=10, seed=3407)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert (len(train), len(val), len(test)) == (60, 20, 10)
