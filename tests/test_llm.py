from personas.llm import LlamaCppCompletion


class Decode:
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    repeat_penalty = 1.05
    n_predict = 1024
    seed = 3407


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Captures the posted body so we can assert on the exact request."""

    def __init__(self):
        self.posts = []

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self.posts.append((url, json))
        return FakeResponse({"content": "## Síntese\ntexto gerado"})

    def close(self):
        return None


def test_posts_the_prompt_verbatim_to_completion():
    client = FakeClient()
    llm = LlamaCppCompletion("http://localhost:8080", Decode(), client=client)
    out = llm.invoke("<|im_start|>user\nNome: Ana<|im_end|>\n")
    assert out == "## Síntese\ntexto gerado"
    url, body = client.posts[0]
    assert url.endswith("/completion"), "must use raw /completion, never /v1 (ADR 0006)"
    assert body["prompt"] == "<|im_start|>user\nNome: Ana<|im_end|>\n"


def test_sends_the_configured_decoding_parameters():
    client = FakeClient()
    llm = LlamaCppCompletion("http://localhost:8080", Decode(), client=client)
    llm.invoke("hello")
    _, body = client.posts[0]
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.8
    assert body["top_k"] == 20
    assert body["repeat_penalty"] == 1.05
    assert body["n_predict"] == 1024
    assert body["seed"] == 3407
    assert body["cache_prompt"] is False


def test_batch_returns_one_result_per_prompt():
    client = FakeClient()
    llm = LlamaCppCompletion("http://localhost:8080", Decode(), client=client)
    results = llm.batch(["a", "b", "c"])
    assert len(results) == 3
    assert len(client.posts) == 3
