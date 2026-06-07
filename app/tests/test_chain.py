from app.chain import steps
from app.chain.steps import PromptBuilder, LLMRunner, ResponseParser, PromptInput, PromptOutput
import pandas as pd

def test_prompt_builder():
    df = pd.DataFrame({"x": [1, 2, 3]})
    inp = PromptInput(question="Vad är medelvärdet?", df=df)
    out = PromptBuilder().invoke(inp)

    assert "Vad är medelvärdet?" in out.prompt
    assert "x" in out.prompt

def test_response_parser():
    raw = "Svar: 10"
    parsed = ResponseParser().invoke(type("obj", (), {"raw": raw}))
    assert parsed.answer == "Svar: 10"

def test_llm_runner_returns_clear_error_when_torch_is_missing(monkeypatch):
    monkeypatch.setattr(steps, "torch", None)
    runner = LLMRunner()

    out = runner.invoke(PromptOutput(prompt="Test"))

    assert out.raw == "AI-modellen kunde inte startas."

def test_llm_runner_initializes_phi_1_5_pipeline_without_unsupported_kwargs(monkeypatch):
    calls = []

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("tokenizer", args, kwargs))
            return "tokenizer"

    class FakeModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("model", args, kwargs))
            return "model"

    def fake_pipeline(*args, **kwargs):
        calls.append(("pipeline", args, kwargs))

        def generate(_prompt, **_kwargs):
            return [{"generated_text": "Svar"}]

        return generate

    class FakeTorch:
        float32 = "float32"

    monkeypatch.setattr(steps, "torch", FakeTorch())
    monkeypatch.setattr(steps, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(steps, "AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(steps, "pipeline", fake_pipeline)
    runner = LLMRunner()

    out = runner.invoke(PromptOutput(prompt="Test"))

    assert out.raw == "Svar"
    assert calls == [
        (
            "tokenizer",
            (steps.MODEL_NAME,),
            {},
        ),
        (
            "model",
            (steps.MODEL_NAME,),
            {
                "torch_dtype": "float32",
                "device_map": "cpu",
            },
        ),
        (
            "pipeline",
            ("text-generation",),
            {"model": "model", "tokenizer": "tokenizer", "max_new_tokens": 200},
        ),
    ]

def test_llm_runner_returns_fallback_when_pipeline_startup_fails(monkeypatch):
    def broken_pipeline(*_args, **_kwargs):
        raise ValueError("unsupported argument")

    monkeypatch.setattr(steps, "torch", object())
    monkeypatch.setattr(steps, "pipeline", broken_pipeline)
    runner = LLMRunner()

    out = runner.invoke(PromptOutput(prompt="Test"))

    assert out.raw.startswith("AI-modellen kunde inte startas.")

def test_full_chain_returns_parsed_fallback_when_torch_is_missing(monkeypatch):
    monkeypatch.setattr(steps, "torch", None)
    df = pd.DataFrame({"x": [1, 2, 3]})
    prompt = PromptBuilder().invoke(PromptInput(question="Test?", df=df))
    llm_output = LLMRunner().invoke(prompt)
    parsed = ResponseParser().invoke(llm_output)

    assert parsed.answer == "AI-modellen kunde inte startas."
