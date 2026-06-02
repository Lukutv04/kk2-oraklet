from app.chain.steps import PromptBuilder, LLMRunner, ResponseParser, PromptInput
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
