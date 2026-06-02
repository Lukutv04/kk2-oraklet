from pydantic import BaseModel
from transformers import pipeline
from app.chain.runnable import Runnable


# Prompt Builder


class PromptInput(BaseModel):
    question: str
    df: object

class PromptOutput(BaseModel):
    prompt: str

class PromptBuilder(Runnable[PromptInput, PromptOutput]):
    def invoke(self, inp: PromptInput) -> PromptOutput:
        stats = inp.df.describe().to_string()
        prompt = f"""
Du är ett dataorakel. Här är statistik från datasetet:

{stats}

Fråga: {inp.question}
Svara kort och tydligt.
"""
        return PromptOutput(prompt=prompt)


# LLM Runner


class LLMOutput(BaseModel):
    raw: str

class LLMRunner(Runnable[PromptOutput, LLMOutput]):
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-135M-Instruct"
        )

    def invoke(self, inp: PromptOutput) -> LLMOutput:
        out = self.pipe(inp.prompt, max_new_tokens=150)[0]["generated_text"]
        return LLMOutput(raw=out)


# Response Parser


class ParsedOutput(BaseModel):
    question: str
    answer: str
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"

class ResponseParser(Runnable[LLMOutput, ParsedOutput]):
    def invoke(self, inp: LLMOutput) -> ParsedOutput:
        return ParsedOutput(
            question="",
            answer=inp.raw.strip()
        )
