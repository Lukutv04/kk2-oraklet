from pydantic import BaseModel
from app.chain.runnable import Runnable

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


# Prompt Builder

class PromptInput(BaseModel):
    question: str
    df: object

class PromptOutput(BaseModel):
    prompt: str


class PromptBuilder(Runnable[PromptInput, PromptOutput]):
    def invoke(self, inp: PromptInput) -> PromptOutput:

        # --- Fallback: svara på enkla datasetfrågor direkt ---
        q = inp.question.lower()

        if "kolumn" in q and "hur många" in q:
            return PromptOutput(prompt=f"ANTAL_KOLUMNER:{len(inp.df.columns)}")

        if "rader" in q and "hur många" in q:
            return PromptOutput(prompt=f"ANTAL_RADER:{len(inp.df)}")

        if "artist" in q and "flest" in q:
            top_artist = inp.df['artist'].value_counts().idxmax()
            return PromptOutput(prompt=f"TOPP_ARTIST:{top_artist}")

        # --- Normal prompt för modellen ---
        stats = inp.df.describe().to_string()

        prompt = f"""
Du är en enkel assistent. Här är statistik från datasetet:

{stats}

Fråga: {inp.question}

Svara kort på svenska. Inga upprepningar. Ingen kod. Bara ett kort svar.
"""

        return PromptOutput(prompt=prompt)


# LLM Runner

class LLMOutput(BaseModel):
    raw: str


class LLMRunner(Runnable[PromptOutput, LLMOutput]):
    def __init__(self):
        self.pipeline = None
        self.load_error = None

    def _load_pipeline(self):
        if torch is None:
            self.load_error = "PyTorch saknas."
            return None

        if pipeline is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            self.load_error = "Transformers saknas."
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=60,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=2.0
            )
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return None

    def invoke(self, inp: PromptOutput) -> LLMOutput:

        # --- Fallback-svar direkt ---
        if inp.prompt.startswith("ANTAL_KOLUMNER:"):
            return LLMOutput(raw=f"Datasetet har {inp.prompt.split(':')[1]} kolumner.")

        if inp.prompt.startswith("ANTAL_RADER:"):
            return LLMOutput(raw=f"Datasetet har {inp.prompt.split(':')[1]} rader.")

        if inp.prompt.startswith("TOPP_ARTIST:"):
            return LLMOutput(raw=f"Artisten med flest låtar är {inp.prompt.split(':')[1]}.")

        # --- Annars kör modellen ---
        if self.pipeline is None:
            self.pipeline = self._load_pipeline()

        if self.pipeline is None:
            return LLMOutput(raw="AI-modellen kunde inte startas.")

        try:
            out = self.pipeline(inp.prompt, return_full_text=False)
            return LLMOutput(raw=out[0]["generated_text"])
        except Exception:
            return LLMOutput(raw="AI-modellen kunde inte generera ett svar.")


# Response Parser

class ParsedOutput(BaseModel):
    question: str
    answer: str
    model: str = MODEL_NAME


class ResponseParser(Runnable[LLMOutput, ParsedOutput]):
    def invoke(self, inp: LLMOutput) -> ParsedOutput:
        return ParsedOutput(
            question="",
            answer=inp.raw.strip()
        )
