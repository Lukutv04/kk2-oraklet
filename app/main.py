from fastapi import FastAPI, UploadFile, HTTPException
from app.data import DataStore
from app.schemas import AskRequest, AskResponse
from app.chain.pipeline import oraklet
from app.chain.steps import PromptInput

app = FastAPI(title="KK2 – Oraklet", version="1.0.0")
store = DataStore()


@app.get("/")
def root():
    return {"message": "Välkommen till KK2 – Oraklet!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/data/upload")
async def upload(file: UploadFile):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Endast CSV-filer tillåtna.")

    try:
        df = store.load_csv(file)
        return store.metadata(df)
    except Exception as e:
        raise HTTPException(500, f"Kunde inte läsa CSV-filen: {e}")


@app.get("/data/stats")
def stats():
    df = store.get()
    if df is None:
        raise HTTPException(404, "Inget dataset uppladdat.")

    try:
        return df.describe().to_dict()
    except Exception as e:
        raise HTTPException(500, f"Kunde inte generera statistik: {e}")


@app.post("/ai/ask", response_model=AskResponse)
def ask(req: AskRequest):
    df = store.get()
    if df is None:
        raise HTTPException(400, "Ladda upp dataset först.")

    try:
        result = oraklet.invoke(
            PromptInput(
                question=req.question,
                df=df
            )
        )
    except Exception as e:
        raise HTTPException(500, f"AI-modellen kunde inte generera ett svar: {e}")

    # Säkerställ att svaret alltid är en sträng
    answer = getattr(result, "answer", None)
    if not isinstance(answer, str):
        answer = str(answer)

    return AskResponse(answer=answer)
