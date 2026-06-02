from fastapi import FastAPI, UploadFile, HTTPException
from app.data import DataStore
from app.schemas import AskRequest, AskResponse
from app.chain.pipeline import oraklet

app = FastAPI()
store = DataStore()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/data/upload")
async def upload(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Endast CSV-filer tillåtna.")
    df = store.load_csv(file)
    return store.metadata(df)

@app.get("/data/stats")
def stats():
    df = store.get()
    if df is None:
        raise HTTPException(404, "Inget dataset uppladdat.")
    return df.describe().to_dict()

@app.post("/ai/ask", response_model=AskResponse)
def ask(req: AskRequest):
    df = store.get()
    if df is None:
        raise HTTPException(400, "Ladda upp dataset först.")
    result = oraklet.invoke({"question": req.question, "df": df})
    return result
