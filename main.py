from fastapi import FastAPI, UploadFile, HTTPException
from app.data import DataStore

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

