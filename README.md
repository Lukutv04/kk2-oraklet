# KK2 – Oraklet
Ett FastAPI‑baserat backendprojekt som laddar upp dataset, visar statistik och använder en kedja av “Runnable”-steg för att generera AI‑svar baserat på datan.
Projektet innehåller även fallback‑logik som besvarar enkla datasetfrågor korrekt, vilket gör systemet stabilt även med en liten lokal modell.

##  Funktionalitet
Projektet erbjuder tre huvuddelar:

### 1. Ladda upp dataset
Endpoint: POST /data/upload  
Tar emot en CSV‑fil, läser in den och sparar den i en intern datalagring.

### 2. Visa statistik
Endpoint: GET /data/stats  
Returnerar df.describe() som dictionary.

### 3. Ställa frågor till Oraklet
Endpoint: POST /ai/ask  
Kör en kedja av tre steg:

- PromptBuilder – bygger en prompt baserat på datasetets statistik och hanterar fallback‑frågor (t.ex. antal kolumner, antal rader, vanligaste artist).

- LLMRunner – kör en lokal textgenereringsmodell med optimerade inställningar (temperature=0, repetition_penalty, max_new_tokens).

- ResponseParser – formaterar svaret till rätt struktur.

Resultatet returneras som:

{
  "question": "...",
  "answer": "...",
  "model": "SmolLM2 + fallback"
}

## Exempel på frågor som stöds
“Hur många kolumner finns i datasetet?”

“Hur många rader finns i datasetet?”

“Vilken artist har flest låtar?”

“Vad visar statistiken?”

##  Projektstruktur
app/
 main.py
 data.py
 schemas.py
 chain/
  runnable.py
  steps.py
    pipeline.py
      tests/
         test_endpoints.py
         test_chain.py
         test_ai_mocked.py


##  Kedjan (Runnable‑design)
Projektet använder ett enkelt designmönster där varje steg är en “Runnable”:

Runnable – basklass

RunnableLambda – wrapper för funktioner

RunnableSequence – kedjar ihop flera steg med |‑operatorn

Kedjan defenieras som :

oraklet = RunnableSequence(
    PromptBuilder(),
    LLMRunner(),
    ResponseParser()
)


##  Tester
Projektet innehåller tre typer av tester:

### Endpoint‑tester
Testar /health, /data/upload, /data/stats.

### Kedje‑tester
Testar PromptBuilder och ResponseParser isolerat.

### Mockad LLM‑test
Testar hela /ai/ask utan att ladda modellen (monkeypatch).

Alla tester passerar:
5 passed, 1 warning



##  Köra projektet
Installera beroenden: 

uv sync


Starta Servern :

uv run uvicorn app.main:app --reload


Köra tester :

uv run pytest app/tests -v


##  Tekniker
- FastAPI

- Pydantic

- Pandas

- Transformers (HuggingFace)

- Pytest

- UV (package manager)

##  Övrigt

- Modellen laddas “lazy‑load” i LLMRunner för att undvika PyTorch‑krav i tester.

- Temperatur=0, repetition_penalty och max_new_tokens används för att undvika loops och hålla svaren stabila.

- En fallback‑logik hanterar enkla datasetfrågor för att säkerställa korrekta svar.

- Projektet är strukturerat för att vara lätt att utöka med fler steg i kedjan.
