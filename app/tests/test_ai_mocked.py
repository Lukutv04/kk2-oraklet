from fastapi.testclient import TestClient
from app.main import app
from app.chain.pipeline import oraklet

client = TestClient(app)

def fake_invoke(_):
    return {
        "question": "Test?",
        "answer": "Detta är ett mockat svar.",
        "model": "mock"
    }

def test_ai_ask_mocked(monkeypatch):
    # mocka kedjan
    monkeypatch.setattr(oraklet, "invoke", fake_invoke)

    # ladda dataset
    csv_content = "a,b\n1,2\n3,4\n"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    client.post("/data/upload", files=files)

    # fråga modellen
    res = client.post("/ai/ask", json={"question": "Test?"})

    assert res.status_code == 200
    assert res.json()["answer"] == "Detta är ett mockat svar."
