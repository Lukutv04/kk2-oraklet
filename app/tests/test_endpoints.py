from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_upload_and_stats():
    csv_content = "a,b,c\n1,2,3\n4,5,6\n"
    files = {"file": ("test.csv", csv_content, "text/csv")}

    upload_res = client.post("/data/upload", files=files)
    assert upload_res.status_code == 200
    assert upload_res.json()["rows"] == 2

    stats_res = client.get("/data/stats")
    assert stats_res.status_code == 200
    assert "a" in stats_res.json()
