"""检查 API 的成功请求、输入错误和文档页面。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_generate_text():
    response = client.post("/generate", json={"start_word": "WE", "length": 3})
    assert response.status_code == 200
    assert response.json()["generated_text"] in {"we are generating", "we are simple"}


def test_generation_stops_at_terminal_word():
    response = client.post("/generate", json={"start_word": "effective", "length": 10})
    assert response.status_code == 200
    assert response.json() == {"generated_text": "effective"}


def test_unknown_start_word():
    response = client.post("/generate", json={"start_word": "unicorn", "length": 10})
    assert response.status_code == 422
    assert "unicorn" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"start_word": "we"},
        {"start_word": "", "length": 10},
        {"start_word": "   ", "length": 10},
        {"start_word": "we are", "length": 10},
        {"start_word": "we", "length": 0},
        {"start_word": "we", "length": 201},
        {"start_word": "we", "length": 1.5},
        {"start_word": "we", "length": True},
    ],
)
def test_invalid_request(payload):
    assert client.post("/generate", json=payload).status_code == 422


def test_interactive_documentation():
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "post" in schema["paths"]["/generate"]
