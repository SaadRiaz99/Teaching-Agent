from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "service" in data

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "docs" in data


class TestChatEndpoint:
    def test_chat_empty_message(self):
        response = client.post("/api/v1/chat", json={"message": ""})
        assert response.status_code == 422

    def test_chat_valid_request(self):
        response = client.post(
            "/api/v1/chat", json={"message": "What is Python?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "conversation_id" in data
        assert "sources" in data


class TestQuizEndpoint:
    def test_quiz_valid_request(self):
        response = client.post(
            "/api/v1/quiz",
            json={
                "num_questions": 3,
                "difficulty": "easy",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "questions" in data
        assert "total_questions" in data


class TestUploadEndpoint:
    def test_upload_no_file(self):
        response = client.post("/api/v1/upload")
        assert response.status_code == 422

    def test_upload_unsupported_type(self):
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.exe", b"fake content", "application/x-msdownload")},
        )
        assert response.status_code == 400
