import unittest
import os
from fastapi.testclient import TestClient

# Mock DB Path for Testing
import db_manager
db_manager.METADATA_DB_PATH = "backend/test_api_metadata.db"

from main import app

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(db_manager.METADATA_DB_PATH):
            os.remove(db_manager.METADATA_DB_PATH)
        db_manager.init_db()
        cls.client = TestClient(app)
        cls.test_email = "api_user@example.com"
        cls.test_password = "secure_api_password_123"
        cls.token = None

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(db_manager.METADATA_DB_PATH):
            os.remove(db_manager.METADATA_DB_PATH)

    def test_01_register(self):
        response = self.client.post(
            "/api/auth/register",
            json={"email": self.test_email, "password": self.test_password}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())

    def test_02_login(self):
        response = self.client.post(
            "/api/auth/login",
            json={"email": self.test_email, "password": self.test_password}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        TestAPIEndpoints.token = data["access_token"]

    def test_03_get_files(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        response = self.client.get("/api/files", headers=headers)
        self.assertEqual(response.status_code, 200)
        # Auth registration seeds mock data automatically, so we should have files
        files = response.json()
        self.assertTrue(len(files) > 0)

    def test_04_unauthorized_access(self):
        # Should return 401 when no token is provided
        response = self.client.get("/api/files")
        self.assertEqual(response.status_code, 401)

if __name__ == "__main__":
    unittest.main()
