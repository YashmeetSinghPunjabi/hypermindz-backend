import unittest
import os
import sqlite3
import json
import shutil
from unittest.mock import patch, MagicMock

# Set test database path before importing db_manager
import db_manager
import auth
import main

class TestBackendPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Override metadata database path to a test db
        db_manager.METADATA_DB_PATH = "backend/test_metadata.db"
        main.METADATA_DB_PATH = "backend/test_metadata.db"
        if os.path.exists("backend/test_metadata.db"):
            os.remove("backend/test_metadata.db")
        db_manager.init_db()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("backend/test_metadata.db"):
            os.remove("backend/test_metadata.db")
            
        # Clean up any test database files created
        for f in os.listdir("backend"):
            if f.startswith("db_test_") and f.endswith(".sqlite"):
                os.remove(os.path.join("backend", f))

    def test_01_user_creation_and_auth(self):
        email = "test@example.com"
        password = "secret_password"
        
        # Test hashing
        pw_hash = auth.get_password_hash(password)
        self.assertNotEqual(password, pw_hash)
        self.assertTrue(auth.verify_password(password, pw_hash))
        
        # Test creation
        user_id = db_manager.create_user(email, pw_hash)
        self.assertIsNotNone(user_id)
        
        # Test fetch
        user = db_manager.get_user_by_email(email)
        self.assertIsNotNone(user)
        self.assertEqual(user["id"], user_id)
        self.assertEqual(user["email"], email)
        
        # Test JWT token
        token = auth.create_access_token({"sub": user_id})
        self.assertIsNotNone(token)
        
        payload = auth.verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], user_id)

    def test_02_sql_security_validator(self):
        # Test valid queries
        self.assertEqual(
            main.SQLSecurityValidator.validate_query("SELECT * FROM data_table;"),
            "SELECT * FROM data_table;"
        )
        self.assertEqual(
            main.SQLSecurityValidator.validate_query("WITH summary AS (SELECT val FROM table) SELECT * FROM summary"),
            "WITH summary AS (SELECT val FROM table) SELECT * FROM summary"
        )
        
        # Test mutations (should raise HTTPException)
        with self.assertRaises(Exception) as ctx:
            main.SQLSecurityValidator.validate_query("DROP TABLE data_table;")
        
        with self.assertRaises(Exception) as ctx:
            main.SQLSecurityValidator.validate_query("UPDATE data_table SET col = 1;")
            
        with self.assertRaises(Exception) as ctx:
            main.SQLSecurityValidator.validate_query("SELECT * FROM data_table; DELETE FROM data_table;")

    def test_03_file_metadata_and_chat_history(self):
        user_id = "test_user_123"
        file_id = "test_file_456"
        columns = ["id", "val", "label"]
        sample_rows = [{"id": 1, "val": 100, "label": "A"}]
        
        # Add file
        fid = db_manager.add_file(
            user_id=user_id,
            file_name="test_data.csv",
            table_name="data_test_file_456",
            row_count=1,
            columns=columns,
            sample_rows=sample_rows,
            custom_file_id=file_id
        )
        self.assertEqual(fid, file_id)
        
        # Retrieve file
        files = db_manager.get_files_by_user(user_id)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["file_name"], "test_data.csv")
        self.assertEqual(files[0]["columns"], columns)
        
        # Chat history
        db_manager.add_chat_message(user_id, file_id, "user", "Hello?")
        db_manager.add_chat_message(user_id, file_id, "model", "Hi there!")
        
        chat = db_manager.get_chat_history(user_id, file_id)
        self.assertEqual(len(chat), 2)
        self.assertEqual(chat[0]["role"], "user")
        self.assertEqual(chat[1]["role"], "model")
        
        # Clear chat
        db_manager.clear_chat_history(user_id, file_id)
        chat_cleared = db_manager.get_chat_history(user_id, file_id)
        self.assertEqual(len(chat_cleared), 0)

if __name__ == "__main__":
    unittest.main()
