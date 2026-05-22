import json
from fastapi.testclient import TestClient
from main import app
import db_manager
import auth

client = TestClient(app)

def run_test():
    email = "test_api2@example.com"
    pwd = "testpassword123"
    
    # 1. Register user
    res = client.post("/api/auth/register", json={"email": email, "password": pwd})
    if res.status_code != 200:
        res = client.post("/api/auth/login", json={"email": email, "password": pwd})
        
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Get files to get a file_id
    res = client.get("/api/files", headers=headers)
    files = res.json()
    if not files:
        print("No files found!")
        return
        
    file_id = files[0]["id"]
    print(f"Using file_id: {file_id}")
    
    # 3. Test suggestions
    print("Testing /api/files/{file_id}/suggestions")
    sug_res = client.get(f"/api/files/{file_id}/suggestions", headers=headers)
    print(f"Suggestions Status: {sug_res.status_code}")
    print(sug_res.json())
    
    # 4. Test query
    print("\nTesting /api/query")
    payload = {
        "file_id": file_id,
        "natural_language_query": "What is the total revenue?"
    }
    q_res = client.post("/api/query", json=payload, headers=headers)
    print(f"Query Status: {q_res.status_code}")
    try:
        print(json.dumps(q_res.json(), indent=2)[:500] + "...")
    except:
        print(q_res.text)

if __name__ == "__main__":
    run_test()
