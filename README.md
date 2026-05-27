# HyperMindZ Backend Architecture 🐍

The HyperMindZ Analytics Engine backend is a robust, modular API constructed with **FastAPI** and powered by **LangChain** and Google's **Gemini 2.5 Flash Lite** model. It bridges the gap between natural language processing and deterministic relational database querying.

## 🏗️ Architecture Overview

The backend follows a service-oriented architecture, strictly decoupling routing, data validation, and core AI processing. 

### Core Components
1. **API Routing Layer (`routers/`)**: 
   - `auth_routes.py`: Manages secure JWT session provisioning and user authentication.
   - `files_routes.py`: Handles multipart file uploads (CSV/Excel), safely parsing and injecting tabular data into dynamically provisioned SQLite databases via pandas.
   - `query_routes.py`: Exposes the AI execution pipeline. It takes natural language queries and passes them to the LangChain SQL agent.
2. **AI Execution Engine (`query_routes.py`)**:
   - Uses `create_sql_agent` to construct an iterative execution loop.
   - Connects securely to the isolated SQLite sandbox via `SQLDatabase.from_uri`.
   - Injects the `gemini-2.5-flash-lite` LLM with specific prompt engineering to ensure non-hallucinated SQL generation.
3. **Data Schemas (`schemas.py`)**: Defines all incoming and outgoing data payloads using Pydantic, ensuring strict type-safety and OpenAPI documentation generation.
4. **Security & Utilities (`utils.py`)**: Handles password hashing, JWT decoding, and secure file I/O operations.

## ⚙️ How It Works (The Query Lifecycle)

1. **User Request**: The client sends a natural language question (e.g., "What was our highest revenue month?") along with an active `file_id`.
2. **Context Binding**: The backend identifies the specific SQLite database associated with the user's `file_id`.
3. **Agent Invocation**: The `SQLDatabaseToolkit` inspects the schema of the SQLite database and passes the context to the LLM.
4. **SQL Generation**: The Gemini model formulates a valid `SELECT` query based on the table schema.
5. **Execution & Formatting**: The agent executes the SQL securely, extracts the resulting rows, and formulates a human-readable explanation.
6. **Visualization Heuristics**: The backend dynamically analyzes the resulting data types and appends a `visualization_config` payload (e.g., recommending a Bar chart for categorical vs. numeric data).

## 🚀 Setup & Deployment

1. **Install Python Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Setup:**
   Create a local `.env` file at the root of the backend folder:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   ```
3. **Start the API Server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   *The Swagger UI documentation is automatically generated at `http://localhost:8000/docs`.*
