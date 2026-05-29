import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

import db_manager
import auth
from schemas import QueryExecutionPayload, QueryResultResponse
from utils import get_user_db_path, SQLSecurityValidator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import langchain
langchain.debug = True

router = APIRouter(
    prefix="/api",
    tags=["query"]
)

@router.get("/history")
def get_user_history(file_id: Optional[str] = None, user_id: str = Depends(auth.get_current_user_id)):
    return db_manager.get_query_history(user_id, file_id)

@router.delete("/chat/{file_id}")
def reset_chat_thread(file_id: str, user_id: str = Depends(auth.get_current_user_id)):
    db_manager.clear_chat_history(user_id, file_id)
    return {"status": "success", "message": "Conversational memory reset successfully."}

def handle_unanswerable(payload: QueryExecutionPayload, user_id: str, explanation: str, file_name: str) -> QueryResultResponse:
    db_manager.add_chat_message(user_id, payload.file_id, "user", payload.natural_language_query)
    db_manager.add_chat_message(user_id, payload.file_id, "model", explanation)
    return QueryResultResponse(
        sql_query="",
        explanation=explanation,
        visualization_config={"recommended": False, "type": "none", "x_axis_key": None, "y_axis_key": None},
        data=[],
        source_file=file_name
    )

@router.post("/query", response_model=QueryResultResponse)
def query_tabular_data(
    payload: QueryExecutionPayload,
    user_id: str = Depends(auth.get_current_user_id)
):
    file_info = db_manager.get_file_by_id(payload.file_id, user_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    db_path = get_user_db_path(user_id, payload.file_id)
    db_uri = f"sqlite:///{db_path}"
    db = SQLDatabase.from_uri(db_uri)

    import random
    
    # Collect all fallback keys to load-balance and prevent rate limits
    gemini_keys = []
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY") and v.strip():
            gemini_keys.append(v.strip())
            
    if not gemini_keys:
        raise HTTPException(status_code=400, detail="Gemini API Key is missing. Please configure it in your Settings or environment.")
        
    api_key = random.choice(gemini_keys)
    
    llm = ChatGoogleGenerativeAI(
        model=payload.ai_model or "gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True
    )

    try:
        # Check if the query is a raw SQL query
        query_stripped = payload.natural_language_query.strip().strip("`").replace("sql\n", "").strip()
        if payload.query_mode == "sql":
            sanitized_sql = SQLSecurityValidator.validate_query(query_stripped)
            conn = sqlite3.connect(db_path)
            df_result = pd.read_sql_query(sanitized_sql, conn)
            conn.close()

            actual_cols = list(df_result.columns)
            x_key = actual_cols[0] if actual_cols else None
            y_key = None
            if len(actual_cols) > 1:
                # Find the first numeric column for y-axis
                for col in actual_cols[1:]:
                    if pd.api.types.is_numeric_dtype(df_result[col]):
                        y_key = col
                        break
                if not y_key:
                    y_key = actual_cols[1]
            
            viz_cfg = {
                "recommended": len(actual_cols) >= 2,
                "type": "bar" if len(actual_cols) >= 2 else "none",
                "x_axis_key": x_key,
                "y_axis_key": y_key
            }

            db_manager.add_chat_message(user_id, payload.file_id, "user", payload.natural_language_query)
            db_manager.add_chat_message(user_id, payload.file_id, "model", f"Executed direct SQL query.")
            
            db_manager.add_query_history(
                user_id=user_id,
                file_id=payload.file_id,
                question=payload.natural_language_query,
                sql_query=sanitized_sql,
                explanation="Executed direct SQL query custom request.",
                viz_config=viz_cfg
            )

            return QueryResultResponse(
                sql_query=sanitized_sql,
                explanation=f"Successfully executed raw SQL query: `{sanitized_sql}`",
                visualization_config=viz_cfg,
                data=df_result.replace({np.nan: None}).to_dict(orient="records"),
                source_file=file_info["file_name"]
            )

        chat_history = db_manager.get_chat_history(user_id, payload.file_id)
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-5:]])

        agent_executor = create_sql_agent(
            llm, 
            db=db, 
            agent_type="tool-calling", 
            verbose=True, 
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        agent_prompt = f"""
You are an expert data analyst and an AI SQL Agent. Your job is to answer the user's question using the database.

CRITICAL OUTPUT INSTRUCTIONS:
1. You MUST use your SQL tools to query the database and analyze the results.
2. Provide a short, friendly analytical explanation based on the data you found.
3. You MUST include the raw SQLite query you used wrapped exactly inside a markdown code block like this:
```sql
SELECT * FROM table_name;
```
4. Do not use functions that do not exist in SQLite.
5. NEVER say "I don't know". If you are unsure, write the best possible query to estimate the answer.

DATABASE SCHEMA:
The ONLY table you need to query is exactly named: `{file_info["table_name"]}`
The columns available in this table are exactly: {file_info["columns"]}
Do not guess table names. Use EXACTLY `{file_info["table_name"]}`.

Chat History for context:
{history_str}

User Question: {payload.natural_language_query}
"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"\n========== GEMINI REQUEST ==========\n{agent_prompt}\n====================================")

        response = agent_executor.invoke({"input": agent_prompt})
        
        logger.info(f"\n========== GEMINI RESPONSE ==========\n{response.get('output', '')}\n=====================================")
        
        raw_output = response.get("output", "")
        if isinstance(raw_output, list):
            text_parts = []
            for part in raw_output:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            explanation = "".join(text_parts)
        else:
            explanation = str(raw_output)

        sql_query = ""
        for action, observation in response.get("intermediate_steps", []):
            tool_name = getattr(action, "tool", str(action))
            if "sql_db_query" in tool_name:
                tool_input = getattr(action, "tool_input", {})
                if isinstance(tool_input, dict):
                    sql_query = tool_input.get("query", "")
                else:
                    sql_query = str(tool_input)
                break
        import re
        # First fallback: Check explanation for markdown blocks
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", explanation, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(1)
        else:
            # Second fallback: Aggressive regex to extract any SELECT statement
            fallback_match = re.search(r"(SELECT\s+.*)", explanation, re.IGNORECASE | re.DOTALL)
            if fallback_match:
                sql_query = fallback_match.group(1)
                
            # Third fallback: Did the AI just spit out raw SQL text directly?
            if not sql_query.strip() and explanation.strip().upper().startswith("SELECT "):
                sql_query = explanation.strip()

        if not sql_query.strip():
            return handle_unanswerable(payload, user_id, explanation, file_info["file_name"])

        import re
        # Clean the explanation so the frontend doesn't double-print the SQL query in the chat bubble
        explanation = re.sub(r"```(?:sql)?\s*.*?\s*```", "", explanation, flags=re.DOTALL | re.IGNORECASE).strip()
        # Also clean up any raw SELECT queries that were dumped
        if sql_query in explanation:
            explanation = explanation.replace(sql_query, "").strip()
            
        if len(explanation) < 5:
            explanation = "Here is the data you requested:"

        sanitized_sql = SQLSecurityValidator.validate_query(sql_query)
        
        conn = sqlite3.connect(db_path)
        df_result = pd.read_sql_query(sanitized_sql, conn)
        conn.close()

        actual_cols = list(df_result.columns)
        x_key = actual_cols[0] if actual_cols else None
        y_key = None
        if len(actual_cols) > 1:
            # Find the first numeric column for y-axis
            for col in actual_cols[1:]:
                if pd.api.types.is_numeric_dtype(df_result[col]):
                    y_key = col
                    break
            if not y_key:
                y_key = actual_cols[1]
                
        viz_cfg = {
            "recommended": len(actual_cols) >= 2,
            "type": "bar" if len(actual_cols) >= 2 else "none",
            "x_axis_key": x_key,
            "y_axis_key": y_key
        }

        db_manager.add_chat_message(user_id, payload.file_id, "user", payload.natural_language_query)
        db_manager.add_chat_message(user_id, payload.file_id, "model", f"Generated SQL: {sanitized_sql}. Explanation: {explanation}")
        
        db_manager.add_query_history(
            user_id=user_id,
            file_id=payload.file_id,
            question=payload.natural_language_query,
            sql_query=sanitized_sql,
            explanation=explanation,
            viz_config=viz_cfg
        )

        return QueryResultResponse(
            sql_query=sanitized_sql,
            explanation=explanation,
            visualization_config=viz_cfg,
            data=df_result.replace({np.nan: None}).to_dict(orient="records"),
            source_file=file_info["file_name"]
        )

    except HTTPException as e:
        db_manager.add_chat_message(user_id, payload.file_id, "user", payload.natural_language_query)
        db_manager.add_chat_message(user_id, payload.file_id, "model", f"Error: {e.detail}")
        raise e
    except Exception as e:
        db_manager.add_chat_message(user_id, payload.file_id, "user", payload.natural_language_query)
        error_str = str(e)
        
        if "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
            clean_msg = "Google Gemini Free-Tier Quota Exceeded (Max 20 requests per minute). Please wait 60 seconds and try again, or upgrade your API Key in Settings."
            db_manager.add_chat_message(user_id, payload.file_id, "model", clean_msg)
            raise HTTPException(status_code=429, detail=clean_msg)
            
        db_manager.add_chat_message(user_id, payload.file_id, "model", f"Error: {error_str}")
        raise HTTPException(status_code=500, detail=f"Agent Error: {error_str}")
