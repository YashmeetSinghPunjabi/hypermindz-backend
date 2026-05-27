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

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is missing. Please configure it in your Settings or environment.")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True
    )

    try:
        # Check if the query is a raw SQL query
        query_stripped = payload.natural_language_query.strip().strip("`").replace("sql\n", "").strip()
        upper_query = query_stripped.upper()
        if upper_query.startswith("SELECT") or upper_query.startswith("WITH"):
            sanitized_sql = SQLSecurityValidator.validate_query(query_stripped)
            conn = sqlite3.connect(db_path)
            df_result = pd.read_sql_query(sanitized_sql, conn)
            conn.close()

            viz_prompt = ChatPromptTemplate.from_messages([
                ("system", 'Analyze this SQL query and recommend a UI chart. Return ONLY JSON matching this schema: {{"visualization_recommended": true/false, "chart_type": "bar"/"line"/"pie"/"none", "x_axis_key": "column_name_or_null", "y_axis_key": "column_name_or_null"}}'),
                ("human", "SQL: {sql}\nQuestion: Custom Query")
            ])
            viz_chain = viz_prompt | llm | JsonOutputParser()
            try:
                viz_config = viz_chain.invoke({"sql": sanitized_sql})
            except Exception:
                viz_config = {"visualization_recommended": False, "chart_type": "none", "x_axis_key": None, "y_axis_key": None}

            actual_cols = list(df_result.columns)
            x_key = viz_config.get("x_axis_key")
            y_key = viz_config.get("y_axis_key")
            
            if x_key not in actual_cols:
                x_key = actual_cols[0] if actual_cols else None
            if y_key not in actual_cols:
                y_key = actual_cols[1] if len(actual_cols) > 1 else (actual_cols[0] if actual_cols else None)
                
            viz_cfg = {
                "recommended": viz_config.get("visualization_recommended", False),
                "type": viz_config.get("chart_type", "none"),
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
You are an expert data analyst. Translate the user's natural language question into an optimized, executable SQLite query and answer it.

CRITICAL INSTRUCTIONS:
1. DO NOT just explain what you are going to do. You MUST actually provide the SQL query.
2. You MUST either use the `sql_db_query` tool to execute the query, OR output the raw query wrapped exactly in a ```sql ... ``` markdown block in your final answer.
3. Stop talking and just output the SQL.

Chat History for context:
{history_str}

User Question: {payload.natural_language_query}
"""
        response = agent_executor.invoke({"input": agent_prompt})
        explanation = response.get("output", "")
        
        sql_query = ""
        for action, observation in response.get("intermediate_steps", []):
            if action.tool == "sql_db_query":
                if isinstance(action.tool_input, dict):
                    sql_query = action.tool_input.get("query", "")
                else:
                    sql_query = str(action.tool_input)
                break
                
        if not sql_query.strip():
            import re
            match = re.search(r"```sql\n(.*?)\n```", explanation, re.DOTALL | re.IGNORECASE)
            if match:
                sql_query = match.group(1)
            else:
                return handle_unanswerable(payload, user_id, explanation, file_info["file_name"])

        sanitized_sql = SQLSecurityValidator.validate_query(sql_query)
        
        conn = sqlite3.connect(db_path)
        df_result = pd.read_sql_query(sanitized_sql, conn)
        conn.close()

        viz_prompt = ChatPromptTemplate.from_messages([
            ("system", 'Analyze this SQL query and its results context to recommend a UI chart. Return ONLY JSON matching this schema: {{"visualization_recommended": true/false, "chart_type": "bar"/"line"/"pie"/"none", "x_axis_key": "column_name_or_null", "y_axis_key": "column_name_or_null"}}'),
            ("human", "SQL: {sql}\nQuestion: {question}")
        ])
        viz_chain = viz_prompt | llm | JsonOutputParser()
        viz_config = viz_chain.invoke({"sql": sanitized_sql, "question": payload.natural_language_query})
        
        actual_cols = list(df_result.columns)
        x_key = viz_config.get("x_axis_key")
        y_key = viz_config.get("y_axis_key")
        
        if x_key not in actual_cols:
            x_key = actual_cols[0] if actual_cols else None
        if y_key not in actual_cols:
            y_key = actual_cols[1] if len(actual_cols) > 1 else (actual_cols[0] if actual_cols else None)
            
        viz_cfg = {
            "recommended": viz_config.get("visualization_recommended", False),
            "type": viz_config.get("chart_type", "none"),
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
