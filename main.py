import os
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import pandas as pd
import numpy as np
import json
import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware

# Initialize Google GenAI API Client tools
from google import genai
from google.genai import types

# Import DB and Auth modules
from db_manager import init_db
import db_manager
import auth

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


app = FastAPI(title="HyperMindZ Tabular NL-to-SQL Engine")

# Configure relaxed CORS settings for Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    # Ensure any sample data is generated/seeded if missing

# ==========================================
# 1. PYDANTIC RESPONSE & REQUEST SCHEMAS
# ==========================================

class UserRegisterPayload(BaseModel):
    email: str
    password: str

class UserLoginPayload(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    email: str
    user_id: str

class SQLGenerationResponse(BaseModel):
    sql_query: str = Field(..., description="The executable SQLite read-only query. Output ONLY the query, no formatting.")
    explanation: str = Field(..., description="Plain English description of what the query calculates or a helpful error message if the query is unanswerable.")
    visualization_recommended: bool = Field(..., description="True if the query results are best interpreted as a chart (bar, line, pie, etc.), false if it fits a simple table or single value.")
    chart_type: str = Field(..., description="Type of chart to display: 'bar', 'line', 'pie', 'area', 'scatter', or 'none'.")
    x_axis_key: Optional[str] = Field(None, description="The column key from the database rows to map onto the chart X-Axis (e.g. date, category).")
    y_axis_key: Optional[str] = Field(None, description="The column key from the database rows to map onto the chart Y-Axis (e.g. revenue, quantity).")

class QueryExecutionPayload(BaseModel):
    file_id: str
    natural_language_query: str

class QueryResultResponse(BaseModel):
    sql_query: str
    explanation: str
    visualization_config: Dict[str, Any]
    data: List[Dict[str, Any]]
    source_file: str

class FileItem(BaseModel):
    id: str
    file_name: str
    table_name: str
    row_count: int
    columns: List[str]
    uploaded_at: str

class ColumnProfile(BaseModel):
    name: str
    type: str
    count: int
    unique_count: int
    null_percentage: float
    mean: Optional[float] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    top_values: Optional[List[Dict[str, Any]]] = None

class FilePreviewResponse(BaseModel):
    file_info: FileItem
    preview_data: List[Dict[str, Any]]

# ==========================================
# 2. PROGRAMMATIC SECURITY ENFORCEMENT
# ==========================================

class SQLSecurityValidator:
    """
    Enforces a strict programmatic read-only sandbox.
    Blocks hazardous commands and structural mutations.
    """
    FORBIDDEN_KEYWORDS = {'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'REPLACE', 'ATTACH', 'DETACH'}

    @classmethod
    def validate_query(cls, sql: str) -> str:
        # Clean up code blocks if any slipped through
        cleaned_sql = sql.strip().strip("`").replace("sql\n", "").strip()
        upper_sql = cleaned_sql.upper()

        # Rule 1: Block obvious stacked queries (semi-colon injection)
        if ";" in cleaned_sql and cleaned_sql.rstrip(";").count(";") > 0:
            raise HTTPException(status_code=400, detail="Security Violation: Multi-statement executions are forbidden.")

        # Rule 2: Token keyword assertion to block modification
        # Split by whitespace, punctuation, etc.
        import re
        tokens = set(re.findall(r'\b\w+\b', upper_sql))
        intercepted = cls.FORBIDDEN_KEYWORDS.intersection(tokens)
        if intercepted:
            raise HTTPException(status_code=400, detail=f"Security Violation: Structural mutations or database attachments are blocked: {list(intercepted)}")

        # Rule 3: Only permit safe analytical reading queries
        if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
            raise HTTPException(status_code=400, detail="Security Violation: Only explicit SELECT analytical queries are allowed.")

        return cleaned_sql

# ==========================================
# 3. HELPER UTILITIES
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_user_db_path(user_id: str, file_id: str) -> str:
    """Maps user file directly to an isolated SQLite file path."""
    return os.path.join(BASE_DIR, f"db_{user_id}_{file_id}.sqlite")

def get_db_connection(user_id: str, file_id: str) -> sqlite3.Connection:
    db_path = get_user_db_path(user_id, file_id)
    return sqlite3.connect(db_path)

def extract_schema_context(conn: sqlite3.Connection, table_name: str) -> tuple:
    """Inspects structural layout and fetches small data footprint for LLM positioning."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    columns_info = cursor.fetchall()
    schema_str = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])
    
    # Fetch a 3-row sample footprint so Gemini knows what the data actually looks like
    df_sample = pd.read_sql_query(f"SELECT * FROM `{table_name}` LIMIT 3;", conn)
    sample_rows = df_sample.to_dict(orient="records")
    return schema_str, sample_rows

def ensure_sample_files_exist():
    """Generates the three domain-specific sample CSV datasets if they do not exist."""
    # 1. Sales Data Sample
    sales_path = os.path.join(BASE_DIR, "sales_data_sample.csv")
    if not os.path.exists(sales_path):
        import csv
        import random
        from datetime import datetime, timedelta
        random.seed(42)
        categories = {
            "Electronics": ["Wireless Headphones", "Smart Watch", "Bluetooth Speaker", "Charging Dock", "Laptop Stand"],
            "Home & Kitchen": ["Air Fryer", "Coffee Maker", "Vacuum Cleaner", "Scented Candle", "Silicon SpatulaSet"],
            "Clothing": ["Cotton T-Shirt", "Denim Jacket", "Athletic Socks", "Woolen Sweater", "Baseball Cap"],
            "Sports & Outdoors": ["Yoga Mat", "Water Bottle", "Dumbbells (Pair)", "Camping Tent", "Running Shoes"],
            "Beauty & Personal Care": ["Face Moisturizer", "Sunscreen SPF 50", "Electric Toothbrush", "Lip Balm", "Hair Dryer"]
        }
        regions = ["North", "South", "East", "West"]
        customer_first = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
        customer_last = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        start_date = datetime(2025, 1, 1)
        with open(sales_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["order_id", "order_date", "customer_name", "product_category", "product_name", "revenue", "quantity", "region", "is_mobile"])
            for i in range(1, 201):
                order_id = f"ORD{1000 + i}"
                days_offset = random.randint(0, 90)
                order_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                customer_name = f"{random.choice(customer_first)} {random.choice(customer_last)}"
                product_category = random.choice(list(categories.keys()))
                product_name = random.choice(categories[product_category])
                quantity = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
                base_price = random.uniform(10, 100)
                revenue = round(base_price * quantity, 2)
                region = random.choice(regions)
                is_mobile = random.choice([0, 1])
                writer.writerow([order_id, order_date, customer_name, product_category, product_name, revenue, quantity, region, is_mobile])

    # 2. Marketing Campaign Performance Data Sample
    marketing_path = os.path.join(BASE_DIR, "marketing_data_sample.csv")
    if not os.path.exists(marketing_path):
        import csv
        import random
        from datetime import datetime, timedelta
        random.seed(100)
        channels = ["Google Ads", "Meta Ads", "LinkedIn Ads", "TikTok Ads", "YouTube Ads"]
        campaign_names = {
            "Google Ads": ["Search_Brand_Core", "Search_NonBrand_Generic", "PerformanceMax_Retail"],
            "Meta Ads": ["Prospecting_Lookalike_US", "Retargeting_AddToCart", "Instagram_Stories_Promo"],
            "LinkedIn Ads": ["B2B_DecisionMakers", "JobHolders_TechSkill"],
            "TikTok Ads": ["InFeedVideo_GenZ_Trend", "SparkAds_Influencer"],
            "YouTube Ads": ["PreRoll_BrandAwareness", "BumperAds_SummerSale"]
        }
        start_date = datetime(2025, 3, 1)
        with open(marketing_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["campaign_id", "date", "campaign_name", "channel", "spend", "impressions", "clicks", "conversions", "revenue"])
            for i in range(1, 151):
                campaign_id = f"CMP{2000 + i}"
                days_offset = random.randint(0, 60)
                date_str = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                channel = random.choice(channels)
                campaign_name = random.choice(campaign_names[channel])
                spend = round(random.uniform(50, 1500), 2)
                cpc = random.uniform(0.15, 4.5)
                clicks = max(1, int(spend / cpc))
                ctr = random.uniform(0.005, 0.05)
                impressions = int(clicks / ctr)
                conv_rate = random.uniform(0.01, 0.08)
                conversions = int(clicks * conv_rate)
                avg_order_value = random.uniform(40, 180)
                revenue = round(conversions * avg_order_value, 2)
                writer.writerow([campaign_id, date_str, campaign_name, channel, spend, impressions, clicks, conversions, revenue])

    # 3. Product Engagement Data Sample
    product_path = os.path.join(BASE_DIR, "product_data_sample.csv")
    if not os.path.exists(product_path):
        import csv
        import random
        from datetime import datetime, timedelta
        random.seed(200)
        devices = ["iOS", "Android", "Web"]
        features = ["dashboard", "catalog", "settings", "query_compiler", "export_pdf"]
        start_date = datetime(2025, 4, 1)
        with open(product_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "date", "device", "session_duration_mins", "pages_viewed", "actions_performed", "active_feature", "churned"])
            for i in range(1, 151):
                user_id = f"USR{5000 + random.randint(1, 50)}"
                days_offset = random.randint(0, 30)
                date_str = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                device = random.choice(devices)
                session_duration_mins = round(random.uniform(2.5, 45.0), 1)
                pages_viewed = random.randint(1, int(session_duration_mins * 0.8) + 2)
                actions_performed = random.randint(pages_viewed, pages_viewed * 5)
                active_feature = random.choice(features)
                churned = random.choices([0, 1], weights=[92, 8])[0]
                writer.writerow([user_id, date_str, device, session_duration_mins, pages_viewed, actions_performed, active_feature, churned])

def seed_sample_data(user_id: str):
    """Automatically seeds the three domain-specific sample datasets for a user."""
    ensure_sample_files_exist()
    
    datasets = [
        {
            "filename": "sales_data_sample.csv",
            "file_id": "sample_ecommerce_sales",
            "display_name": "sales_data_sample.csv (Seeded E-Commerce)",
            "table_name": "data_sample_ecommerce_sales"
        },
        {
            "filename": "marketing_data_sample.csv",
            "file_id": "sample_marketing_campaigns",
            "display_name": "marketing_data_sample.csv (Seeded Ad Performance)",
            "table_name": "data_sample_marketing_campaigns"
        },
        {
            "filename": "product_data_sample.csv",
            "file_id": "sample_product_engagement",
            "display_name": "product_data_sample.csv (Seeded App Engagement)",
            "table_name": "data_sample_product_engagement"
        }
    ]
    
    for ds in datasets:
        unique_file_id = f"{user_id}_{ds['file_id']}"
        # Check if the dataset metadata is already seeded for this user
        if db_manager.get_file_by_id(unique_file_id, user_id):
            continue
            
        sample_csv_path = os.path.join(BASE_DIR, ds["filename"])
        if not os.path.exists(sample_csv_path):
            continue
            
        try:
            df = pd.read_csv(sample_csv_path)
            table_name = ds["table_name"]
            
            # Clean column headers
            df.columns = [c.strip().replace(' ', '_').replace('-', '_').lower() for c in df.columns]
            df.columns = [''.join(e for e in c if e.isalnum() or e == '_') for c in df.columns]
            
            # Establish isolated SQLite database
            db_path = get_user_db_path(user_id, unique_file_id)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            # Record file metadata in central database
            db_manager.add_file(
                user_id=user_id,
                file_name=ds["display_name"],
                table_name=table_name,
                row_count=len(df),
                columns=list(df.columns),
                sample_rows=df.head(3).to_dict(orient="records"),
                custom_file_id=unique_file_id
            )
        except Exception as e:
            print(f"Error seeding sample dataset {ds['filename']}: {e}")

# ==========================================
# 4. API ROUTE ENDPOINTS
# ==========================================

# --- Auth Routes ---

@app.post("/api/auth/register", response_model=AuthResponse)
def register_user(payload: UserRegisterPayload):
    existing = db_manager.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="A user with this email already exists.")
    
    # Hash password and store user
    pwd_hash = auth.get_password_hash(payload.password)
    user_id = db_manager.create_user(payload.email, pwd_hash)
    
    # Automatically seed sample database for instant experience
    seed_sample_data(user_id)
    
    # Generate token
    token = auth.create_access_token({"sub": user_id})
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        email=payload.email,
        user_id=user_id
    )

@app.post("/api/auth/login", response_model=AuthResponse)
def login_user(payload: UserLoginPayload):
    user = db_manager.get_user_by_email(payload.email)
    if not user or not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect email or password.")
    
    token = auth.create_access_token({"sub": user["id"]})
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        email=user["email"],
        user_id=user["id"]
    )

# --- File Management Routes ---

@app.get("/api/files", response_model=List[FileItem])
def get_user_files(user_id: str = Depends(auth.get_current_user_id)):
    files = db_manager.get_files_by_user(user_id)
    return [
        FileItem(
            id=f["id"],
            file_name=f["file_name"],
            table_name=f["table_name"],
            row_count=f["row_count"],
            columns=f["columns"],
            uploaded_at=f["uploaded_at"]
        ) for f in files
    ]

@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    user_id: str = Depends(auth.get_current_user_id)
):
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid extension format. Only .csv files are allowed.")

    try:
        # Check file size (Read chunks to verify size doesn't exceed 10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds maximum limit of 10MB.")
        
        # Reset file read cursor
        await file.seek(0)
        
        # Read into Pandas with robust encoding support
        try:
            df = pd.read_csv(file.file, encoding='utf-8')
        except UnicodeDecodeError:
            await file.seek(0)
            df = pd.read_csv(file.file, encoding='latin-1')
            
        if len(df) > 100000:
            df = df.head(100000)

        # Standardize column headers to avoid SQLite issues
        df.columns = [str(c).strip().replace(' ', '_').replace('-', '_').lower() for c in df.columns]
        df.columns = [''.join(e for e in str(c) if e.isalnum() or e == '_') for c in df.columns]

        file_id = str(uuid.uuid4())
        table_name = f"data_{file_id.replace('-', '_')}"
        
        # Establish isolated SQLite database
        db_path = get_user_db_path(user_id, file_id)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()

        # Handle NaNs for JSON serialization
        import numpy as np
        safe_df = df.head(3).replace({np.nan: None})

        # Add to metadata catalog
        db_manager.add_file(
            user_id=user_id,
            file_name=file.filename,
            table_name=table_name,
            row_count=len(df),
            columns=list(df.columns),
            sample_rows=safe_df.to_dict(orient="records"),
            custom_file_id=file_id
        )

        return {
            "status": "success",
            "file_id": file_id,
            "table_name": table_name,
            "row_count": len(df),
            "columns": list(df.columns)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@app.get("/api/files/{file_id}/preview", response_model=FilePreviewResponse)
def preview_file(file_id: str, user_id: str = Depends(auth.get_current_user_id)):
    file_info = db_manager.get_file_by_id(file_id, user_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="Requested dataset does not exist.")
        
    db_path = get_user_db_path(user_id, file_id)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file missing.")
        
    try:
        conn = sqlite3.connect(db_path)
        df_preview = pd.read_sql_query(f"SELECT * FROM `{file_info['table_name']}` LIMIT 20;", conn)
        conn.close()
        
        # Replace NaN with None for clean JSON serialization
        df_preview = df_preview.replace({np.nan: None})
        
        return FilePreviewResponse(
            file_info=FileItem(
                id=file_info["id"],
                file_name=file_info["file_name"],
                table_name=file_info["table_name"],
                row_count=file_info["row_count"],
                columns=file_info["columns"],
                uploaded_at=file_info["uploaded_at"]
            ),
            preview_data=df_preview.to_dict(orient="records")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch preview: {str(e)}")

@app.get("/api/files/{file_id}/suggestions")
def get_file_suggestions(file_id: str, user_id: str = Depends(auth.get_current_user_id)):
    file_info = db_manager.get_file_by_id(file_id, user_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
        
    # Format table name to be human readable
    clean_table_name = file_info["table_name"].replace("data_", "").replace("_", " ").title()
    columns = file_info["columns"]
    
    numeric_cols = [c for c in columns if any(k in c.lower() for k in ['revenue', 'spend', 'price', 'amount', 'val', 'count', 'qty', 'quantity', 'total', 'salary', 'age'])]
    cat_cols = [c for c in columns if any(k in c.lower() for k in ['category', 'region', 'type', 'status', 'gender', 'name', 'city', 'state', 'country', 'date', 'department', 'role'])]
    
    num = numeric_cols[0] if numeric_cols else columns[0]
    cat = cat_cols[0] if cat_cols else (columns[1] if len(columns) > 1 else columns[0])
    
    suggestions = [
        {"text": f"What is the total {num} by {cat} in {clean_table_name}?", "category": "Aggregation"},
        {"text": f"Show top 5 {cat} by highest {num}", "category": "Sorting"},
        {"text": f"What is the average {num} across all {clean_table_name}?", "category": "Summary"},
        {"text": f"Show the distribution of {cat}", "category": "Analysis"}
    ]
    
    return {"suggestions": suggestions}

@app.delete("/api/files/{file_id}")
def delete_user_file(file_id: str, user_id: str = Depends(auth.get_current_user_id)):
    file_info = db_manager.get_file_by_id(file_id, user_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="Dataset not found or unauthorized.")
        
    # Delete metadata record (cascades to query logs and chat thread)
    deleted = db_manager.delete_file(file_id, user_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to clear catalog metadata.")
        
    # Remove SQLite file from disk
    db_path = get_user_db_path(user_id, file_id)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception as e:
            # Non-blocking log, metadata was successfully removed
            print(f"Warning: Failed to delete database file {db_path}: {e}")
            
    return {"status": "success", "message": f"Successfully deleted {file_info['file_name']}"}

@app.get("/api/files/{file_id}/profile", response_model=List[ColumnProfile])
def profile_file(file_id: str, user_id: str = Depends(auth.get_current_user_id)):
    file_info = db_manager.get_file_by_id(file_id, user_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="Dataset not found.")
        
    db_path = get_user_db_path(user_id, file_id)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file missing.")
        
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM `{file_info['table_name']}`;", conn)
        conn.close()
        
        profiles = []
        for col_name in df.columns:
            series = df[col_name]
            unique_vals = series.dropna().unique()
            null_count = int(series.isnull().sum())
            total_count = len(series)
            
            # Detect type
            if pd.api.types.is_numeric_dtype(series):
                col_type = "numeric"
                mean_val = float(series.mean()) if not series.isnull().all() else None
                min_val = float(series.min()) if not series.isnull().all() else None
                max_val = float(series.max()) if not series.isnull().all() else None
            else:
                col_type = "categorical" if len(unique_vals) < 30 else "text"
                mean_val, min_val, max_val = None, None, None
                
            # Top values
            top_vals = []
            if len(unique_vals) > 0:
                value_counts = series.value_counts().head(5).to_dict()
                top_vals = [{"value": str(k), "count": int(v)} for k, v in value_counts.items()]
                
            profiles.append(ColumnProfile(
                name=col_name,
                type=col_type,
                count=total_count - null_count,
                unique_count=len(unique_vals),
                null_percentage=round((null_count / total_count) * 100, 2),
                mean=mean_val,
                min=min_val,
                max=max_val,
                top_values=top_vals
            ))
            
        return profiles
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate profile stats: {str(e)}")

# --- Query & Interactive Analytics Routes ---

@app.get("/api/history")
def get_user_history(file_id: Optional[str] = None, user_id: str = Depends(auth.get_current_user_id)):
    return db_manager.get_query_history(user_id, file_id)

@app.delete("/api/chat/{file_id}")
def reset_chat_thread(file_id: str, user_id: str = Depends(auth.get_current_user_id)):
    db_manager.clear_chat_history(user_id, file_id)
    return {"status": "success", "message": "Conversational memory reset successfully."}

# ... (Previous imports and schemas remain the same)

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

@app.post("/api/query", response_model=QueryResultResponse)
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
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True
    )

    try:
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
IMPORTANT: You MUST use the `sql_db_query` tool to execute the query before providing your final answer.
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