import os
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

import db_manager
import auth
from schemas import FileItem, ColumnProfile, FilePreviewResponse
from utils import get_user_db_path

router = APIRouter(
    prefix="/api",
    tags=["files"]
)

@router.get("/files", response_model=List[FileItem])
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

@router.post("/upload")
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

@router.get("/files/{file_id}/preview", response_model=FilePreviewResponse)
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

@router.get("/files/{file_id}/suggestions")
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

@router.delete("/files/{file_id}")
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

@router.get("/files/{file_id}/profile", response_model=List[ColumnProfile])
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
