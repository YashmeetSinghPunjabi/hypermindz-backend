from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

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
    ai_model: Optional[str] = "gemini-2.5-flash"
    query_mode: Optional[str] = "nl"

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
