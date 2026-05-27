import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db_manager import init_db

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

# ==========================================
# ROUTER REGISTRATION
# ==========================================

from routers import auth_routes, files_routes, query_routes
app.include_router(auth_routes.router)
app.include_router(files_routes.router)
app.include_router(query_routes.router)