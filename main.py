import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db_manager import init_db
import langchain
import logging
import sys

# Configure clean, organized logging
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': '\033[93m',
        'INFO': '\033[94m',
        'DEBUG': '\033[92m',
        'CRITICAL': '\033[91m',
        'ERROR': '\033[91m'
    }
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        reset = '\033[0m'
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter('\n[%(asctime)s] %(levelname)s [%(name)s]: %(message)s', datefmt='%H:%M:%S'))

logging.basicConfig(level=logging.INFO, handlers=[handler])

# Silence the extremely noisy third-party logs
for logger_name in ["httpx", "httpcore", "passlib", "asyncio", "google_genai"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Leave LangChain debugging on to see AI thoughts, but without the HTTP noise
langchain.debug = True

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