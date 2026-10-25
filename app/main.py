from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.config import settings
from app.routes import health, analysis, benchmark

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Deterministic Non-ML Digital Image Forensics and Manipulation Detection Web Application."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure writable runtime directories exist prior to mounting
try:
    settings.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# Mount runtime generated files first so /static/generated resolves to the writable GENERATED_DIR
if settings.GENERATED_DIR.exists():
    app.mount("/static/generated", StaticFiles(directory=str(settings.GENERATED_DIR)), name="generated")

# Mount static UI assets (CSS, JS) if directory exists
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Mount reports directory for PDF download
if settings.REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(settings.REPORTS_DIR)), name="reports")

# Include API routers
app.include_router(health.router, prefix="/api", tags=["Health & Capabilities"])
app.include_router(analysis.router, prefix="/api", tags=["Forensic Analysis"])
app.include_router(benchmark.router, prefix="/api", tags=["Benchmark & CG-1050"])

# Root frontend entrypoint
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = settings.BASE_DIR / "templates" / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>ForenSight Backend Active.</h2>")
