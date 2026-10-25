from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.config import settings
from app.schemas import BenchmarkReport, BenchmarkRunRequest
from app.services.dataset_loader import inspect_cg1050_dataset
from app.services.benchmark_evaluation import run_held_out_benchmark

router = APIRouter()

BENCHMARK_CACHE = {}

@router.get("/dataset/status")
def get_dataset_status(dataset_dir: Optional[str] = None):
    target = Path(dataset_dir) if dataset_dir else settings.CG1050_DIR
    manifest, summary = inspect_cg1050_dataset(target)
    return {
        "summary": summary,
        "sample_manifest_items": [item.to_dict() for item in manifest[:5]]
    }

@router.post("/benchmark/run", response_model=BenchmarkReport)
def run_benchmark_endpoint(req: BenchmarkRunRequest):
    target_dir = Path(req.dataset_dir) if req.dataset_dir else settings.CG1050_DIR
    try:
        report = run_held_out_benchmark(
            dataset_dir=target_dir,
            test_ratio=req.test_ratio,
            split_seed=req.split_seed,
            max_samples=req.max_samples
        )
        BENCHMARK_CACHE[report.benchmark_id] = report
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")

@router.get("/benchmark/{benchmark_id}", response_model=BenchmarkReport)
def get_benchmark_report(benchmark_id: str):
    if benchmark_id not in BENCHMARK_CACHE:
        raise HTTPException(status_code=404, detail="Benchmark report not found.")
    return BENCHMARK_CACHE[benchmark_id]
