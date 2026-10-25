import uuid
import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.config import settings
from app.schemas import AnalysisReport, ModuleStatus
from app.services.image_validation import validate_and_load_image, ImageValidationError
from app.services.metadata_analysis import analyze_metadata
from app.services.ela_analysis import analyze_ela
from app.services.frequency_analysis import analyze_fft
from app.services.copy_move_analysis import analyze_copy_move
from app.services.noise_analysis import analyze_noise
from app.services.edge_analysis import analyze_edges
from app.services.resampling_analysis import analyze_resampling
from app.services.jpeg_dct_analysis import analyze_jpeg_dct
from app.services.provenance_verification import verify_c2pa
from app.services.evidence_fusion import fuse_evidence
from app.services.report_generation import generate_pdf_report

router = APIRouter()

# In-memory store for session reports
ANALYSIS_CACHE = {}

@router.post("/analyze", response_model=AnalysisReport)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, executes full non-ML forensic pipeline,
    and returns a structured, publication-grade analysis report.
    """
    # 1. Read bytes safely
    raw_bytes = await file.read()
    filename = file.filename or "upload.bin"

    # 2. Validation and security checks
    try:
        img, metadata = validate_and_load_image(raw_bytes, filename)
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing error: {str(e)}")

    analysis_id = str(uuid.uuid4())[:12]

    # 3. Execute deterministic forensic modules
    results = {}

    # Module A: Metadata
    results["metadata"] = analyze_metadata(img, raw_bytes, metadata.format)

    # Module B: ELA
    results["ela"] = analyze_ela(img, metadata.format, analysis_id, settings.ELA_QUALITY, settings.ELA_SCALE)

    # Module C: Frequency FFT
    results["fft"] = analyze_fft(img, analysis_id)

    # Module D: Resampling
    results["resampling"] = analyze_resampling(img, analysis_id)

    # Module E: Copy-Move
    results["copy_move"] = analyze_copy_move(img, analysis_id)

    # Module F: Noise
    results["noise"] = analyze_noise(img, analysis_id)

    # Module G: Edge
    results["edge"] = analyze_edges(img, analysis_id)

    # Module H: JPEG DCT
    results["jpeg_dct"] = analyze_jpeg_dct(img, raw_bytes, metadata.format)

    # Module I: C2PA
    prov_status, prov_details = verify_c2pa(raw_bytes, filename)

    # 4. Deterministic Evidence Fusion
    verdict, verdict_summary = fuse_evidence(results, prov_status)

    # Counts
    comp = sum(1 for m in results.values() if m.status == ModuleStatus.COMPLETED)
    inconc = sum(1 for m in results.values() if m.status == ModuleStatus.INCONCLUSIVE)
    na = sum(1 for m in results.values() if m.status == ModuleStatus.NOT_APPLICABLE)
    fail = sum(1 for m in results.values() if m.status == ModuleStatus.FAILED)

    disclaimer = (
        "Scientific Integrity Notice: ForenSight uses deterministic signal processing, mathematical feature matching, "
        "and physical noise statistics. Results do not represent legal certitude. Inconclusive findings or absence of "
        "detected anomalies must NEVER be interpreted as absolute proof that an image is unaltered or authentic. "
        "Furthermore, this tool is not an AI-generated media classifier; synthetic images may lack traditional editing seams."
    )

    report = AnalysisReport(
        analysis_id=analysis_id,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_version=settings.APP_VERSION,
        image_metadata=metadata,
        verdict=verdict,
        verdict_summary=verdict_summary,
        provenance_status=prov_status,
        provenance_details=prov_details,
        module_results=results,
        tests_completed=comp,
        tests_inconclusive=inconc,
        tests_not_applicable=na,
        tests_failed=fail,
        disclaimer=disclaimer
    )

    # Save to cache
    ANALYSIS_CACHE[analysis_id] = report

    # Generate PDF
    try:
        generate_pdf_report(report)
    except Exception as e:
        # Non-fatal if PDF generation encounters system font issues
        pass

    return report

@router.get("/analysis/{analysis_id}", response_model=AnalysisReport)
def get_analysis_report(analysis_id: str):
    if analysis_id not in ANALYSIS_CACHE:
        raise HTTPException(status_code=404, detail="Analysis report not found.")
    return ANALYSIS_CACHE[analysis_id]

@router.get("/analysis/{analysis_id}/pdf")
def download_pdf_report(analysis_id: str):
    pdf_path = settings.REPORTS_DIR / f"forensic_report_{analysis_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found.")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=f"ForenSight_{analysis_id}.pdf")
