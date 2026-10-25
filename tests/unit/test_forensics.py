import pytest
import io
from pathlib import Path
from PIL import Image

from app.services.image_validation import validate_and_load_image, ImageValidationError
from app.services.metadata_analysis import analyze_metadata
from app.services.ela_analysis import analyze_ela
from app.services.frequency_analysis import analyze_fft
from app.services.copy_move_analysis import analyze_copy_move
from app.services.noise_analysis import analyze_noise
from app.services.edge_analysis import analyze_edges
from app.services.resampling_analysis import analyze_resampling
from app.services.jpeg_dct_analysis import analyze_jpeg_dct
from app.services.evidence_fusion import fuse_evidence
from app.services.benchmark_evaluation import compute_classification_metrics
from app.schemas import ModuleStatus, Verdict, ProvenanceStatus

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

def test_image_validation_valid():
    jpg_path = FIXTURES_DIR / "pristine_sample.jpg"
    raw = jpg_path.read_bytes()
    img, meta = validate_and_load_image(raw, "test.jpg")
    assert meta.format == "JPEG"
    assert meta.width == 400
    assert meta.height == 400
    assert len(meta.sha256) == 64

def test_image_validation_corrupt():
    corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # Incomplete JPEG header
    with pytest.raises(ImageValidationError):
        validate_and_load_image(corrupt_bytes, "corrupt.jpg")

def test_image_validation_invalid_signature():
    fake_png = b"NOT_A_VALID_HEADER_FOR_IMAGE"
    with pytest.raises(ImageValidationError):
        validate_and_load_image(fake_png, "fake.png")

def test_ela_jpeg_vs_png():
    jpg_path = FIXTURES_DIR / "pristine_sample.jpg"
    with Image.open(jpg_path) as img:
        res_jpg = analyze_ela(img, "JPEG", "test_ela_jpg")
        assert res_jpg.status == ModuleStatus.COMPLETED
        assert "mean_error" in res_jpg.measurements

    png_path = FIXTURES_DIR / "pristine_sample.png"
    with Image.open(png_path) as img:
        res_png = analyze_ela(img, "PNG", "test_ela_png")
        assert res_png.status == ModuleStatus.NOT_APPLICABLE

def test_fft_frequency_valid():
    jpg_path = FIXTURES_DIR / "pristine_sample.jpg"
    with Image.open(jpg_path) as img:
        res = analyze_fft(img, "test_fft")
        assert res.status == ModuleStatus.COMPLETED
        assert "spectral_entropy" in res.measurements
        assert "low_frequency_energy_pct" in res.measurements

def test_copy_move_detection():
    # Pristine image should NOT trigger copy-move clusters
    pristine_p = FIXTURES_DIR / "pristine_sample.jpg"
    with Image.open(pristine_p) as img:
        res_clean = analyze_copy_move(img, "test_cm_clean")
        assert res_clean.status == ModuleStatus.COMPLETED
        assert res_clean.verdict_contribution == "NO_COPY_MOVE_CLUSTERS_DETECTED"

    # Synthetic copy-move image has identical cloned object
    cm_p = FIXTURES_DIR / "copymove_sample.jpg"
    with Image.open(cm_p) as img:
        res_cm = analyze_copy_move(img, "test_cm_tampered")
        assert res_cm.status == ModuleStatus.COMPLETED
        assert res_cm.measurements["ratio_test_matches"] > 0

def test_noise_analysis():
    jpg_path = FIXTURES_DIR / "pristine_sample.jpg"
    with Image.open(jpg_path) as img:
        res = analyze_noise(img, "test_noise")
        assert res.status == ModuleStatus.COMPLETED
        assert "global_median_noise_sigma" in res.measurements

def test_edge_analysis():
    jpg_path = FIXTURES_DIR / "pristine_sample.jpg"
    with Image.open(jpg_path) as img:
        res = analyze_edges(img, "test_edge")
        assert res.status == ModuleStatus.COMPLETED
        assert "global_edge_density_pct" in res.measurements

def test_resampling_analysis():
    resampled_p = FIXTURES_DIR / "resampled_sample.jpg"
    with Image.open(resampled_p) as img:
        res = analyze_resampling(img, "test_resample")
        assert res.status == ModuleStatus.COMPLETED
        assert "max_spectral_correlation" in res.measurements

def test_classification_metrics_math():
    # Hand-verifiable metrics test
    # y_true = [1, 1, 0, 0]
    # y_pred = [1, 0, 0, 1]
    # TP=1, FN=1, TN=1, FP=1 -> Acc = 2/4 = 0.50
    m = compute_classification_metrics([1, 1, 0, 0], [1, 0, 0, 1], inconclusive_count=0)
    assert m.accuracy == 0.50
    assert m.tp == 1
    assert m.tn == 1
    assert m.fp == 1
    assert m.fn == 1
    assert m.precision == 0.50
    assert m.recall == 0.50
    assert m.f1_score == 0.50

def test_evidence_fusion_verdict():
    # If copy-move detects strong duplication, overall verdict must be POTENTIAL_MANIPULATION
    jpg_path = FIXTURES_DIR / "pristine_sample.jpg"
    with Image.open(jpg_path) as img:
        cm_mod = analyze_copy_move(img, "test_cm")
        # Artificially set contribution to test rule
        cm_mod.verdict_contribution = "EVIDENCE_OF_COPY_MOVE_DUPLICATION"
        cm_mod.measurements["geometrically_consistent_matches"] = 15
        
        verdict, summary = fuse_evidence({"copy_move": cm_mod}, ProvenanceStatus.CREDENTIALS_ABSENT)
        assert verdict == Verdict.POTENTIAL_MANIPULATION
