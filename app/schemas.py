from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ModuleStatus(str, Enum):
    COMPLETED = "COMPLETED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED = "FAILED"

class Verdict(str, Enum):
    POTENTIAL_MANIPULATION = "POTENTIAL_MANIPULATION"
    NO_SIGNIFICANT_ANOMALIES_DETECTED = "NO_SIGNIFICANT_ANOMALIES_DETECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    VERIFIED_PROVENANCE = "VERIFIED_PROVENANCE"

class ProvenanceStatus(str, Enum):
    CREDENTIALS_ABSENT = "CREDENTIALS_ABSENT"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    SIGNATURE_VALID = "SIGNATURE_VALID"
    TRUST_STATUS_UNKNOWN = "TRUST_STATUS_UNKNOWN"
    CREDENTIALS_VERIFIED = "CREDENTIALS_VERIFIED"

class ModuleResult(BaseModel):
    module_name: str
    status: ModuleStatus
    verdict_contribution: Optional[str] = None
    measurements: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    candidate_regions: List[Dict[str, Any]] = Field(default_factory=list)
    visualizations: Dict[str, str] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    confounding_factors: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None

class ImageMetadata(BaseModel):
    filename: str
    sha256: str
    file_size_bytes: int
    format: str
    width: int
    height: int
    color_mode: str
    exif: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

class AnalysisReport(BaseModel):
    analysis_id: str
    created_at: str
    config_version: str
    image_metadata: ImageMetadata
    verdict: Verdict
    verdict_summary: str
    provenance_status: ProvenanceStatus
    provenance_details: Dict[str, Any] = Field(default_factory=dict)
    module_results: Dict[str, ModuleResult] = Field(default_factory=dict)
    tests_completed: int = 0
    tests_inconclusive: int = 0
    tests_not_applicable: int = 0
    tests_failed: int = 0
    disclaimer: str

class BenchmarkRunRequest(BaseModel):
    dataset_dir: Optional[str] = None
    split_seed: int = 42
    test_ratio: float = 0.3
    max_samples: Optional[int] = None

class BenchmarkMetrics(BaseModel):
    evaluated_images: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    fpr: float
    f1_score: float
    inconclusive_count: int
    inconclusive_rate: float
    confusion_matrix: List[List[int]]
    confidence_interval_95: Dict[str, float]
    per_type_metrics: Dict[str, Any] = Field(default_factory=dict)
    localization_metrics: Optional[Dict[str, Any]] = None

class BenchmarkReport(BaseModel):
    benchmark_id: str
    created_at: str
    dataset_path: str
    total_manifest_pairs: int
    split_seed: int
    calibration_metrics: BenchmarkMetrics
    held_out_metrics: BenchmarkMetrics
    frozen_thresholds: Dict[str, Any]
    target_80_achieved: bool
    notes: List[str]
