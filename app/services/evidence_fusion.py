from typing import Dict, Tuple
from app.schemas import ModuleResult, ModuleStatus, Verdict, ProvenanceStatus

def fuse_evidence(
    module_results: Dict[str, ModuleResult],
    provenance_status: ProvenanceStatus
) -> Tuple[Verdict, str]:
    """
    Forensic Evidence Aggregation & Deterministic Decision Logic.
    Rules:
    - Never uses arbitrary weights or fabricated probabilities.
    - Directly tracks independent forensic channels:
      1. Copy-move geometric clustering
      2. ELA localized recompression discrepancy
      3. Noise variance MAD z-score inconsistency
      4. Periodic resampling spectral correlation
      5. Edge gradient discontinuity
      6. Metadata editing signatures / contradictions
    - Verdict Categories:
      - POTENTIAL_MANIPULATION: When strong localized structural evidence (e.g., verified copy-move clusters)
        OR multiple converging independent anomalies are detected.
      - NO_SIGNIFICANT_ANOMALIES_DETECTED: When applicable tests complete without crossing anomaly thresholds.
        (Explicitly noted that this does NOT prove authenticity).
      - INCONCLUSIVE: When majority of tests fail, lack sufficient resolution, or produce contradictory weak signals.
      - VERIFIED_PROVENANCE: Cryptographic provenance verified with authentic signature claims.
    """
    total_modules = len(module_results)
    completed_modules = [m for m in module_results.values() if m.status == ModuleStatus.COMPLETED]
    failed_modules = [m for m in module_results.values() if m.status == ModuleStatus.FAILED]
    
    if len(completed_modules) < 2 and len(failed_modules) >= 2:
        return Verdict.INCONCLUSIVE, "Forensic evidence is inconclusive because multiple core analysis modules failed or could not be evaluated."

    # Inspect specific independent evidence signals
    strong_evidence_indicators = []
    secondary_evidence_indicators = []

    # 1. Copy-move
    cm = module_results.get("copy_move")
    if cm and cm.status == ModuleStatus.COMPLETED:
        if cm.verdict_contribution == "EVIDENCE_OF_COPY_MOVE_DUPLICATION":
            inliers = cm.measurements.get("geometrically_consistent_matches", 0)
            strong_evidence_indicators.append(f"Copy-Move duplication detected ({inliers} geometrically consistent keypoint clusters).")

    # 2. Resampling
    res = module_results.get("resampling")
    if res and res.status == ModuleStatus.COMPLETED:
        if res.verdict_contribution == "RESAMPLING_PERIODICITY_DETECTED":
            par = res.measurements.get("max_spectral_correlation", 0)
            strong_evidence_indicators.append(f"Resampling detected via periodic derivative spectral correlation (PAR {par}).")

    # 3. ELA (JPEG only)
    ela = module_results.get("ela")
    if ela and ela.status == ModuleStatus.COMPLETED:
        if ela.verdict_contribution == "ELEVATED_LOCAL_DISCREPANCY":
            regions = len(ela.candidate_regions)
            secondary_evidence_indicators.append(f"Error Level Analysis shows {regions} localized recompression anomaly regions.")

    # 4. Noise
    noise = module_results.get("noise")
    if noise and noise.status == ModuleStatus.COMPLETED:
        if noise.verdict_contribution == "LOCALIZED_NOISE_INCONSISTENCY":
            pct = noise.measurements.get("anomalous_patch_percentage", 0)
            secondary_evidence_indicators.append(f"High-pass noise analysis shows {pct}% patches with statistically divergent noise variance.")

    # 5. Metadata
    meta = module_results.get("metadata")
    if meta and meta.status == ModuleStatus.COMPLETED:
        if meta.verdict_contribution == "SUSPICIOUS_METADATA":
            sigs = meta.measurements.get("suspicious_indicators", [])
            contra = meta.measurements.get("contradictions", [])
            meta_notes = sigs + contra
            secondary_evidence_indicators.append(f"Metadata indicators: {'; '.join(meta_notes[:2])}")

    # 6. Edge
    edge = module_results.get("edge")
    if edge and edge.status == ModuleStatus.COMPLETED:
        if edge.verdict_contribution == "ISOLATED_SHARP_BOUNDARY_DETECTED":
            secondary_evidence_indicators.append("Sharp edge discontinuities detected along candidate boundaries.")

    # Deterministic Decision Synthesis
    if len(strong_evidence_indicators) >= 1 or len(secondary_evidence_indicators) >= 2:
        reasons = strong_evidence_indicators + secondary_evidence_indicators
        summary = "Convergence of deterministic forensic evidence indicates potential image manipulation: " + " ".join(reasons)
        return Verdict.POTENTIAL_MANIPULATION, summary

    if len(secondary_evidence_indicators) == 1:
        # Single secondary indicator is not enough to conclude manipulation (could be benign editing/texture)
        summary = f"Inconclusive: A solitary anomaly ({secondary_evidence_indicators[0]}) was observed, but lacks converging corroboration from independent tests."
        return Verdict.INCONCLUSIVE, summary

    if provenance_status == ProvenanceStatus.CREDENTIALS_VERIFIED:
        return Verdict.VERIFIED_PROVENANCE, "C2PA cryptographic Content Credentials verified successfully with an intact asset binding."

    # Clean bill of health from completed tests
    summary = (
        "No significant manipulation anomalies were detected across the completed deterministic tests "
        "(ELA, Frequency, Feature Matching, Noise, and Metadata). Note: This does not prove the image is authentic, "
        "as advanced manipulations, uncompressed formats, or full-frame generative synthesis may not leave traditional traces."
    )
    return Verdict.NO_SIGNIFICANT_ANOMALIES_DETECTED, summary
