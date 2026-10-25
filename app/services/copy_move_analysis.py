from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2
from app.config import settings
from app.schemas import ModuleResult, ModuleStatus

def analyze_copy_move(img: Image.Image, analysis_id: str) -> ModuleResult:
    """
    Forensic Algorithm E: Traditional Local Feature-Based Copy-Move Forgery Detection.
    Uses SIFT (or ORB fallback), extracts keypoints & descriptors, performs 2NN ratio test
    with self-matching prevention, enforces spatial separation and geometric consistency.
    """
    citations = [
        "Amerini, I., Ballan, L., Caldelli, R., Del Bimbo, A., & Turchi, L. (2011). A SIFT-based forensic method for copy-move attack detection and transformation recovery. IEEE TIFS, 6(3), 1099-1110.",
        "Silva, E., Carvalho, T., Ferreira, A., & Rocha, A. (2015). Going further into copy-move forgery detection: Combining local features and dense field approaches. IEEE TIFS."
    ]
    limitations = [
        "Identical natural textures (repeated bricks, windows in facades, identical tree leaves) can produce genuine visual matches.",
        "Small or highly smoothed/blurred duplicated patches may lack sufficient salient keypoints.",
        "Does not prove malicious intent: repeated elements may be natural architectural symmetry."
    ]
    confounding_factors = [
        "High-contrast repeating geometric patterns, wallpaper, textile fabric motifs, and tiled floors."
    ]

    try:
        # Convert to OpenCV grayscale
        orig_rgb = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Use SIFT if available, otherwise ORB
        try:
            detector = cv2.SIFT_create(nfeatures=settings.SIFT_MAX_FEATURES)
            use_sift = True
        except Exception:
            detector = cv2.ORB_create(nfeatures=settings.SIFT_MAX_FEATURES)
            use_sift = False

        keypoints, descriptors = detector.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return ModuleResult(
                module_name="Copy-Move Detection",
                status=ModuleStatus.INCONCLUSIVE,
                verdict_contribution="INSUFFICIENT_KEYPOINTS",
                measurements={"keypoints_detected": len(keypoints) if keypoints else 0, "descriptor_type": "SIFT" if use_sift else "ORB"},
                thresholds={"min_features_required": 10},
                candidate_regions=[],
                visualizations={},
                limitations=limitations,
                confounding_factors=confounding_factors,
                citations=citations
            )

        # In-image feature matching with k-Nearest Neighbors (k=3 or 4 to skip identical self-match)
        if use_sift:
            # FLANN matcher for float descriptors
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Find 4 nearest neighbors for each descriptor within the same image
        raw_matches = matcher.knnMatch(descriptors, descriptors, k=min(4, len(descriptors)))

        good_matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint, float]] = []
        min_dist_sq = settings.MIN_KEYPOINT_DISTANCE ** 2

        for match_group in raw_matches:
            if len(match_group) < 2:
                continue
            # match_group[0] is always the keypoint itself (dist ~ 0)
            # Compare neighbor 1 and neighbor 2 for Lowe's ratio test
            self_m = match_group[0]
            m1 = match_group[1]
            m2 = match_group[2] if len(match_group) > 2 else None

            kp_source = keypoints[self_m.queryIdx]
            kp_target = keypoints[m1.trainIdx]

            # 1. Enforce spatial distance between query and match (reject trivial neighbor keypoints)
            dx = kp_source.pt[0] - kp_target.pt[0]
            dy = kp_source.pt[1] - kp_target.pt[1]
            spatial_dist_sq = dx * dx + dy * dy

            if spatial_dist_sq < min_dist_sq:
                # If m1 was too close physically, check m2
                if m2 is not None:
                    kp_target = keypoints[m2.trainIdx]
                    dx = kp_source.pt[0] - kp_target.pt[0]
                    dy = kp_source.pt[1] - kp_target.pt[1]
                    spatial_dist_sq = dx * dx + dy * dy
                    if spatial_dist_sq >= min_dist_sq:
                        # Ratio test against 3rd neighbor if exists
                        m3_dist = match_group[3].distance if len(match_group) > 3 else (m2.distance / settings.SIFT_RATIO_THRESH)
                        if m2.distance < settings.SIFT_RATIO_THRESH * m3_dist:
                            good_matches.append((kp_source, kp_target, m2.distance))
                continue

            # Standard Lowe's ratio test: m1.distance < ratio * m2.distance
            if m2 is not None and (m1.distance < settings.SIFT_RATIO_THRESH * m2.distance):
                good_matches.append((kp_source, kp_target, m1.distance))

        # Filter out symmetric duplicates (A->B and B->A)
        unique_pairs = []
        seen_coords = set()
        for kp1, kp2, dist in good_matches:
            p1 = (round(kp1.pt[0], 1), round(kp1.pt[1], 1))
            p2 = (round(kp2.pt[0], 1), round(kp2.pt[1], 1))
            pair_key = tuple(sorted([p1, p2]))
            if pair_key not in seen_coords and p1 != p2:
                seen_coords.add(pair_key)
                unique_pairs.append((kp1, kp2, dist))

        # Geometric consistency verification using RANSAC Affine/Homography
        geometrically_consistent_matches = []
        candidate_regions = []

        if len(unique_pairs) >= 4:
            src_pts = np.float32([p[0].pt for p in unique_pairs]).reshape(-1, 1, 2)
            dst_pts = np.float32([p[1].pt for p in unique_pairs]).reshape(-1, 1, 2)

            # RANSAC homography estimation
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                matches_mask = mask.ravel().tolist()
                for idx, is_inlier in enumerate(matches_mask):
                    if is_inlier:
                        geometrically_consistent_matches.append(unique_pairs[idx])

                # If we have consistent clusters, extract bounding regions
                if len(geometrically_consistent_matches) >= settings.MIN_COPY_MOVE_MATCHES:
                    inlier_src = np.float32([p[0].pt for p in geometrically_consistent_matches])
                    inlier_dst = np.float32([p[1].pt for p in geometrically_consistent_matches])

                    x1, y1, w1, h1 = cv2.boundingRect(np.int32(inlier_src))
                    x2, y2, w2, h2 = cv2.boundingRect(np.int32(inlier_dst))

                    candidate_regions.append({
                        "label": "Source Region",
                        "bounding_box": [int(x1), int(y1), int(w1), int(h1)],
                        "inlier_points_count": len(inlier_src)
                    })
                    candidate_regions.append({
                        "label": "Duplicated Region",
                        "bounding_box": [int(x2), int(y2), int(w2), int(h2)],
                        "inlier_points_count": len(inlier_dst)
                    })

        # Save diagnostic visualization
        viz_img = orig_rgb.copy()
        # Draw lines for consistent matches in neon green, others in orange
        draw_matches = geometrically_consistent_matches if len(geometrically_consistent_matches) > 0 else unique_pairs[:50]
        
        for kp1, kp2, _ in draw_matches:
            pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
            pt2 = (int(kp2.pt[0]), int(kp2.pt[1]))
            color = (0, 255, 0) if len(geometrically_consistent_matches) >= settings.MIN_COPY_MOVE_MATCHES else (255, 165, 0)
            cv2.circle(viz_img, pt1, 4, (255, 0, 0), -1)
            cv2.circle(viz_img, pt2, 4, (0, 0, 255), -1)
            cv2.line(viz_img, pt1, pt2, color, 1, cv2.LINE_AA)

        # Draw candidate bounding boxes
        for reg in candidate_regions:
            rx, ry, rw, rh = reg["bounding_box"]
            cv2.rectangle(viz_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        viz_filename = f"copymove_{analysis_id}.jpg"
        viz_path = settings.GENERATED_DIR / viz_filename
        cv2.imwrite(str(viz_path), cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))

        # Contribution
        num_consistent = len(geometrically_consistent_matches)
        if num_consistent >= settings.MIN_COPY_MOVE_MATCHES:
            contribution = "EVIDENCE_OF_COPY_MOVE_DUPLICATION"
        elif len(unique_pairs) > 25:
            contribution = "POSSIBLE_REPETITIVE_TEXTURE_OR_DUPLICATION"
        else:
            contribution = "NO_COPY_MOVE_CLUSTERS_DETECTED"

        measurements = {
            "detector_used": "SIFT" if use_sift else "ORB",
            "total_keypoints": len(keypoints),
            "ratio_test_matches": len(unique_pairs),
            "geometrically_consistent_matches": num_consistent,
            "min_spatial_distance_threshold_px": settings.MIN_KEYPOINT_DISTANCE,
            "ransac_inlier_ratio": round(num_consistent / max(1, len(unique_pairs)), 3)
        }

        return ModuleResult(
            module_name="Copy-Move Duplication Analysis",
            status=ModuleStatus.COMPLETED,
            verdict_contribution=contribution,
            measurements=measurements,
            thresholds={
                "ratio_threshold": settings.SIFT_RATIO_THRESH,
                "min_spatial_distance": settings.MIN_KEYPOINT_DISTANCE,
                "min_inliers_for_detection": settings.MIN_COPY_MOVE_MATCHES,
                "ransac_reproj_threshold": 5.0
            },
            candidate_regions=candidate_regions,
            visualizations={
                "copymove_matches": f"/static/generated/{viz_filename}"
            },
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
    except Exception as e:
        return ModuleResult(
            module_name="Copy-Move Duplication Analysis",
            status=ModuleStatus.FAILED,
            error_message=str(e),
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
