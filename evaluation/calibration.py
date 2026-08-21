"""
Recognition evaluation and threshold calibration.

 This tooling measures the similarity distributions of:
  - GENUINE pairs (same identity, different samples)
  - IMPOSTOR pairs (different identities)

 It computes:
  - False Match Rate (FMR) at various thresholds
  - False Non-Match Rate (FNMR) at various thresholds
  - Equal Error Rate (EER)
  - Recommended operating points

 Usage:
    python -m evaluation.calibration --db data/biometric.db
    # or: python evaluation/calibration.py --db data/biometric.db

 IMPORTANT: This tooling requires enrolled identities with multiple samples
 to produce meaningful results. With too few identities or samples, the
 statistics are not reliable. The tool will report sample size and
 confidence level.

 This is the ONLY correct way to choose a recognition threshold.
 The default threshold in config is a conservative starting point that
 MUST be calibrated with real enrollment data using this tool.
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from config import Config
from storage.db import BiometricDB

logger = logging.getLogger(__name__)


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalized embeddings."""
    return float(np.dot(a.flatten(), b.flatten()))


def evaluate_genuine_impostor(db: BiometricDB) -> dict:
    """Compute genuine and impostor similarity distributions.

    Returns:
        {
            'genuine_sims': list[float],
            'impostor_sims': list[float],
            'num_identities': int,
            'num_genuine_pairs': int,
            'num_impostor_pairs': int,
        }
    """
    identities = db.get_all_identities()
    genuine_sims = []
    impostor_sims = []

    # Collect all sample embeddings per identity
    all_embeddings: dict[str, list[np.ndarray]] = {}
    for ident in identities:
        embs = db.get_embeddings_for_identity(ident["identity_id"])
        if len(embs) >= 2:
            all_embeddings[ident["identity_id"]] = embs

    identity_ids = list(all_embeddings.keys())

    # Genuine pairs: all pairs within the same identity
    for sid, embs in all_embeddings.items():
        n = len(embs)
        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_cosine_similarity(embs[i], embs[j])
                genuine_sims.append(sim)

    # Impostor pairs: best template-to-template similarity across identities
    templates = db.get_all_templates()
    template_map = {tid: vec for tid, vec in templates}
    for i in range(len(identity_ids)):
        for j in range(i + 1, len(identity_ids)):
            sid_a = identity_ids[i]
            sid_b = identity_ids[j]
            if sid_a in template_map and sid_b in template_map:
                sim = compute_cosine_similarity(
                    template_map[sid_a], template_map[sid_b]
                )
                impostor_sims.append(sim)

    # Also compute sample-to-template impostor pairs (more comprehensive)
    # For each identity's samples vs other identities' templates
    for sid_a, embs in all_embeddings.items():
        for sid_b, _ in all_embeddings.items():
            if sid_a == sid_b:
                continue
            if sid_b not in template_map:
                continue
            template_b = template_map[sid_b]
            # Use the best sample-to-template similarity (max)
            best_sim = max(
                compute_cosine_similarity(emb, template_b) for emb in embs
            )
            impostor_sims.append(best_sim)

    return {
        "genuine_sims": genuine_sims,
        "impostor_sims": impostor_sims,
        "num_identities": len(all_embeddings),
        "num_genuine_pairs": len(genuine_sims),
        "num_impostor_pairs": len(impostor_sims),
    }


def compute_fmr_fnmr(genuine_sims: list[float],
                     impostor_sims: list[float],
                     thresholds: np.ndarray | None = None) -> dict:
    """Compute False Match Rate and False Non-Match Rate at various thresholds.

    FMR = fraction of impostor pairs that score >= threshold (false accept)
    FNMR = fraction of genuine pairs that score < threshold (false reject)
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 0.8, 0.01)

    genuine = np.array(genuine_sims)
    impostor = np.array(impostor_sims)

    results = []
    for t in thresholds:
        if len(genuine) > 0:
            fnmr = float(np.mean(genuine < t))
        else:
            fnmr = 0.0
        if len(impostor) > 0:
            fmr = float(np.mean(impostor >= t))
        else:
            fmr = 0.0
        results.append({
            "threshold": float(t),
            "fmr": fmr,
            "fnmr": fnmr,
        })

    # Find EER (Equal Error Rate)
    eer = None
    eer_threshold = None
    for r in results:
        if abs(r["fmr"] - r["fnmr"]) < 0.01:
            eer = (r["fmr"] + r["fnmr"]) / 2
            eer_threshold = r["threshold"]
            break

    return {
        "curves": results,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }


def recommend_threshold(genuine_sims: list[float],
                        impostor_sims: list[float]) -> dict:
    """Recommend a conservative acceptance threshold.

    Policy: choose the threshold that minimizes FMR while keeping FNMR
    below 5% (i.e., ≤5% of genuine pairs are falsely rejected).

    If insufficient data, return the config default with a warning.
    """
    if len(genuine_sims) < 5 or len(impostor_sims) < 5:
        return {
            "recommended": Config.get("recognition", "accept_threshold"),
            "eer": None,
            "confidence": "LOW — insufficient data (need ≥5 genuine and ≥5 impostor pairs)",
            "genuine_count": len(genuine_sims),
            "impostor_count": len(impostor_sims),
        }

    thresholds = np.arange(0.0, 0.8, 0.01)
    curves = compute_fmr_fnmr(genuine_sims, impostor_sims, thresholds)

    # Find the threshold where FNMR ≤ 5% and FMR is minimized
    best_t = None
    best_fmr = 1.0
    for r in curves["curves"]:
        if r["fnmr"] <= 0.05 and r["fmr"] < best_fmr:
            best_t = r["threshold"]
            best_fmr = r["fmr"]

    if best_t is None:
        # All thresholds have FNMR > 5% — distributions overlap badly
        best_t = Config.get("recognition", "accept_threshold")

    return {
        "recommended": best_t,
        "eer": curves["eer"],
        "eer_threshold": curves["eer_threshold"],
        "confidence": "MEDIUM" if len(genuine_sims) >= 20 else "LOW",
        "genuine_count": len(genuine_sims),
        "impostor_count": len(impostor_sims),
        "fmr_at_recommended": best_fmr,
    }


def run_evaluation(db_path: str | None = None) -> dict:
    """Run the full evaluation. Returns a results dict."""
    db = BiometricDB(db_path) if db_path else BiometricDB.safe_open()

    eval_data = evaluate_genuine_impostor(db)
    recommendation = recommend_threshold(
        eval_data["genuine_sims"],
        eval_data["impostor_sims"],
    )

    return {
        "evaluation": eval_data,
        "recommendation": recommendation,
    }


def main():
    parser = argparse.ArgumentParser(
        description="gen2 recognition calibration tool"
    )
    parser.add_argument("--db", default=None, help="Path to biometric.db")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    print("=" * 60)
    print("gen2 Recognition Calibration")
    print("=" * 60)

    results = run_evaluation(args.db)
    ev = results["evaluation"]
    rec = results["recommendation"]

    print(f"\nIdentities with ≥2 samples: {ev['num_identities']}")
    print(f"Genuine pairs: {ev['num_genuine_pairs']}")
    print(f"Impostor pairs: {ev['num_impostor_pairs']}")

    if ev["genuine_sims"]:
        g = np.array(ev["genuine_sims"])
        print(f"\nGenuine similarity: mean={g.mean():.3f}, "
              f"std={g.std():.3f}, min={g.min():.3f}, max={g.max():.3f}")
    if ev["impostor_sims"]:
        i = np.array(ev["impostor_sims"])
        print(f"Impostor similarity: mean={i.mean():.3f}, "
              f"std={i.std():.3f}, min={i.min():.3f}, max={i.max():.3f}")

    print(f"\nRecommended accept threshold: {rec['recommended']:.3f}")
    if rec.get("eer"):
        print(f"EER: {rec['eer']:.3f} at threshold {rec['eer_threshold']:.3f}")
    print(f"Confidence: {rec['confidence']}")
    print(f"Current config threshold: {Config.get('recognition', 'accept_threshold')}")

    if ev["num_genuine_pairs"] < 5:
        print("\n⚠️  Insufficient data for reliable calibration.")
        print("   Enroll more identities (each with ≥5 samples) and re-run.")


if __name__ == "__main__":
    main()
