"""
Script 5: Master Pipeline — NeuroEmo QPP Analysis
Runs full pipeline for all subjects automatically.

HOW TO USE:
  1. Set START_SUBJECT and END_SUBJECT below
  2. Run: python scripts/05_run_pipeline.py
  3. Leave it running — it saves after each subject
  4. If it crashes, just run again — skips already done subjects
"""

import os
import sys
import time
import numpy as np

# Add scripts folder to path so we can import our other scripts
sys.path.insert(0, os.path.dirname(__file__))

from download_01      import download_subject, download_metadata
from preprocess_02    import preprocess_subject
from extract_qpp_03   import run_qpp_for_subject
from group_analysis_04 import plot_group_summary

# ══════════════════════════════════════════════════════════════
# SETTINGS — CHANGE THESE
# ══════════════════════════════════════════════════════════════
START_SUBJECT = 1   # ← first subject to process
END_SUBJECT   = 40 # ← last subject (start with 5 to test!)

TR_REST      = 2.0   # TR for resting state (check bold.json)
TR_EMOTION   = 2.0   # TR for emotion task  (check bold.json)

SKIP_DOWNLOAD     = False  # True = skip download (already downloaded)
SKIP_PREPROCESS   = False  # True = skip preprocessing (already done)
SKIP_QPP          = False  # True = skip QPP extraction

# ══════════════════════════════════════════════════════════════

def log(message, log_file="logs/pipeline.log"):
    """Write message to both terminal and log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg  = f"[{timestamp}] {message}"
    print(full_msg)
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")


def run_pipeline():

    subject_ids = [f"sub-{i:02d}" for i in range(START_SUBJECT, END_SUBJECT + 1)]

    log("="*60)
    log(f"NeuroEmo QPP Pipeline Starting")
    log(f"Subjects : {subject_ids}")
    log(f"Tasks    : rest + emotion (fe)")
    log("="*60)

    # Track results
    results = {
        "success": [],
        "failed" : [],
        "skipped": []
    }

    # Download metadata once
    if not SKIP_DOWNLOAD:
        try:
            download_metadata()
        except Exception as e:
            log(f"WARNING: metadata download failed: {e}")

    # ── Loop through subjects ──────────────────────────────────
    for sub_id in subject_ids:
        log(f"\n{'='*60}")
        log(f"Processing {sub_id}...")
        log(f"{'='*60}")
        start_time = time.time()

        try:
            # ── Step 1: Download ───────────────────────────────
            if not SKIP_DOWNLOAD:
                log(f"  {sub_id} | Step 1: Downloading...")
                ok = download_subject(int(sub_id.replace("sub-", "")))
                if not ok:
                    log(f"  {sub_id} | FAILED: download error")
                    results["failed"].append(sub_id)
                    continue
            else:
                log(f"  {sub_id} | Step 1: Download SKIPPED")

            # ── Step 2: Preprocess REST ────────────────────────
            if not SKIP_PREPROCESS:
                log(f"  {sub_id} | Step 2a: Preprocessing REST...")
                bold_rest, mask_rest, tr_rest = preprocess_subject(
                    sub_id, task="rest", tr=TR_REST
                )
                if bold_rest is None:
                    log(f"  {sub_id} | WARNING: REST preprocessing failed")
            else:
                log(f"  {sub_id} | Step 2a: Preprocessing SKIPPED")

            # ── Step 3: Preprocess EMOTION ─────────────────────
            if not SKIP_PREPROCESS:
                log(f"  {sub_id} | Step 2b: Preprocessing EMOTION...")
                bold_fe, mask_fe, tr_fe = preprocess_subject(
                    sub_id, task="fe", tr=TR_EMOTION
                )
                if bold_fe is None:
                    log(f"  {sub_id} | WARNING: EMOTION preprocessing failed")
            else:
                log(f"  {sub_id} | Step 2b: Preprocessing SKIPPED")

            # ── Step 4: QPP REST ───────────────────────────────
            if not SKIP_QPP:
                log(f"  {sub_id} | Step 3a: QPP extraction REST...")
                info_rest = run_qpp_for_subject(
                    sub_id, task="rest", tr=TR_REST
                )
            else:
                log(f"  {sub_id} | Step 3a: QPP SKIPPED")
                info_rest = None

            # ── Step 5: QPP EMOTION ────────────────────────────
            if not SKIP_QPP:
                log(f"  {sub_id} | Step 3b: QPP extraction EMOTION...")
                info_fe = run_qpp_for_subject(
                    sub_id, task="fe", tr=TR_EMOTION
                )
            else:
                log(f"  {sub_id} | Step 3b: QPP SKIPPED")
                info_fe = None

            # ── Log results ────────────────────────────────────
            elapsed = time.time() - start_time
            log(f"  {sub_id} | DONE in {elapsed/60:.1f} mins")

            if info_rest:
                log(f"  {sub_id} | REST   → {info_rest['n_peaks']} QPPs | "
                    f"r={info_rest['mean_corr']:.3f}")
            if info_fe:
                log(f"  {sub_id} | EMOTION→ {info_fe['n_peaks']} QPPs | "
                    f"r={info_fe['mean_corr']:.3f}")

            results["success"].append(sub_id)

        except Exception as e:
            log(f"  {sub_id} | CRASHED: {e}")
            results["failed"].append(sub_id)
            import traceback
            traceback.print_exc()
            continue   # keep going with next subject!

    # ── Group Analysis ─────────────────────────────────────────
    log("\n" + "="*60)
    log("Running group analysis...")
    log("="*60)

    if len(results["success"]) > 0:
        plot_group_summary(results["success"])

    # ── Final Summary ──────────────────────────────────────────
    log("\n" + "="*60)
    log("PIPELINE COMPLETE!")
    log("="*60)
    log(f"Successful : {len(results['success'])} subjects done")
    log(f"Failed     : {len(results['failed'])} subjects failed: {results['failed']}")
    log(f"Log saved  → logs/pipeline.log")


if __name__ == "__main__":
    run_pipeline()
