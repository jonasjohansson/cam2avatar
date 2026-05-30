#!/usr/bin/env python3
"""
Test suite: sweep every abba test clip through the mocap pipeline, classify
which are single-person / clearly visible (the canonical test set), and report
tracking quality. The qualifying clips are the ones our test procedure should
run, because single-camera mocap only works with one clearly-framed subject.

Run:  python3 eval/run_suite.py [legsMode] [seconds]
Output: eval/out/<clip>_*.png + eval/out/suite_report.json + a printed table.
"""
import sys, os, json
from evaluate import run

LEGS = sys.argv[1] if len(sys.argv) > 1 else "webcam"
SECS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
BASE = "/org/jonasjohansson/abba/assets/test-clips"

# We only use single-person, clearly-framed clips. Add a candidate here to vet
# it — if it scores single_person_clear, promote it into test_procedure.py.
CLIPS = [
    f"{BASE}/Aerobics/pixabay-149195_aerobics-fitness-health-exercise.mp4",
    # --- excluded (group / subject too small), kept for reference, not run ---
    # f"{BASE}/Dancing/pexels-36131698_dancing.mp4",
    # f"{BASE}/Dancing/pexels-6761220_dancing.mp4",
    # f"{BASE}/Disco-Dancing/pexels-34236697_disco-dancing.mp4",
    # f"{BASE}/Disco-Dancing/pexels-34630354_disco-dancing.mp4",
    # f"{BASE}/Aerobics/pixabay-75644_gymnast-aerobics-exercise-fitness-workou.mp4",
]

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

reports = []
for clip in CLIPS:
    print(f"--- evaluating {clip.split('/')[-1]} ---", flush=True)
    try:
        reports.append(run(clip, LEGS, SECS))
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

# Table
print(f"\n{'clip':<46}{'pose%':>7}{'legVis':>8}{'legMove':>9}  single-person")
print("-" * 86)
for r in reports:
    legmove = round(sum(r["bone_movement"][b] for b in
                        ["leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg"]), 3)
    mark = "YES" if r["single_person_clear"] else "no"
    print(f"{r['clip']:<46}{r['detect_%']['pose']:>6}%{r['legVis_avg']:>8}{legmove:>9}  {mark}")

qualifying = [r["source"] for r in reports if r["single_person_clear"]]
print("\nCanonical test set (single-person, clearly visible):")
for q in qualifying:
    print("  -", q)

with open(os.path.join(OUT, "suite_report.json"), "w") as f:
    json.dump({"legsMode": LEGS, "seconds": SECS, "reports": reports,
               "canonical_set": qualifying}, f, indent=2)
print(f"\nWrote {OUT}/suite_report.json")
