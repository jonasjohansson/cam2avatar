#!/usr/bin/env python3
"""
Test procedure (regression): run the canonical single-person clips through the
mocap pipeline and assert tracking quality against thresholds. Exit code 0 if
all pass, 1 otherwise — so it can gate changes.

The canonical set is the clips that are single-person and clearly full-body
framed (established by eval/run_suite.py). Single-camera mocap only works with
one clearly-framed subject, so group/partial clips are intentionally excluded.

Run:  python3 eval/test_procedure.py [legsMode] [seconds]
"""
import sys, os, json
from evaluate import run

BASE = "/org/jonasjohansson/abba/assets/test-clips"
CANONICAL = [
    ("Aerobics", f"{BASE}/Aerobics/pixabay-149195_aerobics-fitness-health-exercise.mp4"),
]
# Pass thresholds — derived from observed good runs with headroom.
THRESH = {"pose": 90, "legVis": 0.55, "legMove": 0.05}

LEGS = sys.argv[1] if len(sys.argv) > 1 else "webcam"
SECS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

results, fails = [], 0
for name, url in CANONICAL:
    print(f"--- {name} ---", flush=True)
    r = run(url, LEGS, SECS)
    legmove = round(sum(r["bone_movement"][b] for b in
                        ["leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg"]), 3)
    pose = r["detect_%"]["pose"]
    ok = pose >= THRESH["pose"] and r["legVis_avg"] >= THRESH["legVis"] and legmove >= THRESH["legMove"]
    fails += 0 if ok else 1
    results.append({"name": name, "pass": ok, "pose": pose, "legVis": r["legVis_avg"], "legMove": legmove})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: pose={pose}% legVis={r['legVis_avg']} legMove={legmove}", flush=True)

print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILED"))
out = os.path.join(os.path.dirname(__file__), "out", "procedure_result.json")
with open(out, "w") as f:
    json.dump({"legsMode": LEGS, "seconds": SECS, "thresholds": THRESH,
               "results": results, "passed": fails == 0}, f, indent=2)
sys.exit(1 if fails else 0)
