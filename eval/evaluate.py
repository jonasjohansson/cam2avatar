#!/usr/bin/env python3
"""
Evaluation harness for the VRM x Mixamo webcam-mocap pipeline.

Feeds one of the abba test clips (real full-body human motion) through the same
MediaPipe Holistic -> Kalidokit -> VRM pipeline the app uses, then samples:
  - landmark detection rates (face / pose / hands / world)
  - leg visibility from MediaPipe (knees + ankles)
  - whether the VRM leg bones actually MOVE (variance of their quaternions)
and saves periodic screenshots.

Usage:
  python3 eval/evaluate.py [clip] [legsMode] [seconds]
    clip     : Dancing | Disco | Tai-Chi | Aerobics   (default Dancing)
    legsMode : off | webcam | idle                    (default webcam)
    seconds  : capture duration                       (default 12)

Requires the localhost server (serves ~/Documents/GitHub) to be running, and
playwright (chromium). Output -> eval/out/.
"""
import sys, os, json, statistics
from playwright.sync_api import sync_playwright

APP = "http://localhost/org/jonasjohansson/vrm-mixamo/index.html"
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

clip = sys.argv[1] if len(sys.argv) > 1 else "Dancing"
legs = sys.argv[2] if len(sys.argv) > 2 else "webcam"
secs = int(sys.argv[3]) if len(sys.argv) > 3 else 12

BONES = ["leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg",
         "leftUpperArm", "rightUpperArm", "neck", "spine"]

READ_RIG = """() => {
  const vrm = window.__getVrm && window.__getVrm();
  if (!vrm || !vrm.humanoid) return null;
  const q = (n) => { const b = vrm.humanoid.getNormalizedBoneNode(n);
    return b ? [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w] : null; };
  const out = {}; %s.forEach(n => out[n] = q(n));
  out.__sceneY = vrm.scene.position.y;
  return out;
}""" % json.dumps(BONES)

def movement(series):
    """Sum of per-component stddev across samples = how much the bone moved."""
    vals = [s for s in series if s]
    if len(vals) < 2:
        return 0.0
    return round(sum(statistics.pstdev([v[c] for v in vals]) for c in range(4)), 4)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1100, "height": 720})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(APP, wait_until="networkidle", timeout=60000)

    for _ in range(40):
        if page.locator("#status").inner_text().startswith(("▶", "✗")):
            break
        page.wait_for_timeout(1000)

    page.check("#opt-skeleton")
    # Pick the source option whose label contains the clip keyword
    val = page.evaluate(
        """(kw) => { const o=[...document.querySelectorAll('#source-select option')]
            .find(o=>o.textContent.toLowerCase().includes(kw.toLowerCase()));
            return o ? o.value : null; }""", clip)
    if not val:
        print(f"no clip matching '{clip}'"); browser.close(); sys.exit(1)
    page.select_option("#source-select", val)
    page.locator("#webcam-btn").click()
    # Legs selector only appears once tracking is running.
    page.wait_for_selector("#mocap-opts", state="visible", timeout=15000)
    page.select_option("#opt-legs", legs)

    # Wait until the pipeline reports a pose
    got = False
    for _ in range(40):
        last = page.evaluate("() => window.__mocapLast || null")
        if last and last.get("pose"):
            got = True; break
        page.wait_for_timeout(1000)

    samples, rig_series = [], {b: [] for b in BONES}
    shots = 0
    for i in range(secs):
        last = page.evaluate("() => window.__mocapLast || null")
        rig = page.evaluate(READ_RIG)
        if last:
            samples.append(last)
        if rig:
            for b in BONES:
                rig_series[b].append(rig.get(b))
        if i % 3 == 0:
            page.screenshot(path=os.path.join(OUT, f"{clip}_{legs}_{i:02d}.png"))
            shots += 1
        page.wait_for_timeout(1000)

    def rate(key):
        return round(100 * sum(1 for s in samples if s.get(key)) / max(1, len(samples)))

    legvis = [s.get("legVis", 0) for s in samples if s]
    report = {
        "clip": clip, "legsMode": legs, "samples": len(samples),
        "pose_started": got,
        "detect_%": {k: rate(k) for k in ["face", "pose", "world", "lhand", "rhand"]},
        "legVis_avg": round(statistics.mean(legvis), 2) if legvis else 0,
        "bone_movement": {b: movement(rig_series[b]) for b in BONES},
        "page_errors": errors or "none",
        "screenshots": shots,
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(OUT, f"{clip}_{legs}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    browser.close()
