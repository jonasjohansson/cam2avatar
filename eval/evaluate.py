#!/usr/bin/env python3
"""
Evaluation harness for the VRM x Mixamo webcam-mocap pipeline.

Feeds a test clip (real human motion) through the same MediaPipe Holistic ->
Kalidokit -> VRM pipeline the app uses, then samples:
  - landmark detection rates (face / pose / hands / world)
  - leg visibility from MediaPipe (knees + ankles)
  - whether the VRM leg bones actually MOVE (variance of their quaternions)
and saves periodic screenshots.

Usage:
  python3 eval/evaluate.py <clip> [legsMode] [seconds]
    clip     : a keyword that matches a source dropdown label (e.g. Tai-Chi),
               OR an absolute web path to a video (e.g. /org/.../foo.mp4)
    legsMode : off | webcam | idle   (default webcam)
    seconds  : capture duration      (default 12)

Requires the localhost server (serves ~/Documents/GitHub) + playwright chromium.
Output -> eval/out/.
"""
import sys, os, json, statistics, re
from playwright.sync_api import sync_playwright

APP = "http://localhost/org/jonasjohansson/vrm-mixamo/index.html"
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

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


def _movement(series):
    vals = [s for s in series if s]
    if len(vals) < 2:
        return 0.0
    return round(sum(statistics.pstdev([v[c] for v in vals]) for c in range(4)), 4)


def _label(clip):
    if clip.startswith("/") or clip.startswith("http"):
        return re.sub(r"[^a-zA-Z0-9]+", "-", os.path.splitext(os.path.basename(clip))[0])[:40]
    return clip


def evaluate(page, clip, legs="webcam", secs=12, shot_prefix=None):
    prefix = shot_prefix or _label(clip)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(APP, wait_until="networkidle", timeout=60000)

    for _ in range(40):
        if page.locator("#status").inner_text().startswith(("▶", "✗")):
            break
        page.wait_for_timeout(1000)

    page.check("#opt-skeleton")

    # Resolve the source: keyword -> dropdown value, or inject a path option.
    if clip.startswith("/") or clip.startswith("http"):
        page.evaluate("""(url) => { const s = document.getElementById('source-select');
            let o = [...s.options].find(o => o.value === url);
            if (!o) { o = document.createElement('option'); o.value = url; o.textContent = 'custom'; s.appendChild(o); }
            s.value = url; }""", clip)
    else:
        val = page.evaluate(
            """(kw) => { const o=[...document.querySelectorAll('#source-select option')]
                .find(o=>o.textContent.toLowerCase().includes(kw.toLowerCase()));
                if (o) document.getElementById('source-select').value = o.value;
                return o ? o.value : null; }""", clip)
        if not val:
            raise SystemExit(f"no clip matching '{clip}'")

    page.locator("#webcam-btn").click()
    page.wait_for_selector("#mocap-opts", state="visible", timeout=90000)  # model download
    page.select_option("#opt-legs", legs)

    got = False
    for _ in range(40):
        last = page.evaluate("() => window.__mocapLast || null")
        if last and last.get("pose"):
            got = True
            break
        page.wait_for_timeout(1000)

    samples, rig_series, shots = [], {b: [] for b in BONES}, 0
    for i in range(secs):
        last = page.evaluate("() => window.__mocapLast || null")
        rig = page.evaluate(READ_RIG)
        if last:
            samples.append(last)
        if rig:
            for b in BONES:
                rig_series[b].append(rig.get(b))
        if i % 4 == 0:
            page.screenshot(path=os.path.join(OUT, f"{prefix}_{legs}_{i:02d}.png"))
            shots += 1
        page.wait_for_timeout(1000)

    def rate(key):
        return round(100 * sum(1 for s in samples if s.get(key)) / max(1, len(samples)))

    legvis = [s.get("legVis", 0) for s in samples if s]
    pose_pct = rate("pose")
    legvis_avg = round(statistics.mean(legvis), 2) if legvis else 0
    # "single-person, clearly visible" = reliable full-body detection.
    qualifies = pose_pct >= 90 and legvis_avg >= 0.55
    return {
        "clip": _label(clip),
        "source": clip,
        "legsMode": legs,
        "samples": len(samples),
        "pose_started": got,
        "detect_%": {k: rate(k) for k in ["face", "pose", "world", "lhand", "rhand"]},
        "legVis_avg": legvis_avg,
        "bone_movement": {b: _movement(rig_series[b]) for b in BONES},
        "single_person_clear": qualifies,
        "page_errors": errors or "none",
        "screenshots": shots,
    }


def run(clip, legs="webcam", secs=12):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1100, "height": 720})
        try:
            return evaluate(page, clip, legs, secs)
        finally:
            browser.close()


if __name__ == "__main__":
    clip = sys.argv[1] if len(sys.argv) > 1 else "Tai-Chi"
    legs = sys.argv[2] if len(sys.argv) > 2 else "webcam"
    secs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    report = run(clip, legs, secs)
    print(json.dumps(report, indent=2))
    with open(os.path.join(OUT, f"{report['clip']}_{legs}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
