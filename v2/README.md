# cam2avatar v2 — WebGPU 3D sandbox

A standalone experiment (doesn't touch the main app): a **second, independent 3D
pose estimate** running in the browser on **WebGPU** via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/), compared live
against MediaPipe's own monocular 3D.

Open at `/v2/` (e.g. https://cam2avatar.jonasjohansson.se/v2/).

## What it does
- Runs **MediaPipe PoseLandmarker** live (webcam or the Aerobics clip) → **blue**
  3D skeleton (its monocular 3D — weakest on the depth axis).
- Feeds MediaPipe's 2D keypoints into a **SimpleBaseline3D** 2D→3D lifter
  (Martinez et al. 2017, from the [MMPose](https://github.com/open-mmlab/mmpose)
  model zoo, Apache-2.0) on **onnxruntime-web / WebGPU** → **orange** 3D skeleton.
- Both render together, root-aligned, so you can **orbit and compare the depth**
  the two methods produce on the same pose. That's the A/B.

The lifter is **bundled** at `models/simplebaseline3d_h36m.onnx` (17 MB) and
auto-loads on Start. See `models/EXPORT_NOTES.md` for provenance.

## How it works (pipeline)
1. MediaPipe gives 2D landmarks (image-space).
2. Map them to the **Human3.6M 17-joint** order (pelvis / thorax / spine / head
   are synthesized from mid-hip, mid-shoulder and the nose) and scale to pixels →
   `[1,17,2]`.
3. ORT runs the lifter → `[1,17,3]`: **metric (m), root-relative 3D**, already in
   MediaPipe's world convention (y-down, +z away), so driving avatar B is a pure
   reindex — no scale fudge or sign flip.
4. One-Euro smooth the 17 joints → render the orange skeleton + drive the avatar.

## Comparison tooling
- **Scoreboard** (in the panel): a self-supervised quality proxy per method —
  **bone-length CV** (a correct 3D pose keeps bone lengths stable frame-to-frame;
  lower is better), per-frame **jitter**, and **depth spread** (zσ). Note bone-CV
  measures *consistency*, not *correctness* — a collapsed/static skeleton also
  scores low — so read it alongside the side view and depth spread.
- **Front / Side / Top** view buttons — the side view is where depth differences
  read most clearly.
- **Overlay** toggle — drops the orange skeleton onto the blue one so divergence
  is obvious.

## Swapping the lifter
Paste another lifter's `.onnx` URL in the panel and Load. It must take the H36M
17-joint layout: input `[1,17,2]` (pixel-scale 2D), output `[1,17,3]` (metric,
root-relative 3D).

## Notes / caveats
- **WebGPU needs a real GPU.** Headless/SwiftShader can hang the WebGPU run — on a
  real Chrome/Edge it's fine. If WebGPU misbehaves, append **`?ep=wasm`** to the
  URL to force the CPU (WASM) backend (slower but always works).
- SimpleBaseline3D lifts **2D → 3D**, so its quality rides on MediaPipe's 2D
  (strong) and is decoupled from the hard part (depth). Trained on Human3.6M, so
  it's most confident on everyday standing/locomotion poses.

## Stack
three.js · @mediapipe/tasks-vision · onnxruntime-web@1.26 (WebGPU EP) ·
SimpleBaseline3D (H36M) ONNX
