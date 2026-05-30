# cam2avatar v2 — WebGPU 3D-lift sandbox

A standalone experiment (doesn't touch the main app) for pushing depth accuracy
past MediaPipe's monocular guess, **in the browser, live**, via a 2D→3D *lifting*
model run on **WebGPU** with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).

Open at `/v2/` (e.g. https://cam2avatar.jonasjohansson.se/v2/).

## What it does today
- Runs **MediaPipe PoseLandmarker** live (webcam or the Aerobics clip).
- Renders MediaPipe's own **monocular 3D** world-landmark skeleton in three.js (blue)
  — so you can see the depth quality of what we already have.
- Initializes **onnxruntime-web on WebGPU** and reports status (WebGPU support, ORT
  version) — the runtime is ready.
- A **model loader**: paste an `.onnx` lifting-model URL; once loaded, its lifted 3D
  renders in orange next to MediaPipe's for direct A/B.

## The honest gap
There is **no public, browser-ready ONNX 2D→3D lifting model** to drop in. The strong
lifters (SimpleBaseline3D, VideoPose3D, MotionBERT) live in
[MMPose](https://mmpose.readthedocs.io/en/latest/model_zoo/body_3d_keypoint.html) as
**PyTorch** — so this needs a one-time **export-to-ONNX** step (which requires Python/
PyTorch, not doable in this browser sandbox).

### To complete the experiment
1. Export a lifter to ONNX. Easiest: **SimpleBaseline3D** (a small MLP, 17 2D joints →
   17 3D joints) via MMPose's ONNX export, or VideoPose3D.
2. Host the `.onnx` (drop it in this folder, or any CDN).
3. Paste the URL into the model box here. The lifted skeleton appears in orange.
4. Adapt `lift()` in `main.js` to the model's exact input/output layout (joint count,
   order, normalization) — MediaPipe's 33 landmarks may need remapping to the model's
   17-joint convention.

## Caveats / open questions
- A lifter may add a **temporal-window latency** (some need a sequence of frames).
- It may **not beat** MediaPipe's BlazePose 3D for live VTuber-style use — that's exactly
  what this sandbox is for measuring (A/B the blue vs orange skeletons).
- WebGPU must be available in the browser (Chrome/Edge; the status panel reports it).

## Stack
three.js · @mediapipe/tasks-vision · onnxruntime-web@1.26 (WebGPU EP)
