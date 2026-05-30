# cam2avatar v2 — WebGPU 3D sandbox

A standalone experiment (doesn't touch the main app): a **second, independent 3D
pose estimate** running in the browser on **WebGPU** via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/), compared live
against MediaPipe's own monocular 3D.

Open at `/v2/` (e.g. https://cam2avatar.jonasjohansson.se/v2/).

## What it does
- Runs **MediaPipe PoseLandmarker** live (webcam or the Aerobics clip) → **blue**
  3D skeleton (its monocular 3D — weakest on the depth axis).
- Crops the person and runs the **Mobile Human Pose** model (a 3DMPPE image→3D
  network, [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo)
  `156_MobileHumanPose`) on **onnxruntime-web / WebGPU** → **orange** 3D skeleton.
- Both render together, root-aligned, so you can **orbit and compare the depth**
  the two methods produce on the same pose. That's the A/B.

The model is **bundled** at `models/mobile_human_pose_256x256.onnx` (13 MB) and
auto-loads on Start.

## How it works (pipeline)
1. MediaPipe gives 2D landmarks → a person bounding box.
2. Crop + resize the person to **256×256**, raw 0–255 RGB, CHW (no normalization).
3. ORT runs the model → `[1,672,32,32]` 3D heatmap (21 joints × 32 depth × 32×32).
4. **Soft-argmax** decode → 21 3D joints; rendered as the orange skeleton.

## Notes / caveats
- **WebGPU needs a real GPU.** Headless/SwiftShader hangs the WebGPU run — on a real
  Chrome/Edge it's fine. If WebGPU misbehaves, append **`?ep=wasm`** to the URL to
  force the CPU (WASM) backend (slower but always works).
- It's **image→3D** (not a 2D→3D lift), and trained on forward-facing, unoccluded
  people — expect it to wobble on hard poses. Whether it beats MediaPipe's depth is
  exactly what this sandbox is for judging.
- Soft-argmax over 21×32³ runs every frame in JS; fine, but it's the main CPU cost.

## Stack
three.js · @mediapipe/tasks-vision · onnxruntime-web@1.26 (WebGPU EP) ·
Mobile Human Pose (3DMPPE) ONNX
