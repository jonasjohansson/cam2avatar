# simplebaseline3d_h36m.onnx — export notes

**What:** SimpleBaseline3D (Martinez et al., ICCV 2017) — a single-frame 2D→3D
human-pose *lifter*. Takes 17 2D keypoints, outputs 17 3D keypoints. Replaces the
image→3D Mobile Human Pose model with a model fed by MediaPipe's reliable 2D.

**Source weights:** mmpose model zoo, `image-pose-lift_tcn_8xb64-200e_h36m`
(Apache-2.0). Checkpoint:
`https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth`
Config (also the source of the normalization stats):
`https://github.com/open-mmlab/mmpose/blob/main/configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m.py`
Reported accuracy on Human3.6M GT-2D: MPJPE 43.4 mm / P-MPJPE 34.3 mm.

**Why this instead of weigq/3d_pose_baseline_pytorch:** that repo's pretrained
`gt_ckpt_best.pth.tar` (Google Drive) and the una-dinosauria `h36m.zip` (Dropbox)
are both dead/permission-restricted (403 / "cannot retrieve public link"). The
mmpose checkpoint is on a stable public CDN and is the same architecture.

**Architecture:** mmpose `TCN` backbone with `kernel_sizes=(1,1,1)`, which for a
single frame is mathematically a plain residual MLP — identical to Martinez:
`Linear(34→1024)+BN+ReLU` → 2× residual blocks `[Linear+BN+ReLU+Dropout]×2 + skip`
→ `Linear(1024→48)`. Rebuilt as Conv1d(k=1) in plain PyTorch from the state_dict
(no mmpose install), normalization folded into the graph, exported with the
TorchScript exporter.

**The ONNX has normalization BAKED IN.** Input is raw H36M-order 2D *pixel*
coordinates; output is metric 3D. The graph internally does
`(kp-keypoints_mean)/keypoints_std` → MLP → `*target_std+target_mean` → reinsert
zero root. So `simplebaseline3d_h36m_stats.json` is reference only; not needed at
runtime.

- **Input** `keypoints2d`: float32 `[1,17,2]` — H36M joint order, x/y in H36M
  image pixels (image ≈ 1000×1000, y down).
- **Output** `keypoints3d`: float32 `[1,17,3]` — H36M order, **meters**,
  root-relative (Pelvis at index 0 = (0,0,0)). +x right, +y down, +z = depth
  (H36M camera convention).
- opset 17, ir_version 8, ~17 MB float32, fully static shapes, ops:
  Conv/Add/Relu/Mul/Div/Sub/Reshape/Concat/Constant — all supported by
  onnxruntime-web 1.26 (WASM + WebGPU).

**Validation (CPU EP):** mean-2D pose → thigh 397 / shin 421 / uarm 234 / stature
1472 mm; synthetic T-pose → thigh 409 / shin 398 / uarm 363 / farm 383, 1.9 m
wingspan; simulated MediaPipe pose (scale≈1000) → thigh ~400 / shin ~420 / stature
~1.7 m. No NaN, no collapse. Anatomically sensible.

**H36M joint order (index → name):**
0 Pelvis, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle, 7 Spine,
8 Thorax, 9 Neck/Nose, 10 Head, 11 LShoulder, 12 LElbow, 13 LWrist,
14 RShoulder, 15 RElbow, 16 RWrist.
