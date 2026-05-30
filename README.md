# VRM × Mixamo playground

Retarget [Mixamo](https://www.mixamo.com) animations onto [VRM](https://vrm.dev) avatars,
entirely in the browser. Built on [three.js](https://threejs.org) +
[`@pixiv/three-vrm`](https://github.com/pixiv/three-vrm). No build step — just static files.

## Run

It's plain static files served from the local web root:

```
http://localhost/org/jonasjohansson/vrm-mixamo/
```

Or serve the folder any way you like (e.g. `python3 -m http.server` from this dir,
then open the printed URL). It must be served over HTTP, not opened as a `file://` —
ES module imports and the importmap won't work otherwise.

## Use

- **Character** dropdown — switch between preloaded VRM avatars (VRoid samples + a VRM 1.0 robot).
- **Animation** dropdown — built-in Mixamo clips (Samba, default).
- **Load your own** — pick a `.vrm` or `.fbx`, or just **drag the file anywhere** on the page.
- Same animation retargets live onto whichever character you pick — that's the point: try a bunch.

### Webcam mocap (beta)

Click **Start tracking** to puppet the current VRM live from your camera (or a test
clip). Stack:
[MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe) — PoseLandmarker (Heavy) +
HandLandmarker on the **GPU** → [Kalidokit](https://github.com/yeemachine/kalidokit)
(kinematics) → VRM bones, with **One-Euro filtering** (low latency + low jitter).

- Drives **body** (spine, shoulders, arms), **hands/fingers**, and **legs** (opt-in).
- **Tracking source**: live webcam, or a single-person test clip.
- **Legs**: Off (pinned) / Webcam + ground contact / Idle. Webcam legs need your whole
  body in frame; see `eval/README.md`.
- **Face** is an opt-in toggle (52 ARKit blendshapes via FaceLandmarker). Off by default
  so the body runs at max framerate — face needs your face large/close in frame.
- Turning it on **pauses the Mixamo clip** (same rig, one driver at a time).
- First activation downloads the models (~35 MB) and asks for camera permission.
  Served over `localhost`, the browser treats it as a secure context.
- Jittery vs laggy? Use the **Smooth ↔ Snappy** slider (One-Euro `minCutoff`).

Validate tracking objectively with the harness in `eval/` (see `eval/README.md`).

### Getting your own assets

- **Animations** — sign in free at [mixamo.com](https://www.mixamo.com), choose an animation,
  download **FBX (without skin)**, drop the file in. (In-place clips read best on a fixed camera.)
- **Characters** — build one in [VRoid Studio](https://vroid.com/en/studio) and export VRM,
  or grab free avatars from [VRoid Hub](https://hub.vroid.com) / [Booth](https://booth.pm).

## How it works

A Mixamo FBX carries the `mixamorig*` skeleton. `loadMixamoAnimation.js` maps each Mixamo
bone to its VRM humanoid equivalent (`mixamoVRMRigMap.js`), rebases every rotation track from
the Mixamo rest pose into the VRM's normalized bone space, scales hip translation by the height
ratio, and flips the sign for VRM 0.x models (which face +Z). The result is a `THREE.AnimationClip`
addressing the VRM's normalized bones, played through a standard `AnimationMixer`. Because the
retarget depends on each avatar's rest pose, switching characters re-runs the retarget.

The bone map + retarget math come from the canonical pixiv/three-vrm example; the only local
change is a fallback so FBX clips not named `mixamo.com` still play.

## Pinned versions

- `three@0.180.0`
- `@pixiv/three-vrm@3.5.3`

## Asset credits / licensing

Preloaded samples are hotlinked, not bundled:

- VRoid sample avatars via [`madjin/vrm-samples`](https://github.com/madjin/vrm-samples)
  (Pixiv VRoid sample models — free for testing/use per VRoid's sample-model terms).
- VRM 1.0 robot via [`pixiv/three-vrm`](https://github.com/pixiv/three-vrm) examples.
- FBX animations via [`mrdoob/three.js`](https://github.com/mrdoob/three.js) examples (Mixamo).

Respect each avatar's embedded VRM license metadata (and Mixamo's terms) for anything beyond
local experimentation.
