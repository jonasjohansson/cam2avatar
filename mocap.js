// Realtime mocap: MediaPipe Tasks Vision (GPU) -> Kalidokit / 52 blendshapes -> VRM.
//
// Upgrade over the legacy Holistic build:
//   - PoseLandmarker (Heavy) + HandLandmarker + FaceLandmarker on the GPU delegate
//   - Face expressions driven by FaceLandmarker's 52 ARKit blendshapes (perfect-sync
//     when the VRM has those shapes, else mapped to standard VRM expressions)
//   - One-Euro filtering on every driven value (low latency AND low jitter)
//   - requestVideoFrameCallback loop
//
// Kalidokit still solves body + hand bone rotations (best lightweight VRM solver);
// the head rotation also comes from Kalidokit's face solve.

import {
	FilesetResolver,
	PoseLandmarker,
	HandLandmarker,
	FaceLandmarker,
	DrawingUtils,
} from '@mediapipe/tasks-vision';

const TV = '0.10.35';
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TV}/wasm`;
// Pose model quality: lite (fastest) | full (balanced) | heavy (most accurate).
const POSE_MODEL = (q) => `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${q}/float16/latest/pose_landmarker_${q}.task`;
const MODELS = {
	hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
	face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
};

// --- One-Euro filter --------------------------------------------------------
class OneEuro {
	constructor(minCutoff = 1.0, beta = 0.02, dcutoff = 1.0) {
		this.minCutoff = minCutoff; this.beta = beta; this.dcutoff = dcutoff;
		this.x = null; this.dx = 0; this.t = null;
	}
	_a(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
	filter(x, t) {
		if (this.x === null) { this.x = x; this.t = t; return x; }
		const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
		const dx = (x - this.x) / dt;
		const ad = this._a(this.dcutoff, dt);
		this.dx = ad * dx + (1 - ad) * this.dx;
		const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
		const a = this._a(cutoff, dt);
		this.x = a * x + (1 - a) * this.x;
		return this.x;
	}
}

export function createMocap({ THREE, video, guideCanvas, getVrm, log }) {

	const opts = { legsMode: 'off', resp: 1, preview: true, face: false, plantFeet: true, quality: 'full', mirror: false, follow: false, requireFullBody: true, confThreshold: 0.5 };
	let mcanvas = null, mctx = null; // offscreen canvas for the mirrored frame
	let pose = null, hand = null, face = null, fileset = null;
	let running = false, rafId = 0;
	let drawUtils = null;

	const Kalidokit = window.Kalidokit;
	const clamp = Kalidokit?.Utils?.clamp ?? ((v, a, b) => Math.max(a, Math.min(b, v)));
	const vrmBone = (n) => n.charAt(0).toLowerCase() + n.slice(1);

	const _euler = new THREE.Euler();
	const _p = new THREE.Vector3();

	// One-Euro filter bank (one per driven scalar channel).
	const filters = new Map();
	function smooth(key, val) {
		let f = filters.get(key);
		if (!f) { f = new OneEuro(); filters.set(key, f); }
		f.minCutoff = 1.0 * opts.resp; // resp slider -> snappier/smoother
		return f.filter(val, performance.now());
	}

	function rigRotation(boneName, rot = { x: 0, y: 0, z: 0 }, dampener = 1) {
		const node = getVrm()?.humanoid?.getNormalizedBoneNode(boneName);
		if (!node) return;
		const x = smooth(boneName + 'x', (rot.x ?? 0) * dampener);
		const y = smooth(boneName + 'y', (rot.y ?? 0) * dampener);
		const z = smooth(boneName + 'z', (rot.z ?? 0) * dampener);
		_euler.set(x, y, z, rot.rotationOrder || 'XYZ');
		node.quaternion.setFromEuler(_euler);
	}

	function expr(name, val) {
		const em = getVrm()?.expressionManager;
		if (em && em.getExpression?.(name)) em.setValue(name, clamp(smooth('e:' + name, val), 0, 1));
	}

	// --- Face: head from Kalidokit, expressions from 52 blendshapes ----------

	let perfectSync = null; // cached: does this VRM expose ARKit shapes?
	function detectPerfectSync() {
		const em = getVrm()?.expressionManager;
		perfectSync = !!(em && em.getExpression?.('jawOpen'));
	}

	function rigFaceHead(faceLandmarks) {
		const rf = Kalidokit.Face.solve(faceLandmarks, { runtime: 'mediapipe', video });
		if (rf) rigRotation('neck', rf.head, 0.7);
	}

	function applyBlendshapes(categories) {
		const em = getVrm()?.expressionManager;
		if (!em) return;
		const d = {};
		for (const c of categories) d[c.categoryName] = c.score;
		if (perfectSync === null) detectPerfectSync();

		if (perfectSync) {
			// VRM carries the ARKit shapes — drive them 1:1 for max fidelity.
			for (const c of categories) expr(c.categoryName, c.score);
			return;
		}

		// Map ARKit 52 -> standard VRM expressions.
		const avg = (a, b) => ((d[a] || 0) + (d[b] || 0)) / 2;
		expr('blinkLeft', d.eyeBlinkLeft || 0);
		expr('blinkRight', d.eyeBlinkRight || 0);
		// Visemes (approximate, from jaw + mouth shapes)
		expr('aa', d.jawOpen || 0);
		expr('ih', avg('mouthSmileLeft', 'mouthSmileRight'));
		expr('ou', d.mouthPucker || 0);
		expr('ee', avg('mouthStretchLeft', 'mouthStretchRight'));
		expr('oh', d.mouthFunnel || 0);
		// Emotions
		expr('happy', avg('mouthSmileLeft', 'mouthSmileRight'));
		expr('angry', avg('browDownLeft', 'browDownRight'));
		expr('sad', clamp((d.browInnerUp || 0) * 0.6 + avg('mouthFrownLeft', 'mouthFrownRight') * 0.6, 0, 1));
		expr('surprised', clamp((d.eyeWideLeft || 0) * 0.5 + (d.jawOpen || 0) * 0.4 + (d.browInnerUp || 0) * 0.4, 0, 1));
		// Gaze
		expr('lookRight', avg('eyeLookInLeft', 'eyeLookOutRight'));
		expr('lookLeft', avg('eyeLookOutLeft', 'eyeLookInRight'));
		expr('lookUp', avg('eyeLookUpLeft', 'eyeLookUpRight'));
		expr('lookDown', avg('eyeLookDownLeft', 'eyeLookDownRight'));
	}

	// --- Body + hands (Kalidokit) -------------------------------------------

	// Ease one bone toward rest (used for confidence gating).
	function easeBone(name, amt = 0.12) {
		const n = getVrm()?.humanoid?.getNormalizedBoneNode(name);
		if (n) n.quaternion.slerp(_identity, amt);
	}

	function rigPose(rp, skipLegs, lm) {
		rigRotation('hips', rp.Hips.rotation, 0.25);
		rigRotation('chest', rp.Spine, 0.3);
		rigRotation('spine', rp.Spine, 0.45);
		// Per-limb confidence gating: drive a limb only when its landmarks are
		// clearly visible, else ease it to rest (kills flail on occluded/out-of-frame
		// limbs and bad-depth guesses). MediaPipe pose indices: elbows 13/14,
		// wrists 15/16, knees 25/26, ankles 27/28.
		const vis = (i) => lm?.[i]?.visibility ?? 1;
		if ((vis(14) + vis(16)) / 2 > 0.5) { rigRotation('rightUpperArm', rp.RightUpperArm); rigRotation('rightLowerArm', rp.RightLowerArm); }
		else { easeBone('rightUpperArm'); easeBone('rightLowerArm'); }
		if ((vis(13) + vis(15)) / 2 > 0.5) { rigRotation('leftUpperArm', rp.LeftUpperArm); rigRotation('leftLowerArm', rp.LeftLowerArm); }
		else { easeBone('leftUpperArm'); easeBone('leftLowerArm'); }
		if (opts.legsMode === 'webcam' && !skipLegs) {
			if ((vis(25) + vis(27)) / 2 > 0.4) { rigRotation('leftUpperLeg', rp.LeftUpperLeg); rigRotation('leftLowerLeg', rp.LeftLowerLeg); }
			else { easeBone('leftUpperLeg'); easeBone('leftLowerLeg'); }
			if ((vis(26) + vis(28)) / 2 > 0.4) { rigRotation('rightUpperLeg', rp.RightUpperLeg); rigRotation('rightLowerLeg', rp.RightLowerLeg); }
			else { easeBone('rightUpperLeg'); easeBone('rightLowerLeg'); }
		}
	}

	// Procedural step cycle (used while the avatar is actually moving sideways):
	// alternating knee-lift phased by distance travelled, so the feet step
	// instead of footskating. Amplitude scales with speed; blends back to live
	// leg tracking when you stop.
	function applyStepLegs(phase, amp) {
		const swing = Math.sin(phase);
		rigRotation('leftUpperLeg', { x: Math.max(0, swing) * 0.65 * amp });
		rigRotation('leftLowerLeg', { x: -Math.max(0, swing) * 1.0 * amp });
		rigRotation('rightUpperLeg', { x: Math.max(0, -swing) * 0.65 * amp });
		rigRotation('rightLowerLeg', { x: -Math.max(0, -swing) * 1.0 * amp });
	}

	function rigHand(rh, side, wristFromPose) {
		rigRotation(`${side}Hand`, { z: wristFromPose.z, y: rh[`${side}Wrist`].y, x: rh[`${side}Wrist`].x });
		for (const key of Object.keys(rh)) {
			if (key.endsWith('Wrist')) continue;
			rigRotation(vrmBone(key), rh[key]);
		}
	}

	// Head orientation from pose landmarks, so the head turns even with face
	// tracking off. Uses image-plane signals (reliable): ear-line tilt = roll,
	// nose-between-ears = yaw, nose-below-ears = pitch. Signs centralized for
	// easy flipping if it turns the wrong way live.
	const HEAD = { yaw: -0.8, pitch: 0.9, roll: -1.0, pitchBias: 0.18 };
	function rigHeadFromPose(lm) {
		const nose = lm[0], lEar = lm[7], rEar = lm[8];
		if (!nose || !lEar || !rEar) return;
		if ((lEar.visibility ?? 1) < 0.4 || (rEar.visibility ?? 1) < 0.4) return;
		const dx = rEar.x - lEar.x, dy = rEar.y - lEar.y;
		const w = Math.hypot(dx, dy) || 1e-3;
		const roll = Math.atan2(dy, dx);
		const t = ((nose.x - lEar.x) * dx + (nose.y - lEar.y) * dy) / (w * w); // 0..1 along ear line
		const yaw = (t - 0.5) * 2;
		const pitch = (nose.y - (lEar.y + rEar.y) / 2) / w - HEAD.pitchBias;
		rigRotation('neck', {
			x: clamp(pitch * HEAD.pitch, -0.6, 0.6),
			y: clamp(yaw * HEAD.yaw, -0.7, 0.7),
			z: clamp(roll * HEAD.roll, -0.5, 0.5),
		}, 0.8);
	}

	// Confidence gating: when pose is lost (left frame / occluded), ease the
	// driven bones + expressions toward rest instead of freezing in place.
	const MANAGED = ['hips', 'spine', 'chest', 'neck', 'leftUpperArm', 'leftLowerArm',
		'rightUpperArm', 'rightLowerArm', 'leftHand', 'rightHand',
		'leftUpperLeg', 'leftLowerLeg', 'rightUpperLeg', 'rightLowerLeg'];
	const _identity = new THREE.Quaternion();
	function easeToRest() {
		const vrm = getVrm(); if (!vrm) return;
		for (const name of MANAGED) {
			const node = vrm.humanoid?.getNormalizedBoneNode(name);
			if (node) node.quaternion.slerp(_identity, 0.08);
		}
		const em = vrm.expressionManager;
		if (em) ['blink', 'blinkLeft', 'blinkRight', 'aa', 'ih', 'ou', 'ee', 'oh', 'happy', 'angry', 'sad', 'surprised']
			.forEach((n) => { if (em.getExpression?.(n)) em.setValue(n, (em.getValue?.(n) || 0) * 0.85); });
	}

	// --- Per-frame leg helpers ----------------------------------------------
	const ANKLE_HEIGHT = 0.085;
	const FOLLOW_RANGE = 2.5; // world metres spanned across the camera frame
	const STEP_FREQ = 5.5;    // step cycles per metre travelled (phased by distance -> no footskate)
	const STEP_GAIN = 60;     // lateral speed -> step amplitude
	let idleT = 0;
	let rootXPrev = 0, rootSpeed = 0, stepPhase = 0, rootXHold = 0, prevSin = 0;
	function applyIdle(delta) {
		if (opts.legsMode !== 'idle' || !getVrm()) return;
		idleT += delta;
		const sway = Math.sin(idleT * 1.1), breathe = Math.sin(idleT * 2.2);
		rigRotation('hips', { x: 0, y: 0, z: sway * 0.04 });
		rigRotation('spine', { x: breathe * 0.015, y: 0, z: 0 }, 1);
		rigRotation('leftUpperLeg', { x: Math.max(0, sway) * 0.10, y: 0, z: 0 });
		rigRotation('leftLowerLeg', { x: -Math.max(0, sway) * 0.16, y: 0, z: 0 });
		rigRotation('rightUpperLeg', { x: Math.max(0, -sway) * 0.10, y: 0, z: 0 });
		rigRotation('rightLowerLeg', { x: -Math.max(0, -sway) * 0.16, y: 0, z: 0 });
	}
	function groundContact() {
		if (opts.legsMode === 'off' || !opts.plantFeet) return;
		const vrm = getVrm(); if (!vrm) return;
		const fy = (n) => { const b = vrm.humanoid?.getRawBoneNode(n); if (!b) return Infinity; b.getWorldPosition(_p); return _p.y; };
		const lowest = Math.min(fy('leftFoot'), fy('rightFoot'));
		if (!isFinite(lowest)) return;
		const delta = ANKLE_HEIGHT - lowest;
		// Asymmetric: shove up firmly if a foot sinks through the floor, but only
		// drift down gently so a real leg-lift isn't dragged back to the ground.
		const gain = delta > 0 ? 0.5 : 0.03;
		vrm.scene.position.y += delta * gain;
	}

	// --- Preview -------------------------------------------------------------
	function drawGuide(poseR, handR, faceR, input) {
		if (!guideCanvas || !opts.preview) return;
		const ctx = guideCanvas.getContext('2d');
		const { width: w, height: h } = guideCanvas;
		if (!drawUtils) drawUtils = new DrawingUtils(ctx);
		ctx.save();
		ctx.clearRect(0, 0, w, h);
		const src = input || video;
		if (src.width || src.videoWidth) ctx.drawImage(src, 0, 0, w, h);
		const scale = { x: w, y: h }; // landmarks are normalized 0..1
		const lm = (pts) => pts.map((p) => ({ x: p.x, y: p.y, z: p.z }));
		if (poseR?.landmarks?.[0]) {
			drawUtils.drawConnectors(lm(poseR.landmarks[0]), PoseLandmarker.POSE_CONNECTIONS, { color: '#00b0ff', lineWidth: 2 });
		}
		if (faceR?.faceLandmarks?.[0]) {
			drawUtils.drawConnectors(lm(faceR.faceLandmarks[0]), FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: 'rgba(255,255,255,0.2)', lineWidth: 0.5 });
		}
		if (handR?.landmarks) {
			handR.landmarks.forEach((pts, i) => drawUtils.drawConnectors(lm(pts), HandLandmarker.HAND_CONNECTIONS, { color: i ? '#f59e0b' : '#22c55e', lineWidth: 2 }));
		}
		ctx.restore();
	}

	// --- Frame processing ----------------------------------------------------
	let loggedSource = false;
	function process(poseR, handR, faceR, input) {
		drawGuide(poseR, handR, faceR, input);

		const poseLm = poseR?.landmarks?.[0];
		const poseWorld = poseR?.worldLandmarks?.[0];
		let legVis = 0;
		if (poseLm) {
			const idx = [25, 26, 27, 28];
			legVis = idx.reduce((s, i) => s + (poseLm[i]?.visibility ?? 0), 0) / idx.length;
		}
		// Tracking quality: 'full' = whole person in frame (torso + both feet),
		// 'partial' = torso only (legs out of frame), 'none' = no usable person.
		const v = (i) => poseLm?.[i]?.visibility ?? 0;
		const ct = opts.confThreshold;
		const torso = !!poseLm && Math.min(v(11), v(12), v(23), v(24)) > ct;
		const feet = !!poseLm && Math.min(v(27), v(28)) > ct * 0.7; // ankles are harder to see
		const track = !poseLm ? 'none' : (torso && feet ? 'full' : (torso ? 'partial' : 'none'));
		window.__mocapLast = {
			face: !!faceR?.faceLandmarks?.length,
			pose: !!poseLm,
			world: !!poseWorld,
			lhand: !!(handR?.landmarks?.length),
			rhand: !!(handR?.landmarks?.length > 1),
			legVis,
			track,
			ts: performance.now(),
		};

		const vrm = getVrm();
		if (!vrm) return;

		// Gate EVERYTHING (face, head, body, hands) on tracking confidence: only
		// drive the avatar when the full person is confidently in frame. Otherwise
		// hold a solid neutral pose instead of wobbling on low-confidence data.
		const faceLm = faceR?.faceLandmarks?.[0];
		let rp = null;
		if (poseWorld && poseLm && (!opts.requireFullBody || track === 'full')) {
			// Face
			if (faceLm) { try { rigFaceHead(faceLm); } catch (e) { /* */ } }
			const bs = faceR?.faceBlendshapes?.[0]?.categories;
			if (bs) applyBlendshapes(bs);

			rp = Kalidokit.Pose.solve(poseWorld, poseLm, { runtime: 'mediapipe', video });

			// Horizontal follow + procedural walk. Image-plane x is reliable; depth
			// is not, so we only translate x. When moving fast enough, the legs
			// switch to a distance-phased step cycle so it walks instead of gliding.
			let walking = false, stepAmp = 0;
			if (opts.follow) {
				const lh = poseLm[23], rh = poseLm[24]; // hip landmarks
				if (lh && rh) {
					const newX = smooth('rootX', ((lh.x + rh.x) / 2 - 0.5) * FOLLOW_RANGE);
					const dx = newX - rootXPrev;
					rootXPrev = newX;
					rootSpeed = rootSpeed * 0.8 + Math.abs(dx) * 0.2; // smoothed lateral speed
					stepAmp = clamp((rootSpeed - 0.0015) * STEP_GAIN, 0, 1);
					walking = stepAmp > 0.05;
					if (walking) {
						stepPhase = (stepPhase + rootSpeed * STEP_FREQ) % (Math.PI * 2);
						// Hold the body between footfalls, advance at each footfall, so the
						// planted foot doesn't drag (kills most of the skate).
						const sinNow = Math.sin(stepPhase);
						if ((prevSin <= 0) !== (sinNow <= 0)) rootXHold = newX; // footfall -> new hold target
						prevSin = sinNow;
						vrm.scene.position.x += (rootXHold - vrm.scene.position.x) * 0.5;
					} else {
						vrm.scene.position.x = newX; // smooth follow when standing
						rootXHold = newX;
					}
				}
			} else if (vrm.scene.position.x) {
				vrm.scene.position.x *= 0.85; // ease back to center when off
				rootSpeed = 0;
			}

			if (rp) rigPose(rp, walking, poseLm);  // skip live legs while walking; gate by confidence
			if (walking) applyStepLegs(stepPhase, stepAmp);
			if (!faceLm) rigHeadFromPose(poseLm);  // head from pose when face tracking is off

			// Hands (handedness tells which side)
			if (handR?.landmarks && rp) {
				handR.landmarks.forEach((pts, i) => {
					const side = handR.handedness?.[i]?.[0]?.categoryName === 'Left' ? 'Left' : 'Right';
					const rh = Kalidokit.Hand.solve(pts, side);
					if (rh) rigHand(rh, side, side === 'Left' ? rp.LeftHand : rp.RightHand);
				});
			}
		} else {
			easeToRest(); // not confident / not full body -> hold a solid neutral
		}

		if (!loggedSource && poseLm) { loggedSource = true; console.log('[mocap] Tasks Vision tracking live · world:', !!poseWorld); }
	}

	// --- Lifecycle -----------------------------------------------------------
	async function makeLandmarker(Cls, model, extra) {
		try {
			return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath: model, delegate: 'GPU' }, runningMode: 'VIDEO', ...extra });
		} catch (e) {
			console.warn('[mocap] GPU delegate failed, falling back to CPU', e);
			return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath: model, delegate: 'CPU' }, runningMode: 'VIDEO', ...extra });
		}
	}
	async function ensureFace() {
		if (face || !fileset) return;
		log('loading face model…');
		face = await makeLandmarker(FaceLandmarker, MODELS.face, { numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
	}
	async function initModels() {
		if (pose) return;
		fileset = await FilesetResolver.forVisionTasks(WASM);
		log('loading tracking models…');
		// Body + hands only by default — fastest path for realtime. Face is opt-in.
		pose = await makeLandmarker(PoseLandmarker, POSE_MODEL(opts.quality), { numPoses: 1 });
		hand = await makeLandmarker(HandLandmarker, MODELS.hand, { numHands: 2 });
		if (opts.face) await ensureFace();
	}
	async function setQuality(q) {
		if (!fileset) return;
		const old = pose; pose = null;
		pose = await makeLandmarker(PoseLandmarker, POSE_MODEL(q), { numPoses: 1 });
		old?.close?.();
	}

	// Returns the frame to run inference on — the raw video, or a horizontally
	// flipped copy when mirroring (so the avatar acts like a mirror of you).
	function getInput() {
		if (!opts.mirror) return video;
		if (!mcanvas) { mcanvas = document.createElement('canvas'); mctx = mcanvas.getContext('2d'); }
		if (mcanvas.width !== video.videoWidth) { mcanvas.width = video.videoWidth; mcanvas.height = video.videoHeight; }
		mctx.setTransform(-1, 0, 0, 1, mcanvas.width, 0);
		mctx.drawImage(video, 0, 0);
		mctx.setTransform(1, 0, 0, 1, 0, 0);
		return mcanvas;
	}

	let fpsCount = 0, fpsT = 0;
	function loop() {
		if (!running) return;
		if (video.readyState >= 2 && video.videoWidth > 0) {
			const t = performance.now();
			const input = getInput();
			let pR, hR, fR;
			try { if (pose) pR = pose.detectForVideo(input, t); } catch (e) { /* */ }
			try { hR = hand.detectForVideo(input, t); } catch (e) { /* */ }
			if (opts.face && face) { try { fR = face.detectForVideo(input, t); } catch (e) { /* */ } }
			process(pR, hR, fR, input);
			// Tracking FPS (inference throughput), exposed for the readout + harness.
			fpsCount++;
			if (t - fpsT >= 500) { window.__mocapFps = Math.round((fpsCount * 1000) / (t - fpsT)); fpsCount = 0; fpsT = t; }
		}
		if (running) {
			if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(loop);
			else rafId = requestAnimationFrame(loop);
		}
	}

	async function start(source = { type: 'camera' }) {
		if (!window.Kalidokit) throw new Error('Kalidokit did not load.');
		if (running) return;
		await initModels();
		perfectSync = null;

		if (source.type === 'video') {
			video.srcObject = null;
			video.src = source.url;
			video.loop = true; video.muted = true; video.crossOrigin = 'anonymous';
			await video.play();
		} else {
			const v = { width: { ideal: source.width || 1280 }, height: { ideal: source.height || 720 } };
			if (source.deviceId) v.deviceId = { exact: source.deviceId };
			const stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false });
			video.srcObject = stream;
			video.src = '';
			await video.play();
		}
		running = true; loggedSource = false;
		loop();
	}

	function stop() {
		running = false;
		window.__mocapFps = null;
		if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
		if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
		if (video.src) { video.pause(); video.removeAttribute('src'); video.load(); }
		filters.clear();
		rootXPrev = 0; rootSpeed = 0; stepPhase = 0; rootXHold = 0; prevSin = 0;
		const vrm = getVrm();
		if (vrm) { vrm.scene.position.y = 0; vrm.scene.position.x = 0; }
		if (vrm?.expressionManager) {
			['blink', 'blinkLeft', 'blinkRight', 'aa', 'ih', 'ou', 'ee', 'oh', 'happy', 'angry', 'sad', 'surprised',
				'lookUp', 'lookDown', 'lookLeft', 'lookRight'].forEach((n) => {
				if (vrm.expressionManager.getExpression?.(n)) vrm.expressionManager.setValue(n, 0);
			});
		}
		if (guideCanvas) guideCanvas.getContext('2d').clearRect(0, 0, guideCanvas.width, guideCanvas.height);
	}

	function setOptions(patch) {
		const qChanged = patch.quality && patch.quality !== opts.quality;
		Object.assign(opts, patch);
		if (patch.face && running) ensureFace();      // lazy-load face model on enable
		if (qChanged && running) setQuality(opts.quality); // swap pose model live
	}

	return { start, stop, setOptions, applyIdle, groundContact, isRunning: () => running };
}
