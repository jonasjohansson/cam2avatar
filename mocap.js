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
const MODELS = {
	pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
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

	const opts = { legsMode: 'off', resp: 1, preview: true, face: false, plantFeet: true };
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
		const x = smooth(boneName + 'x', rot.x * dampener);
		const y = smooth(boneName + 'y', rot.y * dampener);
		const z = smooth(boneName + 'z', rot.z * dampener);
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

	function rigPose(rp) {
		rigRotation('hips', rp.Hips.rotation, 0.25);
		rigRotation('chest', rp.Spine, 0.3);
		rigRotation('spine', rp.Spine, 0.45);
		rigRotation('rightUpperArm', rp.RightUpperArm);
		rigRotation('rightLowerArm', rp.RightLowerArm);
		rigRotation('leftUpperArm', rp.LeftUpperArm);
		rigRotation('leftLowerArm', rp.LeftLowerArm);
		if (opts.legsMode === 'webcam') {
			rigRotation('leftUpperLeg', rp.LeftUpperLeg);
			rigRotation('leftLowerLeg', rp.LeftLowerLeg);
			rigRotation('rightUpperLeg', rp.RightUpperLeg);
			rigRotation('rightLowerLeg', rp.RightLowerLeg);
		}
	}

	function rigHand(rh, side, wristFromPose) {
		rigRotation(`${side}Hand`, { z: wristFromPose.z, y: rh[`${side}Wrist`].y, x: rh[`${side}Wrist`].x });
		for (const key of Object.keys(rh)) {
			if (key.endsWith('Wrist')) continue;
			rigRotation(vrmBone(key), rh[key]);
		}
	}

	// --- Per-frame leg helpers ----------------------------------------------
	const ANKLE_HEIGHT = 0.085;
	let idleT = 0;
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
	function drawGuide(poseR, handR, faceR) {
		if (!guideCanvas || !opts.preview) return;
		const ctx = guideCanvas.getContext('2d');
		const { width: w, height: h } = guideCanvas;
		if (!drawUtils) drawUtils = new DrawingUtils(ctx);
		ctx.save();
		ctx.clearRect(0, 0, w, h);
		if (video.videoWidth) ctx.drawImage(video, 0, 0, w, h);
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
	function process(poseR, handR, faceR) {
		drawGuide(poseR, handR, faceR);

		const poseLm = poseR?.landmarks?.[0];
		const poseWorld = poseR?.worldLandmarks?.[0];
		let legVis = 0;
		if (poseLm) {
			const idx = [25, 26, 27, 28];
			legVis = idx.reduce((s, i) => s + (poseLm[i]?.visibility ?? 0), 0) / idx.length;
		}
		window.__mocapLast = {
			face: !!faceR?.faceLandmarks?.length,
			pose: !!poseLm,
			world: !!poseWorld,
			lhand: !!(handR?.landmarks?.length),
			rhand: !!(handR?.landmarks?.length > 1),
			legVis,
			ts: performance.now(),
		};

		const vrm = getVrm();
		if (!vrm) return;

		// Face
		const faceLm = faceR?.faceLandmarks?.[0];
		if (faceLm) { try { rigFaceHead(faceLm); } catch (e) { /* */ } }
		const bs = faceR?.faceBlendshapes?.[0]?.categories;
		if (bs) applyBlendshapes(bs);

		// Body
		let rp = null;
		if (poseWorld && poseLm) {
			rp = Kalidokit.Pose.solve(poseWorld, poseLm, { runtime: 'mediapipe', video });
			if (rp) rigPose(rp);
		}
		// Hands (handedness tells which side)
		if (handR?.landmarks && rp) {
			handR.landmarks.forEach((pts, i) => {
				const side = handR.handedness?.[i]?.[0]?.categoryName === 'Left' ? 'Left' : 'Right';
				const rh = Kalidokit.Hand.solve(pts, side);
				if (rh) rigHand(rh, side, side === 'Left' ? rp.LeftHand : rp.RightHand);
			});
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
		pose = await makeLandmarker(PoseLandmarker, MODELS.pose, { numPoses: 1 });
		hand = await makeLandmarker(HandLandmarker, MODELS.hand, { numHands: 2 });
		if (opts.face) await ensureFace();
	}

	let fpsCount = 0, fpsT = 0;
	function loop() {
		if (!running) return;
		if (video.readyState >= 2 && video.videoWidth > 0) {
			const t = performance.now();
			let pR, hR, fR;
			try { pR = pose.detectForVideo(video, t); } catch (e) { /* */ }
			try { hR = hand.detectForVideo(video, t); } catch (e) { /* */ }
			if (opts.face && face) { try { fR = face.detectForVideo(video, t); } catch (e) { /* */ } }
			process(pR, hR, fR);
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
			const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
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
		const vrm = getVrm();
		if (vrm) vrm.scene.position.y = 0;
		if (vrm?.expressionManager) {
			['blink', 'blinkLeft', 'blinkRight', 'aa', 'ih', 'ou', 'ee', 'oh', 'happy', 'angry', 'sad', 'surprised',
				'lookUp', 'lookDown', 'lookLeft', 'lookRight'].forEach((n) => {
				if (vrm.expressionManager.getExpression?.(n)) vrm.expressionManager.setValue(n, 0);
			});
		}
		if (guideCanvas) guideCanvas.getContext('2d').clearRect(0, 0, guideCanvas.width, guideCanvas.height);
	}

	function setOptions(patch) {
		Object.assign(opts, patch);
		if (patch.face && running) ensureFace(); // lazy-load face model on enable
	}

	return { start, stop, setOptions, applyIdle, groundContact, isRunning: () => running };
}
