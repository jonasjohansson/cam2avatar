// Webcam mocap: MediaPipe Holistic -> Kalidokit -> VRM (three-vrm v3).
//
// MediaPipe Holistic + drawing utils + Kalidokit + the Camera helper are loaded
// as classic <script> tags in index.html, so they live on window.
//
// Grounding: a single webcam cannot see the floor, so we DON'T drive hips
// vertical position from the camera, and legs are opt-in. The avatar stays in
// its standing rest pose (feet pinned) and we only drive face + torso + arms +
// hands. That keeps the feet planted instead of floating.

/**
 * @param {object} o
 * @param {THREE}  o.THREE
 * @param {HTMLVideoElement}  o.video        Hidden/visible <video> for the stream.
 * @param {HTMLCanvasElement} o.guideCanvas  Overlay canvas for the landmark preview.
 * @param {() => (object|null)} o.getVrm
 * @param {(msg:string)=>void}  o.log
 */
export function createMocap({ THREE, video, guideCanvas, getVrm, log }) {

	let holistic = null;
	let camera = null;
	let videoRaf = 0;
	let running = false;

	const opts = {
		legsMode: 'off', // 'off' | 'webcam' | 'idle'
		resp: 1,         // responsiveness multiplier (lower = smoother)
		preview: true,
	};

	const Kalidokit = window.Kalidokit;
	const clamp = Kalidokit?.Utils?.clamp ?? ((v, a, b) => Math.max(a, Math.min(b, v)));
	const lerp = Kalidokit?.Vector?.lerp ?? ((a, b, t) => a + (b - a) * t);

	const vrmBone = (name) => name.charAt(0).toLowerCase() + name.slice(1);

	const _euler = new THREE.Euler();
	const _quat = new THREE.Quaternion();
	const _p = new THREE.Vector3();

	function rigRotation(boneName, rotation = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmt = 0.3) {
		const vrm = getVrm();
		const node = vrm?.humanoid?.getNormalizedBoneNode(boneName);
		if (!node) return;
		_euler.set(rotation.x * dampener, rotation.y * dampener, rotation.z * dampener, rotation.rotationOrder || 'XYZ');
		_quat.setFromEuler(_euler);
		node.quaternion.slerp(_quat, clamp(lerpAmt * opts.resp, 0, 1));
	}

	// --- Face ----------------------------------------------------------------

	const oldLook = new THREE.Euler();

	function rigFace(riggedFace) {
		const vrm = getVrm();
		if (!vrm) return;
		rigRotation('neck', riggedFace.head, 0.7);

		const em = vrm.expressionManager;
		if (em) {
			const stabilized = Kalidokit.Face.stabilizeBlink(riggedFace.eye, riggedFace.head.y);
			em.setValue('blink', clamp(1 - stabilized.l, 0, 1));

			const m = riggedFace.mouth.shape;
			em.setValue('aa', m.A);
			em.setValue('ih', m.I);
			em.setValue('ou', m.U);
			em.setValue('ee', m.E);
			em.setValue('oh', m.O);

			const lookX = lerp(oldLook.x, riggedFace.pupil.y, 0.4);
			const lookY = lerp(oldLook.y, riggedFace.pupil.x, 0.4);
			oldLook.set(lookX, lookY, 0);
			em.setValue('lookUp', clamp(-lookX, 0, 1));
			em.setValue('lookDown', clamp(lookX, 0, 1));
			em.setValue('lookLeft', clamp(-lookY, 0, 1));
			em.setValue('lookRight', clamp(lookY, 0, 1));
		}
	}

	// --- Pose + hands --------------------------------------------------------

	function rigPose(riggedPose) {
		// Keep the lower body rooted: only a light hip orientation, NO hip
		// translation (that's what made the avatar float).
		rigRotation('hips', riggedPose.Hips.rotation, 0.25, 0.25);

		// Lean comes from the torso, not the hips, so feet don't swing.
		rigRotation('chest', riggedPose.Spine, 0.3, 0.3);
		rigRotation('spine', riggedPose.Spine, 0.45, 0.3);

		rigRotation('rightUpperArm', riggedPose.RightUpperArm, 1, 0.3);
		rigRotation('rightLowerArm', riggedPose.RightLowerArm, 1, 0.3);
		rigRotation('leftUpperArm', riggedPose.LeftUpperArm, 1, 0.3);
		rigRotation('leftLowerArm', riggedPose.LeftLowerArm, 1, 0.3);

		if (opts.legsMode === 'webcam') {
			// Webcam legs are noisy — damp them a little harder than the arms.
			rigRotation('leftUpperLeg', riggedPose.LeftUpperLeg, 1, 0.25);
			rigRotation('leftLowerLeg', riggedPose.LeftLowerLeg, 1, 0.25);
			rigRotation('rightUpperLeg', riggedPose.RightUpperLeg, 1, 0.25);
			rigRotation('rightLowerLeg', riggedPose.RightLowerLeg, 1, 0.25);
		}
	}

	// --- Per-frame leg helpers (called from the render loop) -----------------

	const ANKLE_HEIGHT = 0.085; // ankle bone sits ~this far above the sole
	let idleT = 0;

	// Procedural idle for the lower body (hybrid mode): a gentle weight-shift
	// + knee softening so the legs look alive while the webcam drives the torso.
	function applyIdle(delta) {
		if (opts.legsMode !== 'idle') return;
		const vrm = getVrm();
		if (!vrm) return;
		idleT += delta;
		const sway = Math.sin(idleT * 1.1);
		const breathe = Math.sin(idleT * 1.1 * 2);
		rigRotation('hips', { x: 0, y: 0, z: sway * 0.04 }, 1, 0.2);
		rigRotation('spine', { x: breathe * 0.015, y: 0, z: 0 }, 1, 0.2);
		// Alternating soft knee bend follows the weight shift.
		rigRotation('leftUpperLeg', { x: Math.max(0, sway) * 0.10, y: 0, z: 0 }, 1, 0.2);
		rigRotation('leftLowerLeg', { x: -Math.max(0, sway) * 0.16, y: 0, z: 0 }, 1, 0.2);
		rigRotation('rightUpperLeg', { x: Math.max(0, -sway) * 0.10, y: 0, z: 0 }, 1, 0.2);
		rigRotation('rightLowerLeg', { x: -Math.max(0, -sway) * 0.16, y: 0, z: 0 }, 1, 0.2);
	}

	// Ground contact: after the skeleton is solved, shift the whole avatar so the
	// lower foot sits on the floor. Not full IK, but it kills the float/slide.
	function groundContact() {
		if (opts.legsMode === 'off') return;
		const vrm = getVrm();
		if (!vrm) return;
		const footY = (name) => {
			const n = vrm.humanoid?.getRawBoneNode(name);
			if (!n) return Infinity;
			n.getWorldPosition(_p);
			return _p.y;
		};
		const lowest = Math.min(footY('leftFoot'), footY('rightFoot'));
		if (!isFinite(lowest)) return;
		// We want the lower ankle at ANKLE_HEIGHT above the floor.
		const delta = (ANKLE_HEIGHT - lowest);
		vrm.scene.position.y += delta * 0.25; // smoothed
	}

	function rigHand(riggedHand, side, wristFromPose) {
		rigRotation(`${side}Hand`, {
			z: wristFromPose.z,
			y: riggedHand[`${side}Wrist`].y,
			x: riggedHand[`${side}Wrist`].x,
		});
		for (const key of Object.keys(riggedHand)) {
			if (key.endsWith('Wrist')) continue;
			rigRotation(vrmBone(key), riggedHand[key], 1, 0.3);
		}
	}

	// --- Landmark preview ----------------------------------------------------

	function drawGuide(results) {
		if (!guideCanvas || !opts.preview) return;
		const ctx = guideCanvas.getContext('2d');
		const { width: w, height: h } = guideCanvas;
		ctx.save();
		ctx.clearRect(0, 0, w, h);
		if (results.image) ctx.drawImage(results.image, 0, 0, w, h);

		const dc = window.drawConnectors;
		const dl = window.drawLandmarks;
		if (dc) {
			if (results.poseLandmarks) {
				dc(ctx, results.poseLandmarks, window.POSE_CONNECTIONS, { color: '#00b0ff', lineWidth: 2 });
				if (dl) dl(ctx, results.poseLandmarks, { color: '#ffffff', lineWidth: 1, radius: 2 });
			}
			if (results.faceLandmarks) {
				dc(ctx, results.faceLandmarks, window.FACEMESH_TESSELATION, { color: 'rgba(255,255,255,0.25)', lineWidth: 1 });
			}
			if (results.leftHandLandmarks) dc(ctx, results.leftHandLandmarks, window.HAND_CONNECTIONS, { color: '#22c55e', lineWidth: 2 });
			if (results.rightHandLandmarks) dc(ctx, results.rightHandLandmarks, window.HAND_CONNECTIONS, { color: '#f59e0b', lineWidth: 2 });
		}
		ctx.restore();
	}

	// --- MediaPipe results ---------------------------------------------------

	let loggedSource = false;

	function onResults(results) {
		drawGuide(results);

		const world = results.poseWorldLandmarks || results.ea || results.za;

		// Expose lightweight per-frame stats for the evaluation harness.
		const pls = results.poseLandmarks;
		let legVis = 0;
		if (pls) {
			const idx = [25, 26, 27, 28]; // knees + ankles
			legVis = idx.reduce((s, i) => s + (pls[i]?.visibility ?? 0), 0) / idx.length;
		}
		window.__mocapLast = {
			face: !!results.faceLandmarks,
			pose: !!pls,
			world: !!world,
			lhand: !!results.leftHandLandmarks,
			rhand: !!results.rightHandLandmarks,
			legVis,
			ts: performance.now(),
		};

		const vrm = getVrm();
		if (!vrm) return;

		if (results.faceLandmarks) {
			const riggedFace = Kalidokit.Face.solve(results.faceLandmarks, { runtime: 'mediapipe', video });
			if (riggedFace) rigFace(riggedFace);
		}

		if (!loggedSource && results.poseLandmarks) {
			loggedSource = true;
			console.log('[mocap] tracking live · world landmarks:', !!world);
		}

		let riggedPose = null;
		if (world && results.poseLandmarks) {
			riggedPose = Kalidokit.Pose.solve(world, results.poseLandmarks, { runtime: 'mediapipe', video });
			if (riggedPose) rigPose(riggedPose);
		}

		if (results.leftHandLandmarks && riggedPose) {
			const h = Kalidokit.Hand.solve(results.leftHandLandmarks, 'Left');
			if (h) rigHand(h, 'Left', riggedPose.LeftHand);
		}
		if (results.rightHandLandmarks && riggedPose) {
			const h = Kalidokit.Hand.solve(results.rightHandLandmarks, 'Right');
			if (h) rigHand(h, 'Right', riggedPose.RightHand);
		}
	}

	// --- Lifecycle -----------------------------------------------------------

	// source: { type: 'camera' } (default) or { type: 'video', url }
	async function start(source = { type: 'camera' }) {
		if (!window.Holistic || !window.Kalidokit) {
			throw new Error('Tracking libraries did not load (check network/CDN).');
		}
		if (running) return;

		holistic = new window.Holistic({
			locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
		});
		holistic.setOptions({
			modelComplexity: 1,
			smoothLandmarks: true,
			minDetectionConfidence: 0.6,
			minTrackingConfidence: 0.6,
			refineFaceLandmarks: true,
		});
		holistic.onResults(onResults);

		running = true;
		loggedSource = false;

		if (source.type === 'video') {
			// Play a test clip through the same pipeline (manual frame pump).
			video.srcObject = null;
			video.src = source.url;
			video.loop = true;
			video.muted = true;
			video.crossOrigin = 'anonymous';
			await video.play();
			const pump = async () => {
				if (!running) return;
				try { if (video.readyState >= 2) await holistic.send({ image: video }); } catch (e) { /* frame skip */ }
				if (running) videoRaf = requestAnimationFrame(pump);
			};
			pump();
		} else {
			if (!window.Camera) throw new Error('Camera helper did not load.');
			camera = new window.Camera(video, {
				onFrame: async () => { if (running) await holistic.send({ image: video }); },
				width: 640,
				height: 480,
			});
			await camera.start();
		}
	}

	function stop() {
		running = false;
		if (videoRaf) { cancelAnimationFrame(videoRaf); videoRaf = 0; }
		if (camera) { camera.stop?.(); camera = null; }
		if (video.srcObject) {
			video.srcObject.getTracks().forEach((t) => t.stop());
			video.srcObject = null;
		}
		if (video.src) { video.pause(); video.removeAttribute('src'); video.load(); }
		if (holistic) { holistic.close?.(); holistic = null; }
		if (guideCanvas) guideCanvas.getContext('2d').clearRect(0, 0, guideCanvas.width, guideCanvas.height);
		const vrm = getVrm();
		if (vrm) vrm.scene.position.y = 0; // undo any ground-contact shift
		if (vrm?.expressionManager) {
			['blink', 'aa', 'ih', 'ou', 'ee', 'oh', 'lookUp', 'lookDown', 'lookLeft', 'lookRight']
				.forEach((n) => vrm.expressionManager.setValue(n, 0));
		}
	}

	function setOptions(patch) { Object.assign(opts, patch); }

	return { start, stop, setOptions, applyIdle, groundContact, isRunning: () => running };
}
