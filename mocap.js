// Webcam mocap: MediaPipe Holistic -> Kalidokit -> VRM (three-vrm v3).
//
// MediaPipe Holistic + Kalidokit + the Camera helper are loaded as classic
// <script> tags in index.html, so they live on window. We read them here.
//
// Rigging logic is adapted from the canonical Kalidokit VRM demo, ported from
// three-vrm v0 (getBoneNode + VRMSchema enums) to v3 (getNormalizedBoneNode +
// lowercase string bone names, expressionManager.setValue).

/**
 * @param {object}   o
 * @param {THREE}    o.THREE   The three namespace (for Euler/Quaternion/Vector3).
 * @param {HTMLVideoElement} o.video  Hidden <video> the webcam streams into.
 * @param {() => (object|null)} o.getVrm  Returns the currently-loaded VRM.
 * @param {(msg:string)=>void}  o.log     Status logger.
 */
export function createMocap({ THREE, video, getVrm, log }) {

	let holistic = null;
	let camera = null;
	let running = false;

	const Kalidokit = window.Kalidokit;
	const remap = Kalidokit?.Utils?.remap;
	const clamp = Kalidokit?.Utils?.clamp;
	const lerp = Kalidokit?.Vector?.lerp;

	// Kalidokit PascalCase part name -> VRM v3 humanoid bone name (camelCase).
	const vrmBone = (name) => name.charAt(0).toLowerCase() + name.slice(1);

	const _euler = new THREE.Euler();
	const _quat = new THREE.Quaternion();
	const _vec = new THREE.Vector3();

	function rigRotation(boneName, rotation = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmt = 0.3) {
		const vrm = getVrm();
		const node = vrm?.humanoid?.getNormalizedBoneNode(boneName);
		if (!node) return;
		_euler.set(rotation.x * dampener, rotation.y * dampener, rotation.z * dampener, rotation.rotationOrder || 'XYZ');
		_quat.setFromEuler(_euler);
		node.quaternion.slerp(_quat, lerpAmt);
	}

	function rigPosition(boneName, position = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmt = 0.3) {
		const vrm = getVrm();
		const node = vrm?.humanoid?.getNormalizedBoneNode(boneName);
		if (!node) return;
		_vec.set(position.x * dampener, position.y * dampener, position.z * dampener);
		node.position.lerp(_vec, lerpAmt);
	}

	// --- Face ----------------------------------------------------------------

	let oldLookTarget = new THREE.Euler();

	function rigFace(riggedFace) {
		const vrm = getVrm();
		if (!vrm) return;
		rigRotation('neck', riggedFace.head, 0.7);

		const em = vrm.expressionManager;
		if (em) {
			// Blink (stabilized so a quick wink doesn't read as a blink)
			const stabilized = Kalidokit.Face.stabilizeBlink(riggedFace.eye, riggedFace.head.y);
			em.setValue('blink', clamp(1 - stabilized.l, 0, 1));

			// Mouth shapes (lip-sync)
			const m = riggedFace.mouth.shape;
			em.setValue('aa', m.A);
			em.setValue('ih', m.I);
			em.setValue('ou', m.U);
			em.setValue('ee', m.E);
			em.setValue('oh', m.O);
		}

		// Eye look direction via lookAt target (smoothed)
		const lookTarget = new THREE.Euler(
			lerp(oldLookTarget.x, riggedFace.pupil.y, 0.4),
			lerp(oldLookTarget.y, riggedFace.pupil.x, 0.4),
			0,
			'XYZ',
		);
		oldLookTarget.copy(lookTarget);
		if (vrm.lookAt && vrm.lookAt.applier) {
			// Drive look direction through expressions when no explicit target is set
			if (em) {
				em.setValue('lookUp', clamp(-lookTarget.x, 0, 1));
				em.setValue('lookDown', clamp(lookTarget.x, 0, 1));
				em.setValue('lookLeft', clamp(-lookTarget.y, 0, 1));
				em.setValue('lookRight', clamp(lookTarget.y, 0, 1));
			}
		}
	}

	// --- Pose + hands --------------------------------------------------------

	function rigPose(riggedPose) {
		rigRotation('hips', riggedPose.Hips.rotation, 0.7);
		rigPosition('hips', {
			x: -riggedPose.Hips.position.x, // reverse axis for mirrored webcam
			y: riggedPose.Hips.position.y + 1, // raise to stand on the floor
			z: -riggedPose.Hips.position.z,
		}, 1, 0.07);

		rigRotation('chest', riggedPose.Spine, 0.25, 0.3);
		rigRotation('spine', riggedPose.Spine, 0.45, 0.3);

		rigRotation('rightUpperArm', riggedPose.RightUpperArm, 1, 0.3);
		rigRotation('rightLowerArm', riggedPose.RightLowerArm, 1, 0.3);
		rigRotation('leftUpperArm', riggedPose.LeftUpperArm, 1, 0.3);
		rigRotation('leftLowerArm', riggedPose.LeftLowerArm, 1, 0.3);

		rigRotation('leftUpperLeg', riggedPose.LeftUpperLeg, 1, 0.3);
		rigRotation('leftLowerLeg', riggedPose.LeftLowerLeg, 1, 0.3);
		rigRotation('rightUpperLeg', riggedPose.RightUpperLeg, 1, 0.3);
		rigRotation('rightLowerLeg', riggedPose.RightLowerLeg, 1, 0.3);
	}

	function rigHand(riggedHand, side, wristFromPose) {
		// Wrist combines pose (z twist) + hand solve (x/y)
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

	// --- MediaPipe results ---------------------------------------------------

	let loggedSource = false;

	function onResults(results) {
		const vrm = getVrm();
		if (!vrm) return;

		if (results.faceLandmarks) {
			const riggedFace = Kalidokit.Face.solve(results.faceLandmarks, {
				runtime: 'mediapipe',
				video,
			});
			if (riggedFace) rigFace(riggedFace);
		}

		// Holistic exposes world pose landmarks under different keys across
		// builds; try the known ones in order.
		const world = results.poseWorldLandmarks || results.ea || results.za;
		if (!loggedSource && results.poseLandmarks) {
			loggedSource = true;
			console.log('[mocap] pose world key present:', !!world);
		}

		let riggedPose = null;
		if (world && results.poseLandmarks) {
			riggedPose = Kalidokit.Pose.solve(world, results.poseLandmarks, {
				runtime: 'mediapipe',
				video,
			});
			if (riggedPose) rigPose(riggedPose);
		}

		if (results.leftHandLandmarks && riggedPose) {
			const riggedLeftHand = Kalidokit.Hand.solve(results.leftHandLandmarks, 'Left');
			if (riggedLeftHand) rigHand(riggedLeftHand, 'Left', riggedPose.LeftHand);
		}
		if (results.rightHandLandmarks && riggedPose) {
			const riggedRightHand = Kalidokit.Hand.solve(results.rightHandLandmarks, 'Right');
			if (riggedRightHand) rigHand(riggedRightHand, 'Right', riggedPose.RightHand);
		}
	}

	// --- Lifecycle -----------------------------------------------------------

	async function start() {
		if (!window.Holistic || !window.Kalidokit || !window.Camera) {
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

		camera = new window.Camera(video, {
			onFrame: async () => {
				if (running) await holistic.send({ image: video });
			},
			width: 640,
			height: 480,
		});
		running = true;
		loggedSource = false;
		await camera.start(); // triggers the browser camera-permission prompt
	}

	function stop() {
		running = false;
		if (camera) { camera.stop?.(); camera = null; }
		if (video.srcObject) {
			video.srcObject.getTracks().forEach((t) => t.stop());
			video.srcObject = null;
		}
		if (holistic) { holistic.close?.(); holistic = null; }
		// Reset facial expressions so the avatar doesn't freeze mid-blink
		const vrm = getVrm();
		if (vrm?.expressionManager) {
			['blink', 'aa', 'ih', 'ou', 'ee', 'oh', 'lookUp', 'lookDown', 'lookLeft', 'lookRight']
				.forEach((n) => vrm.expressionManager.setValue(n, 0));
		}
	}

	return { start, stop, isRunning: () => running };
}
