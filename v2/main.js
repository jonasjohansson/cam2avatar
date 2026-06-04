// cam2avatar v2 — WebGPU 3D sandbox.
// Compares MediaPipe's monocular 3D (blue) against a SimpleBaseline3D 2D->3D
// lifter (orange) — a residual MLP that lifts MediaPipe's reliable 2D keypoints
// to metric 3D, run on WebGPU via onnxruntime-web. Standalone; doesn't touch
// the main app.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web/webgpu';

const Kalidokit = window.Kalidokit;

const TV = '0.10.35';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
const LIFT_MODEL = './models/simplebaseline3d_h36m.onnx';

// Human3.6M 17-joint skeleton (pelvis = 0 is the root). The lifter outputs
// metric (m), root-relative, y-down 3D — the same convention as MediaPipe world.
// 0 Pelvis 1 RHip 2 RKnee 3 RAnkle 4 LHip 5 LKnee 6 LAnkle 7 Spine 8 Thorax
// 9 Neck/Nose 10 Head 11 LShoulder 12 LElbow 13 LWrist 14 RShoulder 15 RElbow 16 RWrist
const LIFT_SKELETON = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]];
const LIFT_ROOT = 0;

const statusEl = document.getElementById('status');
const lines = [];
const log = (k, v, cls) => {
	const i = lines.findIndex((l) => l.k === k);
	const e = { k, v, cls };
	if (i >= 0) lines[i] = e; else lines.push(e);
	statusEl.innerHTML = lines.map((l) => `${l.k}: ${l.cls ? `<span class="pill ${l.cls}">${l.v}</span>` : l.v}`).join('<br>');
};

// --- three.js scene ---------------------------------------------------------
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
let compCanvas = null, compCtx = null; // recording composite
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe3ea);
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.01, 100);
camera.position.set(0.85, 0.9, 4.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.85, 0.9, 0);
scene.add(new THREE.GridHelper(4, 8, 0xaab0bd, 0xc7ccd6));
scene.add(new THREE.HemisphereLight(0xffffff, 0x667, 1.4));
const dir = new THREE.DirectionalLight(0xffffff, 1.4); dir.position.set(1, 2, 1.5); scene.add(dir);

// --- Two VRM avatars: A = MediaPipe (like v1), B = SimpleBaseline3D lifter ---
const VRM_URL = 'https://cdn.jsdelivr.net/gh/madjin/vrm-samples@master/vroid/stable/AvatarSample_C.vrm';
let vrmA = null, vrmB = null;
async function loadOne(x) {
	const loader = new GLTFLoader();
	loader.register((parser) => new VRMLoaderPlugin(parser));
	const gltf = await loader.loadAsync(VRM_URL);
	const v = gltf.userData.vrm;
	VRMUtils.removeUnnecessaryVertices(gltf.scene);
	VRMUtils.combineSkeletons(gltf.scene);
	VRMUtils.rotateVRM0(v);
	v.scene.traverse((o) => { o.frustumCulled = false; });
	v.scene.position.x = x;
	scene.add(v.scene);
	return v;
}
async function loadVRMs() { vrmA = await loadOne(0.1); vrmB = await loadOne(1.6); applyView(); }

const _e = new THREE.Euler(), _ident = new THREE.Quaternion();
const clamp = Kalidokit?.Utils?.clamp ?? ((v, a, b) => Math.max(a, Math.min(b, v)));

// One-Euro filter bank, ported from the v1 app (mocap.js). A fixed per-frame
// slerp couples smoothing to the (variable) frame rate and forces one
// jitter/lag tradeoff at all speeds; One-Euro kills rest jitter via minCutoff
// yet opens the cutoff during fast motion (beta) so quick moves aren't lagged.
class OneEuro {
	constructor(minCutoff = 1.7, beta = 0.25, dcutoff = 1.0) {
		this.minCutoff = minCutoff; this.beta = beta; this.dcutoff = dcutoff;
		this.x = null; this.dx = 0; this.t = null;
	}
	_a(c, dt) { const tau = 1 / (2 * Math.PI * c); return 1 / (1 + tau / dt); }
	filter(x, t) {
		if (this.x === null) { this.x = x; this.t = t; return x; }
		const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
		const dx = (x - this.x) / dt;
		const ad = this._a(this.dcutoff, dt);
		this.dx = ad * dx + (1 - ad) * this.dx;
		const a = this._a(this.minCutoff + this.beta * Math.abs(this.dx), dt);
		this.x = a * x + (1 - a) * this.x;
		return this.x;
	}
}
// Per-avatar filter bank so A (live) and B (MHP) don't share smoothing state.
const _banks = new WeakMap();
function smoothCh(vrm, key, val) {
	let bank = _banks.get(vrm);
	if (!bank) { bank = new Map(); _banks.set(vrm, bank); }
	let f = bank.get(key);
	if (!f) { f = new OneEuro(); bank.set(key, f); }
	return f.filter(val, performance.now());
}
function rigRot(vrm, name, rot, damp = 1) {
	const node = vrm?.humanoid?.getNormalizedBoneNode(name);
	if (!node || !rot) return;
	const x = smoothCh(vrm, name + 'x', (rot.x ?? 0) * damp);
	const y = smoothCh(vrm, name + 'y', (rot.y ?? 0) * damp);
	const z = smoothCh(vrm, name + 'z', (rot.z ?? 0) * damp);
	_e.set(x, y, z, rot.rotationOrder || 'XYZ');
	node.quaternion.setFromEuler(_e);
}
function easeBone(vrm, name, amt = 0.12) {
	const n = vrm?.humanoid?.getNormalizedBoneNode(name);
	if (n) n.quaternion.slerp(_ident, amt);
}
// Head orientation from pose landmarks (image-plane signals): ear-line tilt =
// roll, nose-between-ears = yaw, nose-below-ears = pitch. For avatar B the ear
// landmarks (7/8) are unmapped (visibility 0) so the guard disables this.
const HEAD = { yaw: -0.8, pitch: 0.9, roll: -1.0, pitchBias: 0.18 };
function rigHeadFromPose(vrm, lm) {
	const nose = lm[0], lEar = lm[7], rEar = lm[8];
	if (!nose || !lEar || !rEar) return;
	if ((lEar.visibility ?? 1) < 0.4 || (rEar.visibility ?? 1) < 0.4) return;
	const dx = rEar.x - lEar.x, dy = rEar.y - lEar.y, w = Math.hypot(dx, dy) || 1e-3;
	const roll = Math.atan2(dy, dx);
	const t = ((nose.x - lEar.x) * dx + (nose.y - lEar.y) * dy) / (w * w);
	const yaw = (t - 0.5) * 2;
	const pitch = (nose.y - (lEar.y + rEar.y) / 2) / w - HEAD.pitchBias;
	rigRot(vrm, 'neck', {
		x: clamp(pitch * HEAD.pitch, -0.6, 0.6),
		y: clamp(yaw * HEAD.yaw, -0.7, 0.7),
		z: clamp(roll * HEAD.roll, -0.5, 0.5),
	}, 0.8);
}
function driveVRM(vrm, world, lm) {
	if (!vrm || !Kalidokit) return;
	const rp = Kalidokit.Pose.solve(world, lm, { runtime: 'mediapipe', video });
	if (!rp) return;
	const vis = (i) => lm?.[i]?.visibility ?? 1;   // MHP world passes 1 -> gates always open
	const G = 1.15;                                // Kalidokit under-reaches; amplify limbs
	rigRot(vrm, 'hips', rp.Hips.rotation, 0.25);   // hips is the root — damp hard (v1 value)
	rigRot(vrm, 'spine', rp.Spine, 0.45); rigRot(vrm, 'chest', rp.Spine, 0.3);
	// Per-limb confidence gating: drive a limb only when its landmarks are
	// clearly visible, else ease it to rest so occluded limbs don't wobble.
	if ((vis(14) + vis(16)) / 2 > 0.5) { rigRot(vrm, 'rightUpperArm', rp.RightUpperArm, G); rigRot(vrm, 'rightLowerArm', rp.RightLowerArm, G); }
	else { easeBone(vrm, 'rightUpperArm'); easeBone(vrm, 'rightLowerArm'); }
	if ((vis(13) + vis(15)) / 2 > 0.5) { rigRot(vrm, 'leftUpperArm', rp.LeftUpperArm, G); rigRot(vrm, 'leftLowerArm', rp.LeftLowerArm, G); }
	else { easeBone(vrm, 'leftUpperArm'); easeBone(vrm, 'leftLowerArm'); }
	if ((vis(25) + vis(27)) / 2 > 0.4) { rigRot(vrm, 'leftUpperLeg', rp.LeftUpperLeg, G); rigRot(vrm, 'leftLowerLeg', rp.LeftLowerLeg, G); }
	else { easeBone(vrm, 'leftUpperLeg'); easeBone(vrm, 'leftLowerLeg'); }
	if ((vis(26) + vis(28)) / 2 > 0.4) { rigRot(vrm, 'rightUpperLeg', rp.RightUpperLeg, G); rigRot(vrm, 'rightLowerLeg', rp.RightLowerLeg, G); }
	else { easeBone(vrm, 'rightUpperLeg'); easeBone(vrm, 'rightLowerLeg'); }
	rigHeadFromPose(vrm, lm);                      // head from pose (no-op for avatar B)
}
// H36M 17 joints -> MediaPipe-33 world format so the same Kalidokit solver can
// drive avatar B. The lifter output is already metric, pelvis-centred, y-down,
// +z away — exactly MediaPipe's world convention — so this is a pure reindex
// (no scale fudge, no sign flip). Key = MediaPipe index, value = H36M index.
const H2MP = { 0: 9, 11: 11, 12: 14, 13: 12, 14: 15, 15: 13, 16: 16, 23: 4, 24: 1, 25: 5, 26: 2, 27: 6, 28: 3 };
function liftToMpWorld(j) {
	const w = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
	for (const mp in H2MP) { const p = j[H2MP[mp]]; w[mp] = { x: p.x, y: p.y, z: p.z, visibility: 1 }; }
	return w;
}

function makeSkeleton(color, n, conns) {
	const pts = new THREE.BufferGeometry();
	pts.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
	const points = new THREE.Points(pts, new THREE.PointsMaterial({ color, size: 0.045 }));
	const segGeo = new THREE.BufferGeometry();
	segGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(conns.length * 6), 3));
	const segs = new THREE.LineSegments(segGeo, new THREE.LineBasicMaterial({ color }));
	points.frustumCulled = segs.frustumCulled = false;
	scene.add(points, segs);
	return { points, segs, conns, visible: (v) => { points.visible = segs.visible = v; } };
}
const mpSkel = makeSkeleton(0x2b5fd9, 33, PoseLandmarker.POSE_CONNECTIONS.map((c) => [c.start, c.end]));
mpSkel.visible(false); // shown only when the avatar is toggled off
mpSkel.points.position.set(0.1, 0.9, 0); // where the avatar stands
mpSkel.segs.position.set(0.1, 0.9, 0);
function updateMpSkel(world) {
	const pp = mpSkel.points.geometry.attributes.position.array;
	for (let i = 0; i < 33; i++) { const w = world[i]; pp[i * 3] = w.x; pp[i * 3 + 1] = -w.y; pp[i * 3 + 2] = -w.z; }
	mpSkel.points.geometry.attributes.position.needsUpdate = true;
	const sp = mpSkel.segs.geometry.attributes.position.array;
	mpSkel.conns.forEach((c, j) => { const a = world[c[0]], b = world[c[1]]; sp[j*6]=a.x; sp[j*6+1]=-a.y; sp[j*6+2]=-a.z; sp[j*6+3]=b.x; sp[j*6+4]=-b.y; sp[j*6+5]=-b.z; });
	mpSkel.segs.geometry.attributes.position.needsUpdate = true;
}
const showAvatar = document.getElementById('showAvatar');
function applyView() {
	const a = showAvatar.checked;            // avatars vs. skeletons
	if (vrmA) vrmA.scene.visible = a;
	if (vrmB) vrmB.scene.visible = a;
	mpSkel.visible(!a);
	mhpSkel.visible(!a);
}
showAvatar.addEventListener('change', applyView);
const mhpSkel = makeSkeleton(0xf59e0b, 17, LIFT_SKELETON); // orange = lifter
mhpSkel.visible(false);
mhpSkel.points.position.set(1.6, 0.9, 0); // aligned with avatar B
mhpSkel.segs.position.set(1.6, 0.9, 0);

// Render a joint list (x,y,z) into a skeleton, centred on a root joint.
function renderSkeleton(skel, joints, root, scale) {
	const r = joints[root] || { x: 0, y: 0, z: 0 };
	const map = (p) => [(p.x - r.x) * scale, -(p.y - r.y) * scale, -(p.z - r.z) * scale];
	const pp = skel.points.geometry.attributes.position.array;
	joints.forEach((p, i) => { const m = map(p); pp[i * 3] = m[0]; pp[i * 3 + 1] = m[1]; pp[i * 3 + 2] = m[2]; });
	skel.points.geometry.attributes.position.needsUpdate = true;
	const sp = skel.segs.geometry.attributes.position.array;
	skel.conns.forEach((c, j) => {
		const a = map(joints[c[0]] || r), b = map(joints[c[1]] || r);
		sp[j * 6] = a[0]; sp[j * 6 + 1] = a[1]; sp[j * 6 + 2] = a[2];
		sp[j * 6 + 3] = b[0]; sp[j * 6 + 4] = b[1]; sp[j * 6 + 5] = b[2];
	});
	skel.segs.geometry.attributes.position.needsUpdate = true;
}

// --- Self-supervised quality scoreboard ------------------------------------
// No ground truth needed: a correct 3D pose keeps bone LENGTHS stable frame to
// frame, so the coefficient of variation of each bone length (bone±) is a
// method-comparable quality proxy — lower is better. We also show per-frame
// joint velocity (jit) and depth spread (zσ) as method-internal diagnostics.
const WIN = 45; // ~sliding window of frames
const _d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
function makeMetric(bones) { return { bones, hist: bones.map(() => []), prev: null, vsum: 0, vn: 0 }; }
const mBlue = makeMetric([[11,13],[13,15],[12,14],[14,16],[23,25],[25,27],[24,26],[26,28],[11,23],[12,24]]);
const mOrange = makeMetric([[1,2],[2,3],[4,5],[5,6],[11,12],[12,13],[14,15],[15,16]]); // H36M limb bones
function resetMetric(m) { m.hist.forEach((h) => (h.length = 0)); m.prev = null; m.vsum = 0; m.vn = 0; }
function pushMetric(m, J, label) {
	let cv = 0, nb = 0;
	m.bones.forEach((bn, i) => {
		const a = J[bn[0]], b = J[bn[1]]; if (!a || !b) return;
		const h = m.hist[i]; h.push(_d3(a, b)); if (h.length > WIN) h.shift();
		const mean = h.reduce((s, x) => s + x, 0) / h.length;
		const sd = Math.sqrt(h.reduce((s, x) => s + (x - mean) ** 2, 0) / h.length);
		cv += mean > 1e-6 ? sd / mean : 0; nb++;
	});
	cv = nb ? cv / nb : 0;
	if (m.prev) { let v = 0, n = 0; for (let i = 0; i < J.length; i++) { if (J[i] && m.prev[i]) { v += _d3(J[i], m.prev[i]); n++; } } if (n) { m.vsum += v / n; m.vn++; } }
	m.prev = J.map((p) => ({ x: p.x, y: p.y, z: p.z }));
	const zs = J.map((p) => p.z), zmean = zs.reduce((s, x) => s + x, 0) / zs.length;
	const zsd = Math.sqrt(zs.reduce((s, x) => s + (x - zmean) ** 2, 0) / zs.length);
	const jit = m.vn ? m.vsum / m.vn : 0;
	log(label, `bone±${(cv * 100).toFixed(1)}% · jit ${jit.toFixed(3)} · zσ ${zsd.toFixed(2)}`, cv < 0.05 ? 'ok' : cv < 0.12 ? 'wait' : 'bad');
}

// One-Euro bank for the 21 orange joints (xyz), smoothed before render + drive
// so the per-frame ONNX jitter doesn't pass straight through.
const _mhpFilt = Array.from({ length: 17 * 3 }, () => new OneEuro(1.5, 0.3, 1.0)); // 17 joints × xyz
function smoothJoints(joints, t) {
	return joints.map((p, j) => ({ x: _mhpFilt[j*3].filter(p.x, t), y: _mhpFilt[j*3+1].filter(p.y, t), z: _mhpFilt[j*3+2].filter(p.z, t) }));
}
function resetJointFilters() { _mhpFilt.forEach((f) => { f.x = null; f.t = null; f.dx = 0; }); }

// --- Camera views + overlay -------------------------------------------------
// Side view is where depth differences read most clearly; overlay drops the
// orange skeleton onto the blue one so divergence is obvious.
const VIEWS = { front: [0.85, 0.9, 4.6], side: [5.0, 0.9, 0.05], top: [0.85, 4.8, 0.06] };
function setView(name) { const v = VIEWS[name]; if (!v) return; camera.position.set(v[0], v[1], v[2]); controls.target.set(0.85, 0.9, 0); controls.update(); }
['front', 'side', 'top'].forEach((v) => document.getElementById('v-' + v)?.addEventListener('click', () => setView(v)));
const overlayEl = document.getElementById('overlay');
function applyOverlay() {
	const x = overlayEl?.checked ? 0.1 : 1.6; // align orange onto blue, or back beside it
	mhpSkel.points.position.x = mhpSkel.segs.position.x = x;
}
overlayEl?.addEventListener('change', applyOverlay);

addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
const clock = new THREE.Clock();
(function loop() { requestAnimationFrame(loop); const dt = clock.getDelta(); if (vrmA) vrmA.update(dt); if (vrmB) vrmB.update(dt); controls.update(); renderer.render(scene, camera); if (compCtx) drawComp(); })();

// --- onnxruntime-web (WebGPU) + SimpleBaseline3D lifter --------------------
let liftSession = null;

async function loadLifter(url) {
	log('model', 'loading…', 'wait');
	try {
		const ep = new URLSearchParams(location.search).get('ep') || 'webgpu';
		liftSession = await ort.InferenceSession.create(url, { executionProviders: ep === 'wasm' ? ['wasm'] : ['webgpu', 'wasm'] });
		log('model', `loaded · ${liftSession.inputNames[0]} -> ${liftSession.outputNames[0]}`, 'ok');
		applyView();
	} catch (e) { console.error(e); log('model', `failed: ${e.message ?? e}`, 'bad'); }
}

// Build the lifter's [1,17,2] input from MediaPipe image-space landmarks.
// H36M expects pixel-scale coords, so we use real pixels (x*vw, y*vh) to keep
// the true aspect ratio. Pelvis/Thorax/Spine/Head aren't MediaPipe joints, so
// they're synthesized from mid-hip / mid-shoulder / nose. MP2H maps the rest:
// H36M index -> MediaPipe index. (MediaPipe 11 = anatomical left, matching H36M.)
const J = 17;
const MP2H = { 1: 24, 2: 26, 3: 28, 4: 23, 5: 25, 6: 27, 9: 0, 11: 11, 12: 13, 13: 15, 14: 12, 15: 14, 16: 16 };
function buildKeypoints(lm) {
	const vw = video.videoWidth, vh = video.videoHeight;
	const px = (i) => lm[i].x * vw, py = (i) => lm[i].y * vh;
	const kp = new Float32Array(J * 2);
	const set = (h, x, y) => { kp[h * 2] = x; kp[h * 2 + 1] = y; };
	const pelX = (px(23) + px(24)) / 2, pelY = (py(23) + py(24)) / 2;
	const thoX = (px(11) + px(12)) / 2, thoY = (py(11) + py(12)) / 2;
	set(0, pelX, pelY);                                 // Pelvis = mid-hip
	set(8, thoX, thoY);                                 // Thorax = mid-shoulder
	set(7, (pelX + thoX) / 2, (pelY + thoY) / 2);       // Spine
	set(10, px(0), py(0) - 0.06 * vh);                  // Head = nose lifted
	for (const h in MP2H) { const i = MP2H[h]; set(+h, px(i), py(i)); }
	return new ort.Tensor('float32', kp, [1, J, 2]);
}

// Output is [1,17,3] = metric (m), root-relative 3D, H36M order. Just reshape.
function decode(out) {
	const joints = [];
	for (let j = 0; j < J; j++) joints.push({ x: out[j * 3], y: out[j * 3 + 1], z: out[j * 3 + 2] });
	return joints;
}

let busy = false, lastLift = 0;
const LIFT_INTERVAL = 60; // ms — cap the lifter at ~16 Hz so it doesn't waste MediaPipe frames
async function runLifter(lm) {
	const now = performance.now();
	if (!liftSession || busy || now - lastLift < LIFT_INTERVAL) return;
	lastLift = now;
	busy = true;
	try {
		const input = buildKeypoints(lm);
		const out = await liftSession.run({ [liftSession.inputNames[0]]: input });
		const joints = smoothJoints(decode(out[liftSession.outputNames[0]].data), performance.now());
		renderSkeleton(mhpSkel, joints, LIFT_ROOT, 1.0); // metric → scale 1.0, matches blue
		pushMetric(mOrange, joints, 'orange');
		const mw = liftToMpWorld(joints);
		driveVRM(vrmB, mw, mw); // B = SimpleBaseline3D lifter
		input.dispose?.(); out[liftSession.outputNames[0]]?.dispose?.();
	} catch (e) { log('model', `infer error: ${e.message ?? e}`, 'bad'); liftSession = null; }
	busy = false;
}

document.getElementById('loadModel').addEventListener('click', () => {
	const url = document.getElementById('model').value.trim() || LIFT_MODEL;
	loadLifter(url);
});

// --- MediaPipe pose ---------------------------------------------------------
const video = document.getElementById('vid');
const cam2d = document.getElementById('cam2d');
const cctx = cam2d.getContext('2d');
const draw = new DrawingUtils(cctx);
let pose = null, running = false;

async function initPose() {
	log('mediapipe', 'loading…', 'wait');
	const fileset = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TV}/wasm`);
	pose = await PoseLandmarker.createFromOptions(fileset, {
		baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' }, runningMode: 'VIDEO', numPoses: 1,
	});
	log('mediapipe', 'ready', 'ok');
}

function frame() {
	if (!running) return;
	if (video.readyState >= 2 && video.videoWidth) {
		const res = pose.detectForVideo(video, performance.now());
		const world = res.worldLandmarks?.[0];
		const lm = res.landmarks?.[0];
		if (world && lm) { driveVRM(vrmA, world, lm); updateMpSkel(world); pushMetric(mBlue, world, 'blue'); } // A = MediaPipe
		cctx.save(); cctx.clearRect(0, 0, cam2d.width, cam2d.height); cctx.drawImage(video, 0, 0, cam2d.width, cam2d.height);
		if (lm) draw.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#2b5fd9', lineWidth: 2 });
		cctx.restore();
		if (lm) runLifter(lm); // async; orange skeleton
	}
	video.requestVideoFrameCallback ? video.requestVideoFrameCallback(frame) : requestAnimationFrame(frame);
}

async function start() {
	if (running) return;
	if (!vrmA) { log('avatar', 'loading 2 avatars…', 'wait'); await loadVRMs(); log('avatar', 'ready', 'ok'); }
	if (!pose) await initPose();
	if (!liftSession) loadLifter(LIFT_MODEL);
	const src = document.getElementById('src').value;
	if (src === 'camera') { video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false }); video.src = ''; }
	else { video.srcObject = null; video.src = src; video.loop = true; }
	await video.play();
	running = true; log('tracking', 'live', 'ok'); frame();
}
function stop() {
	running = false;
	if (vrmA) _banks.delete(vrmA); if (vrmB) _banks.delete(vrmB); // reset One-Euro state
	resetJointFilters(); resetMetric(mBlue); resetMetric(mOrange); lastLift = 0;
	if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
	if (video.src) { video.pause(); video.removeAttribute('src'); video.load(); }
	log('tracking', 'stopped', 'wait');
}
document.getElementById('start').addEventListener('click', () => start().catch((e) => log('tracking', `error: ${e.message ?? e}`, 'bad')));
document.getElementById('stop').addEventListener('click', stop);

// --- recording (3D scene + camera PiP, like v1) -----------------------------
let recorder = null, chunks = [], recTimer = null;
const recBtn = document.getElementById('rec'), recFmt = document.getElementById('recFmt');
function drawComp() {
	const W = compCanvas.width, H = compCanvas.height;
	compCtx.drawImage(renderer.domElement, 0, 0, W, H);
	if (cam2d.width) {
		const pw = Math.round(W * 0.28), ph = Math.round(pw * cam2d.height / cam2d.width), m = Math.round(W * 0.02);
		compCtx.drawImage(cam2d, W - pw - m, H - ph - m, pw, ph);
		compCtx.strokeStyle = '#fff'; compCtx.lineWidth = 2; compCtx.strokeRect(W - pw - m, H - ph - m, pw, ph);
	}
}
recBtn.addEventListener('click', () => {
	if (!recorder) {
		compCanvas = document.createElement('canvas');
		compCanvas.width = renderer.domElement.width; compCanvas.height = renderer.domElement.height;
		compCtx = compCanvas.getContext('2d');
		const cands = recFmt.value === 'mp4'
			? ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm']
			: ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
		const mime = cands.find((m) => MediaRecorder.isTypeSupported(m));
		const ext = mime.includes('mp4') ? 'mp4' : 'webm';
		recorder = new MediaRecorder(compCanvas.captureStream(30), { mimeType: mime, videoBitsPerSecond: 12_000_000 });
		chunks = [];
		recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
		recorder.onstop = () => {
			const blob = new Blob(chunks, { type: mime }), u = URL.createObjectURL(blob), a = document.createElement('a');
			a.href = u; a.download = `cam2avatar-v2-${Date.now()}.${ext}`; a.click();
			setTimeout(() => URL.revokeObjectURL(u), 1000);
		};
		recorder.start();
		recBtn.style.background = '#d62828'; recBtn.style.color = '#fff';
		const t0 = Date.now(); recBtn.textContent = '■ Stop (0s)';
		recTimer = setInterval(() => { recBtn.textContent = `■ Stop (${Math.floor((Date.now() - t0) / 1000)}s)`; }, 1000);
	} else {
		recorder.stop(); recorder = null; clearInterval(recTimer);
		compCanvas = null; compCtx = null;
		recBtn.textContent = '● Record'; recBtn.style.background = ''; recBtn.style.color = '';
	}
});

// boot
log('webgpu', navigator.gpu ? 'available' : 'NOT available', navigator.gpu ? 'ok' : 'bad');
log('ort', `v${ort.env.versions.common}`, 'ok');
log('mediapipe', 'idle — press Start', 'wait');
document.getElementById('model').value = LIFT_MODEL;
log('model', 'SimpleBaseline3D lifter (bundled) — loads on Start', 'wait');
