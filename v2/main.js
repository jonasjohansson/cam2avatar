// cam2avatar v2 — WebGPU 3D sandbox.
// Compares MediaPipe's monocular 3D (blue) against the Mobile Human Pose model
// (orange) — a 3DMPPE image->3D network run on WebGPU via onnxruntime-web.
// Standalone; doesn't touch the main app.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web/webgpu';

const Kalidokit = window.Kalidokit;

const TV = '0.10.35';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
const LIFT_MODEL = './models/mobile_human_pose_256x256.onnx';

// 3DMPPE / MuCo 21-joint skeleton (pelvis = 14 is the root).
const MHP_SKELETON = [[0,16],[16,1],[1,15],[15,14],[14,8],[14,11],[8,9],[9,10],[10,19],[11,12],[12,13],[13,20],[1,2],[2,3],[3,4],[4,17],[1,5],[5,6],[6,7],[7,18]];
const MHP_ROOT = 14;

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
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe3ea);
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.01, 100);
camera.position.set(0.7, 0.9, 3.8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.7, 0.9, 0);
scene.add(new THREE.GridHelper(4, 8, 0xaab0bd, 0xc7ccd6));
scene.add(new THREE.HemisphereLight(0xffffff, 0x667, 1.4));
const dir = new THREE.DirectionalLight(0xffffff, 1.4); dir.position.set(1, 2, 1.5); scene.add(dir);

// --- VRM avatar (driven by MediaPipe via Kalidokit, like v1) ----------------
let vrm = null;
async function loadVRM() {
	const loader = new GLTFLoader();
	loader.register((parser) => new VRMLoaderPlugin(parser));
	const gltf = await loader.loadAsync('https://cdn.jsdelivr.net/gh/madjin/vrm-samples@master/vroid/stable/AvatarSample_C.vrm');
	vrm = gltf.userData.vrm;
	VRMUtils.removeUnnecessaryVertices(gltf.scene);
	VRMUtils.combineSkeletons(gltf.scene);
	VRMUtils.rotateVRM0(vrm);
	vrm.scene.traverse((o) => { o.frustumCulled = false; });
	vrm.scene.position.x = 0.1; // avatar (clear of the panel), MHP skeleton to its right
	scene.add(vrm.scene);
}

const _e = new THREE.Euler(), _q = new THREE.Quaternion();
function rigRot(name, rot, damp = 1) {
	const node = vrm?.humanoid?.getNormalizedBoneNode(name);
	if (!node || !rot) return;
	_e.set((rot.x ?? 0) * damp, (rot.y ?? 0) * damp, (rot.z ?? 0) * damp, rot.rotationOrder || 'XYZ');
	node.quaternion.slerp(_q.setFromEuler(_e), 0.4);
}
function driveVRM(world, lm) {
	if (!vrm || !Kalidokit) return;
	const rp = Kalidokit.Pose.solve(world, lm, { runtime: 'mediapipe', video });
	if (!rp) return;
	rigRot('hips', rp.Hips.rotation, 0.7);
	rigRot('spine', rp.Spine, 0.45); rigRot('chest', rp.Spine, 0.3);
	rigRot('rightUpperArm', rp.RightUpperArm); rigRot('rightLowerArm', rp.RightLowerArm);
	rigRot('leftUpperArm', rp.LeftUpperArm); rigRot('leftLowerArm', rp.LeftLowerArm);
	rigRot('leftUpperLeg', rp.LeftUpperLeg); rigRot('leftLowerLeg', rp.LeftLowerLeg);
	rigRot('rightUpperLeg', rp.RightUpperLeg); rigRot('rightLowerLeg', rp.RightLowerLeg);
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
mpSkel.visible(false); // hidden — the VRM (left) IS the MediaPipe result
const mhpSkel = makeSkeleton(0xf59e0b, 21, MHP_SKELETON);
mhpSkel.visible(false);
mhpSkel.points.position.set(1.5, 0.9, 0); // beside the avatar, at pelvis height
mhpSkel.segs.position.set(1.5, 0.9, 0);

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

addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
const clock = new THREE.Clock();
(function loop() { requestAnimationFrame(loop); const dt = clock.getDelta(); if (vrm) vrm.update(dt); controls.update(); renderer.render(scene, camera); })();

// --- onnxruntime-web (WebGPU) + Mobile Human Pose --------------------------
let liftSession = null;
const crop = document.createElement('canvas'); crop.width = 256; crop.height = 256;
const cropCtx = crop.getContext('2d', { willReadFrequently: true });

async function loadLifter(url) {
	log('model', 'loading…', 'wait');
	try {
		const ep = new URLSearchParams(location.search).get('ep') || 'webgpu';
		liftSession = await ort.InferenceSession.create(url, { executionProviders: ep === 'wasm' ? ['wasm'] : ['webgpu', 'wasm'] });
		log('model', `loaded · ${liftSession.inputNames[0]} -> ${liftSession.outputNames[0]}`, 'ok');
		mhpSkel.visible(true);
	} catch (e) { console.error(e); log('model', `failed: ${e.message ?? e}`, 'bad'); }
}

// Crop the person (from MediaPipe's landmarks) to a 256x256 raw-RGB CHW tensor.
function preprocess(lm) {
	const vw = video.videoWidth, vh = video.videoHeight;
	let minx = 1, miny = 1, maxx = 0, maxy = 0;
	for (const p of lm) { if ((p.visibility ?? 1) < 0.3) continue; minx = Math.min(minx, p.x); miny = Math.min(miny, p.y); maxx = Math.max(maxx, p.x); maxy = Math.max(maxy, p.y); }
	const cx = (minx + maxx) / 2 * vw, cy = (miny + maxy) / 2 * vh;
	const half = Math.max((maxx - minx) * vw, (maxy - miny) * vh) / 2 * 1.2; // square + pad
	cropCtx.clearRect(0, 0, 256, 256);
	cropCtx.drawImage(video, cx - half, cy - half, half * 2, half * 2, 0, 0, 256, 256);
	const d = cropCtx.getImageData(0, 0, 256, 256).data;
	const t = new Float32Array(3 * 256 * 256);
	const plane = 256 * 256;
	for (let i = 0; i < plane; i++) { t[i] = d[i * 4]; t[plane + i] = d[i * 4 + 1]; t[2 * plane + i] = d[i * 4 + 2]; }
	return new ort.Tensor('float32', t, [1, 3, 256, 256]);
}

// Soft-argmax over the [1,672,32,32] heatmap -> 21 joints (x,y in [0,1], z in [-1,1]).
const J = 21, DIM = 32, HW = DIM * DIM;
function decode(out) {
	const joints = [];
	for (let j = 0; j < J; j++) {
		let max = -Infinity;
		for (let d = 0; d < DIM; d++) { const base = (j * DIM + d) * HW; for (let i = 0; i < HW; i++) { const v = out[base + i]; if (v > max) max = v; } }
		let sum = 0, ex = 0, ey = 0, ez = 0;
		for (let d = 0; d < DIM; d++) {
			const base = (j * DIM + d) * HW;
			for (let h = 0; h < DIM; h++) for (let w = 0; w < DIM; w++) {
				const e = Math.exp(out[base + h * DIM + w] - max);
				sum += e; ex += e * w; ey += e * h; ez += e * d;
			}
		}
		joints.push({ x: ex / sum / DIM, y: ey / sum / DIM, z: (ez / sum / DIM) * 2 - 1 });
	}
	return joints;
}

let busy = false;
async function runLifter(lm) {
	if (!liftSession || busy) return;
	busy = true;
	try {
		const input = preprocess(lm);
		const out = await liftSession.run({ [liftSession.inputNames[0]]: input });
		renderSkeleton(mhpSkel, decode(out[liftSession.outputNames[0]].data), MHP_ROOT, 1.6);
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
		if (world && lm) driveVRM(world, lm); // VRM avatar, like v1
		cctx.save(); cctx.clearRect(0, 0, cam2d.width, cam2d.height); cctx.drawImage(video, 0, 0, cam2d.width, cam2d.height);
		if (lm) draw.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#2b5fd9', lineWidth: 2 });
		cctx.restore();
		if (lm) runLifter(lm); // async; orange skeleton
	}
	video.requestVideoFrameCallback ? video.requestVideoFrameCallback(frame) : requestAnimationFrame(frame);
}

async function start() {
	if (running) return;
	if (!vrm) { log('avatar', 'loading…', 'wait'); await loadVRM(); log('avatar', 'ready', 'ok'); }
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
	if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
	if (video.src) { video.pause(); video.removeAttribute('src'); video.load(); }
	log('tracking', 'stopped', 'wait');
}
document.getElementById('start').addEventListener('click', () => start().catch((e) => log('tracking', `error: ${e.message ?? e}`, 'bad')));
document.getElementById('stop').addEventListener('click', stop);

// boot
log('webgpu', navigator.gpu ? 'available' : 'NOT available', navigator.gpu ? 'ok' : 'bad');
log('ort', `v${ort.env.versions.common}`, 'ok');
log('mediapipe', 'idle — press Start', 'wait');
document.getElementById('model').value = LIFT_MODEL;
log('model', 'Mobile Human Pose (bundled) — loads on Start', 'wait');
