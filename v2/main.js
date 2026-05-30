// cam2avatar v2 — WebGPU 3D-lift sandbox.
// Renders MediaPipe's live monocular 3D pose, and initializes onnxruntime-web on
// WebGPU so a 2D->3D lifting model can be dropped in for A/B. Standalone.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web/webgpu';

const TV = '0.10.35';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

const statusEl = document.getElementById('status');
const lines = [];
const log = (k, v, cls) => {
	const i = lines.findIndex((l) => l.k === k);
	const entry = { k, v, cls };
	if (i >= 0) lines[i] = entry; else lines.push(entry);
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
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 100);
camera.position.set(0, 0, 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
scene.add(new THREE.GridHelper(4, 8, 0xaab0bd, 0xc7ccd6));
scene.add(new THREE.AxesHelper(0.3));

// Skeleton renderer: a set of points + connection lines, updated per frame.
function makeSkeleton(color) {
	const pts = new THREE.BufferGeometry();
	pts.setAttribute('position', new THREE.BufferAttribute(new Float32Array(33 * 3), 3));
	const points = new THREE.Points(pts, new THREE.PointsMaterial({ color, size: 0.04 }));
	const segGeo = new THREE.BufferGeometry();
	segGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(PoseLandmarker.POSE_CONNECTIONS.length * 2 * 3), 3));
	const segs = new THREE.LineSegments(segGeo, new THREE.LineBasicMaterial({ color }));
	points.frustumCulled = segs.frustumCulled = false;
	scene.add(points); scene.add(segs);
	return { points, segs, visible: (v) => { points.visible = segs.visible = v; } };
}
const mpSkel = makeSkeleton(0x2b5fd9);   // MediaPipe 3D (blue)
const liftSkel = makeSkeleton(0xf59e0b); // lifted 3D (orange)
liftSkel.visible(false);

// World landmarks (metres, x-right, y-down, z-depth) -> three (y-up, centred).
function updateSkeleton(skel, world) {
	const pp = skel.points.geometry.attributes.position.array;
	for (let i = 0; i < 33; i++) {
		const w = world[i] || { x: 0, y: 0, z: 0 };
		pp[i * 3] = w.x; pp[i * 3 + 1] = -w.y; pp[i * 3 + 2] = -w.z;
	}
	skel.points.geometry.attributes.position.needsUpdate = true;
	const sp = skel.segs.geometry.attributes.position.array;
	PoseLandmarker.POSE_CONNECTIONS.forEach((c, j) => {
		const a = world[c.start] || { x: 0, y: 0, z: 0 }, b = world[c.end] || { x: 0, y: 0, z: 0 };
		sp[j * 6] = a.x; sp[j * 6 + 1] = -a.y; sp[j * 6 + 2] = -a.z;
		sp[j * 6 + 3] = b.x; sp[j * 6 + 4] = -b.y; sp[j * 6 + 5] = -b.z;
	});
	skel.segs.geometry.attributes.position.needsUpdate = true;
}

addEventListener('resize', () => {
	camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
	renderer.setSize(innerWidth, innerHeight);
});
(function render() { requestAnimationFrame(render); controls.update(); renderer.render(scene, camera); })();

// --- onnxruntime-web (WebGPU) ----------------------------------------------
let liftSession = null;
async function initOrt() {
	log('webgpu', navigator.gpu ? 'available' : 'NOT available', navigator.gpu ? 'ok' : 'bad');
	log('ort', `v${ort.env.versions.common}`, 'ok');
}
document.getElementById('loadModel').addEventListener('click', async () => {
	const url = document.getElementById('model').value.trim();
	if (!url) { log('model', 'paste an .onnx URL first', 'wait'); return; }
	log('model', 'loading…', 'wait');
	try {
		liftSession = await ort.InferenceSession.create(url, { executionProviders: ['webgpu', 'wasm'] });
		log('model', `loaded (${liftSession.inputNames.join(',')} -> ${liftSession.outputNames.join(',')})`, 'ok');
		liftSkel.visible(true);
	} catch (e) {
		console.error(e);
		log('model', `failed: ${e.message ?? e}`, 'bad');
	}
});

// Run the lifter on 2D keypoints. Input/output layout is model-specific — this
// is a best-effort path you adapt to your exported model (e.g. SimpleBaseline3D:
// 17 (x,y) joints in, 17 (x,y,z) out). Wrapped so a mismatch never breaks the app.
async function lift(landmarks2d) {
	if (!liftSession) return null;
	try {
		const flat = new Float32Array(landmarks2d.length * 2);
		landmarks2d.forEach((p, i) => { flat[i * 2] = p.x; flat[i * 2 + 1] = p.y; });
		const input = new ort.Tensor('float32', flat, [1, landmarks2d.length, 2]);
		const out = await liftSession.run({ [liftSession.inputNames[0]]: input });
		const d = out[liftSession.outputNames[0]].data;
		const n = d.length / 3;
		return Array.from({ length: n }, (_, i) => ({ x: d[i * 3], y: d[i * 3 + 1], z: d[i * 3 + 2] }));
	} catch (e) { log('lift', `infer error: ${e.message ?? e}`, 'bad'); liftSession = null; return null; }
}

// --- MediaPipe pose ---------------------------------------------------------
const video = document.getElementById('vid');
const cam2d = document.getElementById('cam2d');
const cctx = cam2d.getContext('2d');
const draw = new DrawingUtils(cctx);
let pose = null, running = false, raf = 0;

async function initPose() {
	log('mediapipe', 'loading…', 'wait');
	const fileset = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TV}/wasm`);
	pose = await PoseLandmarker.createFromOptions(fileset, {
		baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
		runningMode: 'VIDEO', numPoses: 1,
	});
	log('mediapipe', 'ready', 'ok');
}

async function loop() {
	if (!running) return;
	if (video.readyState >= 2 && video.videoWidth) {
		const res = pose.detectForVideo(video, performance.now());
		const world = res.worldLandmarks?.[0];
		const lm = res.landmarks?.[0];
		if (world) updateSkeleton(mpSkel, world);
		// 2D preview
		cctx.save(); cctx.clearRect(0, 0, cam2d.width, cam2d.height);
		cctx.drawImage(video, 0, 0, cam2d.width, cam2d.height);
		if (lm) draw.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#2b5fd9', lineWidth: 2 });
		cctx.restore();
		// optional lift
		if (lm && liftSession) { const l = await lift(lm); if (l) updateSkeleton(liftSkel, l); }
	}
	raf = video.requestVideoFrameCallback ? (video.requestVideoFrameCallback(loop), 0) : requestAnimationFrame(loop);
}

async function start() {
	if (running) return;
	if (!pose) await initPose();
	const src = document.getElementById('src').value;
	if (src === 'camera') {
		video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
		video.src = '';
	} else { video.srcObject = null; video.src = src; video.loop = true; }
	await video.play();
	running = true;
	log('tracking', 'live', 'ok');
	loop();
}
function stop() {
	running = false;
	if (raf) cancelAnimationFrame(raf);
	if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
	if (video.src) { video.pause(); video.removeAttribute('src'); video.load(); }
	log('tracking', 'stopped', 'wait');
}
document.getElementById('start').addEventListener('click', () => start().catch((e) => log('tracking', `error: ${e.message ?? e}`, 'bad')));
document.getElementById('stop').addEventListener('click', stop);

// boot
initOrt();
log('mediapipe', 'idle — press Start', 'wait');
log('model', 'none — paste a lifter .onnx URL', 'wait');
