import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { loadMixamoAnimation } from './loadMixamoAnimation.js';
import { createMocap } from './mocap.js';

// --- Preloaded samples (all hotlinked, verified resolvable) -----------------

const CDN = 'https://cdn.jsdelivr.net/gh';

const CHARACTERS = [
	{ name: 'VRoid — Sample C',      url: `${CDN}/madjin/vrm-samples@master/vroid/stable/AvatarSample_C.vrm` },
	{ name: 'VRoid — Sample A',      url: `${CDN}/madjin/vrm-samples@master/vroid/stable/AvatarSample_A.vrm` },
	{ name: 'VRoid — Sample B',      url: `${CDN}/madjin/vrm-samples@master/vroid/stable/AvatarSample_B.vrm` },
	{ name: 'VRoid — Vita',          url: `${CDN}/madjin/vrm-samples@master/vroid/beta/Vita.vrm` },
	{ name: 'VRoid — Sendagaya Shino', url: `${CDN}/madjin/vrm-samples@master/vroid/beta/Sendagaya_Shino.vrm` },
	{ name: 'VRoid — Victoria Rubin', url: `${CDN}/madjin/vrm-samples@master/vroid/beta/Victoria_Rubin.vrm` },
	{ name: 'VRoid — Vivi',          url: `${CDN}/madjin/vrm-samples@master/vroid/beta/Vivi.vrm` },
	{ name: 'Avatar Orion',          url: `${CDN}/madjin/vrm-samples@master/Avatar_Orion.vrm` },
	{ name: 'Seed-san',              url: `${CDN}/madjin/vrm-samples@master/Seed-san/vrm/Seed-san.vrm` },
	{ name: 'pixiv — VRM 1.0 robot', url: `${CDN}/pixiv/three-vrm@dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm` },
];

const ANIMATIONS = [
	{ name: 'Samba Dancing', url: `${CDN}/mrdoob/three.js@r180/examples/models/fbx/Samba%20Dancing.fbx` },
	{ name: 'Mixamo (default)', url: `${CDN}/mrdoob/three.js@r180/examples/models/fbx/mixamo.fbx` },
];

// --- Scene ------------------------------------------------------------------

const wrap = document.getElementById('canvas-wrap');

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe3ea);

const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 100);
const DEFAULT_CAM = { pos: new THREE.Vector3(0, 1.2, 3), target: new THREE.Vector3(0, 0.95, 0) };
camera.position.copy(DEFAULT_CAM.pos);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(DEFAULT_CAM.target);
controls.enableDamping = true;
controls.update();

// Lights
scene.add(new THREE.HemisphereLight(0xffffff, 0x666677, 1.4));
const dir = new THREE.DirectionalLight(0xffffff, 1.6);
dir.position.set(1, 2, 1.5);
scene.add(dir);

// Floor grid
const grid = new THREE.GridHelper(10, 20, 0xaab0bd, 0xc7ccd6);
scene.add(grid);

// --- State ------------------------------------------------------------------

let currentVrm = null;
let mixer = null;
let action = null;
let currentAnim = null;      // { url } | { file }
let speed = 1;
let playing = true;
const clock = new THREE.Clock();

const statusEl = document.getElementById('status');
const log = (msg) => { statusEl.textContent = msg; };

// Dev hook for the evaluation harness (read current rig state from Playwright).
window.__getVrm = () => currentVrm;

// --- Loading: VRM -----------------------------------------------------------

function makeGltfLoader() {
	const loader = new GLTFLoader();
	loader.register((parser) => new VRMLoaderPlugin(parser));
	return loader;
}

async function loadVRM(src, label) {
	log(`loading character: ${label}…`);
	try {
		const loader = makeGltfLoader();
		const gltf = await loader.loadAsync(src);
		const vrm = gltf.userData.vrm;

		// Cleanup / optimization (per three-vrm guidance)
		VRMUtils.removeUnnecessaryVertices(gltf.scene);
		VRMUtils.combineSkeletons(gltf.scene);
		VRMUtils.combineMorphs(vrm);
		// VRM 0.x faces +Z; rotate so every avatar faces the camera the same way.
		VRMUtils.rotateVRM0(vrm);

		// Swap out the old one
		if (currentVrm) {
			scene.remove(currentVrm.scene);
			VRMUtils.deepDispose(currentVrm.scene);
		}

		// Don't waste cycles on frustum culling tiny humanoid meshes
		vrm.scene.traverse((o) => { o.frustumCulled = false; });

		scene.add(vrm.scene);
		currentVrm = vrm;

		refreshSkeleton(); // rebuild the overlay for the new rig

		const ver = vrm.meta?.metaVersion === '0' ? 'VRM 0.x' : 'VRM 1.0';
		log(`character ready: ${label}  (${ver})`);

		// Re-apply the active animation onto the new rig (retarget is rig-specific)
		if (currentAnim) await applyAnimation();
	} catch (err) {
		console.error(err);
		log(`✗ failed to load character\n${err.message ?? err}`);
	}
}

// --- Loading: Mixamo animation ----------------------------------------------

async function applyAnimation() {
	if (!currentVrm || !currentAnim) return;
	const label = currentAnim.label ?? 'animation';
	log(`retargeting ${label}…`);
	try {
		const url = currentAnim.url ?? URL.createObjectURL(currentAnim.file);
		const clip = await loadMixamoAnimation(url, currentVrm);
		if (currentAnim.file) URL.revokeObjectURL(url);

		if (mixer) mixer.stopAllAction();
		mixer = new THREE.AnimationMixer(currentVrm.scene);
		action = mixer.clipAction(clip);
		action.timeScale = speed;
		action.play();
		if (!playing) action.paused = true;

		log(`▶ ${currentVrm.meta?.name ?? 'character'} · ${label}`);
	} catch (err) {
		console.error(err);
		log(`✗ failed to retarget animation\n${err.message ?? err}`);
	}
}

// --- UI wiring --------------------------------------------------------------

const charSelect = document.getElementById('char-select');
CHARACTERS.forEach((c, i) => {
	const opt = document.createElement('option');
	opt.value = i; opt.textContent = c.name;
	charSelect.appendChild(opt);
});
charSelect.addEventListener('change', () => {
	const c = CHARACTERS[charSelect.value];
	loadVRM(c.url, c.name);
});

const animSelect = document.getElementById('anim-select');
ANIMATIONS.forEach((a, i) => {
	const opt = document.createElement('option');
	opt.value = i; opt.textContent = a.name;
	animSelect.appendChild(opt);
});
animSelect.addEventListener('change', () => {
	const a = ANIMATIONS[animSelect.value];
	currentAnim = { url: a.url, label: a.name };
	applyAnimation();
});

// File pickers
const charFile = document.getElementById('char-file');
document.getElementById('char-file-btn').addEventListener('click', () => charFile.click());
charFile.addEventListener('change', () => {
	if (charFile.files[0]) loadVRMFile(charFile.files[0]);
});

const animFile = document.getElementById('anim-file');
document.getElementById('anim-file-btn').addEventListener('click', () => animFile.click());
animFile.addEventListener('change', () => {
	if (animFile.files[0]) loadAnimFile(animFile.files[0]);
});

function loadVRMFile(file) {
	const url = URL.createObjectURL(file);
	loadVRM(url, file.name).finally(() => URL.revokeObjectURL(url));
}
function loadAnimFile(file) {
	currentAnim = { file, label: file.name };
	applyAnimation();
}

// Play / pause
const playBtn = document.getElementById('play-btn');
playBtn.addEventListener('click', () => {
	playing = !playing;
	if (action) action.paused = !playing;
	playBtn.textContent = playing ? 'Pause' : 'Play';
});

// Record the 3D canvas to a WebM (downloads on stop)
let recorder = null;
let recChunks = [];
let recTimer = null;
const recBtn = document.getElementById('rec-btn');
recBtn.addEventListener('click', () => {
	if (!recorder) {
		const stream = renderer.domElement.captureStream(30);
		// Prefer MP4 where supported (Safari), else WebM (Chrome/Firefox).
		const mime = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
			.find((m) => MediaRecorder.isTypeSupported(m));
		if (!mime) { log('✗ recording not supported in this browser'); return; }
		const ext = mime.includes('mp4') ? 'mp4' : 'webm';
		recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
		recChunks = [];
		recorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
		recorder.onstop = () => {
			const blob = new Blob(recChunks, { type: mime });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `vrm-mixamo-${Date.now()}.${ext}`;
			a.click();
			setTimeout(() => URL.revokeObjectURL(url), 1000);
			log(`saved recording (${(blob.size / 1e6).toFixed(1)} MB) → Downloads`);
		};
		recorder.start();
		recBtn.classList.add('recording');
		const t0 = Date.now();
		recBtn.textContent = '■ Stop (0s)';
		recTimer = setInterval(() => {
			recBtn.textContent = `■ Stop (${Math.floor((Date.now() - t0) / 1000)}s)`;
		}, 1000);
	} else {
		recorder.stop();
		recorder = null;
		clearInterval(recTimer);
		recBtn.classList.remove('recording');
		recBtn.textContent = '● Record';
	}
});

// Reset camera
document.getElementById('reset-cam').addEventListener('click', () => {
	camera.position.copy(DEFAULT_CAM.pos);
	controls.target.copy(DEFAULT_CAM.target);
	controls.update();
});

// Speed
const speedEl = document.getElementById('speed');
const speedVal = document.getElementById('speed-val');
speedEl.addEventListener('input', () => {
	speed = parseFloat(speedEl.value);
	speedVal.textContent = `${speed.toFixed(1)}×`;
	if (action) action.timeScale = speed;
});

// Webcam mocap toggle
const previewEl = document.getElementById('preview');
const mocapOptsEl = document.getElementById('mocap-opts');
const mocap = createMocap({
	THREE,
	video: document.getElementById('webcam'),
	guideCanvas: document.getElementById('guide'),
	getVrm: () => currentVrm,
	log,
});
const webcamBtn = document.getElementById('webcam-btn');
const sourceSelect = document.getElementById('source-select');
let webcamOn = false;
webcamBtn.addEventListener('click', async () => {
	if (!webcamOn) {
		try {
			const src = sourceSelect.value;
			const isCam = src === 'camera';
			webcamBtn.textContent = isCam ? '○ starting camera…' : '○ loading clip…';
			await mocap.start(isCam ? { type: 'camera' } : { type: 'video', url: src });
			webcamOn = true;
			webcamBtn.textContent = '■ Stop tracking';
			mocapOptsEl.hidden = false;
			previewEl.hidden = !optPreview.checked;
			optMirror.checked = isCam;              // mirror live cam by default, clips as-is
			mocap.setOptions({ mirror: isCam });
			if (isCam) {                            // live webcam -> track legs + ground contact by default
				optLegs.value = 'webcam';
				mocap.setOptions({ legsMode: 'webcam' });
			}
			// Webcam drives the rig now — stop the Mixamo clip.
			if (mixer) mixer.stopAllAction();
			playing = false;
			playBtn.textContent = 'Play';
			log(`tracking ON (${sourceSelect.options[sourceSelect.selectedIndex].text}) — face + torso + arms + hands`);
		} catch (err) {
			console.error(err);
			webcamOn = false;
			webcamBtn.textContent = '● Start tracking';
			log(`✗ tracking: ${err.message ?? err}`);
		}
	} else {
		mocap.stop();
		webcamOn = false;
		webcamBtn.textContent = '● Start tracking';
		mocapOptsEl.hidden = true;
		previewEl.hidden = true;
		// Hand the rig back to the Mixamo animation.
		playing = true;
		playBtn.textContent = 'Pause';
		applyAnimation();
		log('webcam off — back to Mixamo clip');
	}
});

// Mocap options
const optPreview = document.getElementById('opt-preview');
const optLegs = document.getElementById('opt-legs');
const optResp = document.getElementById('opt-resp');
optPreview.addEventListener('change', () => {
	mocap.setOptions({ preview: optPreview.checked });
	if (webcamOn) previewEl.hidden = !optPreview.checked;
});
optLegs.addEventListener('change', () => mocap.setOptions({ legsMode: optLegs.value }));
optResp.addEventListener('input', () => mocap.setOptions({ resp: parseFloat(optResp.value) }));

const optFace = document.getElementById('opt-face');
optFace.addEventListener('change', () => mocap.setOptions({ face: optFace.checked }));

const optPlant = document.getElementById('opt-plant');
optPlant.addEventListener('change', () => mocap.setOptions({ plantFeet: optPlant.checked }));

const optQuality = document.getElementById('opt-quality');
optQuality.addEventListener('change', () => mocap.setOptions({ quality: optQuality.value }));

const optMirror = document.getElementById('opt-mirror');
optMirror.addEventListener('change', () => mocap.setOptions({ mirror: optMirror.checked }));

const fpsEl = document.getElementById('fps');

// Skeleton overlay — draws the VRM's actual bones on top of the mesh so you can
// see exactly which bones are (or aren't) being driven.
let skeletonHelper = null;
const optSkel = document.getElementById('opt-skeleton');
function refreshSkeleton() {
	if (skeletonHelper) {
		scene.remove(skeletonHelper);
		skeletonHelper.geometry?.dispose();
		skeletonHelper.material?.dispose();
		skeletonHelper = null;
	}
	if (optSkel.checked && currentVrm) {
		skeletonHelper = new THREE.SkeletonHelper(currentVrm.scene);
		skeletonHelper.material.depthTest = false; // always visible, even through the mesh
		skeletonHelper.material.depthWrite = false;
		skeletonHelper.material.transparent = true;
		skeletonHelper.renderOrder = 999;
		scene.add(skeletonHelper);
	}
}
optSkel.addEventListener('change', refreshSkeleton);

// Drag & drop anywhere
let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('dragleave', (e) => { e.preventDefault(); if (--dragDepth <= 0) document.body.classList.remove('dragging'); });
window.addEventListener('drop', (e) => {
	e.preventDefault();
	dragDepth = 0;
	document.body.classList.remove('dragging');
	for (const file of e.dataTransfer.files) {
		const ext = file.name.split('.').pop().toLowerCase();
		if (ext === 'vrm' || ext === 'glb') loadVRMFile(file);
		else if (ext === 'fbx') loadAnimFile(file);
		else log(`unsupported file: ${file.name}`);
	}
});

// --- Render loop ------------------------------------------------------------

function animate() {
	requestAnimationFrame(animate);
	const delta = clock.getDelta();
	if (mixer) mixer.update(delta);
	if (webcamOn) mocap.applyIdle(delta);   // set leg targets before the rig solves
	if (currentVrm) currentVrm.update(delta);
	if (webcamOn) mocap.groundContact();    // plant feet after world matrices update
	if (webcamOn && window.__mocapFps != null) fpsEl.textContent = `${window.__mocapFps} fps`;
	controls.update();
	renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Boot: load default character + animation -------------------------------

currentAnim = { url: ANIMATIONS[0].url, label: ANIMATIONS[0].name };
loadVRM(CHARACTERS[0].url, CHARACTERS[0].name);
