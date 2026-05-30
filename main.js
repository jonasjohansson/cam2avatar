import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
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
	// three.js glTF characters (non-VRM). Their rest poses don't match the raw
	// Mixamo bind pose, so they use their own built-in animations. Webcam mocap
	// is VRM-only.
	{ name: 'three.js — Xbot',     url: `${CDN}/mrdoob/three.js@r180/examples/models/gltf/Xbot.glb`, type: 'gltf', builtin: 'run' },
	{ name: 'three.js — Soldier',  url: `${CDN}/mrdoob/three.js@r180/examples/models/gltf/Soldier.glb`, type: 'gltf', builtin: 'Walk', faceY: 180 },
	{ name: 'three.js — Michelle', url: `${CDN}/mrdoob/three.js@r180/examples/models/gltf/Michelle.glb`, type: 'gltf', builtin: 'SambaDance' },
	{ name: 'three.js — Robot',    url: `${CDN}/mrdoob/three.js@r180/examples/models/gltf/RobotExpressive/RobotExpressive.glb`, type: 'gltf', builtin: 'Dance' },
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
let currentGltf = null;       // non-VRM glTF character { scene, rig, builtin, name }
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
const plainGltfLoader = new GLTFLoader(); // for non-VRM glTF characters

// Remove + dispose whatever character is currently in the scene.
function disposeCurrent() {
	if (mixer) { mixer.stopAllAction(); mixer = null; action = null; }
	if (currentVrm) { scene.remove(currentVrm.scene); VRMUtils.deepDispose(currentVrm.scene); currentVrm = null; }
	if (currentGltf) { scene.remove(currentGltf.scene); currentGltf.scene.traverse((o) => { o.geometry?.dispose?.(); }); currentGltf = null; }
}

// Route a character entry to the right loader.
function loadCharacter(entry) {
	return entry.type === 'gltf' ? loadGltf(entry) : loadVRM(entry.url, entry.name);
}

// Horizontally flip the whole character (works in every mode), preserving the
// scale magnitude set during normalization.
let avatarMirrored = false;
function applyAvatarMirror() {
	const root = currentVrm?.scene || currentGltf?.scene;
	if (root) root.scale.x = (avatarMirrored ? -1 : 1) * Math.abs(root.scale.x);
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

		disposeCurrent();

		// Don't waste cycles on frustum culling tiny humanoid meshes
		vrm.scene.traverse((o) => { o.frustumCulled = false; });

		scene.add(vrm.scene);
		currentVrm = vrm;

		refreshSkeleton(); // rebuild the overlay for the new rig
		applyAvatarMirror();

		const ver = vrm.meta?.metaVersion === '0' ? 'VRM 0.x' : 'VRM 1.0';
		log(`character ready: ${label}  (${ver})`);

		// Re-apply the active animation onto the new rig (retarget is rig-specific)
		if (currentAnim) await applyAnimation();
	} catch (err) {
		console.error(err);
		log(`✗ failed to load character\n${err.message ?? err}`);
	}
}

// Shared setup for any non-VRM rigged character (glTF or FBX): normalize and
// drive it with its own built-in animations.
function setupObjectCharacter(root, animations, entry) {
	// Auto-detect rig (for naming / future use), else trust the entry.
	const hasMixamo = !!root.getObjectByName('mixamorigHips');
	const rig = entry.rig ?? (hasMixamo ? 'mixamorig' : 'generic');
	// Normalize: face camera, scale to ~1.6 m, drop feet to floor, center on x/z.
	if (entry.faceY) root.rotation.y = (entry.faceY * Math.PI) / 180;
	root.updateMatrixWorld(true);
	const size = new THREE.Vector3();
	new THREE.Box3().setFromObject(root).getSize(size);
	root.scale.setScalar(size.y > 1e-3 ? 1.6 / size.y : 1);
	root.updateMatrixWorld(true);
	const b = new THREE.Box3().setFromObject(root);
	root.position.x -= (b.min.x + b.max.x) / 2;
	root.position.z -= (b.min.z + b.max.z) / 2;
	root.position.y -= b.min.y;
	root.traverse((o) => { o.frustumCulled = false; });

	disposeCurrent();
	scene.add(root);
	const builtin = {};
	(animations || []).forEach((a) => { builtin[a.name] = a; });
	currentGltf = { scene: root, rig, builtin, builtinDefault: entry.builtin, name: entry.name };

	refreshSkeleton();
	applyAvatarMirror();
	const n = Object.keys(builtin).length;
	log(`character ready: ${entry.name}  (${rig} rig, ${n} animation${n === 1 ? '' : 's'})`);
	if (currentAnim) applyAnimation();
}

async function loadGltf(entry, srcOverride) {
	log(`loading character: ${entry.name}…`);
	try {
		const gltf = await plainGltfLoader.loadAsync(srcOverride ?? entry.url);
		setupObjectCharacter(gltf.scene, gltf.animations, entry);
	} catch (err) {
		console.error(err);
		log(`✗ failed to load character\n${err.message ?? err}`);
	}
}

async function loadFbxCharacter(url, name) {
	log(`loading character: ${name}…`);
	try {
		const asset = await new FBXLoader().loadAsync(url);
		setupObjectCharacter(asset, asset.animations, { name });
	} catch (err) {
		console.error(err);
		log(`✗ failed to load character\n${err.message ?? err}`);
	}
}

// A dropped/picked FBX may be a rigged CHARACTER (has a mesh) or a Mixamo
// ANIMATION (skeleton only). Decide by whether it carries skinned geometry.
function fbxHasMesh(obj) {
	let has = false;
	obj.traverse((o) => { if ((o.isMesh || o.isSkinnedMesh) && o.geometry?.attributes?.position?.count > 0) has = true; });
	return has;
}
async function loadFbxFile(file) {
	const url = URL.createObjectURL(file);
	try {
		const asset = await new FBXLoader().loadAsync(url);
		if (fbxHasMesh(asset)) {
			setupObjectCharacter(asset, asset.animations, { name: file.name });
		} else {
			currentAnim = { file, label: file.name }; // skeleton-only -> animate current VRM
			applyAnimation();
		}
	} catch (err) {
		console.error(err);
		log(`✗ ${err.message ?? err}`);
	} finally {
		URL.revokeObjectURL(url);
	}
}

// --- Loading: Mixamo animation ----------------------------------------------

function startClip(root, clip, who, label) {
	if (mixer) mixer.stopAllAction();
	mixer = new THREE.AnimationMixer(root);
	action = mixer.clipAction(clip);
	action.timeScale = speed;
	action.play();
	if (!playing) action.paused = true;
	log(`▶ ${who} · ${label}`);
}

async function applyAnimation() {
	if (!currentAnim) return;
	const label = currentAnim.label ?? 'animation';
	try {
		if (currentVrm) {
			log(`retargeting ${label}…`);
			const url = currentAnim.url ?? URL.createObjectURL(currentAnim.file);
			const clip = await loadMixamoAnimation(url, currentVrm);
			if (currentAnim.file) URL.revokeObjectURL(url);
			startClip(currentVrm.scene, clip, currentVrm.meta?.name ?? 'character', label);
		} else if (currentGltf) {
			// glTF characters use their own built-in animations (their rest poses
			// don't match the raw Mixamo bind pose). Mixamo dropdown doesn't apply.
			const name = (currentGltf.builtinDefault && currentGltf.builtin[currentGltf.builtinDefault])
				? currentGltf.builtinDefault : Object.keys(currentGltf.builtin)[0];
			const clip = currentGltf.builtin[name];
			if (clip) startClip(currentGltf.scene, clip, currentGltf.name, `built-in: ${name}`);
			else log(`${currentGltf.name}: no built-in animation`);
		}
	} catch (err) {
		console.error(err);
		log(`✗ animation failed\n${err.message ?? err}`);
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
	loadCharacter(CHARACTERS[charSelect.value]);
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
	const f = charFile.files[0];
	if (!f) return;
	const ext = f.name.split('.').pop().toLowerCase();
	if (ext === 'vrm') loadVRMFile(f);
	else if (ext === 'glb' || ext === 'gltf') loadGltfFile(f);
	else if (ext === 'fbx') loadFbxFile(f);
	else log(`unsupported character file: ${f.name}`);
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
function loadGltfFile(file) {
	const url = URL.createObjectURL(file);
	loadGltf({ name: file.name }, url).finally(() => URL.revokeObjectURL(url));
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

// Record the 3D canvas (+ camera PiP during tracking) to a video file.
let recorder = null;
let recChunks = [];
let recTimer = null;
let compCanvas = null;   // composite canvas: 3D scene + camera picture-in-picture
let compCtx = null;
const guideEl = document.getElementById('guide');
const recBtn = document.getElementById('rec-btn');
recBtn.addEventListener('click', () => {
	if (!recorder) {
		compCanvas = document.createElement('canvas');
		compCanvas.width = renderer.domElement.width;
		compCanvas.height = renderer.domElement.height;
		compCtx = compCanvas.getContext('2d');
		const stream = compCanvas.captureStream(30);
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
		compCanvas = null;
		compCtx = null;
		clearInterval(recTimer);
		recBtn.classList.remove('recording');
		recBtn.textContent = '● Record';
	}
});

// Paint the composite frame (called from the render loop after renderer.render).
function drawComposite() {
	if (!compCtx) return;
	const W = compCanvas.width, H = compCanvas.height;
	compCtx.drawImage(renderer.domElement, 0, 0, W, H);
	// Camera picture-in-picture (with landmark overlay) during tracking.
	if (webcamOn && !previewEl.hidden && guideEl.width) {
		const pw = Math.round(W * 0.3);
		const ph = Math.round((pw * guideEl.height) / guideEl.width);
		const m = Math.round(W * 0.02);
		const x = W - pw - m, y = H - ph - m;
		compCtx.drawImage(guideEl, x, y, pw, ph);
		compCtx.strokeStyle = '#ffffff';
		compCtx.lineWidth = Math.max(2, Math.round(W * 0.0025));
		compCtx.strokeRect(x, y, pw, ph);
	}
}

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
		if (!currentVrm) {
			log('Webcam mocap needs a VRM character (glTF characters are animation-only for now)');
			return;
		}
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
			// Track legs + ground contact by default for any source (webcam or clip).
			optLegs.value = 'webcam';
			mocap.setOptions({ legsMode: 'webcam' });
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

const optFollow = document.getElementById('opt-follow');
optFollow.addEventListener('change', () => mocap.setOptions({ follow: optFollow.checked }));

const optQuality = document.getElementById('opt-quality');
optQuality.addEventListener('change', () => mocap.setOptions({ quality: optQuality.value }));

const optMirror = document.getElementById('opt-mirror');
optMirror.addEventListener('change', () => mocap.setOptions({ mirror: optMirror.checked }));

const fpsEl = document.getElementById('fps');

// Skeleton overlay — draws the bones on top of the mesh to verify tracking.
// For VRM we draw ONLY the humanoid bones (SkeletonHelper would also draw the
// spring/cloth/accessory bones, which fan out as noise near hips/knees).
let skeletonHelper = null;      // glTF/FBX (SkeletonHelper)
let humanoidOverlay = null;     // VRM (custom, humanoid bones only)
const optSkel = document.getElementById('opt-skeleton');
const _ov = new THREE.Vector3();

// Parent -> child links across the standard VRM humanoid.
const HUMANOID_LINKS = [
	['hips', 'spine'], ['spine', 'chest'], ['chest', 'neck'], ['neck', 'head'],
	['chest', 'leftUpperArm'], ['leftUpperArm', 'leftLowerArm'], ['leftLowerArm', 'leftHand'],
	['chest', 'rightUpperArm'], ['rightUpperArm', 'rightLowerArm'], ['rightLowerArm', 'rightHand'],
	['hips', 'leftUpperLeg'], ['leftUpperLeg', 'leftLowerLeg'], ['leftLowerLeg', 'leftFoot'], ['leftFoot', 'leftToes'],
	['hips', 'rightUpperLeg'], ['rightUpperLeg', 'rightLowerLeg'], ['rightLowerLeg', 'rightFoot'], ['rightFoot', 'rightToes'],
];

function clearOverlays() {
	if (skeletonHelper) { scene.remove(skeletonHelper); skeletonHelper.geometry?.dispose(); skeletonHelper.material?.dispose(); skeletonHelper = null; }
	if (humanoidOverlay) { scene.remove(humanoidOverlay.lines); humanoidOverlay.lines.geometry.dispose(); humanoidOverlay.lines.material.dispose(); humanoidOverlay = null; }
}

function refreshSkeleton() {
	clearOverlays();
	if (!optSkel.checked) return;
	if (currentVrm) {
		const pairs = [];
		for (const [a, b] of HUMANOID_LINKS) {
			const na = currentVrm.humanoid.getRawBoneNode(a);
			const nb = currentVrm.humanoid.getRawBoneNode(b);
			if (na && nb) pairs.push([na, nb]);
		}
		const positions = new Float32Array(pairs.length * 6);
		const geom = new THREE.BufferGeometry();
		geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		const mat = new THREE.LineBasicMaterial({ color: 0x22d3ee, depthTest: false, depthWrite: false, transparent: true });
		const lines = new THREE.LineSegments(geom, mat);
		lines.renderOrder = 999;
		lines.frustumCulled = false;
		humanoidOverlay = { lines, pairs, positions };
		updateHumanoidOverlay();
		scene.add(lines);
	} else if (currentGltf) {
		skeletonHelper = new THREE.SkeletonHelper(currentGltf.scene);
		skeletonHelper.material.depthTest = false;
		skeletonHelper.material.depthWrite = false;
		skeletonHelper.material.transparent = true;
		skeletonHelper.renderOrder = 999;
		scene.add(skeletonHelper);
	}
}
function updateHumanoidOverlay() {
	if (!humanoidOverlay) return;
	const { pairs, positions, lines } = humanoidOverlay;
	for (let i = 0; i < pairs.length; i++) {
		pairs[i][0].getWorldPosition(_ov); positions[i * 6] = _ov.x; positions[i * 6 + 1] = _ov.y; positions[i * 6 + 2] = _ov.z;
		pairs[i][1].getWorldPosition(_ov); positions[i * 6 + 3] = _ov.x; positions[i * 6 + 4] = _ov.y; positions[i * 6 + 5] = _ov.z;
	}
	lines.geometry.attributes.position.needsUpdate = true;
}
optSkel.addEventListener('change', refreshSkeleton);

const optMirrorAvatar = document.getElementById('opt-mirror-avatar');
optMirrorAvatar.addEventListener('change', () => {
	avatarMirrored = optMirrorAvatar.checked;
	applyAvatarMirror();
});

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
		if (ext === 'vrm') loadVRMFile(file);
		else if (ext === 'glb' || ext === 'gltf') loadGltfFile(file);
		else if (ext === 'fbx') loadFbxFile(file); // character (has mesh) or animation (skeleton only)
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
	if (humanoidOverlay) updateHumanoidOverlay();
	if (webcamOn && window.__mocapFps != null) fpsEl.textContent = `${window.__mocapFps} fps`;
	controls.update();
	renderer.render(scene, camera);
	if (compCtx) drawComposite(); // composite the recording frame after render
}
animate();

window.addEventListener('resize', () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Boot: load default character + animation -------------------------------

currentAnim = { url: ANIMATIONS[0].url, label: ANIMATIONS[0].name };
loadCharacter(CHARACTERS[0]);
