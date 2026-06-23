import * as THREE from 'three';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import loadMujoco from '../../demo-assets/mujoco_wasm.js';

window.__go2wDemoModuleLoaded = true;

const ASSET_NAMES = [
  'base_0.obj',
  'base_1.obj',
  'base_2.obj',
  'base_3.obj',
  'base_4.obj',
  'hip_0.obj',
  'hip_1.obj',
  'thigh_0.obj',
  'thigh_1.obj',
  'thigh_mirror_0.obj',
  'thigh_mirror_1.obj',
  'calf_mirror_0.obj',
  'calf_mirror_1.obj',
  'calf.stl',
  'calf_mirror.stl',
  'foot.obj',
  'wheel.stl',
  '336L_with_base.STL',
  '336L_with_safe_base.STL',
  'multi_motion_float_box_terrain_mesh1.stl',
  'multi_motion_float_box_terrain_mesh2.stl',
  'multi_motion_float_box_terrain_mesh3.stl',
  'multi_motion_float_box_terrain_mesh4.stl',
  'multi_motion_float_box_terrain_mesh5.stl',
  'multi_motion_float_box_terrain_mesh6.stl',
  'multi_motion_float_box_terrain_mesh7.stl',
  'multi_motion_float_box_terrain_mesh8.stl',
];

const POLICY_CONFIGS = [
  {
    id: 0,
    name: 'motion_mlp',
    kind: 'mlp',
    url: 'demo-assets/policies/motion_tracking/policy.onnx',
    obsDim: 187,
    vectorHistory: 3,
    imageHistory: -1,
    actionScale: [
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      2.0, 2.0, 2.0, 2.0,
    ],
  },
  {
    id: 1,
    name: 'vtm',
    kind: 'mlp',
    url: 'demo-assets/policies/vtm/student.onnx',
    obsDim: 5449,
    vectorHistory: 5,
    imageHistory: 8,
    actionScale: [
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      5.0, 5.0, 5.0, 5.0,
    ],
  },
  {
    id: 2,
    name: 'vtm_lstm_sru',
    kind: 'split',
    encoderUrl: 'demo-assets/policies/vtm_lstm_sru/student_encoder.onnx',
    memoryUrl: 'demo-assets/policies/vtm_lstm_sru/student_memory.onnx',
    actorUrl: 'demo-assets/policies/vtm_lstm_sru/student_actor.onnx',
    obsDim: 629,
    encodedDim: 181,
    latentDim: 512,
    numLayers: 1,
    hiddenDim: 512,
    memoryType: 'lstm_sru',
    vectorHistory: 1,
    imageHistory: 0,
    actionScale: [
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      5.0, 5.0, 5.0, 5.0,
    ],
  },
  {
    id: 3,
    name: 'vtm_gru_sru',
    kind: 'split',
    encoderUrl: 'demo-assets/policies/vtm_gru_sru/student_encoder.onnx',
    memoryUrl: 'demo-assets/policies/vtm_gru_sru/student_memory.onnx',
    actorUrl: 'demo-assets/policies/vtm_gru_sru/student_actor.onnx',
    obsDim: 629,
    encodedDim: 181,
    latentDim: 512,
    numLayers: 1,
    hiddenDim: 512,
    memoryType: 'gru_sru',
    vectorHistory: 1,
    imageHistory: 0,
    actionScale: [
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      0.125, 0.25, 0.25,
      5.0, 5.0, 5.0, 5.0,
    ],
  },
];

const DEFAULT_POLICY_ID = 0;
const SIM_DT = 0.005;
const POLICY_DECIMATION = 4;
const MOTION_OBS_PREFIX_DIM = 24 + 1 + 3 + 6;
const MOTION_OBS_DIM = 187;
const RAY_WIDTH = 32;
const RAY_HEIGHT = 18;
const RAY_SIZE = RAY_WIDTH * RAY_HEIGHT;
const RAY_MIN_DIST = 0.15;
const RAY_MAX_DIST = 1.5;
const RAY_FOCAL = 1.0;
const RAY_HORIZONTAL_APERTURE = 2.0;
const RAY_VERTICAL_APERTURE = 1.154700538;
const RAY_UPDATE_DT = 0.02;
const RAY_VIS_STRIDE_X = 4;
const RAY_VIS_STRIDE_Y = 3;
const CTRL_LIMIT = 23.7;
const VISUAL_UPDATE_INTERVAL = 20;
const MAX_PHYSICS_STEPS_PER_FRAME = 10;

const LEG_POS_SENSOR_NAMES = [
  'FL_hip_joint_pos',
  'FR_hip_joint_pos',
  'RL_hip_joint_pos',
  'RR_hip_joint_pos',
  'FL_thigh_joint_pos',
  'FR_thigh_joint_pos',
  'RL_thigh_joint_pos',
  'RR_thigh_joint_pos',
  'FL_calf_joint_pos',
  'FR_calf_joint_pos',
  'RL_calf_joint_pos',
  'RR_calf_joint_pos',
];
const DOF_VEL_SENSOR_NAMES = [
  'FL_hip_joint_vel',
  'FR_hip_joint_vel',
  'RL_hip_joint_vel',
  'RR_hip_joint_vel',
  'FL_thigh_joint_vel',
  'FR_thigh_joint_vel',
  'RL_thigh_joint_vel',
  'RR_thigh_joint_vel',
  'FL_calf_joint_vel',
  'FR_calf_joint_vel',
  'RL_calf_joint_vel',
  'RR_calf_joint_vel',
  'FL_wheel_joint_vel',
  'FR_wheel_joint_vel',
  'RL_wheel_joint_vel',
  'RR_wheel_joint_vel',
];
const ACTUATOR_JOINT_ORDER = [
  'FR_hip_joint',
  'FR_thigh_joint',
  'FR_calf_joint',
  'FL_hip_joint',
  'FL_thigh_joint',
  'FL_calf_joint',
  'RR_hip_joint',
  'RR_thigh_joint',
  'RR_calf_joint',
  'RL_hip_joint',
  'RL_thigh_joint',
  'RL_calf_joint',
  'FR_wheel_joint',
  'FL_wheel_joint',
  'RR_wheel_joint',
  'RL_wheel_joint',
];
const JOINT_DEFAULTS = {
  FR_hip_joint: 0.0,
  FR_thigh_joint: 0.8,
  FR_calf_joint: -1.5,
  FL_hip_joint: 0.0,
  FL_thigh_joint: 0.8,
  FL_calf_joint: -1.5,
  RR_hip_joint: 0.0,
  RR_thigh_joint: 0.8,
  RR_calf_joint: -1.5,
  RL_hip_joint: 0.0,
  RL_thigh_joint: 0.8,
  RL_calf_joint: -1.5,
};
const WHEEL_JOINTS = [
  'FR_wheel_joint',
  'FL_wheel_joint',
  'RR_wheel_joint',
  'RL_wheel_joint',
];
const OBS_DEFAULT_DOF_POS = new Float32Array([
  0, 0, 0, 0,
  0.8, 0.8, 0.8, 0.8,
  -1.5, -1.5, -1.5, -1.5,
]);
const ACT_DEFAULT_DOF_POS = new Float32Array([
  0, 0.8, -1.5,
  0, 0.8, -1.5,
  0, 0.8, -1.5,
  0, 0.8, -1.5,
  0, 0, 0, 0,
]);

const MJCAT_ALL = 7;
const MJOBJ_GEOM = 5;
const GEOM_PLANE = 0;
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;
const GEOM_MESH = 7;
const GEOM_LINE = 100;
const TMP_MAT4 = new THREE.Matrix4();

const $ = (id) => document.getElementById(id);
const POLICY_BY_ID = new Map(POLICY_CONFIGS.map((policy) => [policy.id, policy]));

class Go2WDemo {
  constructor() {
    this.canvas = $('go2w-canvas');
    this.rayCanvas = $('ray-canvas');
    this.rayReadout = $('ray-readout');
    this.statusPill = $('status-pill');
    this.policyReadout = $('policy-readout');
    this.errorReadout = $('error-readout');
    this.policyId = DEFAULT_POLICY_ID;
    this.keys = new Set();
    this.keyboardCommandActive = false;
    this.cmd = { x: 0.7, y: 0, yaw: 0 };
    this.realtime = 1.0;
    this.followCamera = true;
    this.clock = new THREE.Clock();
    this.physicsAccumulator = 0;
    this.physicsStep = 0;
    this.policyPending = false;
    this.policyFailed = false;
    this.policyWorkerReady = false;
    this.policyReady = new Set();
    this.policyLoading = new Map();
    this.policySeq = 0;
    this.policyRuns = 0;
    this.policyRequestStartedAt = 0;
    this.lastPolicyDurationMs = 0;
    this.lastRawAction = new Float32Array(16);
    this.currentCtrl = new Float32Array(ACT_DEFAULT_DOF_POS);
    this.jointAdr = new Map();
    this.jointDofAdr = new Map();
    this.sensorAdr = new Map();
    this.geomPool = [];
    this.activeGeoms = 0;
    this.meshGeometries = new Map();
    this.baseBodyId = -1;
    this.cameraId = -1;
    this.cameraBodyId = -1;
    this.frameCount = 0;
    this.lastFrameWallTime = 0;
    this.frameError = null;
    this.frameStage = 'init';
    this.rayImage = new Float32Array(RAY_SIZE);
    this.rayRawImage = new Float32Array(RAY_SIZE);
    this.rayHitPoints = new Array(RAY_SIZE);
    this.rayDirty = false;
    this.rayConvention = null;
    this.rayLastUpdateTime = -Infinity;
    this.rayLocalDirsByConvention = new Map();
    this.rayTerrainMeshes = [];
    this.threeRaycaster = new THREE.Raycaster();
    this.rayCanvasImage = this.rayCanvas?.getContext('2d')?.createImageData(RAY_WIDTH, RAY_HEIGHT) || null;
    this.rayCtx = this.rayCanvas?.getContext('2d') || null;
  }

  async init() {
    this.setStatus('Loading', 'Loading MuJoCo');

    this.mujoco = await loadMujoco();
    window.__go2wApp = this;
    this.ensureDir('/working');
    this.ensureDir('/working/assets');
    await this.loadFiles();

    this.setStatus('Loading', 'Compiling MJCF');
    this.model = this.loadModelXml('/working/scene_parkour.xml');
    this.data = new this.mujoco.MjData(this.model);
    if (window.__go2wFallbackStarted) return;

    this.setupRenderer();
    this.cacheJointAddresses();
    this.cacheSensorAddresses();
    this.setupVisualScene();
    this.setupRayVisualization();
    this.bindControls();
    this.updatePolicyButtons();
    this.setStatus('Loading', `Loading ${this.activePolicy().name}`);
    await this.startPolicyWorker();
    await this.ensurePolicyLoaded(this.policyId);
    this.resetBrowserPose({ resetWorker: true });

    window.__go2wDemoReady = true;
    this.setStatus('Ready', 'MuJoCo + ONNX policy ready', 'ready');
    this.renderer.setAnimationLoop(() => this.safeFrame());
    this.startFrameWatchdog();
  }

  activePolicy() {
    return POLICY_BY_ID.get(this.policyId) || POLICY_CONFIGS[0];
  }

  setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
    });
    window.__go2wWebglStarted = true;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = false;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x151b17);
    this.scene.fog = new THREE.Fog(0x151b17, 18, 46);

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 200);
    this.camera.position.set(-1.8, 1.8, 2.4);
    this.scene.add(this.camera);

    const hemi = new THREE.HemisphereLight(0xdfeee0, 0x2a2f2a, 1.0);
    this.scene.add(hemi);

    const sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(-4, 7, 5);
    this.scene.add(sun);

    this.mujocoRoot = new THREE.Group();
    this.mujocoRoot.name = 'MuJoCo Root';
    this.mujocoRoot.rotation.x = -Math.PI / 2;
    this.scene.add(this.mujocoRoot);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 50),
      new THREE.MeshLambertMaterial({ color: 0xd4ddd0 }),
    );
    ground.name = 'Ground Plane';
    ground.userData.isRayTerrain = true;
    this.groundMesh = ground;
    this.mujocoRoot.add(ground);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.set(0.6, 0.35, 0);
    this.controls.enableDamping = true;
    this.controls.enablePan = true;
    this.controls.dampingFactor = 0.08;
    this.controls.update();

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  bindControls() {
    const sliders = [
      ['cmd-x', 'cmd-x-out', 'x'],
      ['cmd-y', 'cmd-y-out', 'y'],
      ['cmd-yaw', 'cmd-yaw-out', 'yaw'],
    ];

    for (const [inputId, outputId, key] of sliders) {
      const input = $(inputId);
      const output = $(outputId);
      input.addEventListener('input', () => {
        this.cmd[key] = Number(input.value);
        output.value = this.cmd[key].toFixed(2);
      });
      output.value = Number(input.value).toFixed(2);
    }

    $('realtime').addEventListener('input', () => {
      this.realtime = Number($('realtime').value);
      $('realtime-out').value = this.realtime.toFixed(2);
    });
    this.realtime = Number($('realtime').value);
    $('realtime-out').value = this.realtime.toFixed(2);

    $('zero-command').addEventListener('click', () => {
      this.setCommand(0, 0, 0);
    });
    $('reset-sim').addEventListener('click', () => {
      this.resetBrowserPose({ resetWorker: true });
    });
    $('reset-policy').addEventListener('click', () => {
      this.resetActivePolicyState();
    });
    $('follow-camera').addEventListener('click', () => {
      this.setFollowCamera(true);
    });

    for (const button of document.querySelectorAll('[data-policy]')) {
      button.addEventListener('click', () => {
        this.setPolicy(Number(button.dataset.policy));
      });
    }

    document.addEventListener('keydown', (event) => {
      this.keys.add(event.code);
      const handled = this.handleKeyCommand(event.code);
      if (handled) event.preventDefault();
    });

    document.addEventListener('keyup', (event) => {
      this.keys.delete(event.code);
    });
  }

  handleKeyCommand(code) {
    if (code === 'Space') {
      this.setCommand(0, 0, 0);
      return true;
    }
    if (code === 'KeyR') {
      this.resetBrowserPose({ resetWorker: true });
      return true;
    }
    if (code === 'KeyN') {
      this.resetActivePolicyState();
      return true;
    }
    if (code === 'KeyF') {
      this.setFollowCamera(true);
      return true;
    }
    const digit = code.match(/^Digit([1-4])$/);
    if (digit) {
      this.setPolicy(Number(digit[1]) - 1);
      return true;
    }
    return ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE'].includes(code);
  }

  updateKeyboardCommand() {
    const xSign = (this.keys.has('KeyW') ? 1 : 0) - (this.keys.has('KeyS') ? 1 : 0);
    const ySign = (this.keys.has('KeyA') ? 1 : 0) - (this.keys.has('KeyD') ? 1 : 0);
    const yawSign = (this.keys.has('KeyQ') ? 1 : 0) - (this.keys.has('KeyE') ? 1 : 0);
    if (xSign === 0 && ySign === 0 && yawSign === 0) {
      if (this.keyboardCommandActive) {
        this.keyboardCommandActive = false;
        this.setCommand(0, 0, 0);
      }
      return;
    }
    this.keyboardCommandActive = true;
    this.setCommand(xSign * 0.8, ySign * 0.55, yawSign * 1.0);
  }

  setCommand(x, y, yaw) {
    this.cmd.x = clamp(x, -1.0, 1.2);
    this.cmd.y = clamp(y, -0.8, 0.8);
    this.cmd.yaw = clamp(yaw, -1.5, 1.5);
    $('cmd-x').value = this.cmd.x;
    $('cmd-y').value = this.cmd.y;
    $('cmd-yaw').value = this.cmd.yaw;
    $('cmd-x-out').value = this.cmd.x.toFixed(2);
    $('cmd-y-out').value = this.cmd.y.toFixed(2);
    $('cmd-yaw-out').value = this.cmd.yaw.toFixed(2);
  }

  setFollowCamera(enabled) {
    this.followCamera = enabled;
    $('follow-camera')?.classList.toggle('active', enabled);
  }

  async setPolicy(policyId) {
    if (!POLICY_BY_ID.has(policyId)) return;
    if (policyId === this.policyId && this.policyReady.has(policyId)) return;
    this.policyId = policyId;
    this.policySeq += 1;
    this.policyPending = false;
    this.policyRequestStartedAt = 0;
    this.policyFailed = false;
    this.lastRawAction.fill(0);
    this.currentCtrl.set(ACT_DEFAULT_DOF_POS);
    this.policyRuns = 0;
    this.lastPolicyDurationMs = 0;
    this.applyAction(this.lastRawAction);
    this.resetObservationHistory();
    this.updatePolicyButtons();
    this.setStatus('Loading', `Loading ${this.activePolicy().name}`);
    try {
      await this.ensurePolicyLoaded(policyId);
      await this.resetActivePolicyState();
      this.setStatus('Ready', `${this.activePolicy().name} ready`, 'ready');
    } catch (error) {
      this.policyFailed = true;
      this.setStatus('Policy Error', error.message, 'error');
    }
  }

  updatePolicyButtons() {
    for (const button of document.querySelectorAll('[data-policy]')) {
      const id = Number(button.dataset.policy);
      button.classList.toggle('active', id === this.policyId);
      button.disabled = this.policyLoading.has(id);
      if (this.policyLoading.get(id)) {
        button.textContent = `${POLICY_BY_ID.get(id).name}...`;
      } else {
        button.textContent = POLICY_BY_ID.get(id).name;
      }
    }
    const policy = this.activePolicy();
    const status = this.policyReady.has(policy.id) ? 'ready' : 'loading';
    this.policyReadout.textContent = `policy ${policy.id}: ${policy.name} · ${status}`;
  }

  ensureDir(path) {
    if (!this.mujoco.FS.analyzePath(path).exists) {
      this.mujoco.FS.mkdir(path);
    }
  }

  loadModelXml(path) {
    const loader = this.mujoco?.MjModel?.mj_loadXML || this.mujoco?.MjModel?.loadFromXML;
    if (typeof loader !== 'function') {
      throw new Error('MuJoCo WASM does not expose an MJCF XML loader');
    }
    return loader.call(this.mujoco.MjModel, path);
  }

  async loadFiles() {
    const textFiles = ['scene_parkour.xml', 'go2w.xml'].map(async (name) => {
      const text = await fetchWithRetry(`demo-assets/scenes/${name}`, 'text');
      this.mujoco.FS.writeFile(`/working/${name}`, text);
    });
    await Promise.all(textFiles);

    const assetFiles = ASSET_NAMES.map((name) => async () => {
      const buffer = await fetchWithRetry(`demo-assets/scenes/assets/${name}`, 'arrayBuffer');
      this.mujoco.FS.writeFile(`/working/assets/${name}`, new Uint8Array(buffer));
    });
    await runLimited(assetFiles, 4);
  }

  cacheJointAddresses() {
    for (const name of [...Object.keys(JOINT_DEFAULTS), ...WHEEL_JOINTS]) {
      const id = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_JOINT.value, name);
      if (id >= 0) {
        this.jointAdr.set(name, this.model.jnt_qposadr[id]);
        this.jointDofAdr.set(name, this.model.jnt_dofadr[id]);
      }
    }
  }

  cacheSensorAddresses() {
    const sensorNames = [
      'imu_gyro',
      'imu_quat',
      ...LEG_POS_SENSOR_NAMES,
      ...DOF_VEL_SENSOR_NAMES,
    ];
    for (const name of sensorNames) {
      const id = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_SENSOR.value, name);
      if (id < 0) {
        throw new Error(`Missing MuJoCo sensor: ${name}`);
      }
      this.sensorAdr.set(name, {
        adr: this.model.sensor_adr[id],
        dim: this.model.sensor_dim[id],
      });
    }
  }

  async startPolicyWorker() {
    const workerConfigs = POLICY_CONFIGS.map((config) => {
      const out = { ...config };
      for (const key of ['url', 'encoderUrl', 'memoryUrl', 'actorUrl']) {
        if (out[key]) out[key] = new URL(out[key], window.location.href).href;
      }
      delete out.actionScale;
      return out;
    });
    this.policyWorker = new Worker(new URL('policy-worker.js?v=visual-sru-1', import.meta.url), {
      name: 'go2w-policy-worker',
    });
    this.policyWorker.onmessage = (event) => this.handlePolicyWorkerMessage(event.data || {});
    this.policyWorker.onerror = (error) => {
      this.policyFailed = true;
      this.policyPending = false;
      this.setStatus('Policy Error', error.message || 'policy worker failed', 'error');
    };

    await new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        reject(new Error('Timed out while starting policy worker'));
      }, 30000);
      const onMessage = (event) => {
        const data = event.data || {};
        if (data.type === 'ready') {
          window.clearTimeout(timeout);
          this.policyWorker.removeEventListener('message', onMessage);
          this.policyWorkerReady = true;
          resolve();
        } else if (data.type === 'error') {
          window.clearTimeout(timeout);
          this.policyWorker.removeEventListener('message', onMessage);
          reject(new Error(data.message || 'policy worker failed to initialize'));
        }
      };
      this.policyWorker.addEventListener('message', onMessage);
      this.policyWorker.postMessage({ type: 'init', policies: workerConfigs });
    });
  }

  async ensurePolicyLoaded(policyId) {
    if (this.policyReady.has(policyId)) return;
    if (this.policyLoading.has(policyId)) return this.policyLoading.get(policyId);
    const promise = new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        reject(new Error(`Timed out while loading ${POLICY_BY_ID.get(policyId).name}`));
      }, 60000);
      const onMessage = (event) => {
        const data = event.data || {};
        if (data.policyId !== policyId) return;
        if (data.type === 'loaded') {
          window.clearTimeout(timeout);
          this.policyWorker.removeEventListener('message', onMessage);
          this.policyReady.add(policyId);
          resolve();
        } else if (data.type === 'error') {
          window.clearTimeout(timeout);
          this.policyWorker.removeEventListener('message', onMessage);
          reject(new Error(data.message || `Failed to load ${POLICY_BY_ID.get(policyId).name}`));
        }
      };
      this.policyWorker.addEventListener('message', onMessage);
      this.policyWorker.postMessage({ type: 'load', policyId });
    }).finally(() => {
      this.policyLoading.delete(policyId);
      this.updatePolicyButtons();
    });
    this.policyLoading.set(policyId, promise);
    this.updatePolicyButtons();
    return promise;
  }

  async resetActivePolicyState() {
    this.policySeq += 1;
    this.policyPending = false;
    this.policyRequestStartedAt = 0;
    this.lastRawAction.fill(0);
    this.currentCtrl.set(ACT_DEFAULT_DOF_POS);
    this.policyRuns = 0;
    this.lastPolicyDurationMs = 0;
    this.applyAction(this.lastRawAction);
    this.resetObservationHistory();
    if (this.policyWorkerReady) {
      this.policyWorker.postMessage({ type: 'reset', policyId: this.policyId });
    }
    this.policyReadout.textContent = `policy ${this.policyId}: ${this.activePolicy().name} · reset`;
  }

  handlePolicyWorkerMessage(data) {
    if (data.type === 'ready' || data.type === 'loaded' || data.type === 'reset') return;
    if (data.type === 'error') {
      console.error(data.message);
      if (data.seq !== undefined && data.seq !== this.policySeq) return;
      this.policyFailed = true;
      this.policyPending = false;
      this.policyRequestStartedAt = 0;
      this.setStatus('Policy Error', data.message, 'error');
      return;
    }
    if (data.type !== 'result' || data.seq !== this.policySeq || data.policyId !== this.policyId) return;

    const raw = data.action;
    for (let i = 0; i < 16; i += 1) {
      this.lastRawAction[i] = Number.isFinite(raw[i]) ? raw[i] : 0;
    }
    this.lastPolicyDurationMs = data.durationMs || 0;
    this.policyRuns += 1;
    this.applyAction(this.lastRawAction);
    this.policyPending = false;
    this.policyRequestStartedAt = 0;
    const policy = this.activePolicy();
    this.policyReadout.textContent =
      `policy ${policy.id}: ${policy.name} · worker ${this.lastPolicyDurationMs.toFixed(0)}ms`;
  }

  setupVisualScene() {
    this.mjvScene = new this.mujoco.MjvScene(this.model, 10000);
    this.mjvCamera = new this.mujoco.MjvCamera();
    this.mjvOption = new this.mujoco.MjvOption();
    this.mjvPerturb = new this.mujoco.MjvPerturb();
    this.mujoco.mjv_defaultCamera(this.mjvCamera);
    this.mujoco.mjv_defaultOption(this.mjvOption);
    this.mujoco.mjv_defaultPerturb(this.mjvPerturb);
    if (this.mjvOption.geomgroup && this.mjvOption.geomgroup.length > 3) {
      this.mjvOption.geomgroup[3] = 0;
    }
    this.baseBodyId = this.mujoco.mj_name2id(
      this.model,
      this.mujoco.mjtObj.mjOBJ_BODY.value,
      'base_link',
    );
    this.cameraId = this.mujoco.mj_name2id(
      this.model,
      this.mujoco.mjtObj.mjOBJ_CAMERA.value,
      'RayCasterCamera',
    );
    this.cameraBodyId = this.cameraId >= 0 && this.model.cam_bodyid
      ? this.model.cam_bodyid[this.cameraId]
      : this.baseBodyId;
  }

  setupRayVisualization() {
    this.raySampleIndices = [];
    for (let row = 0; row < RAY_HEIGHT; row += RAY_VIS_STRIDE_Y) {
      for (let col = 0; col < RAY_WIDTH; col += RAY_VIS_STRIDE_X) {
        this.raySampleIndices.push(row * RAY_WIDTH + col);
      }
    }
    this.rayLinePositions = new Float32Array(this.raySampleIndices.length * 2 * 3);
    this.rayLineGeometry = new THREE.BufferGeometry();
    this.rayLineGeometry.setAttribute('position', new THREE.BufferAttribute(this.rayLinePositions, 3));
    this.rayLines = new THREE.LineSegments(
      this.rayLineGeometry,
      new THREE.LineBasicMaterial({
        color: 0x88ff74,
        transparent: true,
        opacity: 0.62,
      }),
    );
    this.rayLines.frustumCulled = false;
    this.mujocoRoot.add(this.rayLines);
  }

  updateVisualScene() {
    this.mujoco.mjv_updateScene(
      this.model,
      this.data,
      this.mjvOption,
      this.mjvPerturb,
      this.mjvCamera,
      MJCAT_ALL,
      this.mjvScene,
    );

    for (let i = 0; i < this.activeGeoms; i += 1) {
      this.geomPool[i].visible = false;
    }

    let meshIndex = 0;
    this.rayTerrainMeshes = this.groundMesh ? [this.groundMesh] : [];
    for (let i = 0; i < this.mjvScene.ngeom; i += 1) {
      const geom = this.mjvScene.geoms.get(i);
      try {
        if (!geom || geom.type >= GEOM_LINE) continue;
        if (geom.type === GEOM_PLANE) continue;

        const mesh = this.getOrCreateVisualMesh(meshIndex);
        meshIndex += 1;

        const geometry = this.geometryForVisualGeom(geom);
        if (mesh.geometry !== geometry) mesh.geometry = geometry;
        this.scaleVisualMesh(mesh, geom);

        mesh.position.set(geom.pos[0], geom.pos[1], geom.pos[2]);
        TMP_MAT4.set(
          geom.mat[0], geom.mat[1], geom.mat[2], 0,
          geom.mat[3], geom.mat[4], geom.mat[5], 0,
          geom.mat[6], geom.mat[7], geom.mat[8], 0,
          0, 0, 0, 1,
        );
        mesh.quaternion.setFromRotationMatrix(TMP_MAT4);

        const opacity = geom.rgba[3];
        const transparent = opacity < 1;
        mesh.material.color.setRGB(geom.rgba[0], geom.rgba[1], geom.rgba[2]);
        if (mesh.material.opacity !== opacity || mesh.material.transparent !== transparent) {
          mesh.material.opacity = opacity;
          mesh.material.transparent = transparent;
          mesh.material.needsUpdate = true;
        }
        mesh.userData.isRayTerrain = this.isTerrainRayGeom(geom.objid);
        if (mesh.userData.isRayTerrain) this.rayTerrainMeshes.push(mesh);
        // Record the model geom id for the cheap per-frame pose sync. Only real
        // model geoms (objtype MJOBJ_GEOM) have a stable transform in
        // data.geom_xpos/geom_xmat; decor geoms (objtype != MJOBJ_GEOM, e.g.
        // contact arrows) keep the pose baked at rebuild time.
        mesh.userData.modelGeomId =
          geom.objtype === MJOBJ_GEOM && Number.isInteger(geom.objid) && geom.objid >= 0
            ? geom.objid
            : -1;
        mesh.visible = true;
      } finally {
        if (geom && typeof geom.delete === 'function') geom.delete();
      }
    }

    this.activeGeoms = meshIndex;
  }

  // Cheap per-frame pose sync. The full updateVisualScene() (mjv_updateScene +
  // a per-geom embind walk) costs ~13-16ms and must stay throttled, but the
  // robot's body geoms move every physics step. Reading data.geom_xpos /
  // geom_xmat directly (both refreshed by every mj_step) lets us reposition the
  // already-built meshes every rendered frame for well under 1ms, eliminating
  // the ~20-frame "teleport" jumps without the cost of a full rebuild.
  syncVisualPoses() {
    const data = this.data;
    if (!data) return;
    const xpos = data.geom_xpos;
    const xmat = data.geom_xmat;
    if (!xpos || !xmat) return;
    for (let i = 0; i < this.activeGeoms; i += 1) {
      const mesh = this.geomPool[i];
      const gid = mesh.userData.modelGeomId;
      if (gid === undefined || gid < 0) continue;
      const p = gid * 3;
      const m = gid * 9;
      mesh.position.set(xpos[p], xpos[p + 1], xpos[p + 2]);
      TMP_MAT4.set(
        xmat[m], xmat[m + 1], xmat[m + 2], 0,
        xmat[m + 3], xmat[m + 4], xmat[m + 5], 0,
        xmat[m + 6], xmat[m + 7], xmat[m + 8], 0,
        0, 0, 0, 1,
      );
      mesh.quaternion.setFromRotationMatrix(TMP_MAT4);
    }
  }

  isTerrainRayGeom(geomId) {
    if (!Number.isInteger(geomId) || geomId < 0 || !this.model.geom_bodyid) return false;
    const group = this.model.geom_group?.[geomId] ?? 0;
    return this.model.geom_bodyid[geomId] === 0 && group <= 1;
  }

  getOrCreateVisualMesh(index) {
    if (index < this.geomPool.length) return this.geomPool[index];

    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshLambertMaterial({
        color: 0xd9e1d3,
        side: THREE.DoubleSide,
      }),
    );
    mesh.visible = false;
    this.mujocoRoot.add(mesh);
    this.geomPool.push(mesh);
    return mesh;
  }

  geometryForVisualGeom(geom) {
    if (!this.sharedGeometries) {
      this.sharedGeometries = {
        plane: new THREE.PlaneGeometry(1, 1),
        sphere: new THREE.SphereGeometry(0.5, 24, 16),
        box: new THREE.BoxGeometry(1, 1, 1),
        cylinder: new THREE.CylinderGeometry(0.5, 0.5, 1, 24),
      };
    }

    if (geom.type === GEOM_PLANE) return this.sharedGeometries.plane;
    if (geom.type === GEOM_SPHERE || geom.type === GEOM_ELLIPSOID) return this.sharedGeometries.sphere;
    if (geom.type === GEOM_BOX) return this.sharedGeometries.box;
    if (geom.type === GEOM_CYLINDER) return this.sharedGeometries.cylinder;
    if (geom.type === GEOM_CAPSULE) {
      const radius = geom.size[0] || 0.02;
      const height = (geom.size[1] || 0.1) * 2;
      const key = `capsule:${radius.toFixed(4)}:${height.toFixed(4)}`;
      if (!this.meshGeometries.has(key)) {
        this.meshGeometries.set(key, new THREE.CapsuleGeometry(radius, height, 8, 16));
      }
      return this.meshGeometries.get(key);
    }
    if (geom.type === GEOM_MESH) {
      const meshId = this.meshIdForVisualGeom(geom);
      if (meshId >= 0) return this.meshGeometry(meshId);
    }
    return this.sharedGeometries.box;
  }

  meshIdForVisualGeom(geom) {
    if (Number.isInteger(geom.objid) && geom.objid >= 0 && this.model.geom_dataid) {
      const meshId = this.model.geom_dataid[geom.objid];
      if (Number.isInteger(meshId) && meshId >= 0) return meshId;
    }
    return Number.isInteger(geom.dataid) ? geom.dataid : -1;
  }

  meshGeometry(meshId) {
    const key = `mesh:${meshId}`;
    if (this.meshGeometries.has(key)) return this.meshGeometries.get(key);

    const vertStart = this.model.mesh_vertadr[meshId];
    const vertNum = this.model.mesh_vertnum[meshId];
    const faceStart = this.model.mesh_faceadr[meshId];
    const faceNum = this.model.mesh_facenum[meshId];

    const positions = new Float32Array(vertNum * 3);
    for (let i = 0; i < vertNum; i += 1) {
      const src = (vertStart + i) * 3;
      positions[i * 3] = this.model.mesh_vert[src];
      positions[i * 3 + 1] = this.model.mesh_vert[src + 1];
      positions[i * 3 + 2] = this.model.mesh_vert[src + 2];
    }

    const indices = new Uint32Array(faceNum * 3);
    for (let i = 0; i < faceNum; i += 1) {
      const src = (faceStart + i) * 3;
      indices[i * 3] = this.model.mesh_face[src];
      indices[i * 3 + 1] = this.model.mesh_face[src + 1];
      indices[i * 3 + 2] = this.model.mesh_face[src + 2];
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    this.meshGeometries.set(key, geometry);
    return geometry;
  }

  scaleVisualMesh(mesh, geom) {
    const size = geom.size;
    if (geom.type === GEOM_PLANE) {
      mesh.scale.set(Math.max(size[0] * 2, 80), Math.max(size[1] * 2, 80), 1);
    } else if (geom.type === GEOM_SPHERE) {
      mesh.scale.set(size[0] * 2, size[0] * 2, size[0] * 2);
    } else if (geom.type === GEOM_ELLIPSOID || geom.type === GEOM_BOX) {
      mesh.scale.set(size[0] * 2, size[1] * 2, size[2] * 2);
    } else if (geom.type === GEOM_CYLINDER) {
      mesh.scale.set(size[0] * 2, size[1] * 2, size[0] * 2);
    } else {
      mesh.scale.set(1, 1, 1);
    }
  }

  resetBrowserPose({ resetWorker = false } = {}) {
    this.physicsAccumulator = 0;
    this.physicsStep = 0;
    this.policyPending = false;
    this.policyRequestStartedAt = 0;
    this.policyFailed = false;
    this.policySeq += 1;
    this.policyRuns = 0;
    this.lastPolicyDurationMs = 0;
    this.lastRawAction.fill(0);
    this.currentCtrl.set(ACT_DEFAULT_DOF_POS);
    this.mujoco.mj_resetData(this.model, this.data);
    this.setInitialPose();
    this.applyAction(this.lastRawAction);
    this.mujoco.mj_forward(this.model, this.data);
    this.rayLastUpdateTime = -Infinity;
    this.resetObservationHistory();
    this.refreshRaycasterImage(true);
    this.updateRayVisualization();
    if (resetWorker && this.policyWorkerReady) {
      this.policyWorker.postMessage({ type: 'reset', policyId: null });
    }
    this.setStatus('Ready', 'Simulation reset', 'ready');
  }

  safeFrame() {
    try {
      this.frameStage = 'frame';
      this.frame();
      this.frameError = null;
      if (this.statusPill.textContent === 'Runtime Error') {
        this.setStatus('Ready', 'Recovered after a transient browser runtime error', 'ready');
      }
    } catch (error) {
      this.frameError = error;
      console.error(error);
      this.policyPending = false;
      this.policyRequestStartedAt = 0;
      this.setStatus('Runtime Error', error?.message || String(error), 'error');
    }
  }

  startFrameWatchdog() {
    // Guard against multiple installs (e.g. re-init) stacking intervals.
    if (this.frameWatchdogId) {
      window.clearInterval(this.frameWatchdogId);
      this.frameWatchdogId = 0;
    }
    this.watchdogFramePending = false;
    this.frameWatchdogId = window.setInterval(() => {
      if (!window.__go2wDemoReady || !this.renderer || document.hidden) return;
      // Only inject a recovery frame when the loop appears truly stalled AND we
      // have not already queued one. setAnimationLoop is still the primary
      // driver; injecting extra frames while one is already in flight (or while
      // frames are merely slow) compounds the work and feeds the spiral of
      // death. The pending flag clears itself once the recovery frame runs.
      if (this.watchdogFramePending) return;
      if (performance.now() - this.lastFrameWallTime > 2000) {
        this.watchdogFramePending = true;
        window.requestAnimationFrame(() => {
          this.watchdogFramePending = false;
          this.safeFrame();
        });
      }
    }, 1000);
  }

  stopFrameWatchdog() {
    if (this.frameWatchdogId) {
      window.clearInterval(this.frameWatchdogId);
      this.frameWatchdogId = 0;
    }
    this.watchdogFramePending = false;
  }

  frame() {
    this.lastFrameWallTime = performance.now();
    const dt = Math.min(this.clock.getDelta(), 0.04);
    this.frameCount += 1;
    this.frameStage = 'keyboard';
    this.updateKeyboardCommand();
    this.frameStage = 'physics';
    this.stepSimulation(dt * this.realtime);
    const unsafe = this.needsVisualGuard();
    if (unsafe) {
      if (this.needsSafetyReset()) this.stopUnsafeSimulation();
    } else {
      // Full scene rebuild (mjv_updateScene + per-geom rebuild) is expensive
      // (~13-16ms) so it stays throttled, but the cheap pose sync runs every
      // frame so the robot moves smoothly instead of snapping every 20 frames.
      if (this.frameCount % VISUAL_UPDATE_INTERVAL === 1) {
        this.frameStage = 'visual';
        this.updateVisualScene();
      }
      this.frameStage = 'visual-pose';
      this.syncVisualPoses();
    }
    this.frameStage = 'follow';
    this.followBase(dt);
    this.controls.update();
    this.frameStage = 'ray-vis';
    this.updateRayVisualization();
    this.frameStage = 'render';
    this.renderer.render(this.scene, this.camera);
    this.frameStage = 'stats';
    this.updateRuntimeStats();
    this.frameStage = 'idle';
  }

  setInitialPose() {
    const qpos = this.data.qpos;
    qpos[0] = 0;
    qpos[1] = 0;
    qpos[2] = 0.56;
    qpos[3] = 1;
    qpos[4] = 0;
    qpos[5] = 0;
    qpos[6] = 0;
    for (let i = 0; i < ACTUATOR_JOINT_ORDER.length; i += 1) {
      const adr = this.jointAdr.get(ACTUATOR_JOINT_ORDER[i]);
      if (adr !== undefined) qpos[adr] = ACT_DEFAULT_DOF_POS[i];
    }
    for (let i = 0; i < this.data.qvel.length; i += 1) {
      this.data.qvel[i] = 0;
    }
  }

  stepSimulation(dt) {
    this.physicsAccumulator = Math.min(this.physicsAccumulator + dt, 0.08);
    let steps = 0;
    while (this.physicsAccumulator >= SIM_DT && steps < MAX_PHYSICS_STEPS_PER_FRAME) {
      if (this.needsSafetyReset()) {
        this.stopUnsafeSimulation();
        this.physicsAccumulator = 0;
        break;
      }
      if (this.physicsStep % POLICY_DECIMATION === 0) {
        this.frameStage = 'policy-request';
        this.requestPolicyStep();
      }
      this.frameStage = 'mj-step';
      this.mujoco.mj_step(this.model, this.data);
      this.wrapWheelJointPositions();
      this.physicsStep += 1;
      steps += 1;
      this.physicsAccumulator -= SIM_DT;
    }
    if (steps >= MAX_PHYSICS_STEPS_PER_FRAME) {
      this.physicsAccumulator = 0;
    }
    this.frameStage = 'ray-refresh';
    this.refreshRaycasterImage();
  }

  needsSafetyReset() {
    if (!this.data || this.baseBodyId < 0) return false;
    return hasNonFinite(this.data.qpos) ||
      hasNonFinite(this.data.qvel) ||
      maxAbs(this.data.qvel) > 80 ||
      maxAbs(this.data.ctrl) > 120;
  }

  needsVisualGuard() {
    if (!this.data || this.baseBodyId < 0) return false;
    const baseOffset = this.baseBodyId * 3;
    const baseZ = this.data.xpos[baseOffset + 2];
    return !Number.isFinite(baseZ) || baseZ < 0.36 || this.needsSafetyReset();
  }

  stopUnsafeSimulation() {
    for (let i = 0; i < this.data.ctrl.length; i += 1) this.data.ctrl[i] = 0;
    this.policyPending = false;
    this.policyRequestStartedAt = 0;
  }

  wrapWheelJointPositions() {
    for (const name of WHEEL_JOINTS) {
      const adr = this.jointAdr.get(name);
      if (adr !== undefined) this.data.qpos[adr] = wrapPi(this.data.qpos[adr]);
    }
  }

  requestPolicyStep() {
    const policy = this.activePolicy();
    if (this.policyPending && performance.now() - this.policyRequestStartedAt > 500) {
      this.policyPending = false;
      this.policyRequestStartedAt = 0;
      this.policySeq += 1;
    }
    if (
      !this.policyWorker ||
      !this.policyWorkerReady ||
      !this.policyReady.has(policy.id) ||
      this.policyPending ||
      this.policyFailed
    ) {
      return;
    }

    const obs = policy.id === 0 ? this.buildMotionObservation() : this.buildVisualObservation(policy);
    const seq = this.policySeq + 1;
    this.policySeq = seq;
    this.policyPending = true;
    this.policyRequestStartedAt = performance.now();
    try {
      this.policyWorker.postMessage({
        type: 'run',
        policyId: policy.id,
        seq,
        obs: obs.buffer,
        dims: [1, policy.obsDim],
      }, [obs.buffer]);
    } catch (error) {
      this.policyPending = false;
      this.policyRequestStartedAt = 0;
      throw error;
    }
  }

  applyAction(rawAction) {
    const scale = this.activePolicy().actionScale || POLICY_CONFIGS[0].actionScale;
    for (let i = 0; i < 16 && i < this.data.ctrl.length; i += 1) {
      const target = ACT_DEFAULT_DOF_POS[i] + rawAction[i] * scale[i];
      this.currentCtrl[i] = clampFinite(target, -CTRL_LIMIT, CTRL_LIMIT);
      this.data.ctrl[i] = this.currentCtrl[i];
    }
  }

  resetObservationHistory() {
    this.histBaseAngVel = new HistoryBuffer(3, 3);
    this.histProjectedGravity = new HistoryBuffer(3, 3);
    this.histCommand = new HistoryBuffer(3, 1);
    this.histDofPos = new HistoryBuffer(12, 3);
    this.histDofVel = new HistoryBuffer(16, 3);
    this.histLastAction = new HistoryBuffer(16, 3);
    this.pushObservationTerms();

    const policy = this.activePolicy();
    this.visualHistBaseAngVel = new HistoryBuffer(3, policy.vectorHistory);
    this.visualHistProjectedGravity = new HistoryBuffer(3, policy.vectorHistory);
    this.visualHistCommand = new HistoryBuffer(3, policy.vectorHistory);
    this.visualHistDofPos = new HistoryBuffer(12, policy.vectorHistory);
    this.visualHistDofVel = new HistoryBuffer(16, policy.vectorHistory);
    this.visualHistLastAction = new HistoryBuffer(16, policy.vectorHistory);
    this.visualImageHistory = new ImageHistoryBuffer(RAY_SIZE, Math.max(policy.imageHistory, 0));
    this.pushVisualObservationTerms();
    const image = this.refreshRaycasterImage(true);
    this.visualImageHistory.reset(image);
  }

  buildMotionObservation() {
    this.histLastAction.push(this.lastRawAction);
    this.pushObservationTerms();

    const obs = new Float32Array(MOTION_OBS_DIM);
    let offset = MOTION_OBS_PREFIX_DIM;
    obs.set(this.histBaseAngVel.flat(), offset);
    offset += this.histBaseAngVel.size;
    obs.set(this.histProjectedGravity.flat(), offset);
    offset += this.histProjectedGravity.size;
    obs.set(this.histCommand.flat(), offset);
    offset += this.histCommand.size;
    obs.set(this.histDofPos.flat(), offset);
    offset += this.histDofPos.size;
    obs.set(this.histDofVel.flat(), offset);
    offset += this.histDofVel.size;
    obs.set(this.histLastAction.flat(), offset);
    offset += this.histLastAction.size;
    if (offset !== MOTION_OBS_DIM) {
      throw new Error(`Unexpected motion_mlp obs size: ${offset}`);
    }
    return obs;
  }

  buildVisualObservation(policy) {
    this.pushVisualObservationTerms();
    const image = this.refreshRaycasterImage();
    const imageObs = policy.imageHistory > 0
      ? this.visualImageHistory.flatWithCurrent(image)
      : image;
    if (policy.imageHistory > 0) {
      this.visualImageHistory.push(image);
    }

    const obs = new Float32Array(policy.obsDim);
    let offset = 0;
    obs.set(this.visualHistBaseAngVel.flat(), offset);
    offset += this.visualHistBaseAngVel.size;
    obs.set(this.visualHistProjectedGravity.flat(), offset);
    offset += this.visualHistProjectedGravity.size;
    obs.set(this.visualHistCommand.flat(), offset);
    offset += this.visualHistCommand.size;
    obs.set(this.visualHistDofPos.flat(), offset);
    offset += this.visualHistDofPos.size;
    obs.set(this.visualHistDofVel.flat(), offset);
    offset += this.visualHistDofVel.size;
    obs.set(this.visualHistLastAction.flat(), offset);
    offset += this.visualHistLastAction.size;
    obs.set(imageObs, offset);
    offset += imageObs.length;
    if (offset !== policy.obsDim) {
      throw new Error(`Unexpected ${policy.name} obs size: ${offset}, expected ${policy.obsDim}`);
    }
    return obs;
  }

  pushObservationTerms() {
    const baseAngVel = this.readSensorVector('imu_gyro');
    for (let i = 0; i < baseAngVel.length; i += 1) baseAngVel[i] *= 0.25;
    this.histBaseAngVel.push(baseAngVel);
    this.histProjectedGravity.push(quatRotateInverse(this.readSensorVector('imu_quat'), [0, 0, -1]));
    this.histCommand.push(new Float32Array([this.cmd.x, this.cmd.y, this.cmd.yaw]));
    this.histDofPos.push(this.readDofPos());
    this.histDofVel.push(this.readDofVel());
  }

  pushVisualObservationTerms() {
    if (!this.visualHistBaseAngVel) return;
    // base_ang_vel obs term is scaled by 0.25 for every policy (matches the
    // motion path in pushObservationTerms() and make_base_ang_vel_term() in the
    // C++ reference, mj_env.cpp). Omitting it feeds the visual policies angular
    // velocity at 4x the trained magnitude, causing rapid in-place shaking.
    const baseAngVel = this.readSensorVector('imu_gyro');
    for (let i = 0; i < baseAngVel.length; i += 1) baseAngVel[i] *= 0.25;
    this.visualHistBaseAngVel.push(baseAngVel);
    this.visualHistProjectedGravity.push(quatRotateInverse(this.readSensorVector('imu_quat'), [0, 0, -1]));
    this.visualHistCommand.push(new Float32Array([this.cmd.x, this.cmd.y, this.cmd.yaw]));
    this.visualHistDofPos.push(this.readDofPos());
    this.visualHistDofVel.push(this.readDofVel());
    this.visualHistLastAction.push(this.lastRawAction);
  }

  readDofPos() {
    const values = new Float32Array(LEG_POS_SENSOR_NAMES.length);
    for (let i = 0; i < LEG_POS_SENSOR_NAMES.length; i += 1) {
      values[i] = this.readSensorScalar(LEG_POS_SENSOR_NAMES[i]) - OBS_DEFAULT_DOF_POS[i];
    }
    return values;
  }

  readDofVel() {
    const values = new Float32Array(DOF_VEL_SENSOR_NAMES.length);
    for (let i = 0; i < DOF_VEL_SENSOR_NAMES.length; i += 1) {
      values[i] = this.readSensorScalar(DOF_VEL_SENSOR_NAMES[i]) * 0.05;
    }
    return values;
  }

  readSensorScalar(name) {
    const spec = this.sensorAdr.get(name);
    return spec ? this.data.sensordata[spec.adr] : 0;
  }

  readSensorVector(name) {
    const spec = this.sensorAdr.get(name);
    if (!spec) return new Float32Array(0);
    const values = new Float32Array(spec.dim);
    for (let i = 0; i < spec.dim; i += 1) {
      values[i] = this.data.sensordata[spec.adr + i];
    }
    return values;
  }

  refreshRaycasterImage(force = false) {
    if (!this.data) return this.rayImage;
    if (!force && this.data.time - this.rayLastUpdateTime < RAY_UPDATE_DT) {
      return this.rayImage;
    }
    return this.computeRaycasterImage();
  }

  computeRaycasterImage() {
    if (!this.model || !this.data || this.cameraId < 0) return this.rayImage;
    if (this.needsVisualGuard()) {
      if (this.needsSafetyReset()) this.stopUnsafeSimulation();
      return this.rayImage;
    }
    const pose = this.cameraPoseMujoco();
    if (!pose) return this.rayImage;
    if (!this._rayGeomId) this._rayGeomId = new Int32Array(1);
    if (!this.rayConvention) {
      this.rayConvention = this.chooseRayConvention(pose);
    }

    const centerDir = this.cameraRayDir(0, 0, pose.mat, this.rayConvention);
    const localDirs = this.rayLocalDirs(this.rayConvention);
    const dir = this._rayDir || (this._rayDir = [0, 0, 0]);
    for (let row = 0; row < RAY_HEIGHT; row += 1) {
      for (let col = 0; col < RAY_WIDTH; col += 1) {
        const index = row * RAY_WIDTH + col;
        this.cameraRayDirFromLocal(localDirs, index, pose.mat, dir);
        this.frameStage = 'three-ray';
        const hit = this.raycastTerrain(pose.pos, dir);
        const dist = hit ? hit.distance : -1;
        const planeDist = dist > 0 ? dist * Math.max(0.05, dot3(dir, centerDir)) : 0;
        this.rayRawImage[index] = planeDist;
        this.rayImage[index] = normalizeDepth(planeDist);
        this.rayHitPoints[index] = hit ? hit.pointMujoco : null;
      }
    }
    this.lastRayPose = pose;
    this.frameStage = 'ray-finished';
    this.rayLastUpdateTime = this.data.time;
    this.rayDirty = true;
    return this.rayImage;
  }

  cameraPoseMujoco() {
    if (this.data.cam_xpos && this.data.cam_xmat) {
      const p = this.cameraId * 3;
      const m = this.cameraId * 9;
      return {
        pos: [
          this.data.cam_xpos[p],
          this.data.cam_xpos[p + 1],
          this.data.cam_xpos[p + 2],
        ],
        mat: [
          this.data.cam_xmat[m],
          this.data.cam_xmat[m + 1],
          this.data.cam_xmat[m + 2],
          this.data.cam_xmat[m + 3],
          this.data.cam_xmat[m + 4],
          this.data.cam_xmat[m + 5],
          this.data.cam_xmat[m + 6],
          this.data.cam_xmat[m + 7],
          this.data.cam_xmat[m + 8],
        ],
      };
    }
    if (this.cameraBodyId < 0 || !this.model.cam_pos || !this.model.cam_quat) return null;
    const body = this.cameraBodyId;
    const bodyPos = [
      this.data.xpos[body * 3],
      this.data.xpos[body * 3 + 1],
      this.data.xpos[body * 3 + 2],
    ];
    const bodyMat = [
      this.data.xmat[body * 9],
      this.data.xmat[body * 9 + 1],
      this.data.xmat[body * 9 + 2],
      this.data.xmat[body * 9 + 3],
      this.data.xmat[body * 9 + 4],
      this.data.xmat[body * 9 + 5],
      this.data.xmat[body * 9 + 6],
      this.data.xmat[body * 9 + 7],
      this.data.xmat[body * 9 + 8],
    ];
    const localPos = [
      this.model.cam_pos[this.cameraId * 3],
      this.model.cam_pos[this.cameraId * 3 + 1],
      this.model.cam_pos[this.cameraId * 3 + 2],
    ];
    const posOffset = mulMat3Vec3(bodyMat, localPos);
    return {
      pos: [
        bodyPos[0] + posOffset[0],
        bodyPos[1] + posOffset[1],
        bodyPos[2] + posOffset[2],
      ],
      mat: bodyMat,
    };
  }

  chooseRayConvention(pose) {
    const conventions = [
      { name: '-z', local: (u, v) => normalize3([u, v, -RAY_FOCAL]) },
      { name: '+z', local: (u, v) => normalize3([u, v, RAY_FOCAL]) },
      { name: '+x', local: (u, v) => normalize3([RAY_FOCAL, -u, v]) },
      { name: '-x', local: (u, v) => normalize3([-RAY_FOCAL, u, v]) },
      { name: '+y', local: (u, v) => normalize3([u, RAY_FOCAL, v]) },
      { name: '-y', local: (u, v) => normalize3([u, -RAY_FOCAL, v]) },
    ];
    const cameraLike = conventions.find((convention) => convention.name === '-y');
    if (!this.rayTerrainMeshes.length) return cameraLike || conventions[0];
    let best = conventions[0];
    let bestScore = -Infinity;
    const sample = [-0.6, 0, 0.6];
    for (const convention of conventions) {
      let score = 0;
      for (const u of sample) {
        for (const v of sample) {
          const dir = this.cameraRayDir(u, v, pose.mat, convention);
          const hit = this.raycastTerrain(pose.pos, dir);
          const dist = hit ? hit.distance : -1;
          if (dist > 0 && dist < 4.0) score += 1;
          if (dist >= RAY_MIN_DIST && dist <= RAY_MAX_DIST) score += 2;
        }
      }
      if (score > bestScore) {
        bestScore = score;
        best = convention;
      }
    }
    return best;
  }

  cameraRayDir(u, v, mat, convention) {
    return normalize3(mulMat3Vec3(mat, convention.local(u, v)));
  }

  raycastTerrain(posMujoco, dirMujoco) {
    if (!this.rayTerrainMeshes.length) return null;
    const origin = mujocoToThreePoint(posMujoco);
    const direction = mujocoToThreeVector(dirMujoco);
    this.threeRaycaster.set(origin, direction);
    this.threeRaycaster.near = RAY_MIN_DIST;
    this.threeRaycaster.far = RAY_MAX_DIST;
    const hits = this.threeRaycaster.intersectObjects(this.rayTerrainMeshes, false);
    if (!hits.length) return null;
    const hit = hits[0];
    return {
      distance: hit.distance,
      pointMujoco: threeToMujocoPoint(hit.point),
    };
  }

  rayLocalDirs(convention) {
    const key = convention.name;
    if (this.rayLocalDirsByConvention.has(key)) return this.rayLocalDirsByConvention.get(key);
    const localDirs = new Float32Array(RAY_SIZE * 3);
    for (let row = 0; row < RAY_HEIGHT; row += 1) {
      const v = (0.5 - (row + 0.5) / RAY_HEIGHT) * RAY_VERTICAL_APERTURE;
      for (let col = 0; col < RAY_WIDTH; col += 1) {
        const u = ((col + 0.5) / RAY_WIDTH - 0.5) * RAY_HORIZONTAL_APERTURE;
        const local = convention.local(u, v);
        const offset = (row * RAY_WIDTH + col) * 3;
        localDirs[offset] = local[0];
        localDirs[offset + 1] = local[1];
        localDirs[offset + 2] = local[2];
      }
    }
    this.rayLocalDirsByConvention.set(key, localDirs);
    return localDirs;
  }

  cameraRayDirFromLocal(localDirs, index, mat, out) {
    const offset = index * 3;
    const x = localDirs[offset];
    const y = localDirs[offset + 1];
    const z = localDirs[offset + 2];
    out[0] = mat[0] * x + mat[1] * y + mat[2] * z;
    out[1] = mat[3] * x + mat[4] * y + mat[5] * z;
    out[2] = mat[6] * x + mat[7] * y + mat[8] * z;
    return out;
  }

  updateRayVisualization() {
    if (!this.rayDirty) return;
    this.rayDirty = false;
    if (this.rayCanvasImage && this.rayCtx) {
      const pixels = this.rayCanvasImage.data;
      for (let i = 0; i < RAY_SIZE; i += 1) {
        const value = clamp(this.rayImage[i], 0, 1);
        const color = depthColor(value);
        const p = i * 4;
        pixels[p] = color[0];
        pixels[p + 1] = color[1];
        pixels[p + 2] = color[2];
        pixels[p + 3] = 255;
      }
      this.rayCtx.putImageData(this.rayCanvasImage, 0, 0);
      if (this.rayReadout) {
        const validCount = this.rayImage.reduce((acc, value) => acc + (value > 0 ? 1 : 0), 0);
        this.rayReadout.textContent = `${RAY_WIDTH} x ${RAY_HEIGHT} depth · ${validCount} hits`;
      }
    }
    if (this.rayLines && this.lastRayPose) {
      let p = 0;
      for (const index of this.raySampleIndices) {
        const hit = this.rayHitPoints[index];
        const end = hit || this.lastRayPose.pos;
        this.rayLinePositions[p++] = this.lastRayPose.pos[0];
        this.rayLinePositions[p++] = this.lastRayPose.pos[1];
        this.rayLinePositions[p++] = this.lastRayPose.pos[2];
        this.rayLinePositions[p++] = end[0];
        this.rayLinePositions[p++] = end[1];
        this.rayLinePositions[p++] = end[2];
      }
      this.rayLineGeometry.attributes.position.needsUpdate = true;
      this.rayLineGeometry.computeBoundingSphere();
    }
  }

  followBase(dt) {
    this.followCamera = true;
    if (this.baseBodyId < 0) return;
    const base = new THREE.Vector3(
      this.data.xpos[this.baseBodyId * 3],
      this.data.xpos[this.baseBodyId * 3 + 2],
      -this.data.xpos[this.baseBodyId * 3 + 1],
    );
    const desired = base.clone().add(new THREE.Vector3(-2.4, 1.45, 2.2));
    this.camera.position.lerp(desired, 1 - Math.pow(0.002, dt));
    this.controls.target.lerp(base, 1 - Math.pow(0.001, dt));
  }

  updateRuntimeStats() {
    if (!this.data) return;
    const baseOffset = this.baseBodyId >= 0 ? this.baseBodyId * 3 : -1;
    window.__go2wRuntime = {
      frameCount: this.frameCount,
      simTime: this.data.time,
      activeGeoms: this.activeGeoms,
      policyId: this.policyId,
      policyName: this.activePolicy().name,
      policyPending: this.policyPending,
      policyFailed: this.policyFailed,
      policyRuns: this.policyRuns,
      lastPolicyDurationMs: this.lastPolicyDurationMs,
      frameError: this.frameError?.message || null,
      frameStage: this.frameStage,
      rayConvention: this.rayConvention?.name || null,
      rayMean: meanPositive(this.rayImage),
      qposMaxAbs: maxAbs(this.data.qpos),
      qvelMaxAbs: maxAbs(this.data.qvel),
      ctrlMaxAbs: maxAbs(this.data.ctrl),
      followCamera: this.followCamera,
      base: baseOffset >= 0 ? [
        this.data.xpos[baseOffset],
        this.data.xpos[baseOffset + 1],
        this.data.xpos[baseOffset + 2],
      ] : null,
    };
  }

  setStatus(label, title, klass = '') {
    this.statusPill.textContent = label;
    this.statusPill.title = title;
    this.statusPill.className = `status-pill ${klass}`.trim();
    if (!this.errorReadout) return;
    if (klass === 'error') {
      this.errorReadout.hidden = false;
      this.errorReadout.textContent = title || 'Unknown error';
    } else {
      this.errorReadout.hidden = true;
      this.errorReadout.textContent = '';
    }
  }
}

class HistoryBuffer {
  constructor(width, length) {
    this.width = width;
    this.length = Math.max(1, length);
    this.size = width * this.length;
    this.buffer = new Float32Array(this.size);
    this.pointer = 0;
    this.empty = true;
  }

  push(values) {
    if (values.length !== this.width) {
      throw new Error(`History width mismatch: expected ${this.width}, got ${values.length}`);
    }
    if (this.empty) {
      for (let i = 0; i < this.length; i += 1) {
        this.buffer.set(values, i * this.width);
      }
      this.pointer = 1 % this.length;
      this.empty = false;
      return;
    }
    this.buffer.set(values, this.pointer * this.width);
    this.pointer = (this.pointer + 1) % this.length;
  }

  flat() {
    if (this.length === 1) return new Float32Array(this.buffer);
    const out = new Float32Array(this.size);
    for (let i = 0; i < this.length; i += 1) {
      const src = ((this.pointer + i) % this.length) * this.width;
      out.set(this.buffer.subarray(src, src + this.width), i * this.width);
    }
    return out;
  }
}

class ImageHistoryBuffer {
  constructor(width, historyLength) {
    this.width = width;
    this.historyLength = Math.max(0, historyLength);
    this.buffer = new Float32Array(this.width * this.historyLength);
    this.pointer = 0;
    this.empty = true;
  }

  reset(frame) {
    if (this.historyLength <= 0) return;
    for (let i = 0; i < this.historyLength; i += 1) {
      this.buffer.set(frame, i * this.width);
    }
    this.pointer = 0;
    this.empty = false;
  }

  push(frame) {
    if (this.historyLength <= 0) return;
    if (frame.length !== this.width) {
      throw new Error(`Image history width mismatch: expected ${this.width}, got ${frame.length}`);
    }
    if (this.empty) this.reset(frame);
    this.buffer.set(frame, this.pointer * this.width);
    this.pointer = (this.pointer + 1) % this.historyLength;
  }

  flatWithCurrent(frame) {
    if (this.historyLength <= 0) return new Float32Array(frame);
    const out = new Float32Array(this.width * (this.historyLength + 1));
    for (let i = 0; i < this.historyLength; i += 1) {
      const src = ((this.pointer + i) % this.historyLength) * this.width;
      out.set(this.buffer.subarray(src, src + this.width), i * this.width);
    }
    out.set(frame, this.width * this.historyLength);
    return out;
  }
}

function quatRotateInverse(quat, vec) {
  const w = quat[0];
  const x = quat[1];
  const y = quat[2];
  const z = quat[3];
  const vx = vec[0];
  const vy = vec[1];
  const vz = vec[2];
  const dot = x * vx + y * vy + z * vz;
  const factor = 2 * w * w - 1;
  return new Float32Array([
    vx * factor - 2 * w * (y * vz - z * vy) + 2 * x * dot,
    vy * factor - 2 * w * (z * vx - x * vz) + 2 * y * dot,
    vz * factor - 2 * w * (x * vy - y * vx) + 2 * z * dot,
  ]);
}

function checked(response) {
  if (!response.ok) {
    throw new Error(`${response.status} ${response.url}`);
  }
  return response;
}

async function fetchWithRetry(url, bodyType, attempts = 4) {
  let lastError = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = checked(await fetch(url));
      return await response[bodyType]();
    } catch (error) {
      lastError = error;
      await delay(250 * (attempt + 1));
    }
  }
  throw lastError || new Error(`Failed to fetch ${url}`);
}

async function runLimited(tasks, limit) {
  let next = 0;
  const workers = new Array(Math.min(limit, tasks.length)).fill(0).map(async () => {
    while (next < tasks.length) {
      const index = next;
      next += 1;
      await tasks[index]();
    }
  });
  await Promise.all(workers);
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampFinite(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  return clamp(value, min, max);
}

function normalizeDepth(value) {
  if (!Number.isFinite(value) || value < RAY_MIN_DIST || value > RAY_MAX_DIST) return 0;
  return (value - RAY_MIN_DIST) / (RAY_MAX_DIST - RAY_MIN_DIST);
}

function normalize3(vec) {
  const len = Math.hypot(vec[0], vec[1], vec[2]);
  if (len <= 1.0e-9) return [0, 0, 0];
  return [vec[0] / len, vec[1] / len, vec[2] / len];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function mulMat3Vec3(mat, vec) {
  return [
    mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
    mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
    mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2],
  ];
}

function mujocoToThreePoint(pos) {
  return new THREE.Vector3(pos[0], pos[2], -pos[1]);
}

function mujocoToThreeVector(vec) {
  return new THREE.Vector3(vec[0], vec[2], -vec[1]).normalize();
}

function threeToMujocoPoint(point) {
  return [point.x, -point.z, point.y];
}

function depthColor(value) {
  if (value <= 0) return [10, 14, 12];
  const t = clamp(value, 0, 1);
  const r = Math.round(38 + 205 * t);
  const g = Math.round(92 + 142 * (1 - Math.abs(t - 0.5) * 1.6));
  const b = Math.round(72 + 42 * (1 - t));
  return [r, g, b];
}

function meanPositive(values) {
  let sum = 0;
  let count = 0;
  for (const value of values) {
    if (value > 0) {
      sum += value;
      count += 1;
    }
  }
  return count > 0 ? sum / count : 0;
}

function maxAbs(values) {
  let max = 0;
  for (const value of values) {
    const abs = Math.abs(value);
    if (Number.isFinite(abs) && abs > max) max = abs;
  }
  return max;
}

function hasNonFinite(values) {
  for (const value of values) {
    if (!Number.isFinite(value)) return true;
  }
  return false;
}

function wrapPi(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.atan2(Math.sin(value), Math.cos(value));
}

const demo = new Go2WDemo();
window.__go2wDemo = demo;
demo.init().catch((error) => {
  console.error(error);
  if (!window.__go2wWebglStarted && window.__go2wStartFallback) {
    window.__go2wStartFallback(error.message);
  } else {
    demo.setStatus('Error', error.message, 'error');
  }
});
