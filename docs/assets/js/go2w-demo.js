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

const POLICY_NAMES = ['motion_mlp', 'vtm', 'vtm_lstm_sru', 'vtm_gru_sru'];
const SIM_DT = 0.005;
const POLICY_DECIMATION = 4;
const MOTION_POLICY_URL = 'demo-assets/policies/motion_tracking/policy.onnx';
const MOTION_OBS_PREFIX_DIM = 24 + 1 + 3 + 6;
const MOTION_OBS_DIM = 187;
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
const ACTION_SCALE_MOTION = new Float32Array([
  0.125, 0.25, 0.25,
  0.125, 0.25, 0.25,
  0.125, 0.25, 0.25,
  0.125, 0.25, 0.25,
  2.0, 2.0, 2.0, 2.0,
]);
const MJCAT_ALL = 7;
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

class Go2WDemo {
  constructor() {
    this.canvas = $('go2w-canvas');
    this.statusPill = $('status-pill');
    this.policyReadout = $('policy-readout');
    this.errorReadout = $('error-readout');
    this.policyId = 0;
    this.keys = new Set();
    this.cmd = { x: 0.7, y: 0, yaw: 0 };
    this.realtime = 1.0;
    this.pose = { x: 0, y: 0, z: 0.56, yaw: 0 };
    this.gaitTime = 0;
    this.wheelPhase = 0;
    this.clock = new THREE.Clock();
    this.physicsAccumulator = 0;
    this.physicsStep = 0;
    this.policyPending = false;
    this.policyFailed = false;
    this.policyWorkerReady = false;
    this.policySeq = 0;
    this.policyRuns = 0;
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
    this.frameCount = 0;
  }

  async init() {
    this.setStatus('Loading', 'Loading MuJoCo');

    this.mujoco = await loadMujoco();
    this.ensureDir('/working');
    this.ensureDir('/working/assets');
    await this.loadFiles();

    this.setStatus('Loading', 'Compiling MJCF');
    this.model = this.mujoco.MjModel.loadFromXML('/working/scene_parkour.xml');
    this.data = new this.mujoco.MjData(this.model);
    if (window.__go2wFallbackStarted) return;
    this.setupRenderer();
    this.cacheJointAddresses();
    this.cacheSensorAddresses();
    this.setupVisualScene();
    this.setStatus('Loading', 'Loading motion_mlp ONNX');
    await this.loadMotionPolicy();
    this.resetBrowserPose();
    this.bindControls();

    window.__go2wDemoReady = true;
    this.setStatus('Ready', 'MuJoCo + ONNX policy ready', 'ready');
    this.renderer.setAnimationLoop(() => this.frame());
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
      new THREE.MeshLambertMaterial({
        color: 0xd4ddd0,
      }),
    );
    ground.name = 'Ground Plane';
    this.mujocoRoot.add(ground);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.set(0.6, 0.35, 0);
    this.controls.enableDamping = true;
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
      this.resetBrowserPose();
    });

    for (const button of document.querySelectorAll('[data-policy]')) {
      if (Number(button.dataset.policy) !== 0) {
        button.disabled = true;
        button.title = 'Visual policy support requires ray image + SRU state migration.';
      }
      button.addEventListener('click', () => {
        this.setPolicy(Number(button.dataset.policy));
      });
    }

    document.addEventListener('keydown', (event) => {
      if (event.repeat) return;
      this.keys.add(event.code);
      const handled = this.handleKeyStep(event.code);
      if (handled) event.preventDefault();
    });

    document.addEventListener('keyup', (event) => {
      this.keys.delete(event.code);
    });
  }

  handleKeyStep(code) {
    const step = 0.1;
    if (code === 'KeyW') this.cmd.x += step;
    else if (code === 'KeyS') this.cmd.x -= step;
    else if (code === 'KeyA') this.cmd.y += step;
    else if (code === 'KeyD') this.cmd.y -= step;
    else if (code === 'KeyQ') this.cmd.yaw += step;
    else if (code === 'KeyE') this.cmd.yaw -= step;
    else if (code === 'Space') this.setCommand(0, 0, 0);
    else if (code === 'KeyR') {
      this.resetBrowserPose();
      return true;
    } else {
      const digit = code.match(/^Digit([1-4])$/);
      if (digit) {
        this.setPolicy(Number(digit[1]) - 1);
        return true;
      }
      return false;
    }
    this.setCommand(this.cmd.x, this.cmd.y, this.cmd.yaw);
    return true;
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

  setPolicy(policyId) {
    this.policyId = policyId === 0 ? 0 : this.policyId;
    for (const button of document.querySelectorAll('[data-policy]')) {
      button.classList.toggle('active', Number(button.dataset.policy) === this.policyId);
    }
    this.policyReadout.textContent = `policy ${this.policyId}: ${POLICY_NAMES[this.policyId]} · ONNX ${MOTION_OBS_DIM}D`;
  }

  ensureDir(path) {
    if (!this.mujoco.FS.analyzePath(path).exists) {
      this.mujoco.FS.mkdir(path);
    }
  }

  async loadFiles() {
    const textFiles = ['scene_parkour.xml', 'go2w.xml'].map(async (name) => {
      const text = await fetch(`demo-assets/scenes/${name}`).then((r) => checked(r).text());
      this.mujoco.FS.writeFile(`/working/${name}`, text);
    });
    const assetFiles = ASSET_NAMES.map(async (name) => {
      const buffer = await fetch(`demo-assets/scenes/assets/${name}`).then((r) => checked(r).arrayBuffer());
      this.mujoco.FS.writeFile(`/working/assets/${name}`, new Uint8Array(buffer));
    });
    await Promise.all([...textFiles, ...assetFiles]);
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

  async loadMotionPolicy() {
    const policyUrl = new URL(MOTION_POLICY_URL, window.location.href).href;
    this.setStatus('Loading', 'Starting motion_mlp worker');
    this.motionPolicyWorker = new Worker(new URL('policy-worker.js?v=worker-2', import.meta.url), {
      name: 'go2w-motion-policy',
    });
    this.motionPolicyWorker.onmessage = (event) => this.handlePolicyWorkerMessage(event.data || {});
    this.motionPolicyWorker.onerror = (error) => {
      this.policyFailed = true;
      this.policyPending = false;
      this.setStatus('Policy Error', error.message || 'motion_mlp worker failed', 'error');
    };

    await new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        reject(new Error('Timed out while loading motion_mlp worker'));
      }, 30000);
      const onMessage = (event) => {
        const data = event.data || {};
        if (data.type === 'ready') {
          window.clearTimeout(timeout);
          this.motionPolicyWorker.removeEventListener('message', onMessage);
          this.motionPolicyWorker.removeEventListener('message', onError);
          this.policyWorkerReady = true;
          resolve();
        } else if (data.type === 'error') {
          window.clearTimeout(timeout);
          this.motionPolicyWorker.removeEventListener('message', onMessage);
          this.motionPolicyWorker.removeEventListener('message', onError);
          reject(new Error(data.message || 'motion_mlp worker failed to initialize'));
        }
      };
      const onError = (event) => {
        const data = event.data || {};
        if (data.type !== 'error') return;
        window.clearTimeout(timeout);
        this.motionPolicyWorker.removeEventListener('message', onMessage);
        this.motionPolicyWorker.removeEventListener('message', onError);
        reject(new Error(data.message || 'motion_mlp worker failed to initialize'));
      };
      this.motionPolicyWorker.addEventListener('message', onMessage);
      this.motionPolicyWorker.addEventListener('message', onError);
      this.motionPolicyWorker.postMessage({ type: 'init', policyUrl });
    });
    this.policyReadout.textContent = `policy 0: motion_mlp · ONNX ${MOTION_OBS_DIM}D`;
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
    for (let i = 0; i < this.mjvScene.ngeom; i += 1) {
      const geom = this.mjvScene.geoms.get(i);
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
      mesh.visible = true;
    }

    this.activeGeoms = meshIndex;
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

  resetBrowserPose() {
    this.physicsAccumulator = 0;
    this.physicsStep = 0;
    this.policyPending = false;
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
    this.resetObservationHistory();
  }

  frame() {
    const dt = Math.min(this.clock.getDelta(), 0.04);
    this.frameCount += 1;
    this.stepSimulation(dt * this.realtime);
    this.updateVisualScene();
    this.followBase(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.updateRuntimeStats();
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
    while (this.physicsAccumulator >= SIM_DT && steps < 24) {
      if (this.physicsStep % POLICY_DECIMATION === 0) {
        this.requestPolicyStep();
      }
      this.mujoco.mj_step(this.model, this.data);
      this.physicsStep += 1;
      steps += 1;
      this.physicsAccumulator -= SIM_DT;
    }
  }

  requestPolicyStep() {
    if (!this.motionPolicyWorker || !this.policyWorkerReady || this.policyPending || this.policyFailed || this.policyId !== 0) return;

    const obs = this.buildMotionObservation();
    const seq = this.policySeq + 1;
    this.policySeq = seq;
    this.policyPending = true;
    this.motionPolicyWorker.postMessage({
      type: 'run',
      seq,
      obs: obs.buffer,
      dims: [1, MOTION_OBS_DIM],
    }, [obs.buffer]);
  }

  handlePolicyWorkerMessage(data) {
    if (data.type === 'ready') return;
    if (data.type === 'error') {
      console.error(data.message);
      this.policyFailed = true;
      this.policyPending = false;
      this.setStatus('Policy Error', data.message, 'error');
      return;
    }
    if (data.type !== 'result' || data.seq !== this.policySeq) return;

    const raw = data.action;
    for (let i = 0; i < 16; i += 1) {
      this.lastRawAction[i] = Number.isFinite(raw[i]) ? raw[i] : 0;
    }
    this.lastPolicyDurationMs = data.durationMs || 0;
    this.policyRuns += 1;
    this.applyAction(this.lastRawAction);
    this.policyPending = false;
    this.policyReadout.textContent =
      `policy 0: motion_mlp · worker ${this.lastPolicyDurationMs.toFixed(0)}ms`;
  }

  applyAction(rawAction) {
    for (let i = 0; i < 16 && i < this.data.ctrl.length; i += 1) {
      const target = ACT_DEFAULT_DOF_POS[i] + rawAction[i] * ACTION_SCALE_MOTION[i];
      this.currentCtrl[i] = clampFinite(target, -100, 100);
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

  pushObservationTerms() {
    const baseAngVel = this.readSensorVector('imu_gyro');
    for (let i = 0; i < baseAngVel.length; i += 1) baseAngVel[i] *= 0.25;
    this.histBaseAngVel.push(baseAngVel);
    this.histProjectedGravity.push(quatRotateInverse(this.readSensorVector('imu_quat'), [0, 0, -1]));
    this.histCommand.push(new Float32Array([this.cmd.x, this.cmd.y, this.cmd.yaw]));
    this.histDofPos.push(this.readDofPos());
    this.histDofVel.push(this.readDofVel());
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

  followBase(dt) {
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
      policyPending: this.policyPending,
      policyFailed: this.policyFailed,
      policyRuns: this.policyRuns,
      lastPolicyDurationMs: this.lastPolicyDurationMs,
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
    this.length = length;
    this.size = width * length;
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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampFinite(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  return clamp(value, min, max);
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
