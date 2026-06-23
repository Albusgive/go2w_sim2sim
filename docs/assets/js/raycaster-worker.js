import loadMujoco from '../../demo-assets/mujoco_wasm.js';

let mujoco = null;
let model = null;
let data = null;
let raycaster = null;
let config = null;
let ready = false;

function assetUrl(path) {
  return new URL(`../../demo-assets/${path}`, self.location.href).href;
}

function ensureDir(path) {
  if (!mujoco.FS.analyzePath(path).exists) {
    mujoco.FS.mkdir(path);
  }
}

function loadModelXml(path) {
  const loader = mujoco?.MjModel?.mj_loadXML || mujoco?.MjModel?.loadFromXML;
  if (typeof loader !== 'function') {
    throw new Error('MuJoCo WASM does not expose an MJCF XML loader');
  }
  return loader.call(mujoco.MjModel, path);
}

async function fetchWithRetry(url, responseType, attempts = 5) {
  let lastError = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.url}`);
      return responseType === 'arrayBuffer' ? response.arrayBuffer() : response.text();
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 300 * (attempt + 1)));
    }
  }
  throw lastError || new Error(`Failed to fetch ${url}`);
}

async function runLimited(tasks, limit = 4) {
  let next = 0;
  const workers = new Array(Math.min(limit, tasks.length)).fill(0).map(async () => {
    while (next < tasks.length) {
      const task = tasks[next];
      next += 1;
      await task();
    }
  });
  await Promise.all(workers);
}

async function loadFiles(assetNames) {
  ensureDir('/working');
  ensureDir('/working/assets');
  const textFiles = ['scene_parkour.xml', 'go2w.xml'].map(async (name) => {
    const text = await fetchWithRetry(assetUrl(`scenes/${name}`), 'text');
    mujoco.FS.writeFile(`/working/${name}`, text);
  });
  await Promise.all(textFiles);

  const assetFiles = assetNames.map((name) => async () => {
    const buffer = await fetchWithRetry(assetUrl(`scenes/assets/${name}`), 'arrayBuffer');
    mujoco.FS.writeFile(`/working/assets/${name}`, new Uint8Array(buffer));
  });
  await runLimited(assetFiles, 4);
}

async function initRaycaster(nextConfig) {
  config = nextConfig;
  mujoco = await loadMujoco();
  await loadFiles(config.assetNames || []);
  model = loadModelXml('/working/scene_parkour.xml');
  data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);

  const RayCasterCamera = mujoco.RayCasterCamera;
  if (typeof RayCasterCamera !== 'function') {
    throw new Error('MuJoCo WASM does not expose RayCasterCamera');
  }

  raycaster = new RayCasterCamera(
    model,
    data,
    'RayCasterCamera',
    config.width,
    config.height,
    config.near,
    config.far,
    config.focal,
    config.horizontalAperture,
    config.verticalAperture,
    false,
    config.numThreads || 0,
    0.0,
    0.0,
    0.0,
  );
  ready = true;
  self.postMessage({
    type: 'ready',
    backend: `worker-RayCasterCamera/${Number(raycaster.numThreads) || 0}t`,
  });
}

function computeRaycast(message) {
  if (!ready || !raycaster || !data || !model) return;
  const startedAt = performance.now();
  data.qpos.set(new Float64Array(message.qpos));
  if (message.qvel) data.qvel.set(new Float64Array(message.qvel));
  mujoco.mj_forward(model, data);
  raycaster.compute(model, data);

  const depthView = raycaster.depthView();
  const hitPointView = raycaster.hitPointView?.();
  const depth = new Float32Array(depthView.length);
  depth.set(depthView);
  const hitPoints = hitPointView ? new Float32Array(hitPointView) : new Float32Array(depth.length * 3);

  self.postMessage({
    type: 'result',
    seq: message.seq,
    simTime: message.simTime,
    durationMs: performance.now() - startedAt,
    validCount: Number(raycaster.validCount) || 0,
    backend: `worker-RayCasterCamera/${Number(raycaster.numThreads) || 0}t`,
    depth,
    hitPoints,
  }, [depth.buffer, hitPoints.buffer]);
}

self.onmessage = async (event) => {
  const message = event.data || {};
  try {
    if (message.type === 'init') {
      await initRaycaster(message.config || {});
      return;
    }
    if (message.type === 'compute') {
      computeRaycast(message);
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      seq: message.seq,
      message: error?.message || String(error),
    });
  }
};
