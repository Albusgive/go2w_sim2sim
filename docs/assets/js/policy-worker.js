self.importScripts('../vendor/onnxruntime-web/ort.wasm.min.js');

const policies = new Map();
let configs = new Map();
// Highest run sequence number received from the main thread. Because onmessage
// is async, several 'run' messages can queue up while an inference is awaiting
// (e.g. after the main thread's 500ms stale-pending timeout posts a new run).
// Recording the latest seq synchronously on receipt lets us drop stale runs
// before paying for ONNX execution, so the worker never falls behind.
let latestRunSeq = -Infinity;
let workerCapabilities = {
  hardwareConcurrency: 1,
  sharedArrayBuffer: false,
  crossOriginIsolated: false,
  wasmPthreads: false,
  onnxThreads: 1,
};

function postError(seq, policyId, error) {
  self.postMessage({
    type: 'error',
    seq,
    policyId,
    message: error?.message || String(error),
  });
}

function toConfigMap(rawConfigs) {
  const map = new Map();
  for (const config of rawConfigs || []) {
    map.set(config.id, config);
  }
  return map;
}

async function createSession(url) {
  const modelBuffer = await fetchArrayBufferWithRetry(url);
  return self.ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

async function fetchArrayBufferWithRetry(url, attempts = 5) {
  let lastError = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.url}`);
      return await response.arrayBuffer();
    } catch (error) {
      lastError = error;
      await delay(350 * (attempt + 1));
    }
  }
  throw new Error(`${url}: ${lastError?.message || 'failed to fetch policy model'}`);
}

function delay(ms) {
  return new Promise((resolve) => {
    self.setTimeout(resolve, ms);
  });
}

function chooseThreadCapabilities(requested = {}) {
  const hardwareConcurrency = Math.max(
    1,
    Math.floor(requested.hardwareConcurrency || self.navigator?.hardwareConcurrency || 1),
  );
  const sharedArrayBuffer = typeof self.SharedArrayBuffer !== 'undefined';
  const crossOriginIsolated = self.crossOriginIsolated === true;
  const wasmPthreads = sharedArrayBuffer && crossOriginIsolated;
  const requestedThreads = requested.onnxThreads === undefined || requested.onnxThreads === null
    ? (wasmPthreads ? Math.min(2, hardwareConcurrency) : 1)
    : Math.max(1, Math.floor(requested.onnxThreads));
  return {
    hardwareConcurrency,
    sharedArrayBuffer,
    crossOriginIsolated,
    wasmPthreads,
    onnxThreads: wasmPthreads ? Math.min(requestedThreads, hardwareConcurrency) : 1,
  };
}

function zeros(length) {
  return new Float32Array(length);
}

function resetPolicyState(policy) {
  if (!policy || policy.kind !== 'split') return;
  const stateSize = policy.config.numLayers * policy.batchSize * policy.config.hiddenDim;
  policy.hiddenState = zeros(stateSize);
  policy.cellState = zeros(stateSize);
  // Bump the recurrent-state epoch so any runSplit() that started before this
  // reset and is still suspended at an await will refuse to write its (now
  // stale, pre-reset) next_hidden/next_cell over the freshly-zeroed state.
  // Without this, an in-flight inference clobbers the reset and the hidden
  // state stays corrupted until a reset happens to land with no run in flight.
  policy.stateEpoch = (policy.stateEpoch || 0) + 1;
}

async function loadPolicy(policyId) {
  if (policies.has(policyId)) return policies.get(policyId);

  const config = configs.get(policyId);
  if (!config) throw new Error(`Unknown policy id: ${policyId}`);

  if (config.kind === 'mlp') {
    const session = await createSession(config.url);
    const policy = {
      kind: 'mlp',
      config,
      session,
      inputName: session.inputNames[0],
      outputName: session.outputNames[0],
    };
    policies.set(policyId, policy);
    return policy;
  }

  if (config.kind === 'split') {
    const [encoder, memory, actor] = await Promise.all([
      createSession(config.encoderUrl),
      createSession(config.memoryUrl),
      createSession(config.actorUrl),
    ]);
    const policy = {
      kind: 'split',
      config,
      encoder,
      memory,
      actor,
      batchSize: 1,
      hiddenState: null,
      cellState: null,
    };
    resetPolicyState(policy);
    policies.set(policyId, policy);
    return policy;
  }

  throw new Error(`Unsupported policy kind: ${config.kind}`);
}

async function runMlp(policy, obs, dims) {
  const feeds = {
    [policy.inputName]: new self.ort.Tensor('float32', obs, dims),
  };
  const results = await policy.session.run(feeds);
  return results[policy.outputName]?.data;
}

function pickOutput(results, names, fallbackIndex = 0) {
  for (const name of names) {
    if (results[name]) return results[name].data;
  }
  const values = Object.values(results);
  return values[fallbackIndex]?.data;
}

async function runSplit(policy, obs, dims) {
  // Snapshot the recurrent-state epoch before any await. If a reset (or a newer
  // run, see onmessage) advances the epoch while we are suspended, we must not
  // write this run's state back — it was computed from the pre-reset state and
  // a stale observation.
  const startEpoch = policy.stateEpoch || 0;
  const encoderInputName = policy.encoder.inputNames[0];
  const encoderOutputName = policy.encoder.outputNames[0];
  const encoderResults = await policy.encoder.run({
    [encoderInputName]: new self.ort.Tensor('float32', obs, dims),
  });
  const encoded = encoderResults[encoderOutputName]?.data;
  if (!encoded) throw new Error(`${policy.config.name} encoder returned no encoded_obs`);

  const memoryFeeds = {};
  for (const name of policy.memory.inputNames) {
    if (name === 'encoded_obs') {
      memoryFeeds[name] = new self.ort.Tensor('float32', encoded, [1, policy.config.encodedDim]);
    } else if (name === 'hidden_state') {
      memoryFeeds[name] = new self.ort.Tensor(
        'float32',
        policy.hiddenState,
        [policy.config.numLayers, policy.batchSize, policy.config.hiddenDim],
      );
    } else if (name === 'cell_state') {
      memoryFeeds[name] = new self.ort.Tensor(
        'float32',
        policy.cellState,
        [policy.config.numLayers, policy.batchSize, policy.config.hiddenDim],
      );
    } else {
      throw new Error(`${policy.config.name} unsupported memory input: ${name}`);
    }
  }
  const memoryResults = await policy.memory.run(memoryFeeds);
  const latent = pickOutput(memoryResults, ['latent'], 0);
  const nextHidden = pickOutput(memoryResults, ['next_hidden_state'], 1);
  const nextCell = pickOutput(memoryResults, ['next_cell_state'], 2);
  if (!latent) throw new Error(`${policy.config.name} memory returned no latent`);
  // Only advance the recurrent state if no reset/newer run intervened while we
  // were awaiting. Otherwise drop this stale state update so a reset's zeros
  // (or the newest run's state) survive.
  if (startEpoch === (policy.stateEpoch || 0)) {
    if (nextHidden) policy.hiddenState = new Float32Array(nextHidden);
    if (nextCell) policy.cellState = new Float32Array(nextCell);
  }

  const actorFeeds = {};
  const actorInputName = policy.actor.inputNames[0];
  actorFeeds[actorInputName] = new self.ort.Tensor('float32', latent, [1, policy.config.latentDim]);
  const actorResults = await policy.actor.run(actorFeeds);
  return pickOutput(actorResults, ['actions'], 0);
}

function sanitizeAction(raw) {
  if (!raw || raw.length < 16) {
    throw new Error('Policy returned an invalid action tensor');
  }
  const action = new Float32Array(16);
  for (let i = 0; i < 16; i += 1) {
    action[i] = Number.isFinite(raw[i]) ? raw[i] : 0;
  }
  return action;
}

self.onmessage = async (event) => {
  const data = event.data || {};
  try {
    if (data.type === 'init') {
      const wasmBaseUrl = new URL('../vendor/onnxruntime-web/', self.location.href).href;
      workerCapabilities = chooseThreadCapabilities({
        ...(data.threadCaps || {}),
        onnxThreads: data.onnxThreads,
      });
      self.ort.env.wasm.wasmPaths = wasmBaseUrl;
      self.ort.env.wasm.numThreads = workerCapabilities.onnxThreads;
      self.ort.env.wasm.proxy = false;
      configs = toConfigMap(data.policies);
      self.postMessage({ type: 'ready', capabilities: workerCapabilities });
      return;
    }

    if (data.type === 'load') {
      const policy = await loadPolicy(data.policyId);
      self.postMessage({
        type: 'loaded',
        policyId: data.policyId,
        name: policy.config.name,
        kind: policy.kind,
      });
      return;
    }

    if (data.type === 'reset') {
      if (data.policyId === undefined || data.policyId === null) {
        for (const policy of policies.values()) resetPolicyState(policy);
      } else {
        resetPolicyState(policies.get(data.policyId));
      }
      self.postMessage({ type: 'reset', policyId: data.policyId ?? null });
      return;
    }

    if (data.type === 'run') {
      // Record the newest request synchronously, before any await, so queued
      // older runs can detect that they have been superseded.
      const seq = typeof data.seq === 'number' ? data.seq : latestRunSeq;
      if (seq > latestRunSeq) latestRunSeq = seq;
      const policy = await loadPolicy(data.policyId);
      // A newer run arrived while the session was loading — drop this one. Its
      // result would be discarded by the main thread anyway (it checks seq),
      // and for stateful (split) policies executing it would needlessly advance
      // the recurrent state on stale observations.
      if (seq < latestRunSeq) return;
      const startedAt = performance.now();
      const obs = new Float32Array(data.obs);
      let raw;
      if (policy.kind === 'split') {
        // Claim recurrent-state ownership for this run: bump the epoch so any
        // older split run still suspended at an await will refuse to write its
        // state, leaving the newest run (this one) as the sole writer. runSplit
        // snapshots the epoch AFTER this bump.
        policy.stateEpoch = (policy.stateEpoch || 0) + 1;
        raw = await runSplit(policy, obs, data.dims);
      } else {
        raw = await runMlp(policy, obs, data.dims);
      }
      const action = sanitizeAction(raw);
      self.postMessage({
        type: 'result',
        seq: data.seq,
        policyId: data.policyId,
        action,
        durationMs: performance.now() - startedAt,
      }, [action.buffer]);
    }
  } catch (error) {
    postError(data.seq, data.policyId, error);
  }
};
