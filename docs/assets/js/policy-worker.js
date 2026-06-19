self.importScripts('../vendor/onnxruntime-web/ort.wasm.min.js');

const policies = new Map();
let configs = new Map();

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
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.url}`);
  const modelBuffer = await response.arrayBuffer();
  return self.ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

function zeros(length) {
  return new Float32Array(length);
}

function resetPolicyState(policy) {
  if (!policy || policy.kind !== 'split') return;
  const stateSize = policy.config.numLayers * policy.batchSize * policy.config.hiddenDim;
  policy.hiddenState = zeros(stateSize);
  policy.cellState = zeros(stateSize);
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
  if (nextHidden) policy.hiddenState = new Float32Array(nextHidden);
  if (nextCell) policy.cellState = new Float32Array(nextCell);

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
      self.ort.env.wasm.wasmPaths = wasmBaseUrl;
      self.ort.env.wasm.numThreads = 1;
      self.ort.env.wasm.proxy = false;
      configs = toConfigMap(data.policies);
      self.postMessage({ type: 'ready' });
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
      const policy = await loadPolicy(data.policyId);
      const startedAt = performance.now();
      const obs = new Float32Array(data.obs);
      const raw = policy.kind === 'split'
        ? await runSplit(policy, obs, data.dims)
        : await runMlp(policy, obs, data.dims);
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
