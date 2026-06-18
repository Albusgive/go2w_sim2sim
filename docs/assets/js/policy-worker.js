self.importScripts('../vendor/onnxruntime-web/ort.wasm.min.js');

let session = null;
let inputName = '';
let outputName = '';

function postError(seq, error) {
  self.postMessage({
    type: 'error',
    seq,
    message: error?.message || String(error),
  });
}

self.onmessage = async (event) => {
  const data = event.data || {};
  try {
    if (data.type === 'init') {
      const wasmBaseUrl = new URL('../vendor/onnxruntime-web/', self.location.href).href;
      self.ort.env.wasm.wasmPaths = wasmBaseUrl;
      self.ort.env.wasm.numThreads = 1;
      self.ort.env.wasm.proxy = false;

      const response = await fetch(data.policyUrl);
      if (!response.ok) throw new Error(`${response.status} ${response.url}`);
      const modelBuffer = await response.arrayBuffer();

      session = await self.ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      inputName = session.inputNames[0];
      outputName = session.outputNames[0];
      self.postMessage({ type: 'ready', inputName, outputName });
      return;
    }

    if (data.type === 'run') {
      if (!session) throw new Error('motion_mlp worker session is not ready');
      const startedAt = performance.now();
      const obs = new Float32Array(data.obs);
      const feeds = {
        [inputName]: new self.ort.Tensor('float32', obs, data.dims),
      };
      const results = await session.run(feeds);
      const raw = results[outputName]?.data;
      if (!raw || raw.length < 16) {
        throw new Error('motion_mlp returned an invalid action tensor');
      }
      const action = new Float32Array(16);
      for (let i = 0; i < 16; i += 1) {
        action[i] = Number.isFinite(raw[i]) ? raw[i] : 0;
      }
      self.postMessage({
        type: 'result',
        seq: data.seq,
        action,
        durationMs: performance.now() - startedAt,
      }, [action.buffer]);
    }
  } catch (error) {
    postError(data.seq, error);
  }
};
