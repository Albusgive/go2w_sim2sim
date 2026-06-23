#!/usr/bin/env node
import fs from 'node:fs';
import { spawn } from 'node:child_process';

const DEFAULTS = {
  url: 'https://albusgive.github.io/go2w_sim2sim/demo.html',
  geckodriverPort: 4446,
  expectVersion: 'preload-ray-13',
  expectRayBackend: 'mujoco-mj_ray',
  minRayHits: 100,
  minOnnxThreads: 2,
  policyId: 2,
  readyTimeoutMs: 180000,
  preloadTimeoutMs: 120000,
  switchWaitMs: 12000,
  longRunMs: 30000,
  screenshot: '/tmp/go2w-pages-demo.png',
};

function parseArgs(argv) {
  const out = { ...DEFAULTS, startGeckodriver: true, serveDir: null, servePort: 8088 };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`Missing value for ${arg}`);
      i += 1;
      return argv[i];
    };
    if (arg === '--url') out.url = next();
    else if (arg === '--local') out.serveDir = next();
    else if (arg === '--port') out.servePort = Number(next());
    else if (arg === '--geckodriver-port') out.geckodriverPort = Number(next());
    else if (arg === '--no-start-geckodriver') out.startGeckodriver = false;
    else if (arg === '--expect-version') out.expectVersion = next();
    else if (arg === '--expect-ray-backend') out.expectRayBackend = next();
    else if (arg === '--min-ray-hits') out.minRayHits = Number(next());
    else if (arg === '--min-onnx-threads') out.minOnnxThreads = Number(next());
    else if (arg === '--policy-id') out.policyId = Number(next());
    else if (arg === '--ready-timeout-ms') out.readyTimeoutMs = Number(next());
    else if (arg === '--preload-timeout-ms') out.preloadTimeoutMs = Number(next());
    else if (arg === '--switch-wait-ms') out.switchWaitMs = Number(next());
    else if (arg === '--long-run-ms') out.longRunMs = Number(next());
    else if (arg === '--screenshot') out.screenshot = next();
    else if (arg === '--help') {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (out.serveDir) out.url = `http://127.0.0.1:${out.servePort}/demo.html`;
  return out;
}

function printUsage() {
  console.log(`Usage:
  node tools/verify_go2w_pages_demo.mjs [options]

Options:
  --url URL                       Demo URL to test.
  --local DIR                     Start python3 -m http.server for DIR and test it.
  --port N                        Local server port when using --local. Default: 8088.
  --geckodriver-port N            WebDriver port. Default: 4446.
  --no-start-geckodriver          Reuse an already running geckodriver.
  --expect-version VERSION        Expected optimizer version. Default: preload-ray-13.
  --expect-ray-backend BACKEND    Expected ray backend. Default: mujoco-mj_ray.
  --min-ray-hits N                Minimum valid ray hits. Default: 100.
  --min-onnx-threads N            Minimum ONNX worker threads. Default: 2.
  --policy-id ID                  Policy button id to switch to. Default: 2.
  --switch-wait-ms N              Wait after policy switch. Default: 12000.
  --long-run-ms N                 Additional runtime after switch. Default: 30000.
  --screenshot PATH               Screenshot output path. Default: /tmp/go2w-pages-demo.png.
`);
}

function spawnProcess(command, args, options = {}) {
  const child = spawn(command, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    ...options,
  });
  child.stdout.on('data', (chunk) => process.stderr.write(chunk));
  child.stderr.on('data', (chunk) => process.stderr.write(chunk));
  return child;
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHttp(url, timeoutMs, label) {
  const start = Date.now();
  let lastError = null;
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
      lastError = new Error(`${response.status} ${response.statusText}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  throw new Error(`${label} did not become ready: ${lastError?.message || 'unknown error'}`);
}

function withCacheBuster(url) {
  const parsed = new URL(url);
  parsed.searchParams.set('verify', `go2w-${Date.now()}`);
  return parsed.href;
}

function webdriverClient(port) {
  const base = `http://127.0.0.1:${port}`;
  async function wd(path, method = 'GET', body = null) {
    const response = await fetch(`${base}${path}`, {
      method,
      headers: body ? { 'content-type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await response.text();
    let payload;
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(`${method} ${path} returned non-JSON ${response.status}: ${text.slice(0, 200)}`);
    }
    if (!response.ok) throw new Error(`${method} ${path} ${response.status}: ${text}`);
    return payload.value;
  }
  async function execute(sessionId, script, args = []) {
    return wd(`/session/${sessionId}/execute/sync`, 'POST', { script, args });
  }
  return { wd, execute };
}

async function waitForCondition(execute, sessionId, script, timeoutMs, label) {
  const start = Date.now();
  let last = null;
  while (Date.now() - start < timeoutMs) {
    last = await execute(sessionId, script);
    if (last?.ok) return last.value;
    await sleep(500);
  }
  throw new Error(`${label} timed out; last=${JSON.stringify(last)}`);
}

function assertRuntime(snapshot, options, phase) {
  const rt = snapshot.rt || {};
  const failures = [];
  if (snapshot.status !== 'Ready') failures.push(`status=${snapshot.status}`);
  if (snapshot.error) failures.push(`error=${snapshot.error}`);
  if (rt.optimizerVersion !== options.expectVersion) failures.push(`version=${rt.optimizerVersion}`);
  if (rt.preloadComplete !== true) failures.push('preloadComplete=false');
  if (rt.policyLoadErrors?.length) failures.push(`policyLoadErrors=${rt.policyLoadErrors.join(',')}`);
  if (rt.threadCaps?.onnxThreads < options.minOnnxThreads) {
    failures.push(`onnxThreads=${rt.threadCaps?.onnxThreads}`);
  }
  if (rt.rayBackend !== options.expectRayBackend) failures.push(`rayBackend=${rt.rayBackend}`);
  if (rt.rayHits < options.minRayHits) failures.push(`rayHits=${rt.rayHits}`);
  if (rt.followCamera !== true) failures.push(`followCamera=${rt.followCamera}`);
  if (rt.frameError) failures.push(`frameError=${rt.frameError}`);
  if (failures.length) {
    throw new Error(`${phase} failed: ${failures.join('; ')}`);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const children = [];
  const { wd, execute } = webdriverClient(options.geckodriverPort);
  let sessionId = null;

  try {
    if (options.serveDir) {
      const server = spawnProcess('python3', ['-m', 'http.server', String(options.servePort)], {
        cwd: options.serveDir,
      });
      children.push(server);
      await waitForHttp(`http://127.0.0.1:${options.servePort}/demo.html`, 10000, 'local HTTP server');
    }

    if (options.startGeckodriver) {
      const gecko = spawnProcess('geckodriver', ['--port', String(options.geckodriverPort)]);
      children.push(gecko);
      await waitForHttp(`http://127.0.0.1:${options.geckodriverPort}/status`, 10000, 'geckodriver');
    }

    const session = await wd('/session', 'POST', {
      capabilities: {
        alwaysMatch: {
          browserName: 'firefox',
          'moz:firefoxOptions': { args: ['-headless'] },
        },
      },
    });
    sessionId = session.sessionId;
    await wd(`/session/${sessionId}/url`, 'POST', { url: withCacheBuster(options.url) });

    const ready = await waitForCondition(execute, sessionId, `
      const rt = window.__go2wRuntime || null;
      const status = document.querySelector('.status-pill')?.textContent || '';
      return { ok: !!rt && rt.optimizerVersion === '${options.expectVersion}' && status !== 'Loading',
        value: {
          status,
          error: document.getElementById('error-readout')?.hidden ? null : document.getElementById('error-readout')?.textContent,
          readout: document.getElementById('ray-readout')?.textContent || '',
          crossOriginIsolated: window.crossOriginIsolated,
          serviceWorker: !!navigator.serviceWorker?.controller,
          scripts: [...document.scripts].map((script) => script.src).filter(Boolean),
          rt,
        }
      };
    `, options.readyTimeoutMs, 'initial ready');

    await waitForCondition(execute, sessionId, `
      const rt = window.__go2wRuntime || null;
      return { ok: !!rt && rt.preloadComplete === true,
        value: { status: document.querySelector('.status-pill')?.textContent || '', rt }
      };
    `, options.preloadTimeoutMs, 'policy preload');

    assertRuntime(ready, options, 'initial ready');

    await execute(sessionId, `document.querySelector('[data-policy="${options.policyId}"]')?.click(); return true;`);
    await sleep(options.switchWaitMs);
    const afterSwitch = await execute(sessionId, `
      return {
        status: document.querySelector('.status-pill')?.textContent || '',
        error: document.getElementById('error-readout')?.hidden ? null : document.getElementById('error-readout')?.textContent,
        readout: document.getElementById('ray-readout')?.textContent || '',
        rt: window.__go2wRuntime || null,
      };
    `);
    assertRuntime(afterSwitch, options, 'after policy switch');
    if (afterSwitch.rt.policyId !== options.policyId) {
      throw new Error(`after policy switch failed: policyId=${afterSwitch.rt.policyId}`);
    }

    await sleep(options.longRunMs);
    const longRun = await execute(sessionId, `
      return {
        status: document.querySelector('.status-pill')?.textContent || '',
        error: document.getElementById('error-readout')?.hidden ? null : document.getElementById('error-readout')?.textContent,
        readout: document.getElementById('ray-readout')?.textContent || '',
        rt: window.__go2wRuntime || null,
      };
    `);
    assertRuntime(longRun, options, 'long run');
    if (longRun.rt.policyId !== options.policyId) {
      throw new Error(`long run failed: policyId=${longRun.rt.policyId}`);
    }

    const png = await wd(`/session/${sessionId}/screenshot`, 'GET');
    fs.writeFileSync(options.screenshot, Buffer.from(png, 'base64'));

    console.log(JSON.stringify({
      ok: true,
      url: options.url,
      screenshot: options.screenshot,
      ready,
      afterSwitch,
      longRun,
    }, null, 2));
  } finally {
    if (sessionId) await wd(`/session/${sessionId}`, 'DELETE').catch(() => {});
    for (const child of children.reverse()) child.kill('SIGINT');
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
