# Go2W Sim2Sim Pages

This directory is the GitHub Pages root for the project page and the browser
`lab2mj` demo.

The demo runs the Go2W MJCF scene with MuJoCo WASM and drives the robot through
ONNX Runtime Web. ROS2, `real2sim`, and the native C++ process are not deployed
here. The browser starts from `motion_mlp`, preloads all available policy slots,
and can switch to `vtm`, `vtm_lstm_sru`, and `vtm_gru_sru` without fetching or
compiling ONNX graphs during the run. Visual policies receive a browser
RayCasterCamera depth image (`32 x 18`) and show the ray image in the bottom UI.

## Runtime Optimizations

- Policy sessions are preloaded during startup so policy switching does not
  lazy-load large ONNX files while the simulation is already running.
- ONNX Runtime Web runs in a dedicated Worker. GitHub Pages uses
  `coi-serviceworker.js` to inject COOP/COEP headers, so the page reloads once
  on first visit and then opts into two ONNX WASM threads when
  `SharedArrayBuffer` is available.
- The browser RayCasterCamera uses MuJoCo WASM `mj_ray`, the verified `-z`
  local camera convention, wall-clock throttling, and adaptive backoff when ray
  updates become slow.
- If a custom MuJoCo WASM artifact exports `mujoco.RayCasterCamera`, the demo
  automatically switches to that native binding and reports the threaded native
  backend in `window.__go2wRuntime.rayBackend`.
- The main runtime exports frame, ray, policy, safety, and thread diagnostics to
  `window.__go2wRuntime` for browser-side testing.

The native C++ RayCasterCamera port scaffold and build preflight are tracked in
[`raycaster-wasm-port.md`](raycaster-wasm-port.md) and
[`../tools/raycaster_wasm_port/`](../tools/raycaster_wasm_port/).

## Local Preview

```bash
cd docs
python3 -m http.server 8000
```

Open `http://localhost:8000/demo.html`.

## Verification

The browser demo can be smoke-tested with Firefox/geckodriver:

```bash
node tools/verify_go2w_pages_demo.mjs \
  --url https://albusgive.github.io/go2w_sim2sim/demo.html \
  --screenshot /tmp/go2w-pages-demo.png
```

For a local preview:

```bash
node tools/verify_go2w_pages_demo.mjs --local docs
```

The verifier checks optimizer version, policy preloading, ONNX thread count,
the expected ray backend, nonzero RayCaster hits, policy switching, follow
camera, runtime error state, and a short long-run window. Until a native
RayCasterCamera WASM artifact is built, the expected backend remains
`mujoco-mj_ray`.

## GitHub Pages

In the repository settings, configure Pages to deploy from:

```text
Branch: main
Folder: /docs
```

The browser demo is static and uses only the assets checked into this directory:
MuJoCo WASM, Three.js, ONNX Runtime Web, the Go2W MJCF assets, and the
ONNX checkpoints under `docs/demo-assets/policies/`.
