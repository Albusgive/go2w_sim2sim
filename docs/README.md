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
- ONNX Runtime Web runs in a dedicated Worker. The worker uses one WASM thread
  on GitHub Pages and automatically opts into two ONNX threads only when the
  page is cross-origin isolated and `SharedArrayBuffer` is available.
- The browser RayCasterCamera uses the verified `-z` local camera convention,
  wall-clock throttling, and adaptive backoff when ray updates become slow.
- The main runtime exports frame, ray, policy, safety, and thread diagnostics to
  `window.__go2wRuntime` for browser-side testing.

The native C++ RayCasterCamera port is tracked separately in
[`raycaster-wasm-port.md`](raycaster-wasm-port.md).

## Local Preview

```bash
cd docs
python3 -m http.server 8000
```

Open `http://localhost:8000/demo.html`.

## GitHub Pages

In the repository settings, configure Pages to deploy from:

```text
Branch: main
Folder: /docs
```

The browser demo is static and uses only the assets checked into this directory:
MuJoCo WASM, Three.js, ONNX Runtime Web, the Go2W MJCF assets, and the
ONNX checkpoints under `docs/demo-assets/policies/`.
