# Go2W Sim2Sim Pages

This directory is the GitHub Pages root for the project page and the browser
`lab2mj` demo.

The demo runs the Go2W MJCF scene with MuJoCo WASM and drives the robot through
ONNX Runtime Web. ROS2, `real2sim`, and the native C++ process are not deployed
here. The default browser policy is `vtm_lstm_sru`; the demo also exposes
`motion_mlp`, `vtm`, and `vtm_gru_sru`. Visual policies receive a browser
RayCasterCamera depth image (`32 x 18`) and show the ray image in the bottom UI.

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
