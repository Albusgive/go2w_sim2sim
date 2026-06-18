# Go2W Sim2Sim Pages

This directory is the GitHub Pages root for the project page and the browser
`lab2mj` demo.

The demo runs the Go2W MJCF scene with MuJoCo WASM and drives the robot with the
`motion_mlp` ONNX policy through ONNX Runtime Web. ROS2, `real2sim`, and the
native C++ process are not deployed here. The visual VTM/SRU policies are
intentionally disabled until the browser ray-image pipeline is ported.

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
`policy/motion_tracking/policy.onnx` checkpoint.
