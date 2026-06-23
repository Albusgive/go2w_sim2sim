# RayCasterCamera WASM Port

This directory contains the browser-native port path for
`utils/mujoco_ray_caster/RayCasterCamera`.

The current GitHub Pages demo already uses MuJoCo WASM `mj_ray` as a stable
fallback, but that path still loops over rays in JavaScript. The files here
prepare the stricter port: compile the C++ `RayCasterCamera` class into the same
MuJoCo WASM module and expose a small embind API to JavaScript.

## Why Same-Module WASM

`RayCasterCamera` needs the live `mjModel*` and `mjData*`. A separate WASM
module cannot safely access those pointers from the MuJoCo module, and copying
state to a worker-owned model would risk diverging from the sim state. The
binding include is therefore injected into MuJoCo's generated
`wasm/codegen/generated/bindings.cc`, where the `MjModel` and `MjData` wrapper
classes are visible.

## Build Prerequisites

MuJoCo's WASM README expects Emscripten SDK `4.0.10`.

```sh
git clone https://github.com/emscripten-core/emsdk.git
./emsdk/emsdk install 4.0.10
./emsdk/emsdk activate 4.0.10
source ./emsdk/emsdk_env.sh
```

Then check this repo's port prerequisites:

```sh
node tools/raycaster_wasm_port/prepare_mujoco_wasm_port.mjs
```

The command is read-only by default. To patch a local MuJoCo source tree:

```sh
node tools/raycaster_wasm_port/prepare_mujoco_wasm_port.mjs \
  --mujoco-root /home/albusgive2/software/mujoco \
  --raycaster-root /home/albusgive2/go2w_sim2sim/utils/mujoco_ray_caster \
  --patch
```

The script appends idempotent marker blocks to:

- `wasm/CMakeLists.txt`
- `wasm/codegen/generated/bindings.cc`

`utils/mujoco_ray_caster/` is ignored by this repository, so a clean checkout
can pass `--raycaster-root` to a local clone of the maintained raycaster source
instead of relying on that ignored directory.

After patching, run the printed `emcmake cmake` and `cmake --build` commands,
then copy `wasm/dist/mujoco_wasm.js` and `.wasm` into `docs/demo-assets/`.

## JavaScript API

The injected binding exports `mujoco.RayCasterCamera`:

```js
const raycam = new mujoco.RayCasterCamera(
  model,
  data,
  'RayCasterCamera',
  32,
  18,
  0.1,
  2.0,
  1.0,
  2.0,
  1.1547005,
  false,
  2,
  0.0,
  0.0,
  0.0,
);

raycam.compute(model, data);
const depth = raycam.depthView();
const hitPoints = raycam.hitPointView();
```

The class uses `RayCasterCamera::set_num_thread()`, so the threaded artifact
requires cross-origin isolation (`SharedArrayBuffer`). The demo already
registers a COOP/COEP service worker for GitHub Pages.
