# RayCasterCamera WASM Port Plan

Older browser builds approximated `RayCasterCamera` in JavaScript with Three.js
terrain ray casts. The current shim first uses the MuJoCo WASM `mj_ray` export,
so occlusion follows MJCF collision geometry and excludes the camera parent body
like the native lab2mj plugin. It is still not the same implementation as
`utils/mujoco_ray_caster/RayCasterCamera`, because the current browser binding
does not expose the newer normal-output ray overload used by the plugin for
loss-angle and stereo energy filtering.

The exact port should be implemented as a MuJoCo WASM extension instead of a
runtime JavaScript patch.

## Target Architecture

1. Build a small Emscripten target that links MuJoCo WASM with the C++
   RayCasterCamera sources from lab2mj. The local source used for comparison is
   `/home/albusgive2/mujoco_ray_caster/raycaster_src/RayCasterCamera.*`.
2. Expose a stable C or embind API:
   - create and destroy a camera object
   - configure width, height, near/far range, camera frame, aperture, and noise
   - compute one depth frame from `mjModel*` and `mjData*`
   - return the depth image through a preallocated WASM heap buffer
3. Keep the JS demo API narrow:
   - initialize the native raycaster after MJCF compilation
   - pass the current model/data handles each policy tick
   - copy only the final `32 x 18` `Float32Array` depth image to JS
4. Build two browser artifacts:
   - single-threaded WASM for GitHub Pages compatibility
   - pthread-enabled WASM for hosts that serve COOP/COEP headers

## Deployment Requirements

GitHub Pages is suitable for the single-threaded artifact. The pthread-enabled
artifact requires cross-origin isolation, which means the host must serve, or a
service worker must inject:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without those headers, browsers disable `SharedArrayBuffer`, and Emscripten
pthreads cannot start reliably. The demo includes a COOP/COEP service worker for
GitHub Pages, so the first visit may reload once before ONNX Runtime can choose
the threaded WASM backend.

## Validation Gates

- Compare the WASM RayCasterCamera output against the native lab2mj output for
  the same XML, qpos, qvel, and camera settings.
- Verify the center ray, corner rays, min/max distance clipping, and invalid-hit
  encoding.
- Run a long browser test through all four policy slots with:
  - no runtime abort
  - follow camera enabled after every policy switch
  - stable `window.__go2wRuntime.rayMs`
  - no policy fetch or ONNX session creation after startup
