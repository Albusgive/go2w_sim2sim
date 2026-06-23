(function installGo2WRaycasterRuntimePatch() {
  const VERSION = 'preload-ray-15';
  const SCRIPT_URL = document.currentScript?.src || window.location.href;
  const RAY_WIDTH = 32;
  const RAY_HEIGHT = 18;
  const RAY_SIZE = RAY_WIDTH * RAY_HEIGHT;
  const RAY_MIN_DIST = 0.1;
  const RAY_MAX_DIST = 2.0;
  const RAY_FOCAL = 1.0;
  const RAY_HORIZONTAL_APERTURE = 2.0;
  const RAY_VERTICAL_APERTURE = 1.1547005;
  const RAY_UPDATE_DT = 0.02;
  const RAY_UPDATE_WALL_MS = 50;
  const MAX_BASE_XY = 32.0;
  const MAX_BASE_Y = 14.0;
  const MAX_QVEL = 60;
  const MAX_CTRL = 90;

  const ASSET_NAMES = [
    'base_0.obj',
    'base_1.obj',
    'base_2.obj',
    'base_3.obj',
    'base_4.obj',
    'hip_0.obj',
    'hip_1.obj',
    'thigh_0.obj',
    'thigh_1.obj',
    'thigh_mirror_0.obj',
    'thigh_mirror_1.obj',
    'calf_mirror_0.obj',
    'calf_mirror_1.obj',
    'calf.stl',
    'calf_mirror.stl',
    'foot.obj',
    'wheel.stl',
    '336L_with_base.STL',
    '336L_with_safe_base.STL',
    'multi_motion_float_box_terrain_mesh1.stl',
    'multi_motion_float_box_terrain_mesh2.stl',
    'multi_motion_float_box_terrain_mesh3.stl',
    'multi_motion_float_box_terrain_mesh4.stl',
    'multi_motion_float_box_terrain_mesh5.stl',
    'multi_motion_float_box_terrain_mesh6.stl',
    'multi_motion_float_box_terrain_mesh7.stl',
    'multi_motion_float_box_terrain_mesh8.stl',
  ];

  function normalizeDepth(value) {
    if (!Number.isFinite(value) || value < RAY_MIN_DIST || value > RAY_MAX_DIST) return 0;
    return (value - RAY_MIN_DIST) / (RAY_MAX_DIST - RAY_MIN_DIST);
  }

  function maxAbs(values) {
    let max = 0;
    if (!values) return max;
    for (const value of values) {
      const abs = Math.abs(value);
      if (!Number.isFinite(abs)) return Infinity;
      if (abs > max) max = abs;
    }
    return max;
  }

  function basePosition(demo) {
    if (!demo?.data || demo.baseBodyId < 0) return null;
    const offset = demo.baseBodyId * 3;
    return [
      demo.data.xpos[offset],
      demo.data.xpos[offset + 1],
      demo.data.xpos[offset + 2],
    ];
  }

  function unsafeReason(demo) {
    const base = basePosition(demo);
    if (base) {
      if (!base.every(Number.isFinite)) return 'non-finite base';
      if (Math.abs(base[0]) > MAX_BASE_XY || Math.abs(base[1]) > MAX_BASE_Y) {
        return 'base left demo terrain';
      }
    }
    if (maxAbs(demo.data?.qpos) === Infinity || maxAbs(demo.data?.qvel) === Infinity) {
      return 'non-finite state';
    }
    if (maxAbs(demo.data?.qvel) > MAX_QVEL) return 'excessive velocity';
    if (maxAbs(demo.data?.ctrl) > MAX_CTRL) return 'excessive control';
    return null;
  }

  function snapFollowCameraToBase(demo) {
    const base = basePosition(demo);
    if (!base || !demo.camera || !demo.controls) return;
    demo.controls.target.set(base[0], base[2] + 0.22, -base[1]);
    demo.camera.position.set(base[0] - 1.55, base[2] + 1.05, -base[1] + 1.65);
    demo.controls.update();
  }

  function ensureRayWorkerFields(demo) {
    demo.rayBackend ||= 'three';
    demo.rayValidCount ||= 0;
    demo.rayCenterDepth ||= 0;
    demo.rayWorker ||= null;
    demo.rayWorkerReady ||= false;
    demo.rayWorkerPending ||= false;
    demo.rayWorkerFailed ||= null;
    demo.rayWorkerSeq ||= 0;
    demo.rayWorkerAppliedSeq ||= 0;
    demo.lastRayWorkerDurationMs ||= 0;
    demo.rayWorkerBackend ||= null;
    demo.rayLastUpdateWallTime ||= 0;
    demo.rayUpdateWallMs ||= RAY_UPDATE_WALL_MS;
  }

  function startRaycasterWorker() {
    ensureRayWorkerFields(this);
    if (this.rayWorker || this.rayWorkerFailed || typeof Worker !== 'function') return;
    try {
      this.rayWorker = new Worker(new URL(`raycaster-worker.js?v=${VERSION}`, SCRIPT_URL), {
        type: 'module',
        name: 'go2w-raycaster-worker',
      });
    } catch (error) {
      this.rayWorkerFailed = error?.message || String(error);
      this.rayWorker = null;
      return;
    }

    this.rayWorker.onmessage = (event) => this.handleRaycasterWorkerMessage(event.data || {});
    this.rayWorker.onerror = (event) => {
      this.rayWorkerFailed = event.message || 'raycaster worker failed';
      this.rayWorkerReady = false;
      this.rayWorkerPending = false;
      this.rayWorker?.terminate?.();
      this.rayWorker = null;
    };
    this.rayWorker.postMessage({
      type: 'init',
      config: {
        assetNames: ASSET_NAMES,
        width: RAY_WIDTH,
        height: RAY_HEIGHT,
        near: RAY_MIN_DIST,
        far: RAY_MAX_DIST,
        focal: RAY_FOCAL,
        horizontalAperture: RAY_HORIZONTAL_APERTURE,
        verticalAperture: RAY_VERTICAL_APERTURE,
        numThreads: 0,
      },
    });
  }

  function handleRaycasterWorkerMessage(data) {
    ensureRayWorkerFields(this);
    if (data.type === 'ready') {
      this.rayWorkerReady = true;
      this.rayWorkerBackend = data.backend || 'worker-RayCasterCamera';
      this.rayBackend = this.rayWorkerBackend;
      this.requestRaycasterWorker(true);
      return;
    }
    if (data.type === 'error') {
      this.rayWorkerFailed = data.message || 'raycaster worker failed';
      this.rayWorkerReady = false;
      this.rayWorkerPending = false;
      return;
    }
    if (data.type !== 'result') return;
    this.rayWorkerPending = false;
    if (data.seq < this.rayWorkerAppliedSeq) return;
    this.rayWorkerAppliedSeq = data.seq;
    this.lastRayDurationMs = data.durationMs || 0;
    this.lastRayWorkerDurationMs = this.lastRayDurationMs;
    this.rayBackend = data.backend || this.rayWorkerBackend || 'worker-RayCasterCamera';

    const depth = data.depth || [];
    const hitPoints = data.hitPoints || [];
    let validCount = 0;
    for (let index = 0; index < RAY_SIZE; index += 1) {
      const value = Number(depth[index]) || 0;
      this.rayRawImage[index] = value;
      this.rayImage[index] = normalizeDepth(value);
      if (value > 0) {
        validCount += 1;
        const p = index * 3;
        this.rayHitPoints[index] = Number.isFinite(hitPoints[p])
          ? [hitPoints[p], hitPoints[p + 1], hitPoints[p + 2]]
          : null;
      } else {
        this.rayHitPoints[index] = null;
      }
    }
    const centerIndex = Math.floor(RAY_HEIGHT / 2) * RAY_WIDTH + Math.floor(RAY_WIDTH / 2);
    this.rayValidCount = Number(data.validCount) || validCount;
    this.rayCenterDepth = this.rayRawImage[centerIndex] || 0;
    this.rayLastUpdateTime = Number.isFinite(data.simTime) ? data.simTime : this.data?.time || 0;
    this.rayLastUpdateWallTime = performance.now();
    // Update the ray-line origin to the current camera pose. Only the
    // synchronous raycast paths set lastRayPose; on the async worker path (the
    // default backend) it would otherwise stay frozen at the spawn pose, making
    // the debug rays appear to emanate from the world origin instead of the
    // robot's camera. cameraPoseMujoco() reads the live cam_xpos/cam_xmat.
    if (typeof this.cameraPoseMujoco === 'function') {
      const pose = this.cameraPoseMujoco();
      if (pose) this.lastRayPose = pose;
    }
    this.rayDirty = true;
  }

  // True while the async worker is the live backend: it either already exists
  // (initializing/ready) or can still be (re)started, and has not permanently
  // failed. When this is true we must NOT run the synchronous 576-ray CPU loop
  // on the main thread every frame — we return the last image instead.
  function rayWorkerUsable(demo) {
    if (demo.rayWorkerFailed) return false;
    if (typeof Worker !== 'function') return false;
    return true;
  }

  function requestRaycasterWorker(force = false) {
    ensureRayWorkerFields(this);
    if (!this.rayWorker || !this.rayWorkerReady || this.rayWorkerPending || this.rayWorkerFailed) {
      return false;
    }
    if (!this.data || !this.model) return false;
    const now = performance.now();
    const wallMs = this.rayUpdateWallMs || this.__rayOptimizerWallMs || RAY_UPDATE_WALL_MS;
    if (!force && this.data.time - this.rayLastUpdateTime < RAY_UPDATE_DT) return true;
    if (!force && now - (this.rayLastUpdateWallTime || 0) < wallMs) return true;
    const qpos = new Float64Array(this.data.qpos);
    const qvel = new Float64Array(this.data.qvel);
    const seq = this.rayWorkerSeq + 1;
    this.rayWorkerSeq = seq;
    this.rayWorkerPending = true;
    this.rayBackend = this.rayWorkerBackend || 'worker-RayCasterCamera';
    this.rayWorker.postMessage({
      type: 'compute',
      seq,
      simTime: this.data.time,
      qpos,
      qvel,
    }, [qpos.buffer, qvel.buffer]);
    return true;
  }

  function patch(demo) {
    if (!demo || demo.__raycasterRuntimePatchVersion === VERSION) return Boolean(demo);
    demo.__raycasterRuntimePatchVersion = VERSION;
    ensureRayWorkerFields(demo);

    demo.startRaycasterWorker = startRaycasterWorker;
    demo.handleRaycasterWorkerMessage = handleRaycasterWorkerMessage;
    demo.requestRaycasterWorker = requestRaycasterWorker;

    demo.needsSafetyReset = function needsSafetyResetRayWorkerPatch() {
      const reason = unsafeReason(this);
      this.__unsafeReason = reason;
      return Boolean(reason);
    };
    demo.needsVisualGuard = function needsVisualGuardRayWorkerPatch() {
      if (!this.data || this.baseBodyId < 0) return false;
      const base = basePosition(this);
      return Boolean(base && !base.every(Number.isFinite)) || this.needsSafetyReset();
    };

    const originalSetFollowCamera = demo.setFollowCamera?.bind(demo);
    demo.setFollowCamera = function setFollowCameraRayWorkerPatch() {
      originalSetFollowCamera?.(true);
      this.followCamera = true;
      snapFollowCameraToBase(this);
    };
    demo.snapFollowCameraToBase = function snapFollowCameraRayWorkerPatch() {
      snapFollowCameraToBase(this);
    };

    const originalFrame = demo.frame?.bind(demo);
    if (originalFrame && !demo.__rayPatchFrameWrapped) {
      demo.__rayPatchFrameWrapped = true;
      demo.frame = function frameRayWorkerPatch() {
        this.followCamera = true;
        const result = originalFrame();
        this.followCamera = true;
        return result;
      };
    }

    const originalRefresh = demo.refreshRaycasterImage?.bind(demo);
    if (originalRefresh && !demo.__rayPatchRefreshWrapped) {
      demo.__rayPatchRefreshWrapped = true;
      demo.refreshRaycasterImage = function refreshRaycasterImageRayWorkerPatch(force = false) {
        // A forced one-shot (reset seeding visual history, etc.) needs a fresh
        // image RIGHT NOW. The async worker would only return the stale image
        // and compute later, desyncing visualImageHistory, so compute
        // synchronously this once. __rayForceSync makes the (also patched)
        // computeRaycasterImage skip its own worker routing for this call.
        if (force) {
          this.__rayForceSync = true;
          try {
            return originalRefresh(true);
          } finally {
            this.__rayForceSync = false;
          }
        }
        if (this.requestRaycasterWorker(false)) return this.rayImage;
        // The worker accepted no request. While it is still the live backend
        // (initializing or a request is in flight) keep returning the last
        // image instead of blocking the main thread with the synchronous
        // 576-ray CPU loop. Only a permanent worker failure may fall through to
        // the synchronous path.
        if (rayWorkerUsable(this)) return this.rayImage;
        return originalRefresh(false);
      };
    }

    const originalCompute = demo.computeRaycasterImage?.bind(demo);
    if (originalCompute && !demo.__rayPatchComputeWrapped) {
      demo.__rayPatchComputeWrapped = true;
      demo.computeRaycasterImage = function computeRaycasterImageRayWorkerPatch() {
        // During a forced synchronous refresh, compute on the main thread so
        // the caller gets a fresh image instead of a worker-deferred stale one.
        if (this.__rayForceSync) return originalCompute();
        if (this.requestRaycasterWorker(false)) return this.rayImage;
        if (rayWorkerUsable(this)) return this.rayImage;
        return originalCompute();
      };
    }

    const originalUpdateRayVisualization = demo.updateRayVisualization?.bind(demo);
    if (originalUpdateRayVisualization && !demo.__rayPatchVisualWrapped) {
      demo.__rayPatchVisualWrapped = true;
      demo.updateRayVisualization = function updateRayVisualizationRayWorkerPatch() {
        originalUpdateRayVisualization();
        if (this.rayReadout) {
          const count = this.rayValidCount || this.rayImage.reduce((acc, value) => acc + (value > 0 ? 1 : 0), 0);
          const ms = Number(this.lastRayDurationMs || this.lastRayWorkerDurationMs || 0).toFixed(1);
          this.rayReadout.textContent = `${RAY_WIDTH} x ${RAY_HEIGHT} depth · ${count} hits · ${ms}ms`;
        }
      };
    }

    const originalUpdateRuntimeStats = demo.updateRuntimeStats?.bind(demo);
    if (originalUpdateRuntimeStats && !demo.__rayPatchStatsWrapped) {
      demo.__rayPatchStatsWrapped = true;
      demo.updateRuntimeStats = function updateRuntimeStatsRayWorkerPatch() {
        originalUpdateRuntimeStats();
        if (!window.__go2wRuntime) return;
        window.__go2wRuntime.optimizerVersion = VERSION;
        window.__go2wRuntime.rayBackend = this.rayBackend || null;
        window.__go2wRuntime.rayHits = this.rayValidCount || 0;
        window.__go2wRuntime.rayCenterDepth = this.rayCenterDepth || 0;
        window.__go2wRuntime.rayMs = this.lastRayDurationMs || this.lastRayWorkerDurationMs || 0;
        window.__go2wRuntime.rayWorkerReady = this.rayWorkerReady || false;
        window.__go2wRuntime.rayWorkerPending = this.rayWorkerPending || false;
        window.__go2wRuntime.rayWorkerMs = this.lastRayWorkerDurationMs || 0;
        window.__go2wRuntime.rayWorkerFailure = this.rayWorkerFailed || null;
        window.__go2wRuntime.unsafeReason = this.__unsafeReason || null;
        window.__go2wRuntime.followCamera = true;
      };
    }

    demo.startRaycasterWorker();
    return true;
  }

  function waitForDemo() {
    if (!patch(window.__go2wDemo)) window.setTimeout(waitForDemo, 10);
  }

  waitForDemo();
}());
