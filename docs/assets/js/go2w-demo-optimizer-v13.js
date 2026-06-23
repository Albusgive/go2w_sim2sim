(function installGo2WDemoOptimizer() {
  const VERSION = 'preload-ray-13';
  const RAY_WIDTH = 32;
  const RAY_HEIGHT = 18;
  const RAY_MIN_DIST = 0.1;
  const RAY_MAX_DIST = 2.0;
  const RAY_FOCAL = 1.0;
  const RAY_UPDATE_WALL_MS_MIN = 50;
  const RAY_UPDATE_WALL_MS_MAX = 140;
  const RAY_SLOW_FRAME_MS = 8;
  const RAY_FAST_FRAME_MS = 3;
  const MAX_BASE_XY = 7.0;
  const MAX_BASE_Y = 4.8;
  const MIN_SAFE_BASE_Z = 0.25;
  const MAX_QVEL = 60;
  const MAX_CTRL = 90;
  const RECOVERY_COOLDOWN_MS = 1200;
  const RAY_GEOMGROUP = [1, 1, 0, 0, 0, 0];

  function normalize3(vec) {
    const len = Math.hypot(vec[0], vec[1], vec[2]);
    if (len <= 1.0e-9) return [0, 0, 0];
    return [vec[0] / len, vec[1] / len, vec[2] / len];
  }

  function dot3(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

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

  function detectThreadCapabilities() {
    const hardwareConcurrency = Math.max(1, Math.floor(navigator.hardwareConcurrency || 1));
    const sharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
    const crossOriginIsolated = window.crossOriginIsolated === true;
    const wasmPthreads = sharedArrayBuffer && crossOriginIsolated;
    return {
      hardwareConcurrency,
      sharedArrayBuffer,
      crossOriginIsolated,
      wasmPthreads,
      onnxThreads: wasmPthreads ? Math.min(2, hardwareConcurrency) : 1,
    };
  }

  const cameraConvention = Object.freeze({
    name: '-z',
    local: (u, v) => normalize3([u, v, -RAY_FOCAL]),
  });

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
      if (base[2] < MIN_SAFE_BASE_Z) return 'base too low';
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

  function syncCommandFromControls(demo) {
    if (!demo?.cmd) return;
    const cmdX = document.getElementById('cmd-x');
    const cmdY = document.getElementById('cmd-y');
    const cmdYaw = document.getElementById('cmd-yaw');
    demo.cmd.x = Number(cmdX?.value || 0);
    demo.cmd.y = Number(cmdY?.value || 0);
    demo.cmd.yaw = Number(cmdYaw?.value || 0);
  }

  function installLegacyRuntimePatch(demo) {
    demo.rayConvention = cameraConvention;
    syncCommandFromControls(demo);
    if (demo.mjvOption?.geomgroup && demo.mjvOption.geomgroup.length > 3) {
      demo.mjvOption.geomgroup[2] = 1;
      demo.mjvOption.geomgroup[3] = 1;
      demo.updateVisualScene?.();
    }
    demo.rayLocalDirsByConvention?.clear?.();
    demo.__lastRayOptimizerWallMs = 0;
    demo.__lastRayOptimizerDurationMs = 0;
    demo.__rayOptimizerWallMs = RAY_UPDATE_WALL_MS_MIN;
    demo.__lastFrameOptimizerDurationMs = 0;
    demo.__lastRecoveryAt = 0;
    demo.__recoveryCount = demo.__recoveryCount || 0;
    demo.__lastRecoveryReason = demo.__lastRecoveryReason || null;
    demo.__unsafeReason = null;

    const originalSetFollowCamera = demo.setFollowCamera?.bind(demo);
    if (originalSetFollowCamera && !demo.__optimizerSetFollowWrapped) {
      demo.__optimizerSetFollowWrapped = true;
      demo.setFollowCamera = function setFollowCameraOptimized(enabled) {
        originalSetFollowCamera(enabled);
        if (enabled) snapFollowCameraToBase(this);
      };
    }

    demo.snapFollowCameraToBase = function snapFollowCameraToBaseMethod() {
      snapFollowCameraToBase(this);
    };

    const originalResetBrowserPose = demo.resetBrowserPose?.bind(demo);
    if (originalResetBrowserPose && !demo.__optimizerResetWrapped) {
      demo.__optimizerResetWrapped = true;
      demo.resetBrowserPose = function resetBrowserPoseOptimized(options = {}) {
        const out = originalResetBrowserPose(options);
        if (this.setFollowCamera) this.setFollowCamera(true);
        snapFollowCameraToBase(this);
        this.__unsafeReason = null;
        return out;
      };
    }

    const originalSetPolicy = demo.setPolicy?.bind(demo);
    if (originalSetPolicy && !demo.__optimizerSetPolicyWrapped) {
      demo.__optimizerSetPolicyWrapped = true;
      demo.setPolicy = async function setPolicyOptimized(policyId) {
        const out = await originalSetPolicy(policyId);
        if (!this.policyFailed && this.resetBrowserPose) {
          this.resetBrowserPose({ resetWorker: true });
          if (this.resetActivePolicyState) await this.resetActivePolicyState();
          if (this.setFollowCamera) this.setFollowCamera(true);
          snapFollowCameraToBase(this);
          this.setStatus?.('Ready', `${this.activePolicy?.().name || 'policy'} ready`, 'ready');
        }
        return out;
      };
    }

    const originalNeedsSafetyReset = demo.needsSafetyReset?.bind(demo);
    if (originalNeedsSafetyReset && !demo.__optimizerSafetyWrapped) {
      demo.__optimizerSafetyWrapped = true;
      demo.needsSafetyReset = function needsSafetyResetOptimized() {
        const reason = unsafeReason(this);
        this.__unsafeReason = reason;
        return Boolean(reason) || originalNeedsSafetyReset();
      };
    }

    if (!demo.__optimizerStopWrapped) {
      demo.__optimizerStopWrapped = true;
      demo.stopUnsafeSimulation = function stopUnsafeSimulationOptimized() {
        if (this.data?.ctrl) {
          for (let i = 0; i < this.data.ctrl.length; i += 1) this.data.ctrl[i] = 0;
        }
        this.policyPending = false;
        this.policyRequestStartedAt = 0;
        const now = performance.now();
        if (now - (this.__lastRecoveryAt || 0) < RECOVERY_COOLDOWN_MS) return;
        this.__lastRecoveryAt = now;
        const reason = this.__unsafeReason || unsafeReason(this) || 'unsafe simulation state';
        this.__recoveryCount = (this.__recoveryCount || 0) + 1;
        this.__lastRecoveryReason = reason;
        if (originalResetBrowserPose) originalResetBrowserPose({ resetWorker: true });
        if (this.setFollowCamera) this.setFollowCamera(true);
        snapFollowCameraToBase(this);
        this.setStatus?.('Ready', `Recovered from ${reason}`, 'ready');
      };
    }

    const originalUpdateVisualScene = demo.updateVisualScene?.bind(demo);
    if (originalUpdateVisualScene && !demo.__optimizerVisualWrapped) {
      demo.__optimizerVisualWrapped = true;
      demo.updateVisualScene = function updateVisualSceneOptimized() {
        originalUpdateVisualScene();
        for (const mesh of this.geomPool || []) {
          if (!mesh?.visible || mesh.userData?.isRayTerrain) continue;
          mesh.material.color.setHex(0x263238);
          mesh.material.opacity = 1;
          mesh.material.transparent = false;
          mesh.material.needsUpdate = true;
        }
      };
    }

    const originalSafeFrame = demo.safeFrame?.bind(demo);
    if (originalSafeFrame && !demo.__optimizerSafeFrameWrapped) {
      demo.__optimizerSafeFrameWrapped = true;
      demo.safeFrame = function safeFrameRecovering() {
        originalSafeFrame();
        if (this.statusPill?.textContent !== 'Runtime Error') return;
        const message = this.errorReadout?.textContent || this.frameError?.message || 'runtime error';
        this.policyPending = false;
        this.policyRequestStartedAt = 0;
        this.__recoveryCount = (this.__recoveryCount || 0) + 1;
        this.__lastRecoveryReason = message;
        this.frameError = null;
        this.setStatus?.('Ready', `Recovered from ${message}`, 'ready');
        try {
          if (originalResetBrowserPose) originalResetBrowserPose({ resetWorker: true });
          if (this.setFollowCamera) this.setFollowCamera(true);
          snapFollowCameraToBase(this);
          this.setStatus?.('Ready', `Recovered from ${message}`, 'ready');
        } catch (error) {
          this.__lastRecoveryReason = error?.message || message;
          this.frameError = null;
          this.setStatus?.('Ready', `Recovered from ${message}`, 'ready');
        }
      };
    }

    const originalRefreshRaycasterImage = demo.refreshRaycasterImage?.bind(demo);
    if (originalRefreshRaycasterImage && !demo.__optimizerRayRefreshWrapped) {
      demo.__optimizerRayRefreshWrapped = true;
      demo.refreshRaycasterImage = function refreshRaycasterImageOptimized(force = false) {
        const now = performance.now();
        if (!force && now - (this.__lastRayOptimizerWallMs || 0) < this.__rayOptimizerWallMs) {
          return this.rayImage;
        }
        const startedAt = performance.now();
        const image = originalRefreshRaycasterImage(force);
        const durationMs = performance.now() - startedAt;
        this.__lastRayOptimizerDurationMs = durationMs;
        this.__lastRayOptimizerWallMs = now;
        if (durationMs > RAY_SLOW_FRAME_MS) {
          this.__rayOptimizerWallMs = Math.min(RAY_UPDATE_WALL_MS_MAX, this.__rayOptimizerWallMs + 10);
        } else if (durationMs < RAY_FAST_FRAME_MS) {
          this.__rayOptimizerWallMs = Math.max(RAY_UPDATE_WALL_MS_MIN, this.__rayOptimizerWallMs - 5);
        }
        return image;
      };
    }

    const originalFrame = demo.frame?.bind(demo);
    if (originalFrame && !demo.__optimizerFrameWrapped) {
      demo.__optimizerFrameWrapped = true;
      demo.frame = function frameOptimized() {
        const startedAt = performance.now();
        const result = originalFrame();
        this.__lastFrameOptimizerDurationMs = performance.now() - startedAt;
        return result;
      };
    }

    demo.raycastTerrain = function raycastTerrainOptimized(posMujoco, dirMujoco) {
      if (this.mujoco?.mj_ray && this.model && this.data) {
        if (!this._mjRayPoint) {
          this._mjRayPoint = new Float64Array(3);
          this._mjRayVector = new Float64Array(3);
          this._mjRayGeomGroup = new Uint8Array(RAY_GEOMGROUP);
          this._mjRayGeomId = new Int32Array(1);
        }
        const len = Math.hypot(dirMujoco[0], dirMujoco[1], dirMujoco[2]);
        if (len <= 1.0e-9) return null;
        const scale = RAY_MAX_DIST / len;
        this._mjRayPoint[0] = posMujoco[0];
        this._mjRayPoint[1] = posMujoco[1];
        this._mjRayPoint[2] = posMujoco[2];
        this._mjRayVector[0] = dirMujoco[0] * scale;
        this._mjRayVector[1] = dirMujoco[1] * scale;
        this._mjRayVector[2] = dirMujoco[2] * scale;
        this._mjRayGeomId[0] = -1;
        const bodyExclude = this.cameraBodyId > 0 ? this.cameraBodyId : -1;
        const ratio = this.mujoco.mj_ray(
          this.model,
          this.data,
          this._mjRayPoint,
          this._mjRayVector,
          this._mjRayGeomGroup,
          1,
          bodyExclude,
          this._mjRayGeomId,
        );
        this.rayBackend = 'mujoco-mj_ray';
        if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
          return null;
        }
        const distance = ratio * RAY_MAX_DIST;
        return {
          distance,
          geomid: this._mjRayGeomId[0],
          pointMujoco: [
            posMujoco[0] + dirMujoco[0] * (distance / len),
            posMujoco[1] + dirMujoco[1] * (distance / len),
            posMujoco[2] + dirMujoco[2] * (distance / len),
          ],
        };
      }

      if (!this.rayTerrainMeshes.length) return null;
      this.rayBackend = 'three';
      this.threeRaycaster.ray.origin.set(posMujoco[0], posMujoco[2], -posMujoco[1]);
      this.threeRaycaster.ray.direction.set(dirMujoco[0], dirMujoco[2], -dirMujoco[1]).normalize();
      this.threeRaycaster.near = RAY_MIN_DIST;
      this.threeRaycaster.far = RAY_MAX_DIST;
      const hits = this.threeRaycaster.intersectObjects(this.rayTerrainMeshes, false);
      if (!hits.length) return null;
      const hit = hits[0];
      return {
        distance: hit.distance,
        pointMujoco: [hit.point.x, -hit.point.z, hit.point.y],
      };
    };

    demo.computeRaycasterImage = function computeRaycasterImageOptimized() {
      if (!this.model || !this.data || this.cameraId < 0) return this.rayImage;
      if (this.needsVisualGuard()) {
        if (this.needsSafetyReset()) this.stopUnsafeSimulation();
        return this.rayImage;
      }
      const pose = this.cameraPoseMujoco();
      if (!pose) return this.rayImage;
      const centerDir = this.cameraRayDir(0, 0, pose.mat, cameraConvention);
      const localDirs = this.rayLocalDirs(cameraConvention);
      const dir = this._rayDir || (this._rayDir = [0, 0, 0]);
      let validCount = 0;
      for (let row = 0; row < RAY_HEIGHT; row += 1) {
        for (let col = 0; col < RAY_WIDTH; col += 1) {
          const index = row * RAY_WIDTH + col;
          this.cameraRayDirFromLocal(localDirs, index, pose.mat, dir);
          this.frameStage = this.mujoco?.mj_ray ? 'mujoco-ray' : 'three-ray';
          const hit = this.raycastTerrain(pose.pos, dir);
          const dist = hit ? hit.distance : -1;
          const planeDist = dist > 0 ? dist * Math.max(0.05, dot3(dir, centerDir)) : 0;
          this.rayRawImage[index] = planeDist;
          this.rayImage[index] = normalizeDepth(planeDist);
          this.rayHitPoints[index] = hit ? hit.pointMujoco : null;
          if (planeDist > 0) validCount += 1;
        }
      }
      const centerIndex = Math.floor(RAY_HEIGHT / 2) * RAY_WIDTH + Math.floor(RAY_WIDTH / 2);
      this.rayValidCount = validCount;
      this.rayCenterDepth = this.rayRawImage[centerIndex] || 0;
      this.lastRayPose = pose;
      this.frameStage = 'ray-finished';
      this.rayLastUpdateTime = this.data.time;
      this.rayDirty = true;
      return this.rayImage;
    };
  }

  async function preloadPolicies(demo) {
    if (demo.__optimizerPreloadComplete) return;
    const ids = [demo.policyId, ...[0, 1, 2, 3].filter((id) => id !== demo.policyId)];
    const oldRealtime = demo.realtime;
    demo.realtime = 0;
    demo.__optimizerPreloadErrors = [];
    for (let i = 0; i < ids.length; i += 1) {
      const id = ids[i];
      const name = demo.activePolicy?.call({ policyId: id })?.name || `policy ${id}`;
      demo.setStatus?.('Loading', `Preloading ${name} (${i + 1}/${ids.length})`);
      try {
        await demo.ensurePolicyLoaded(id);
      } catch (error) {
        demo.__optimizerPreloadErrors.push(id);
        if (id === demo.policyId) throw error;
      }
      demo.updatePolicyButtons?.();
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    }
    demo.__optimizerPreloadComplete = true;
    if ('preloadComplete' in demo) demo.preloadComplete = true;
    demo.realtime = oldRealtime;
    demo.setStatus?.('Ready', 'MuJoCo + ONNX policies ready', 'ready');
    demo.updatePolicyButtons?.();
  }

  function startPreloadWhenReady(demo) {
    if (demo.__optimizerPreloadStarted) return;
    demo.__optimizerPreloadStarted = true;
    const wait = () => {
      if (!window.__go2wDemoReady || !demo.policyWorkerReady || !demo.policyWorker) {
        window.setTimeout(wait, 50);
        return;
      }
      preloadPolicies(demo).catch((error) => {
        demo.setStatus?.('Policy Error', error?.message || String(error), 'error');
      });
    };
    wait();
  }

  function wrapRuntimeStats(demo) {
    const originalUpdateRuntimeStats = demo.updateRuntimeStats?.bind(demo);
    if (!originalUpdateRuntimeStats || demo.__optimizerStatsWrapped) return;
    demo.__optimizerStatsWrapped = true;
    demo.updateRuntimeStats = function updateRuntimeStatsOptimized() {
      originalUpdateRuntimeStats();
      if (!window.__go2wRuntime) return;
      window.__go2wRuntime.optimizerVersion = VERSION;
      window.__go2wRuntime.preloadComplete = this.preloadComplete ?? this.__optimizerPreloadComplete ?? false;
      window.__go2wRuntime.policyLoadErrors = this.policyLoadErrors
        ? Array.from(this.policyLoadErrors.keys())
        : (this.__optimizerPreloadErrors || []);
      window.__go2wRuntime.threadCaps = this.threadCaps || detectThreadCapabilities();
      window.__go2wRuntime.frameMs = this.lastFrameDurationMs || this.__lastFrameOptimizerDurationMs || 0;
      window.__go2wRuntime.rayMs = this.lastRayDurationMs || this.__lastRayOptimizerDurationMs || 0;
      window.__go2wRuntime.rayHits = this.rayValidCount || 0;
      window.__go2wRuntime.rayCenterDepth = this.rayCenterDepth || 0;
      window.__go2wRuntime.rayBackend = this.rayBackend || null;
      window.__go2wRuntime.rayUpdateWallMs = this.rayUpdateWallMs || this.__rayOptimizerWallMs || null;
      window.__go2wRuntime.unsafeReason = this.unsafeReason || this.__unsafeReason || null;
      window.__go2wRuntime.recoveryCount = this.recoveryCount ?? this.__recoveryCount ?? 0;
      window.__go2wRuntime.lastRecoveryReason = this.lastRecoveryReason || this.__lastRecoveryReason || null;
    };
  }

  function patch(demo) {
    if (!demo || demo.__go2wOptimizerVersion === VERSION) return Boolean(demo);
    demo.__go2wOptimizerVersion = VERSION;
    installLegacyRuntimePatch(demo);
    wrapRuntimeStats(demo);
    startPreloadWhenReady(demo);
    return true;
  }

  function waitForDemo() {
    if (!patch(window.__go2wDemo)) window.setTimeout(waitForDemo, 10);
  }

  waitForDemo();
}());
