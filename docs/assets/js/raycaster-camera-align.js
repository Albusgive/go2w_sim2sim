(function installRayCasterCameraAlign() {
  const RAY_WIDTH = 32;
  const RAY_HEIGHT = 18;
  const RAY_MIN_DIST = 0.1;
  const RAY_MAX_DIST = 2.0;
  const RAY_FOCAL = 1.0;
  const RAY_UPDATE_WALL_MS = 50;
  const MAX_BASE_XY = 12.0;
  const MAX_BASE_Y = 5.5;
  const MIN_BASE_Z = 0.15;
  const MAX_QVEL = 60;
  const MAX_CTRL = 90;
  const RECOVERY_COOLDOWN_MS = 1200;

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

  const cameraConvention = Object.freeze({
    name: '-z',
    local: (u, v) => normalize3([u, v, -RAY_FOCAL]),
  });

  function maxAbs(values) {
    let max = 0;
    if (!values) return max;
    for (const value of values) {
      const abs = Math.abs(value);
      if (Number.isFinite(abs) && abs > max) max = abs;
      if (!Number.isFinite(abs)) return Infinity;
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
    if (!base) return null;
    if (!base.every(Number.isFinite)) return 'non-finite base';
    if (Math.abs(base[0]) > MAX_BASE_XY || Math.abs(base[1]) > MAX_BASE_Y) return 'base left demo terrain';
    if (base[2] < MIN_BASE_Z) return 'base too low';
    if (maxAbs(demo.data?.qpos) === Infinity || maxAbs(demo.data?.qvel) === Infinity) return 'non-finite state';
    if (maxAbs(demo.data?.qvel) > MAX_QVEL) return 'excessive velocity';
    if (maxAbs(demo.data?.ctrl) > MAX_CTRL) return 'excessive control';
    return null;
  }

  function snapFollowCameraToBase(demo) {
    const base = basePosition(demo);
    if (!base || !demo.camera || !demo.controls) return;
    demo.controls.target.set(base[0], base[2], -base[1]);
    demo.camera.position.set(base[0] - 2.4, base[2] + 1.45, -base[1] + 2.2);
    demo.controls.update();
  }

  function patch(demo) {
    if (!demo || demo.__rayCasterCameraAligned) return Boolean(demo);
    demo.__rayCasterCameraAligned = true;
    demo.rayConvention = cameraConvention;
    demo.rayLocalDirsByConvention?.clear?.();
    demo.__rayPatchVersion = 'ray-z-3';
    demo.__lastRayPatchWallMs = 0;
    demo.__lastRayPatchDurationMs = 0;
    demo.__lastFramePatchDurationMs = 0;
    demo.__lastRecoveryAt = 0;
    demo.__unsafeReason = null;

    const originalSetFollowCamera = demo.setFollowCamera?.bind(demo);
    if (originalSetFollowCamera) {
      demo.setFollowCamera = function setFollowCameraAligned(enabled) {
        originalSetFollowCamera(enabled);
        if (enabled) snapFollowCameraToBase(this);
      };
    }

    demo.snapFollowCameraToBase = function snapFollowCameraToBaseMethod() {
      snapFollowCameraToBase(this);
    };

    const originalResetBrowserPose = demo.resetBrowserPose?.bind(demo);
    if (originalResetBrowserPose) {
      demo.resetBrowserPose = function resetBrowserPoseAligned(options = {}) {
        const out = originalResetBrowserPose(options);
        if (this.setFollowCamera) this.setFollowCamera(true);
        snapFollowCameraToBase(this);
        this.__unsafeReason = null;
        return out;
      };
    }

    const originalSetPolicy = demo.setPolicy?.bind(demo);
    if (originalSetPolicy) {
      demo.setPolicy = async function setPolicyAligned(policyId) {
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
    if (originalNeedsSafetyReset) {
      demo.needsSafetyReset = function needsSafetyResetAligned() {
        const reason = unsafeReason(this);
        this.__unsafeReason = reason;
        return Boolean(reason) || originalNeedsSafetyReset();
      };
    }

    demo.stopUnsafeSimulation = function stopUnsafeSimulationAligned() {
      if (this.data?.ctrl) {
        for (let i = 0; i < this.data.ctrl.length; i += 1) this.data.ctrl[i] = 0;
      }
      this.policyPending = false;
      this.policyRequestStartedAt = 0;
      const now = performance.now();
      if (now - (this.__lastRecoveryAt || 0) < RECOVERY_COOLDOWN_MS) return;
      this.__lastRecoveryAt = now;
      const reason = this.__unsafeReason || unsafeReason(this) || 'unsafe simulation state';
      if (originalResetBrowserPose) originalResetBrowserPose({ resetWorker: true });
      if (this.setFollowCamera) this.setFollowCamera(true);
      snapFollowCameraToBase(this);
      this.setStatus?.('Ready', `Recovered from ${reason}`, 'ready');
    };

    const originalRefreshRaycasterImage = demo.refreshRaycasterImage?.bind(demo);
    if (originalRefreshRaycasterImage) {
      demo.refreshRaycasterImage = function refreshRaycasterImageAligned(force = false) {
        const now = performance.now();
        if (!force && now - (this.__lastRayPatchWallMs || 0) < RAY_UPDATE_WALL_MS) {
          return this.rayImage;
        }
        const startedAt = performance.now();
        const image = originalRefreshRaycasterImage(force);
        this.__lastRayPatchDurationMs = performance.now() - startedAt;
        this.__lastRayPatchWallMs = now;
        return image;
      };
    }

    const originalFrame = demo.frame?.bind(demo);
    if (originalFrame) {
      demo.frame = function frameAligned() {
        const startedAt = performance.now();
        const result = originalFrame();
        this.__lastFramePatchDurationMs = performance.now() - startedAt;
        return result;
      };
    }

    const originalUpdateRuntimeStats = demo.updateRuntimeStats?.bind(demo);
    if (originalUpdateRuntimeStats) {
      demo.updateRuntimeStats = function updateRuntimeStatsAligned() {
        originalUpdateRuntimeStats();
        if (window.__go2wRuntime) {
          window.__go2wRuntime.patchVersion = this.__rayPatchVersion;
          window.__go2wRuntime.frameMs = this.__lastFramePatchDurationMs || 0;
          window.__go2wRuntime.rayMs = this.__lastRayPatchDurationMs || 0;
          window.__go2wRuntime.unsafeReason = this.__unsafeReason || null;
        }
      };
    }

    demo.raycastTerrain = function raycastTerrainAligned(posMujoco, dirMujoco) {
      if (!this.rayTerrainMeshes.length) return null;
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

    demo.computeRaycasterImage = function computeRaycasterImageAligned() {
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
      for (let row = 0; row < RAY_HEIGHT; row += 1) {
        for (let col = 0; col < RAY_WIDTH; col += 1) {
          const index = row * RAY_WIDTH + col;
          this.cameraRayDirFromLocal(localDirs, index, pose.mat, dir);
          this.frameStage = 'three-ray';
          const hit = this.raycastTerrain(pose.pos, dir);
          const dist = hit ? hit.distance : -1;
          const planeDist = dist > 0 ? dist * Math.max(0.05, dot3(dir, centerDir)) : 0;
          this.rayRawImage[index] = planeDist;
          this.rayImage[index] = normalizeDepth(planeDist);
          this.rayHitPoints[index] = hit ? hit.pointMujoco : null;
        }
      }
      this.lastRayPose = pose;
      this.frameStage = 'ray-finished';
      this.rayLastUpdateTime = this.data.time;
      this.rayDirty = true;
      return this.rayImage;
    };
    return true;
  }

  function waitForDemo() {
    if (!patch(window.__go2wDemo)) {
      window.setTimeout(waitForDemo, 10);
    }
  }

  waitForDemo();
}());
