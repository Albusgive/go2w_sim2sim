(function installRayCasterCameraAlign() {
  const RAY_WIDTH = 32;
  const RAY_HEIGHT = 18;
  const RAY_MIN_DIST = 0.1;
  const RAY_MAX_DIST = 2.0;
  const RAY_FOCAL = 1.0;

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

  function patch(demo) {
    if (!demo || demo.__rayCasterCameraAligned) return Boolean(demo);
    demo.__rayCasterCameraAligned = true;
    demo.rayConvention = cameraConvention;
    demo.rayLocalDirsByConvention?.clear?.();

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
