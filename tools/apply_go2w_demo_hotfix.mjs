#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(new URL('..', import.meta.url).pathname);

function file(relPath) {
  return path.join(repoRoot, relPath);
}

function edit(relPath, updater) {
  const target = file(relPath);
  const before = fs.readFileSync(target, 'utf8');
  const after = updater(before);
  if (after !== before) fs.writeFileSync(target, after);
}

function replaceAll(source, from, to) {
  return source.split(from).join(to);
}

function ensureAfter(source, anchor, insertion) {
  if (source.includes(insertion.trim())) return source;
  if (!source.includes(anchor)) throw new Error(`Missing anchor: ${anchor.slice(0, 80)}`);
  return source.replace(anchor, `${anchor}${insertion}`);
}

function removeBlock(source, block) {
  return source.includes(block) ? source.replace(block, '') : source;
}

edit('docs/demo.html', (source) => {
  let out = source;
  out = replaceAll(out, 'Follow Cam', 'Snap Cam');
  out = replaceAll(out, 'Drag mouse: orbit camera', 'Camera follows base');
  out = replaceAll(out, 'preload-ray-13', 'preload-ray-14');
  out = out.replace(
    'src="assets/js/go2w-demo-optimizer-v13.js"',
    'src="assets/js/go2w-demo-optimizer-v13.js?v=preload-ray-14"',
  );
  return out;
});

edit('tools/verify_go2w_pages_demo.mjs', (source) => (
  replaceAll(source, 'preload-ray-13', 'preload-ray-14')
));

edit('docs/assets/js/go2w-demo.js', (source) => {
  let out = source;
  out = ensureAfter(out, 'const VISUAL_UPDATE_INTERVAL = 20;\n', 'const MAX_PHYSICS_STEPS_PER_FRAME = 10;\n');
  out = out.replace('const MIN_SAFE_BASE_Z = 0.25;\n', '');
  out = removeBlock(out, `    this.controls.addEventListener('start', () => {
      this.setFollowCamera(false);
    });
`);
  out = replaceAll(out, 'this.setFollowCamera(!this.followCamera);', 'this.setFollowCamera(true);');
  out = out.replace(
    `  setFollowCamera(enabled) {
    this.followCamera = enabled;
    $('follow-camera')?.classList.toggle('active', enabled);
    if (enabled) this.snapFollowCameraToBase();
  }
`,
    `  setFollowCamera(enabled) {
    this.followCamera = true;
    $('follow-camera')?.classList.toggle('active', true);
    this.snapFollowCameraToBase();
  }
`,
  );
  out = removeBlock(out, `        if (!mesh.userData.isRayTerrain) {
          mesh.material.color.setHex(0x263238);
          mesh.material.opacity = 1;
          mesh.material.transparent = false;
        }
`);
  out = ensureAfter(out, '    this.frameCount += 1;\n', '    this.followCamera = true;\n');
  out = out.replace(
    '    while (this.physicsAccumulator >= SIM_DT && steps < 24) {',
    '    while (this.physicsAccumulator >= SIM_DT && steps < MAX_PHYSICS_STEPS_PER_FRAME) {',
  );
  out = ensureAfter(out, '      this.physicsAccumulator -= SIM_DT;\n    }\n', `    if (steps >= MAX_PHYSICS_STEPS_PER_FRAME) {
      this.physicsAccumulator = 0;
    }
`);
  out = out.replace(
    `    return !Number.isFinite(baseZ) || baseZ < MIN_SAFE_BASE_Z || this.needsSafetyReset();
`,
    `    return !Number.isFinite(baseZ) || this.needsSafetyReset();
`,
  );
  out = out.replace(`      if (base[2] < MIN_SAFE_BASE_Z) return 'base too low';
`, '');
  out = out.replace(
    `  followBase(dt) {
    if (!this.followCamera || this.baseBodyId < 0) return;
`,
    `  followBase(dt) {
    this.followCamera = true;
    if (this.baseBodyId < 0) return;
`,
  );
  return out;
});

function patchOptimizer(relPath) {
  edit(relPath, (source) => {
    let out = source;
    out = out.replace("const VERSION = 'preload-ray-13';", "const VERSION = 'preload-ray-14';");
    out = out.replace('  const MIN_SAFE_BASE_Z = 0.25;\n', '');
    out = out.replace(`      if (base[2] < MIN_SAFE_BASE_Z) return 'base too low';
`, '');
    out = out.replace(
      `      demo.setFollowCamera = function setFollowCameraOptimized(enabled) {
        originalSetFollowCamera(enabled);
        if (enabled) snapFollowCameraToBase(this);
      };
`,
      `      demo.setFollowCamera = function setFollowCameraOptimized(enabled) {
        originalSetFollowCamera(true);
        this.followCamera = true;
        snapFollowCameraToBase(this);
      };
`,
    );
    out = out.replace(
      '        return Boolean(reason) || originalNeedsSafetyReset();',
      '        return Boolean(reason);',
    );
    out = ensureAfter(out, `    if (originalNeedsSafetyReset && !demo.__optimizerSafetyWrapped) {
      demo.__optimizerSafetyWrapped = true;
      demo.needsSafetyReset = function needsSafetyResetOptimized() {
        const reason = unsafeReason(this);
        this.__unsafeReason = reason;
        return Boolean(reason);
      };
    }
`, `
    if (!demo.__optimizerVisualGuardWrapped) {
      demo.__optimizerVisualGuardWrapped = true;
      demo.needsVisualGuard = function needsVisualGuardOptimized() {
        if (!this.data || this.baseBodyId < 0) return false;
        const base = basePosition(this);
        return Boolean(base && !base.every(Number.isFinite)) || this.needsSafetyReset();
      };
    }
`);
    out = out.replace(
      `      demo.updateVisualScene = function updateVisualSceneOptimized() {
        originalUpdateVisualScene();
        for (const mesh of this.geomPool || []) {
          if (!mesh?.visible || mesh.userData?.isRayTerrain) continue;
          mesh.material.color.setHex(0x263238);
          mesh.material.opacity = 1;
          mesh.material.transparent = false;
          mesh.material.needsUpdate = true;
        }
      };
`,
      `      demo.updateVisualScene = function updateVisualSceneOptimized() {
        originalUpdateVisualScene();
      };
`,
    );
    out = out.replace(
      `      demo.frame = function frameOptimized() {
        const startedAt = performance.now();
        const result = originalFrame();
        this.__lastFrameOptimizerDurationMs = performance.now() - startedAt;
        return result;
      };
`,
      `      demo.frame = function frameOptimized() {
        const startedAt = performance.now();
        this.followCamera = true;
        const result = originalFrame();
        this.followCamera = true;
        this.__lastFrameOptimizerDurationMs = performance.now() - startedAt;
        return result;
      };
`,
    );
    return out;
  });
}

patchOptimizer('docs/assets/js/go2w-demo-optimizer-v13.js');
patchOptimizer('docs/assets/js/go2w-demo-optimizer.js');
