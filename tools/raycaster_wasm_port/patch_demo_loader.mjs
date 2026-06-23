#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(new URL('../..', import.meta.url).pathname);
const demoPath = path.join(repoRoot, 'docs/assets/js/go2w-demo.js');

let source = fs.readFileSync(demoPath, 'utf8');
source = source.replace(
  "this.model = this.mujoco.MjModel.loadFromXML('/working/scene_parkour.xml');",
  "this.model = this.loadModelXml('/working/scene_parkour.xml');",
);

if (!source.includes('loadModelXml(path)')) {
  const marker = `  ensureDir(path) {
    if (!this.mujoco.FS.analyzePath(path).exists) {
      this.mujoco.FS.mkdir(path);
    }
  }
`;
  const replacement = `${marker}
  loadModelXml(path) {
    const loader = this.mujoco?.MjModel?.mj_loadXML || this.mujoco?.MjModel?.loadFromXML;
    if (typeof loader !== 'function') {
      throw new Error('MuJoCo WASM does not expose an MJCF XML loader');
    }
    return loader.call(this.mujoco.MjModel, path);
  }
`;
  if (!source.includes(marker)) {
    throw new Error(`Could not find ensureDir block in ${demoPath}`);
  }
  source = source.replace(marker, replacement);
}

fs.writeFileSync(demoPath, source);
