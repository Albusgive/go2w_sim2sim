#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const MARKER_BEGIN = '// GO2W_RAYCASTER_WASM_BEGIN';
const MARKER_END = '// GO2W_RAYCASTER_WASM_END';

function parseArgs(argv) {
  const out = {
    patch: false,
    repoRoot: path.resolve(new URL('../..', import.meta.url).pathname),
    mujocoRoot: process.env.MUJOCO_WASM_SOURCE || '/home/albusgive2/software/mujoco',
    raycasterRoot: null,
    buildDir: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`Missing value for ${arg}`);
      i += 1;
      return argv[i];
    };
    if (arg === '--patch') out.patch = true;
    else if (arg === '--repo-root') out.repoRoot = path.resolve(next());
    else if (arg === '--mujoco-root') out.mujocoRoot = path.resolve(next());
    else if (arg === '--raycaster-root') out.raycasterRoot = path.resolve(next());
    else if (arg === '--build-dir') out.buildDir = path.resolve(next());
    else if (arg === '--help') {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  out.buildDir ||= path.join(out.mujocoRoot, 'build-go2w-raycaster-wasm');
  out.raycasterRoot ||= process.env.RAYCASTER_ROOT
    ? path.resolve(process.env.RAYCASTER_ROOT)
    : path.join(out.repoRoot, 'utils/mujoco_ray_caster');
  return out;
}

function printUsage() {
  console.log(`Usage:
  node tools/raycaster_wasm_port/prepare_mujoco_wasm_port.mjs [options]

Options:
  --patch                Patch MuJoCo's local wasm CMake/bindings files.
  --mujoco-root DIR      MuJoCo source root. Default: $MUJOCO_WASM_SOURCE or /home/albusgive2/software/mujoco.
  --raycaster-root DIR   RayCasterCamera source root. Default: $RAYCASTER_ROOT or <repo>/utils/mujoco_ray_caster.
  --repo-root DIR        go2w_sim2sim repo root. Default: this repository.
  --build-dir DIR        CMake build directory. Default: <mujoco-root>/build-go2w-raycaster-wasm.
`);
}

function commandPath(name) {
  const result = spawnSync('bash', ['-lc', `command -v ${name}`], {
    encoding: 'utf8',
  });
  return result.status === 0 ? result.stdout.trim() : '';
}

function requireFile(label, filePath, failures) {
  if (!fs.existsSync(filePath)) failures.push(`${label} missing: ${filePath}`);
}

function replaceMarkedBlock(source, block) {
  const escapedBegin = MARKER_BEGIN.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const escapedEnd = MARKER_END.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`\\n?${escapedBegin}[\\s\\S]*?${escapedEnd}\\n?`, 'm');
  if (re.test(source)) {
    return `${source.replace(re, `\n${block}\n`)}`;
  }
  return `${source.trimEnd()}\n\n${block}\n`;
}

function patchCMake(cmakePath, raycasterRoot) {
  const normalizedRayRoot = raycasterRoot.replaceAll('\\', '/');
  const block = `${MARKER_BEGIN}
set(GO2W_RAYCASTER_ROOT "${normalizedRayRoot}" CACHE PATH "go2w RayCasterCamera source root")
target_sources(mujoco_wasm PRIVATE
  "\${GO2W_RAYCASTER_ROOT}/raycaster_src/RayCaster.cpp"
  "\${GO2W_RAYCASTER_ROOT}/raycaster_src/RayCasterCamera.cpp"
)
target_include_directories(mujoco_wasm PRIVATE
  "\${GO2W_RAYCASTER_ROOT}"
  "\${GO2W_RAYCASTER_ROOT}/raycaster_src"
)
${MARKER_END}`;
  const source = fs.readFileSync(cmakePath, 'utf8');
  if (!source.includes('add_executable(mujoco_wasm')) {
    throw new Error(`Could not find mujoco_wasm target in ${cmakePath}`);
  }
  fs.writeFileSync(cmakePath, replaceMarkedBlock(source, block));
}

function patchBindings(bindingsPath, repoRoot) {
  const includePath = path
    .join(repoRoot, 'tools/raycaster_wasm_port/raycaster_camera_bindings.inc.cc')
    .replaceAll('\\', '/');
  const block = `${MARKER_BEGIN}
#include "${includePath}"
${MARKER_END}`;
  const source = fs.readFileSync(bindingsPath, 'utf8');
  fs.writeFileSync(bindingsPath, replaceMarkedBlock(source, block));
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const wasmDir = path.join(options.mujocoRoot, 'wasm');
  const cmakePath = path.join(wasmDir, 'CMakeLists.txt');
  const bindingsPath = path.join(wasmDir, 'codegen/generated/bindings.cc');
  const bindingInclude = path.join(
    options.repoRoot,
    'tools/raycaster_wasm_port/raycaster_camera_bindings.inc.cc',
  );
  const rayCasterHeader = path.join(
    options.raycasterRoot,
    'raycaster_src/RayCasterCamera.h',
  );
  const rayCasterCpp = path.join(options.raycasterRoot, 'raycaster_src/RayCaster.cpp');
  const rayCasterCameraCpp = path.join(
    options.raycasterRoot,
    'raycaster_src/RayCasterCamera.cpp',
  );

  const failures = [];
  requireFile('MuJoCo wasm CMakeLists', cmakePath, failures);
  requireFile('MuJoCo generated bindings', bindingsPath, failures);
  requireFile('go2w raycaster binding include', bindingInclude, failures);
  requireFile('RayCasterCamera header', rayCasterHeader, failures);
  requireFile('RayCaster implementation', rayCasterCpp, failures);
  requireFile('RayCasterCamera implementation', rayCasterCameraCpp, failures);

  const tools = {
    emcc: commandPath('emcc'),
    emcmake: commandPath('emcmake'),
    cmake: commandPath('cmake'),
    ninja: commandPath('ninja'),
    node: commandPath('node'),
  };
  if (!tools.emcc) failures.push('emcc not found; source emsdk_env.sh for Emscripten 4.0.10 first');
  if (!tools.emcmake) failures.push('emcmake not found; source emsdk_env.sh for Emscripten 4.0.10 first');
  if (!tools.cmake) failures.push('cmake not found');

  if (options.patch) {
    if (!fs.existsSync(cmakePath) || !fs.existsSync(bindingsPath)) {
      throw new Error('Cannot patch because MuJoCo wasm files are missing');
    }
    patchCMake(cmakePath, options.raycasterRoot);
    patchBindings(bindingsPath, options.repoRoot);
  }

  const generator = tools.ninja ? '-G Ninja ' : '';
  const commands = [
    `cd ${shellQuote(options.mujocoRoot)}`,
    `emcmake cmake -S . -B ${shellQuote(options.buildDir)} ${generator}`.trim(),
    `cmake --build ${shellQuote(options.buildDir)} --target mujoco_wasm`,
    `cp ${shellQuote(path.join(options.mujocoRoot, 'wasm/dist/mujoco_wasm.js'))} ${shellQuote(path.join(options.repoRoot, 'docs/demo-assets/mujoco_wasm.js'))}`,
    `cp ${shellQuote(path.join(options.mujocoRoot, 'wasm/dist/mujoco_wasm.wasm'))} ${shellQuote(path.join(options.repoRoot, 'docs/demo-assets/mujoco_wasm.wasm'))}`,
  ];

  const result = {
    ok: failures.length === 0,
    patched: options.patch,
    mujocoRoot: options.mujocoRoot,
    repoRoot: options.repoRoot,
    raycasterRoot: options.raycasterRoot,
    buildDir: options.buildDir,
    tools,
    failures,
    commands,
  };
  console.log(JSON.stringify(result, null, 2));
  if (failures.length) process.exitCode = 1;
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", "'\\''")}'`;
}

main();
