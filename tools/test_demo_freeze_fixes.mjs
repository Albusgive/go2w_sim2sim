#!/usr/bin/env node
// Unit tests for the demo freeze fixes (no browser required).
//
// These cover the pure decision logic that was changed to stop the
// "spiral of death" freeze:
//   1. raycaster-runtime-patch.js  — stale-image fallback gating
//   2. policy-worker.js            — stale 'run' message dropping
//   3. go2w-demo.js                — watchdog re-entrancy guard
//   4. go2w-demo-optimizer-v13.js  — realtime restored on preload failure
//
// They re-implement the exact branch conditions from the source so we can
// assert the behaviour deterministically. Run: node tools/test_demo_freeze_fixes.mjs

let passed = 0;
let failed = 0;
function check(name, cond) {
  if (cond) {
    passed += 1;
    console.log(`  ok  - ${name}`);
  } else {
    failed += 1;
    console.error(`FAIL  - ${name}`);
  }
}

// ---------------------------------------------------------------------------
// 1. Raycaster fallback gating (raycaster-runtime-patch.js)
//
// The wrapped refreshRaycasterImage must NOT run the synchronous 576-ray loop
// when the async worker is still usable. It returns the stale image instead.
// Sync (originalRefresh) is only reached on a forced one-shot or permanent
// worker failure.
// ---------------------------------------------------------------------------
function rayWorkerUsable(demo, hasWorkerCtor = true) {
  if (demo.rayWorkerFailed) return false;
  if (!hasWorkerCtor) return false;
  return true;
}

// Mirrors the wrapped refreshRaycasterImage decision after the Codex-review
// follow-up. Returns 'worker' when a request was accepted, 'stale' when the
// last image is reused, 'sync' when it computes synchronously on the main
// thread. A forced refresh ALWAYS computes synchronously (so reset seeding of
// visual history gets a fresh image, never a worker-deferred stale one).
function refreshDecision(demo, force, requestAccepted, hasWorkerCtor = true) {
  if (force) return 'sync';
  if (requestAccepted) return 'worker';
  if (rayWorkerUsable(demo, hasWorkerCtor)) return 'stale';
  return 'sync';
}

console.log('Raycaster fallback gating:');
// Worker initializing (ready=false): request not accepted, not forced -> stale, never sync.
check('initializing worker returns stale (not sync)',
  refreshDecision({ rayWorkerFailed: null }, false, false) === 'stale');
// Worker has a request in flight (pending): not accepted -> stale.
check('in-flight worker returns stale (not sync)',
  refreshDecision({ rayWorkerFailed: null }, false, false) === 'stale');
// Worker accepted the request -> worker path, image refreshed asynchronously.
check('ready worker that accepts request uses worker path',
  refreshDecision({ rayWorkerFailed: null }, false, true) === 'worker');
// Forced one-shot (reset) must compute synchronously NOW for a fresh image,
// even if the worker is ready and would accept the request.
check('forced refresh always computes synchronously (fresh image)',
  refreshDecision({ rayWorkerFailed: null }, true, true) === 'sync');
check('forced refresh syncs even when worker would not accept',
  refreshDecision({ rayWorkerFailed: null }, true, false) === 'sync');
// Permanent worker failure -> sync fallback (degraded but functional).
check('permanently failed worker falls through to sync',
  refreshDecision({ rayWorkerFailed: 'boom' }, false, false) === 'sync');
// No Worker constructor in environment -> sync fallback.
check('no Worker support falls through to sync',
  refreshDecision({ rayWorkerFailed: null }, false, false, false) === 'sync');

// ---------------------------------------------------------------------------
// 2. Policy worker stale-run dropping (policy-worker.js)
//
// latestRunSeq is updated synchronously on receipt. After the async session
// load, a run whose seq is now behind latestRunSeq must be dropped.
// ---------------------------------------------------------------------------
console.log('Policy worker stale-run dropping:');

function makeWorkerHarness() {
  let latestRunSeq = -Infinity;
  const executed = [];
  // Simulates receiving a 'run' and (after an await) deciding to execute it.
  return {
    receive(seqMaybe) {
      const seq = typeof seqMaybe === 'number' ? seqMaybe : latestRunSeq;
      if (seq > latestRunSeq) latestRunSeq = seq;
      return seq; // captured pre-await
    },
    // Called after the simulated await; returns true if it would execute.
    resolve(seq) {
      if (seq < latestRunSeq) return false; // dropped
      executed.push(seq);
      return true;
    },
    get executed() { return executed; },
    get latest() { return latestRunSeq; },
  };
}

// Three runs queue up (1,2,3) before any resolves. Only the newest executes.
{
  const w = makeWorkerHarness();
  const s1 = w.receive(1);
  const s2 = w.receive(2);
  const s3 = w.receive(3);
  // They resolve in order; 1 and 2 are now stale.
  const r1 = w.resolve(s1);
  const r2 = w.resolve(s2);
  const r3 = w.resolve(s3);
  check('older queued runs (seq 1,2) are dropped', r1 === false && r2 === false);
  check('newest queued run (seq 3) executes', r3 === true);
  check('only newest seq executed', w.executed.length === 1 && w.executed[0] === 3);
  check('latestRunSeq tracks the maximum', w.latest === 3);
}

// A single run executes normally.
{
  const w = makeWorkerHarness();
  const s = w.receive(7);
  check('single run executes', w.resolve(s) === true);
}

// undefined seq falls back to latestRunSeq and still executes (does not crash / NaN).
{
  const w = makeWorkerHarness();
  w.receive(5);            // establishes latestRunSeq = 5
  const s = w.receive(undefined); // -> treated as current latest (5)
  check('undefined seq does not advance latest', w.latest === 5);
  check('undefined seq still resolves (not dropped as NaN)', w.resolve(s) === true);
}

// ---------------------------------------------------------------------------
// 3. Watchdog re-entrancy guard (go2w-demo.js)
//
// While a recovery frame is pending, the watchdog must not queue another.
// ---------------------------------------------------------------------------
console.log('Watchdog re-entrancy guard:');

function makeWatchdog(now, lastFrameWallTime) {
  let watchdogFramePending = false;
  let injected = 0;
  const STALL_MS = 2000;
  function tick(currentNow, lastFrame) {
    if (watchdogFramePending) return;
    if (currentNow - lastFrame > STALL_MS) {
      watchdogFramePending = true;
      injected += 1;
    }
  }
  function frameRan() { watchdogFramePending = false; }
  return { tick, frameRan, get injected() { return injected; } };
}

{
  const w = makeWatchdog();
  // Loop is healthy (last frame 100ms ago) -> no injection.
  w.tick(5000, 4900);
  check('healthy loop injects no recovery frame', w.injected === 0);
}
{
  const w = makeWatchdog();
  // Stalled for 3s -> inject once.
  w.tick(5000, 2000);
  check('stalled loop injects one recovery frame', w.injected === 1);
  // Still stalled, recovery frame not yet run -> must NOT inject again.
  w.tick(6000, 2000);
  check('no double injection while recovery pending', w.injected === 1);
  // Recovery frame runs, loop still stalled -> may inject again.
  w.frameRan();
  w.tick(7000, 2000);
  check('can inject again after recovery frame ran', w.injected === 2);
}

// ---------------------------------------------------------------------------
// 4. Preload realtime restore (go2w-demo-optimizer-v13.js)
//
// realtime must be restored to its original value whether preload succeeds or
// throws.
// ---------------------------------------------------------------------------
console.log('Preload realtime restore:');

// Mirrors preloadPolicies: pause at realtime=0, restore in finally — but only
// if realtime is still the 0 sentinel (slider min is 0.05, so 0 means no user
// change landed during preload). `userSetTo` simulates the slider firing
// mid-preload. `throwOn` simulates a policy load failing.
async function preloadSim({ throwOn, userSetTo } = {}) {
  const demo = { realtime: 1.0 };
  const oldRealtime = demo.realtime;
  demo.realtime = 0;
  try {
    for (const id of [0, 1, 2, 3]) {
      if (userSetTo !== undefined && id === 1) demo.realtime = userSetTo; // user moved slider
      if (throwOn === id) throw new Error(`load failed for ${id}`);
    }
  } catch (e) {
    // swallow so we can assert the finally's effect
  } finally {
    if (demo.realtime === 0) demo.realtime = oldRealtime;
  }
  return demo;
}

{
  const demo = await preloadSim();
  check('realtime restored after successful preload', demo.realtime === 1.0);
}
{
  const demo = await preloadSim({ throwOn: 0 });
  check('realtime restored even when preload throws', demo.realtime === 1.0);
}
{
  const demo = await preloadSim({ userSetTo: 0.3 });
  check('user speed change mid-preload is NOT clobbered', demo.realtime === 0.3);
}
{
  const demo = await preloadSim({ userSetTo: 0.5, throwOn: 2 });
  check('user speed change preserved even when preload throws', demo.realtime === 0.5);
}

// ---------------------------------------------------------------------------
console.log('');
console.log(`Passed: ${passed}, Failed: ${failed}`);
process.exit(failed === 0 ? 0 : 1);
