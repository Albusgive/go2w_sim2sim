(function () {
const FALLBACK_DELAY_MS = 8000;
  const POLICY_NAMES = ['motion_mlp', 'vtm', 'vtm_lstm_sru', 'vtm_gru_sru'];
  const $ = (id) => document.getElementById(id);

  let started = false;
  let rafId = 0;
  let canvas = null;
  let ctx = null;
  let lastFrame = 0;

  const state = {
    cmd: { x: 0.7, y: 0, yaw: 0 },
    pose: { x: 0, y: 0, yaw: 0 },
    realtime: 1,
    gait: 0,
    wheel: 0,
    policyId: 0,
  };

  const terrain = [
    { x: 1.4, y: -1.4, w: 0.5, h: 0.6, color: '#526156' },
    { x: 2.2, y: -1.4, w: 0.5, h: 0.6, color: '#5d6b60' },
    { x: 3.0, y: -1.4, w: 0.5, h: 0.6, color: '#68766a' },
    { x: 4.1, y: 0.0, w: 1.2, h: 0.55, color: '#58675f' },
    { x: 5.2, y: 0.1, w: 0.75, h: 0.85, color: '#3f4c46' },
    { x: 6.3, y: 1.3, w: 1.4, h: 0.32, color: '#6f806f' },
    { x: 7.8, y: 1.3, w: 0.9, h: 0.5, color: '#5c6b60' },
  ];

  function startFallback(reason) {
    if (started || window.__go2wDemoReady || window.__go2wWebglStarted) return;
    canvas = $('go2w-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    if (!ctx) return;

    started = true;
    window.__go2wFallbackStarted = true;
    bindControls();
    resize();

    const status = $('status-pill');
    if (status) {
      status.textContent = '2D';
      status.title = reason || '3D MuJoCo viewer did not load; running controlled 2D fallback.';
      status.className = 'status-pill';
    }

    lastFrame = performance.now();
    rafId = requestAnimationFrame(frame);
  }

  function bindControls() {
    const sliders = [
      ['cmd-x', 'cmd-x-out', 'x'],
      ['cmd-y', 'cmd-y-out', 'y'],
      ['cmd-yaw', 'cmd-yaw-out', 'yaw'],
    ];

    for (const [inputId, outputId, key] of sliders) {
      const input = $(inputId);
      const output = $(outputId);
      if (!input || !output) continue;
      const sync = () => {
        state.cmd[key] = Number(input.value);
        output.value = state.cmd[key].toFixed(2);
      };
      input.addEventListener('input', sync);
      sync();
    }

    const realtime = $('realtime');
    const realtimeOut = $('realtime-out');
    if (realtime && realtimeOut) {
      const syncRealtime = () => {
        state.realtime = Number(realtime.value);
        realtimeOut.value = state.realtime.toFixed(2);
      };
      realtime.addEventListener('input', syncRealtime);
      syncRealtime();
    }

    $('zero-command')?.addEventListener('click', () => setCommand(0, 0, 0));
    $('reset-sim')?.addEventListener('click', resetPose);

    for (const button of document.querySelectorAll('[data-policy]')) {
      button.addEventListener('click', () => setPolicy(Number(button.dataset.policy)));
    }

    document.addEventListener('keydown', (event) => {
      if (event.repeat) return;
      if (handleKey(event.code)) event.preventDefault();
    });
  }

  function handleKey(code) {
    const step = 0.1;
    if (code === 'KeyW') setCommand(state.cmd.x + step, state.cmd.y, state.cmd.yaw);
    else if (code === 'KeyS') setCommand(state.cmd.x - step, state.cmd.y, state.cmd.yaw);
    else if (code === 'KeyA') setCommand(state.cmd.x, state.cmd.y + step, state.cmd.yaw);
    else if (code === 'KeyD') setCommand(state.cmd.x, state.cmd.y - step, state.cmd.yaw);
    else if (code === 'KeyQ') setCommand(state.cmd.x, state.cmd.y, state.cmd.yaw + step);
    else if (code === 'KeyE') setCommand(state.cmd.x, state.cmd.y, state.cmd.yaw - step);
    else if (code === 'Space') setCommand(0, 0, 0);
    else if (code === 'KeyR') resetPose();
    else {
      const digit = code.match(/^Digit([1-4])$/);
      if (!digit) return false;
      setPolicy(Number(digit[1]) - 1);
    }
    return true;
  }

  function setCommand(x, y, yaw) {
    state.cmd.x = clamp(x, -1.0, 1.2);
    state.cmd.y = clamp(y, -0.8, 0.8);
    state.cmd.yaw = clamp(yaw, -1.5, 1.5);
    updateSlider('cmd-x', 'cmd-x-out', state.cmd.x);
    updateSlider('cmd-y', 'cmd-y-out', state.cmd.y);
    updateSlider('cmd-yaw', 'cmd-yaw-out', state.cmd.yaw);
  }

  function updateSlider(inputId, outputId, value) {
    const input = $(inputId);
    const output = $(outputId);
    if (input) input.value = value;
    if (output) output.value = value.toFixed(2);
  }

  function setPolicy(policyId) {
    state.policyId = clamp(policyId, 0, POLICY_NAMES.length - 1);
    for (const button of document.querySelectorAll('[data-policy]')) {
      button.classList.toggle('active', Number(button.dataset.policy) === state.policyId);
    }
    const readout = $('policy-readout');
    if (readout) readout.textContent = `policy ${state.policyId}: ${POLICY_NAMES[state.policyId]}`;
  }

  function resetPose() {
    state.pose = { x: 0, y: 0, yaw: 0 };
    state.gait = 0;
    state.wheel = 0;
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function frame(now) {
    const dt = Math.min((now - lastFrame) / 1000, 0.04) * state.realtime;
    lastFrame = now;
    step(dt);
    draw();
    rafId = requestAnimationFrame(frame);
  }

  function step(dt) {
    state.pose.yaw += state.cmd.yaw * dt;
    const c = Math.cos(state.pose.yaw);
    const s = Math.sin(state.pose.yaw);
    state.pose.x += (state.cmd.x * c - state.cmd.y * s) * dt;
    state.pose.y += (state.cmd.x * s + state.cmd.y * c) * dt;
    const speed = Math.hypot(state.cmd.x, state.cmd.y);
    state.gait += dt * Math.max(0.25, speed * 2.4);
    state.wheel += dt * (state.cmd.x * 7 + state.cmd.yaw * 2);
  }

  function worldToScreen(x, y) {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const compact = w < 700;
    const scale = compact ? 58 : 72;
    const cx = w * (compact ? 0.52 : 0.66);
    const cy = h * (compact ? 0.72 : 0.56);
    const dx = x - state.pose.x;
    const dy = y - state.pose.y;
    return {
      x: cx + (dx - dy) * scale,
      y: cy + (dx + dy) * scale * 0.38,
    };
  }

  function draw() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, '#182019');
    grad.addColorStop(0.55, '#0b0e0c');
    grad.addColorStop(1, '#141712');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    drawGrid();
    drawTerrain();
    const compact = w < 700;
    drawRobot(w * (compact ? 0.55 : 0.66), h * (compact ? 0.72 : 0.5));
  }

  function drawGrid() {
    ctx.save();
    ctx.strokeStyle = 'rgba(127, 209, 122, 0.22)';
    ctx.lineWidth = 1;
    for (let i = -16; i <= 22; i += 1) {
      drawWorldLine(i, -6, i, 6);
    }
    for (let j = -6; j <= 6; j += 1) {
      drawWorldLine(-16, j, 22, j);
    }
    ctx.restore();
  }

  function drawTerrain() {
    for (const block of terrain) {
      drawBlock(block);
    }
  }

  function drawBlock(block) {
    const a = worldToScreen(block.x - block.w / 2, block.y - block.h / 2);
    const b = worldToScreen(block.x + block.w / 2, block.y - block.h / 2);
    const c = worldToScreen(block.x + block.w / 2, block.y + block.h / 2);
    const d = worldToScreen(block.x - block.w / 2, block.y + block.h / 2);
    ctx.save();
    ctx.fillStyle = block.color;
    ctx.strokeStyle = 'rgba(213, 231, 205, 0.32)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.lineTo(c.x, c.y);
    ctx.lineTo(d.x, d.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function drawWorldLine(x1, y1, x2, y2) {
    const a = worldToScreen(x1, y1);
    const b = worldToScreen(x2, y2);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function drawRobot(cx, cy) {
    const speed = Math.min(Math.hypot(state.cmd.x, state.cmd.y), 1);
    const stride = Math.sin(state.gait * Math.PI * 2) * speed;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(state.pose.yaw);

    ctx.fillStyle = '#d9e1d3';
    ctx.strokeStyle = '#0b0e0c';
    ctx.lineWidth = 5;
    roundedRect(-92, -34, 184, 68, 16);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#7fd17a';
    roundedRect(34, -48, 74, 28, 9);
    ctx.fill();

    const wheels = [
      [-78, -56, stride], [78, -56, -stride], [-78, 56, -stride], [78, 56, stride],
    ];
    ctx.strokeStyle = '#c8d1c4';
    ctx.lineWidth = 8;
    for (const [x, y, phase] of wheels) {
      ctx.beginPath();
      ctx.moveTo(x * 0.58, y * 0.34);
      ctx.lineTo(x + phase * 8, y);
      ctx.stroke();
    }
    for (const [x, y, phase] of wheels) {
      ctx.save();
      ctx.translate(x + phase * 8, y);
      ctx.rotate(state.wheel);
      ctx.beginPath();
      ctx.ellipse(0, 0, 24, 14, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#161a18';
      ctx.fill();
      ctx.strokeStyle = '#9aa59a';
      ctx.lineWidth = 4;
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();
  }

  function roundedRect(x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  window.addEventListener('resize', () => {
    if (started) resize();
  });

  window.__go2wStartFallback = startFallback;

  if (new URLSearchParams(window.location.search).has('fallback')) {
    setTimeout(() => startFallback('Forced 2D fallback for testing.'), 0);
    return;
  }

  window.addEventListener('error', (event) => {
    if (!window.__go2wDemoReady && !window.__go2wWebglStarted && event.target?.tagName === 'SCRIPT') {
      startFallback('3D module failed to load.');
    }
  }, true);

  window.addEventListener('unhandledrejection', () => {
    if (!window.__go2wDemoReady && !window.__go2wWebglStarted) startFallback('3D module failed to load.');
  });

  setTimeout(() => {
    if (!window.__go2wDemoReady && !window.__go2wWebglStarted) {
      startFallback('3D viewer did not become ready; running controlled 2D fallback.');
    }
  }, FALLBACK_DELAY_MS);

  window.addEventListener('beforeunload', () => cancelAnimationFrame(rafId));
}());
