/**
 * script.js
 * Three.js LiDAR truck bed reconstruction visualiser (v3).
 * Connects to FastAPI /lidar-data endpoint.
 */

"use strict";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────
const API_URL      = "/lidar-data";
const REFRESH_MS   = 5000;
const FETCH_TIMEOUT = 12000;
const MAX_LOG      = 80;

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let scene, camera, renderer, controls;
let pointsMesh = null, surfaceMesh = null;
let currentMode = "both";
let logLines    = [];
let fetching    = false;
let frameId     = null;
let pointSize   = 0.035;

// FPS tracking
let fpsFrames = 0, fpsLast = performance.now();

// ─────────────────────────────────────────────
// Colour map  (deep blue → cyan → green → amber → red)
// ─────────────────────────────────────────────
const COLOR_STOPS = [
  { t: 0.00, r: 0.00, g: 0.15, b: 0.85 },
  { t: 0.20, r: 0.00, g: 0.60, b: 1.00 },
  { t: 0.40, r: 0.00, g: 0.90, b: 0.80 },
  { t: 0.55, r: 0.18, g: 1.00, b: 0.54 },
  { t: 0.75, r: 1.00, g: 0.72, b: 0.00 },
  { t: 1.00, r: 1.00, g: 0.24, b: 0.35 },
];

function heightColor(t) {
  t = Math.max(0, Math.min(1, t));
  let lo = COLOR_STOPS[0], hi = COLOR_STOPS[COLOR_STOPS.length - 1];
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    if (t <= COLOR_STOPS[i + 1].t) { lo = COLOR_STOPS[i]; hi = COLOR_STOPS[i + 1]; break; }
  }
  const f = (t - lo.t) / (hi.t - lo.t + 1e-9);
  return {
    r: lo.r + (hi.r - lo.r) * f,
    g: lo.g + (hi.g - lo.g) * f,
    b: lo.b + (hi.b - lo.b) * f,
  };
}

// ─────────────────────────────────────────────
// Three.js init
// ─────────────────────────────────────────────
function initThree() {
  const wrap   = document.getElementById("viewport-wrap");
  const canvas = document.getElementById("three-canvas");

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060a0f);
  scene.fog = new THREE.FogExp2(0x060a0f, 0.045);

  // Camera
  camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.1, 200);
  camera.position.set(5, -5, 6);
  camera.up.set(0, 0, 1);

  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  renderer.toneMapping = THREE.LinearToneMapping;
  renderer.toneMappingExposure = 1.0;

  // Orbit controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping  = true;
  controls.dampingFactor  = 0.06;
  controls.minDistance    = 2;
  controls.maxDistance    = 30;
  controls.target.set(0, 0, 0.3);
  controls.update();

  // Lighting
  const ambient = new THREE.AmbientLight(0x1a2a40, 1.5);
  scene.add(ambient);

  const dir1 = new THREE.DirectionalLight(0x80d0ff, 0.9);
  dir1.position.set(4, 3, 8);
  scene.add(dir1);

  const dir2 = new THREE.DirectionalLight(0x4080ff, 0.3);
  dir2.position.set(-3, -2, 4);
  scene.add(dir2);

  // Grid
  addGrid();

  // Truck bed outline
  addTruckBedOutline();

  // Axes helper (small, subtle)
  const axes = new THREE.AxesHelper(0.6);
  axes.position.set(-2.5, -1.7, 0);
  axes.material.opacity = 0.4;
  axes.material.transparent = true;
  scene.add(axes);

  // Resize handler
  const onResize = () => {
    camera.aspect = wrap.clientWidth / wrap.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  };
  window.addEventListener("resize", onResize);

  animLoop();
}

function addGrid() {
  // Custom grid with fading
  const gridSize = 12;
  const gridDiv  = 24;
  const grid = new THREE.GridHelper(gridSize, gridDiv, 0x1a3050, 0x0f1a28);
  grid.rotation.x = Math.PI / 2;
  grid.material.opacity = 0.6;
  grid.material.transparent = true;
  scene.add(grid);
}

function addTruckBedOutline() {
  const mat = new THREE.LineBasicMaterial({
    color: 0x1a4060,
    transparent: true,
    opacity: 0.55,
  });

  // Bottom rectangle
  const bottom = [
    [-2, -1.2, 0], [2, -1.2, 0], [2, 1.2, 0], [-2, 1.2, 0], [-2, -1.2, 0]
  ];
  // Top rectangle
  const top = [
    [-2, -1.2, 0.6], [2, -1.2, 0.6], [2, 1.2, 0.6], [-2, 1.2, 0.6], [-2, -1.2, 0.6]
  ];

  [bottom, top].forEach(ring => {
    const verts = [];
    ring.forEach(([x, y, z]) => verts.push(x, y, z));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    scene.add(new THREE.Line(geo, mat));
  });

  // Vertical pillars
  [[-2,-1.2],[2,-1.2],[2,1.2],[-2,1.2]].forEach(([x, y]) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute([x,y,0, x,y,0.6], 3));
    scene.add(new THREE.Line(geo, mat));
  });

  // Glow outline (slightly brighter dashed)
  const glowMat = new THREE.LineDashedMaterial({
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.08,
    dashSize: 0.15,
    gapSize: 0.1,
  });
  const glowVerts = [];
  bottom.forEach(([x, y, z]) => glowVerts.push(x, y, z));
  const glowGeo = new THREE.BufferGeometry();
  glowGeo.setAttribute("position", new THREE.Float32BufferAttribute(glowVerts, 3));
  const glowLine = new THREE.Line(glowGeo, glowMat);
  glowLine.computeLineDistances();
  scene.add(glowLine);
}

// ─────────────────────────────────────────────
// Animation loop
// ─────────────────────────────────────────────
function animLoop() {
  frameId = requestAnimationFrame(animLoop);
  controls.update();
  renderer.render(scene, camera);

  // FPS counter
  fpsFrames++;
  const now = performance.now();
  if (now - fpsLast >= 1000) {
    const fps = Math.round(fpsFrames * 1000 / (now - fpsLast));
    const el = document.getElementById("fps-counter");
    if (el) el.textContent = fps + " FPS";
    fpsFrames = 0;
    fpsLast = now;
  }
}

// ─────────────────────────────────────────────
// Point cloud rendering
// ─────────────────────────────────────────────
function buildPointCloud(positions) {
  if (pointsMesh) {
    scene.remove(pointsMesh);
    pointsMesh.geometry.dispose();
    pointsMesh.material.dispose();
  }

  const n = positions.length;
  if (n === 0) return;

  const posArr = new Float32Array(n * 3);
  const colArr = new Float32Array(n * 3);

  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const z = positions[i][2];
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  const zRange = zMax - zMin || 1;

  for (let i = 0; i < n; i++) {
    posArr[i * 3]     = positions[i][0];
    posArr[i * 3 + 1] = positions[i][1];
    posArr[i * 3 + 2] = positions[i][2];
    const c = heightColor((positions[i][2] - zMin) / zRange);
    colArr[i * 3]     = c.r;
    colArr[i * 3 + 1] = c.g;
    colArr[i * 3 + 2] = c.b;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
  geo.setAttribute("color",    new THREE.BufferAttribute(colArr, 3));

  const mat = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.92,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  pointsMesh = new THREE.Points(geo, mat);
  scene.add(pointsMesh);
}

// ─────────────────────────────────────────────
// Surface mesh rendering
// ─────────────────────────────────────────────
function buildMesh(vertices, faces) {
  if (surfaceMesh) {
    scene.remove(surfaceMesh);
    surfaceMesh.geometry.dispose();
    surfaceMesh.material.dispose();
  }
  if (!vertices.length || !faces.length) return;

  let zMin = Infinity, zMax = -Infinity;
  vertices.forEach(v => { if (v[2] < zMin) zMin = v[2]; if (v[2] > zMax) zMax = v[2]; });
  const zRange = zMax - zMin || 1;

  const posArr = new Float32Array(vertices.length * 3);
  const colArr = new Float32Array(vertices.length * 3);
  vertices.forEach((v, i) => {
    posArr[i * 3] = v[0]; posArr[i * 3 + 1] = v[1]; posArr[i * 3 + 2] = v[2];
    const c = heightColor((v[2] - zMin) / zRange);
    colArr[i * 3] = c.r; colArr[i * 3 + 1] = c.g; colArr[i * 3 + 2] = c.b;
  });

  const idxArr = new Uint32Array(faces.length * 3);
  faces.forEach((f, i) => { idxArr[i*3]=f[0]; idxArr[i*3+1]=f[1]; idxArr[i*3+2]=f[2]; });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
  geo.setAttribute("color",    new THREE.BufferAttribute(colArr, 3));
  geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
  geo.computeVertexNormals();

  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.65,
    shininess: 80,
    specular: new THREE.Color(0x305080),
    flatShading: false,
  });

  surfaceMesh = new THREE.Mesh(geo, mat);
  scene.add(surfaceMesh);
}

// ─────────────────────────────────────────────
// Visibility based on mode toggle
// ─────────────────────────────────────────────
function applyMode() {
  if (pointsMesh)  pointsMesh.visible  = currentMode === "points" || currentMode === "both";
  if (surfaceMesh) surfaceMesh.visible = currentMode === "mesh"   || currentMode === "both";
}

// ─────────────────────────────────────────────
// Stats UI
// ─────────────────────────────────────────────
function updateStats(stats) {
  const volEl = document.getElementById("stat-volume");
  if (volEl) {
    const vol = stats.volume_m3 != null ? stats.volume_m3.toFixed(3) : "–";
    volEl.innerHTML = vol + '<span class="unit">m³</span>';
  }

  setText("stat-points",   stats.total_points   ?? "–");
  setText("stat-load-pts", stats.load_point_count ?? "–");
  setText("stat-backend",  (stats.backend ?? "–").toUpperCase());

  updateBalance("bal-x", stats.cx ?? 0, stats.balance_x ?? "–");
  updateBalance("bal-y", stats.cy ?? 0, stats.balance_y ?? "–");
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function updateBalance(id, cx, label) {
  const fill = document.getElementById(id + "-fill");
  const tag  = document.getElementById(id + "-tag");
  if (!fill) return;

  const norm = (cx + 2) / 4;
  const pct  = Math.max(0, Math.min(100, norm * 100));
  const centre = 50;

  const isWarning = Math.abs(cx) > 0.6;
  fill.style.background = isWarning ? "var(--accent-amber)" : "var(--accent-cyan)";
  fill.style.boxShadow  = isWarning ? "var(--glow-amber)"   : "0 0 6px rgba(0,212,255,0.3)";

  if (pct < centre) {
    fill.style.left  = pct + "%";
    fill.style.width = (centre - pct) + "%";
  } else {
    fill.style.left  = centre + "%";
    fill.style.width = (pct - centre) + "%";
  }

  if (tag) {
    tag.textContent = label;
    tag.style.color = isWarning ? "var(--accent-amber)" : "var(--accent-cyan)";
  }
}

// ─────────────────────────────────────────────
// Scan animation
// ─────────────────────────────────────────────
function triggerScan() {
  const bar = document.getElementById("scan-bar");
  if (!bar) return;
  bar.classList.remove("active");
  void bar.offsetWidth;
  bar.classList.add("active");
  bar.addEventListener("animationend", () => bar.classList.remove("active"), { once: true });
}

// ─────────────────────────────────────────────
// Log console
// ─────────────────────────────────────────────
function log(msg, type = "inf") {
  const ts = new Date().toLocaleTimeString("en-GB", { hour12: false });
  logLines.push({ ts, msg, type });
  if (logLines.length > MAX_LOG) logLines.shift();

  const out = document.getElementById("log-output");
  if (!out) return;
  out.innerHTML = logLines.map(l =>
    `<span class="log-entry ${l.type}"><span class="ts">[${l.ts}]</span> <span class="msg">${l.msg}</span></span>`
  ).join("\n");
  out.scrollTop = out.scrollHeight;
}

// ─────────────────────────────────────────────
// Data fetch + render
// ─────────────────────────────────────────────
async function fetchAndRender() {
  if (fetching) return;
  fetching = true;
  triggerScan();
  log("Requesting scan data…", "inf");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

  try {
    const t0  = performance.now();
    const res = await fetch(API_URL, { signal: controller.signal });
    clearTimeout(timer);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const dt   = Math.round(performance.now() - t0);

    buildPointCloud(data.points?.positions ?? []);
    buildMesh(data.mesh?.vertices ?? [], data.mesh?.faces ?? []);
    applyMode();
    updateStats({ ...data.stats, backend: data.backend });
    updateClock();

    const pts = data.points?.count ?? 0;
    const tri = data.mesh?.faces?.length ?? 0;
    log(`OK — ${pts} pts, ${tri} tri, ${dt}ms (${data.backend ?? "?"})`, "ok");

    document.getElementById("loading")?.classList.add("hidden");
    const toast = document.getElementById("error-toast");
    if (toast) toast.style.display = "none";

    // Update status indicator
    const status = document.getElementById("status-indicator");
    if (status) { status.textContent = "SENSOR ACTIVE"; status.classList.remove("error"); }

  } catch (e) {
    clearTimeout(timer);
    const msg = e.name === "AbortError" ? "Request timed out" : e.message;
    log("ERROR: " + msg, "err");

    const toast = document.getElementById("error-toast");
    if (toast) {
      toast.textContent = "⚠ Cannot reach API — run: python app.py";
      toast.style.display = "block";
    }

    const status = document.getElementById("status-indicator");
    if (status) { status.textContent = "DISCONNECTED"; status.classList.add("error"); }

  } finally {
    fetching = false;
    setTimeout(fetchAndRender, REFRESH_MS);
  }
}

// ─────────────────────────────────────────────
// Clock
// ─────────────────────────────────────────────
function updateClock() {
  const c = document.getElementById("clock");
  if (c) c.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
}
setInterval(updateClock, 1000);

// ─────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initThree();

  // View toggle buttons
  document.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;
      applyMode();
      log(`Display mode: ${currentMode.toUpperCase()}`, "inf");
    });
  });

  // Point size slider
  const slider = document.getElementById("point-size-slider");
  const sLabel = document.getElementById("point-size-label");
  if (slider) {
    slider.addEventListener("input", () => {
      const val = parseFloat(slider.value);
      pointSize = val * 0.012;
      if (sLabel) sLabel.textContent = val.toFixed(1);
      if (pointsMesh) pointsMesh.material.size = pointSize;
    });
  }

  updateClock();
  fetchAndRender();
  log("LiDAR system initialised (FastAPI + Three.js r160)", "ok");
  log(`Auto-refresh: ${REFRESH_MS / 1000}s`, "inf");
});
