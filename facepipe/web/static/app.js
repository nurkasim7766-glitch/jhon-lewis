/* facepipe web UI — dependency-free SPA. Every view reads real API data; nothing is simulated. */
"use strict";

// ---------------------------------------------------------------- utilities
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (ms) => { const s = Math.max(0, ms / 1000); const m = Math.floor(s / 60); const r = s - m * 60; return `${m}:${r.toFixed(1).padStart(4, "0")}`; };
const fmtClock = (ms) => { const s = Math.floor(ms / 1000); return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; };
const fmtBytes = (b) => { if (!b && b !== 0) return "—"; const u = ["B", "KB", "MB", "GB"]; let i = 0; while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; } return `${b.toFixed(i ? 1 : 0)} ${u[i]}`; };
const ago = (t) => { const d = Date.now() / 1000 - t; if (d < 60) return "just now"; if (d < 3600) return `${Math.floor(d / 60)} min ago`; if (d < 86400) return `${Math.floor(d / 3600)} h ago`; return `${Math.floor(d / 86400)} d ago`; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const api = {
  async req(method, url, body, isForm) {
    const opts = { method, headers: {} };
    if (body !== undefined) { if (isForm) opts.body = body; else { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); } }
    const r = await fetch(url, opts);
    if (r.status === 204) return null;
    const data = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
    if (!r.ok) throw new Error(data?.detail || (typeof data === "string" ? data : r.statusText));
    return data;
  },
  get: (u) => api.req("GET", u), post: (u, b) => api.req("POST", u, b), patch: (u, b) => api.req("PATCH", u, b), del: (u) => api.req("DELETE", u),
};

function toast(msg, kind = "info", ms = 4200) {
  const el = document.createElement("div"); el.className = `toast ${kind}`; el.textContent = msg;
  $("#toasts").appendChild(el); setTimeout(() => el.remove(), ms);
}
function modal(html, onMount) {
  const root = $("#modal-root"); root.innerHTML = `<div class="modal-back"><div class="modal">${html}</div></div>`;
  const close = () => (root.innerHTML = "");
  $(".modal-back", root).addEventListener("click", (e) => { if (e.target.classList.contains("modal-back")) close(); });
  onMount?.(root, close); return close;
}
function confirmModal(title, text, okLabel = "Delete") {
  return new Promise((resolve) => {
    modal(`<h2>${esc(title)}</h2><p class="muted">${esc(text)}</p><div class="foot"><button class="btn" data-x>Cancel</button><button class="btn btn-danger" data-ok>${esc(okLabel)}</button></div>`,
      (root, close) => { $("[data-x]", root).onclick = () => { close(); resolve(false); }; $("[data-ok]", root).onclick = () => { close(); resolve(true); }; });
  });
}
const statusPill = (s) => {
  const map = { new: ["", "New"], uploaded: ["info", "Needs transcript"], ready: ["info", "Ready"], analyzing: ["accent", "Analyzing"], analyzed: ["ok", "Moments found"], processing: ["accent", "Processing"], error: ["err", "Error"],
    candidate: ["", "Candidate"], rendered: ["ok", "Rendered"], queued: ["warn", "Queued"], running: ["accent", "Running"], done: ["ok", "Done"], failed: ["err", "Failed"], cancelled: ["", "Cancelled"], interrupted: ["warn", "Interrupted"] };
  const [k, l] = map[s] || ["", s]; const spin = ["analyzing", "processing", "running"].includes(s) ? '<span class="spin"></span>' : "";
  return `<span class="pill ${k}">${spin}${esc(l)}</span>`;
};
const FACE_COLORS = ["#ffc850", "#3c96ff", "#78ff78", "#e66eff", "#5affff", "#ff82aa"];
const faceColor = (id) => FACE_COLORS[(id ?? 0) % FACE_COLORS.length];

// ---------------------------------------------------------------- state & router
const state = { health: null, pollers: new Set(), steps: [] };
function stopPollers() { for (const p of state.pollers) clearInterval(p); state.pollers.clear(); }
function poll(fn, ms) { const id = setInterval(fn, ms); state.pollers.add(id); return id; }

const routes = [
  [/^#?\/?$/, "dashboard", viewDashboard],
  [/^#\/projects$/, "projects", viewProjects],
  [/^#\/upload$/, "upload", viewUpload],
  [/^#\/exports$/, "exports", viewExports],
  [/^#\/p\/([a-z0-9]+)$/, "projects", viewProject],
  [/^#\/p\/([a-z0-9]+)\/upload$/, "projects", viewUpload],
  [/^#\/p\/([a-z0-9]+)\/processing$/, "projects", viewProcessing],
  [/^#\/p\/([a-z0-9]+)\/clips$/, "projects", viewClips],
  [/^#\/p\/([a-z0-9]+)\/edit\/([a-z0-9]+)$/, "projects", viewEditor],
  [/^#\/p\/([a-z0-9]+)\/settings$/, "projects", viewSettings],
];
async function route() {
  stopPollers();
  const hash = location.hash || "#/";
  for (const [re, nav, fn] of routes) {
    const m = hash.match(re);
    if (m) {
      $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === nav));
      try { await fn(...m.slice(1)); } catch (e) { $("#view").innerHTML = `<div class="banner err">⚠ ${esc(e.message)}</div>`; }
      return;
    }
  }
  location.hash = "#/";
}
function crumbs(parts) { $("#crumbs").innerHTML = parts.map((p, i) => i === parts.length - 1 ? `<span class="cur">${esc(p[0])}</span>` : `<a href="${p[1]}">${esc(p[0])}</a><span class="sep">/</span>`).join(""); }
function projectTabs(p, cur) {
  const t = [["Overview", `#/p/${p.id}`, "overview"], ["Upload", `#/p/${p.id}/upload`, "upload"], ["Processing", `#/p/${p.id}/processing`, "processing"], ["Clips", `#/p/${p.id}/clips`, "clips"], ["Settings", `#/p/${p.id}/settings`, "settings"]];
  return `<div class="tabs">${t.map(([l, h, k]) => `<a href="${h}" class="${k === cur ? "active" : ""}">${l}</a>`).join("")}</div>`;
}

// ---------------------------------------------------------------- dashboard
async function viewDashboard() {
  crumbs([["Dashboard"]]);
  const render = async () => {
    const d = await api.get("/api/dashboard");
    const s = d.stats;
    $("#view").innerHTML = `
      <div class="grid cols-4">
        <div class="card stat"><div class="label">Projects</div><div class="value">${s.projects}</div><div class="sub">${s.videos} with video</div></div>
        <div class="card stat"><div class="label">Videos processed</div><div class="value">${s.videos}</div><div class="sub">uploaded sources</div></div>
        <div class="card stat"><div class="label">Clips generated</div><div class="value">${s.clips_rendered}</div><div class="sub">${s.candidates} candidates discovered</div></div>
        <div class="card stat"><div class="label">Processing</div><div class="value">${s.active_jobs.length}</div><div class="sub">${s.active_jobs.length ? "job" + (s.active_jobs.length > 1 ? "s" : "") + " active" : "idle"}</div></div>
      </div>
      <div class="grid cols-2 section">
        <div class="card pad-0">
          <div class="card-head" style="padding:16px 18px 0"><h2>Recent projects</h2><a class="btn btn-sm btn-primary" href="#/upload">⇪ Quick upload</a></div>
          ${d.recent_projects.length ? `<table class="table">${d.recent_projects.map((p) => `<tr class="row-link" data-href="#/p/${p.id}"><td><b>${esc(p.name)}</b><div class="dim small">${p.source_filename ? esc(p.source_filename) : "no video yet"}</div></td><td>${statusPill(p.status)}</td><td class="dim small">${p.clip_counts.rendered}/${p.clip_counts.total} clips</td><td class="dim small">${ago(p.updated_at)}</td></tr>`).join("")}</table>` : `<div class="empty" style="margin:18px"><div class="big">▦</div>No projects yet. <a href="#/upload" style="color:var(--accent-2)">Upload a video</a> to get started.</div>`}
        </div>
        <div class="card">
          <div class="card-head"><h2>Processing status</h2></div>
          ${s.active_jobs.length ? s.active_jobs.map((j) => jobSummary(j)).join("") : `<p class="muted">No jobs running.</p>`}
          <div class="card-head" style="margin-top:18px"><h2>Recent activity</h2></div>
          <div class="list">${d.activity.map((a) => `<div class="list-item"><span class="act-ico">${{ upload: "⇪", job: "⚙", analysis: "✦", render: "▶", error: "⚠", project: "▤" }[a.kind] || "•"}</span><div><div>${esc(a.message)}</div><div class="dim small">${a.project_name ? esc(a.project_name) : ""}</div></div><span class="when">${ago(a.at)}</span></div>`).join("") || '<p class="muted">Nothing yet.</p>'}</div>
        </div>
      </div>`;
    $$("[data-href]").forEach((r) => (r.onclick = () => (location.hash = r.dataset.href)));
  };
  await render(); poll(render, 4000);
}
function jobSummary(j) {
  const pr = j.progress || {}; const steps = pr.steps || {}; const done = Object.values(steps).filter((s) => ["done", "reused"].includes(s.status)).length;
  return `<div class="list-item"><span class="act-ico">⚙</span><div style="flex:1"><div><b>${esc(j.kind)}</b> ${statusPill(j.status)} <span class="dim small">${esc(pr.message || "")}</span></div><div class="progress" style="margin-top:6px"><i style="width:${Math.round(100 * done / Math.max(1, state.steps.length))}%"></i></div></div><a class="btn btn-sm" href="#/p/${j.project_id}/processing">Open</a></div>`;
}

// ---------------------------------------------------------------- projects list
async function viewProjects() {
  crumbs([["Projects"]]);
  const ps = await api.get("/api/projects");
  $("#view").innerHTML = `
    <div class="section-title"><h1>Projects</h1><div class="actions"><button class="btn" id="new-p">＋ Empty project</button><a class="btn btn-primary" href="#/upload">⇪ Upload video</a></div></div>
    ${ps.length ? `<div class="card pad-0"><table class="table"><thead><tr><th>Project</th><th>Status</th><th>Source</th><th>Clips</th><th>Updated</th><th></th></tr></thead>${ps.map((p) => `<tr class="row-link" data-href="#/p/${p.id}"><td><b>${esc(p.name)}</b></td><td>${statusPill(p.status)}</td><td class="small muted">${p.source_info ? `${p.source_info.width}×${p.source_info.height} · ${fmtClock(p.source_info.duration * 1000)}` : "—"}</td><td class="small muted">${p.clip_counts.rendered} rendered / ${p.clip_counts.total}</td><td class="small dim">${ago(p.updated_at)}</td><td><button class="btn btn-sm btn-ghost btn-danger" data-del="${p.id}" title="Delete">✕</button></td></tr>`).join("")}</table></div>` : `<div class="empty"><div class="big">▤</div>No projects yet.</div>`}`;
  $$("[data-href]").forEach((r) => (r.onclick = (e) => { if (!e.target.closest("[data-del]")) location.hash = r.dataset.href; }));
  $$("[data-del]").forEach((b) => (b.onclick = async (e) => { e.stopPropagation(); if (await confirmModal("Delete project?", "This removes the uploaded video, all intermediate files and rendered clips.")) { try { await api.del(`/api/projects/${b.dataset.del}`); toast("Project deleted", "ok"); viewProjects(); } catch (err) { toast(err.message, "err"); } } }));
  $("#new-p").onclick = async () => { const p = await api.post("/api/projects", { name: "Untitled project" }); location.hash = `#/p/${p.id}`; };
}

// ---------------------------------------------------------------- upload
async function viewUpload(pid) {
  let project = pid ? await api.get(`/api/projects/${pid}`) : null;
  crumbs(project ? [["Projects", "#/projects"], [project.name, `#/p/${project.id}`], ["Upload"]] : [["New upload"]]);
  const h = state.health;
  const fmts = (h?.video_formats || []).join(", ");
  $("#view").innerHTML = `
    ${project ? projectTabs(project, "upload") : `<h1 style="margin-bottom:16px">New project</h1>`}
    <div class="grid cols-2">
      <div class="card">
        <div class="card-head"><h2>1 · Project</h2></div>
        <div class="field"><label>Project name</label><input class="input" id="pname" value="${esc(project?.name || "")}" placeholder="e.g. Episode 42 — founder interview"></div>
      </div>
      <div class="card">
        <div class="card-head"><h2>2 · Source video</h2>${project?.source_filename ? statusPill("done") : ""}</div>
        <div class="dropzone ${project?.source_filename ? "done" : ""}" id="dz-video">
          ${project?.source_filename ? `<div class="big">✓</div><b>${esc(project.source_filename)}</b><div class="muted small">${project.source_info.width}×${project.source_info.height} · ${fmtClock(project.source_info.duration * 1000)} · ${project.source_info.fps?.toFixed(2)} fps · ${fmtBytes(project.source_info.size_bytes)} · ${project.source_info.has_audio ? "audio" : "no audio"}</div><div class="dim small" style="margin-top:6px">Drop another file to replace it (clears existing clips)</div>` : `<div class="big">⇪</div><b>Drag &amp; drop a video</b> or click to browse<div class="dim small" style="margin-top:6px">${esc(fmts)} · up to ${fmtBytes(h?.max_upload_bytes)}</div>`}
          <input type="file" id="f-video" accept="video/*,${(h?.video_formats || []).join(",")}" hidden>
        </div>
        <div id="video-progress" style="margin-top:10px;display:none"><div class="progress"><i style="width:0"></i></div><div class="actions" style="margin-top:8px;justify-content:space-between"><span class="small muted" id="vp-text">0%</span><button class="btn btn-sm btn-danger" id="vp-cancel">Cancel upload</button></div></div>
        <div id="video-error"></div>
      </div>
      <div class="card">
        <div class="card-head"><h2>3 · Transcript</h2>${project?.transcript_filename ? statusPill("done") : ""}</div>
        <div class="dropzone ${project?.transcript_filename ? "done" : ""}" id="dz-tr">
          ${project?.transcript_filename ? `<div class="big">✓</div><b>${esc(project.transcript_filename)}</b><div class="muted small">${project.transcript_info.sentences} sentences · ${project.transcript_info.words} words · speakers: ${project.transcript_info.speakers.join(", ") || "none"} · ${fmtClock(project.transcript_info.duration_ms)}</div>` : `<div class="big">≡</div><b>Drop a transcript JSON</b> or click to browse<div class="dim small" style="margin-top:6px">Whisper / faster-whisper / WhisperX <code>segments</code> format (word timings optional)</div>`}
          <input type="file" id="f-tr" accept=".json,application/json" hidden>
        </div>
        <details style="margin-top:10px"><summary class="muted small" style="cursor:pointer">…or paste transcript JSON</summary><textarea class="input" id="tr-text" placeholder='{"segments":[{"start":0.0,"end":3.2,"text":"Hello there.","speaker":"A"}]}'></textarea><button class="btn btn-sm" id="tr-paste" style="margin-top:8px">Use pasted transcript</button></details>
        <div id="tr-error"></div>
      </div>
      <div class="card">
        <div class="card-head"><h2>4 · Analyze</h2></div>
        <p class="muted">Analysis selects sentence-aligned moments locally (no cloud calls), then processes the selected clips: face &amp; landmark tracking, active-speaker detection, speaker segmentation, crop planning and rendering.</p>
        <div class="actions" style="margin-top:12px"><button class="btn btn-primary" id="go" ${project && project.status === "ready" || project?.status === "analyzed" || project?.status === "error" ? "" : "disabled"}>✦ Start AI analysis</button>${project ? `<a class="btn" href="#/p/${project.id}/settings">Settings</a>` : ""}</div>
        <p class="help" style="margin-top:10px">${project ? (project.source_filename && project.transcript_filename ? "Ready to analyze." : "Upload both a video and a transcript to enable analysis.") : "Create the project by naming it and uploading a video."}</p>
      </div>
    </div>`;

  const ensureProject = async () => { if (project) return project; const name = $("#pname").value.trim() || "Untitled project"; project = await api.post("/api/projects", { name }); history.replaceState(null, "", `#/p/${project.id}/upload`); return project; };
  $("#pname").onchange = async () => { if (project) { project = await api.patch(`/api/projects/${project.id}`, { name: $("#pname").value }); toast("Renamed", "ok"); } };

  // video upload with progress + cancel (XHR for upload progress events)
  const dzV = $("#dz-video"), fV = $("#f-video");
  dzV.onclick = () => fV.click();
  ["dragenter", "dragover"].forEach((ev) => dzV.addEventListener(ev, (e) => { e.preventDefault(); dzV.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dzV.addEventListener(ev, (e) => { e.preventDefault(); dzV.classList.remove("over"); }));
  dzV.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadVideo(f); });
  fV.onchange = () => { if (fV.files[0]) uploadVideo(fV.files[0]); };
  async function uploadVideo(file) {
    $("#video-error").innerHTML = "";
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!(h?.video_formats || []).includes(ext)) { $("#video-error").innerHTML = `<div class="banner err" style="margin-top:10px">Unsupported format ${esc(ext)}. Supported: ${esc(fmts)}</div>`; return; }
    if (file.size > (h?.max_upload_bytes || Infinity)) { $("#video-error").innerHTML = `<div class="banner err" style="margin-top:10px">File is ${fmtBytes(file.size)}; the limit is ${fmtBytes(h.max_upload_bytes)}.</div>`; return; }
    const p = await ensureProject();
    const box = $("#video-progress"); box.style.display = "block";
    const xhr = new XMLHttpRequest(); const fd = new FormData(); fd.append("file", file);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { const pct = Math.round(100 * e.loaded / e.total); $("#video-progress i").style.width = pct + "%"; $("#vp-text").textContent = `${pct}% · ${fmtBytes(e.loaded)} of ${fmtBytes(e.total)}${pct === 100 ? " · validating on server…" : ""}`; } };
    xhr.onload = () => { if (xhr.status >= 200 && xhr.status < 300) { toast("Video uploaded and validated", "ok"); viewUpload(p.id); } else { let msg = xhr.statusText; try { msg = JSON.parse(xhr.responseText).detail; } catch {} box.style.display = "none"; $("#video-error").innerHTML = `<div class="banner err" style="margin-top:10px">⚠ ${esc(msg)}</div>`; } };
    xhr.onerror = () => { box.style.display = "none"; $("#video-error").innerHTML = `<div class="banner err" style="margin-top:10px">Network error during upload.</div>`; };
    xhr.onabort = () => { box.style.display = "none"; toast("Upload cancelled"); };
    $("#vp-cancel").onclick = () => xhr.abort();
    xhr.open("POST", `/api/projects/${p.id}/video`); xhr.send(fd);
  }
  // transcript
  const dzT = $("#dz-tr"), fT = $("#f-tr");
  dzT.onclick = () => fT.click();
  ["dragenter", "dragover"].forEach((ev) => dzT.addEventListener(ev, (e) => { e.preventDefault(); dzT.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dzT.addEventListener(ev, (e) => { e.preventDefault(); dzT.classList.remove("over"); }));
  dzT.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadTranscript(f); });
  fT.onchange = () => { if (fT.files[0]) uploadTranscript(fT.files[0]); };
  async function uploadTranscript(file) {
    $("#tr-error").innerHTML = "";
    try { const p = await ensureProject(); const fd = new FormData(); fd.append("file", file); await api.req("POST", `/api/projects/${p.id}/transcript`, fd, true); toast("Transcript parsed", "ok"); viewUpload(p.id); }
    catch (e) { $("#tr-error").innerHTML = `<div class="banner err" style="margin-top:10px">⚠ ${esc(e.message)}</div>`; }
  }
  $("#tr-paste").onclick = async () => { $("#tr-error").innerHTML = ""; try { const p = await ensureProject(); await api.post(`/api/projects/${p.id}/transcript`, { text: $("#tr-text").value }); toast("Transcript parsed", "ok"); viewUpload(p.id); } catch (e) { $("#tr-error").innerHTML = `<div class="banner err" style="margin-top:10px">⚠ ${esc(e.message)}</div>`; } };
  $("#go").onclick = async () => { try { await api.post(`/api/projects/${project.id}/analyze`); location.hash = `#/p/${project.id}/processing`; } catch (e) { toast(e.message, "err"); } };
}

// ---------------------------------------------------------------- project overview
async function viewProject(pid) {
  const p = await api.get(`/api/projects/${pid}`);
  crumbs([["Projects", "#/projects"], [p.name]]);
  const si = p.source_info, ti = p.transcript_info;
  $("#view").innerHTML = `
    ${projectTabs(p, "overview")}
    <div class="section-title"><div><h1 id="title" contenteditable="true" spellcheck="false" title="Click to rename">${esc(p.name)}</h1><div class="muted small">created ${ago(p.created_at)} · ${statusPill(p.status)}</div></div>
      <div class="actions">
        ${p.clip_counts.total ? `<a class="btn" href="#/p/${p.id}/clips">✦ ${p.clip_counts.total} moments</a>` : ""}
        ${["ready", "analyzed", "error"].includes(p.status) ? `<button class="btn btn-primary" id="analyze">✦ ${p.clip_counts.total ? "Re-run moment selection" : "Start AI analysis"}</button>` : ""}
        ${["analyzing", "processing"].includes(p.status) ? `<a class="btn btn-primary" href="#/p/${p.id}/processing">View processing</a>` : ""}
        <button class="btn btn-danger" id="del">Delete</button>
      </div></div>
    ${p.error ? `<div class="banner err" style="margin-bottom:16px">⚠ Last job failed: ${esc(p.error)}</div>` : ""}
    <div class="grid cols-3">
      <div class="card"><div class="card-head"><h2>Source video</h2>${si ? statusPill("done") : ""}</div>${si ? `<dl class="kv"><dt>File</dt><dd>${esc(p.source_filename)}</dd><dt>Resolution</dt><dd>${si.width}×${si.height}</dd><dt>Duration</dt><dd>${fmtClock(si.duration * 1000)}</dd><dt>Frame rate</dt><dd>${si.fps?.toFixed(2)} fps</dd><dt>Audio</dt><dd>${si.has_audio ? "yes" : "none"}</dd><dt>Size</dt><dd>${fmtBytes(si.size_bytes)}</dd></dl>` : `<p class="muted">No video yet.</p><a class="btn btn-sm" href="#/p/${p.id}/upload">Upload</a>`}</div>
      <div class="card"><div class="card-head"><h2>Transcript</h2>${ti ? statusPill("done") : ""}</div>${ti ? `<dl class="kv"><dt>File</dt><dd>${esc(p.transcript_filename)}</dd><dt>Sentences</dt><dd>${ti.sentences}</dd><dt>Words</dt><dd>${ti.words}</dd><dt>Speakers</dt><dd>${ti.speakers.join(", ") || "—"}</dd><dt>Covers</dt><dd>${fmtClock(ti.duration_ms)}</dd></dl>` : `<p class="muted">No transcript yet.</p><a class="btn btn-sm" href="#/p/${p.id}/upload">Add transcript</a>`}</div>
      <div class="card"><div class="card-head"><h2>Results</h2></div><dl class="kv"><dt>Moments</dt><dd>${p.clip_counts.total}</dd><dt>Selected</dt><dd>${p.clip_counts.selected}</dd><dt>Rendered</dt><dd>${p.clip_counts.rendered}</dd><dt>Output</dt><dd>${p.settings.output_width}×${p.settings.output_height}</dd><dt>Clip length</dt><dd>${p.settings.min_clip_s}–${p.settings.max_clip_s} s</dd><dt>Speaker model</dt><dd>${p.settings.asd_backend}</dd></dl></div>
    </div>
    <div class="section"><div class="section-title"><h2>Job history</h2></div>
      <div class="card pad-0">${p.jobs.length ? `<table class="table"><thead><tr><th>Job</th><th>Status</th><th>Started</th><th>Duration</th><th>Result</th></tr></thead>${p.jobs.map((j) => `<tr><td><b>${esc(j.kind)}</b> <span class="dim mono small">${j.id}</span></td><td>${statusPill(j.status)}</td><td class="small dim">${j.started_at ? ago(j.started_at) : "—"}</td><td class="small mono">${j.started_at && j.finished_at ? (j.finished_at - j.started_at).toFixed(1) + "s" : "—"}</td><td class="small muted">${esc(j.error || j.progress?.message || "")}</td></tr>`).join("")}</table>` : `<p class="muted" style="padding:16px">No jobs yet.</p>`}</div></div>`;
  $("#title").onblur = async () => { const n = $("#title").textContent.trim(); if (n && n !== p.name) { await api.patch(`/api/projects/${pid}`, { name: n }); toast("Renamed", "ok"); } };
  $("#title").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); $("#title").blur(); } };
  $("#analyze") && ($("#analyze").onclick = async () => { try { await api.post(`/api/projects/${pid}/analyze`); location.hash = `#/p/${pid}/processing`; } catch (e) { toast(e.message, "err"); } });
  $("#del").onclick = async () => { if (await confirmModal("Delete project?", "This removes the uploaded video, all intermediate files and rendered clips.")) { try { await api.del(`/api/projects/${pid}`); toast("Project deleted", "ok"); location.hash = "#/projects"; } catch (e) { toast(e.message, "err"); } } };
  if (["analyzing", "processing"].includes(p.status)) poll(() => viewProject(pid), 4000);
}

// ---------------------------------------------------------------- processing workspace
async function viewProcessing(pid) {
  const p = await api.get(`/api/projects/${pid}`);
  crumbs([["Projects", "#/projects"], [p.name, `#/p/${pid}`], ["Processing"]]);
  $("#view").innerHTML = `${projectTabs(p, "processing")}<div id="proc"></div>`;
  const render = async () => {
    const proj = await api.get(`/api/projects/${pid}`);
    const job = proj.jobs[0];
    const clips = proj.clips;
    const pr = job?.progress || {}; const steps = pr.steps || {};
    const live = job && ["queued", "running"].includes(job.status);
    const clipOf = (id) => clips.find((c) => c.id === id);
    const stepRows = state.steps.map((s, i) => {
      const st = steps[s.id] || {}; const status = st.status || "pending";
      const meta = [st.message, st.clip && clipOf(st.clip) ? `clip ${clipOf(st.clip).rank + 1}` : "", st.done && st.total ? `${st.done}/${st.total} frames` : "", st.started_at && st.finished_at ? `${(st.finished_at - st.started_at).toFixed(1)}s` : ""].filter(Boolean).join(" · ");
      const bar = status === "running" && st.total ? `<div class="bar progress"><i style="width:${Math.round(100 * st.done / st.total)}%"></i></div>` : "";
      return `<div class="step ${status}"><span class="n">${["done", "reused"].includes(status) ? "✓" : status === "failed" ? "!" : i + 1}</span><div><div>${esc(s.label)}${status === "reused" ? ' <span class="pill" style="margin-left:6px">reused</span>' : ""}</div><div class="meta">${esc(meta)}</div></div><div>${status === "running" ? '<span class="pill accent"><span class="spin"></span>running</span>' : ""}</div>${bar}</div>`;
    }).join("");
    $("#proc").innerHTML = `
      <div class="editor" style="grid-template-columns: minmax(0,1fr) 380px">
        <div class="card">
          <div class="card-head"><h2>Pipeline</h2>${job ? statusPill(job.status) : '<span class="pill">idle</span>'}</div>
          ${job ? `<div class="muted small" style="margin-bottom:12px">${esc(job.kind)} job <span class="mono">${job.id}</span> · ${esc(pr.message || "")}${pr.clip_total ? ` · clip ${pr.clip_index}/${pr.clip_total}` : ""}</div>` : `<p class="muted">No job yet. Start analysis from the overview or upload tab.</p>`}
          <div class="steps">${stepRows}</div>
          ${job?.error ? `<div class="banner err" style="margin-top:14px">⚠ ${esc(job.error)}</div>${pr.traceback ? `<details style="margin-top:8px"><summary class="dim small">traceback</summary><pre class="mono small" style="white-space:pre-wrap;color:var(--text-2)">${esc(pr.traceback)}</pre></details>` : ""}` : ""}
          <div class="actions" style="margin-top:16px">
            ${live ? `<button class="btn btn-danger" id="cancel">■ Cancel job</button>` : ""}
            ${!live && ["ready", "analyzed", "error"].includes(proj.status) ? `<button class="btn btn-primary" id="analyze">✦ ${clips.length ? "Re-run moment selection" : "Start AI analysis"}</button>` : ""}
            ${!live && clips.some((c) => c.selected) ? `<button class="btn" id="process">▶ Process ${clips.filter((c) => c.selected).length} selected clip(s)</button>` : ""}
            ${clips.length ? `<a class="btn" href="#/p/${pid}/clips">Open clips</a>` : ""}
          </div>
          <p class="help" style="margin-top:12px">Steps 1–3 run once per project; steps 4–11 run per selected clip. Completed intermediates are reused — re-rendering never repeats face detection or moment selection.</p>
        </div>
        <div class="card">
          <div class="card-head"><h2>Clips</h2></div>
          ${clips.length ? `<div class="list">${clips.map((c) => `<div class="list-item"><div style="width:56px;height:32px;border-radius:6px;background:#000 url(/api/projects/${pid}/clips/${c.id}/thumbnail) center/cover;flex:none"></div><div style="flex:1;min-width:0"><div class="small"><b>Clip ${c.rank + 1}</b> <span class="mono dim">${fmtTime(c.start_ms)}–${fmtTime(c.end_ms)}</span></div><div class="dim small" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(c.text)}</div></div>${statusPill(c.status)}</div>`).join("")}</div>` : `<p class="muted">Moments appear here after analysis.</p>`}
          <div class="card-head" style="margin-top:18px"><h2>Job history</h2></div>
          <div class="list">${proj.jobs.slice(0, 8).map((j) => `<div class="list-item"><span class="act-ico">⚙</span><div class="small"><b>${esc(j.kind)}</b> ${statusPill(j.status)}<div class="dim">${j.started_at ? ago(j.started_at) : "queued"}${j.error ? " · " + esc(j.error.slice(0, 80)) : ""}</div></div></div>`).join("") || '<p class="muted small">—</p>'}</div>
        </div>
      </div>`;
    $("#cancel") && ($("#cancel").onclick = async () => { await api.post(`/api/jobs/${job.id}/cancel`); toast("Cancellation requested — stops at the next frame/stage boundary"); });
    $("#analyze") && ($("#analyze").onclick = async () => { try { await api.post(`/api/projects/${pid}/analyze`); render(); } catch (e) { toast(e.message, "err"); } });
    $("#process") && ($("#process").onclick = async () => { try { await api.post(`/api/projects/${pid}/process`, {}); render(); } catch (e) { toast(e.message, "err"); } });
  };
  await render(); poll(render, 1500);
}

// ---------------------------------------------------------------- clip discovery
const REASONS = { hook: "strong hook words", intensity: "emotional intensity", question: "opens with a question", numbers: "concrete numbers", density: "dense, varied language", speech_rate: "energetic pace", dialogue: "back-and-forth dialogue", duration_fit: "ideal length", strong_opener: "clean opening", clean_ending: "complete ending", llm_rank: "LLM ranked" };
function reasonsFor(c) {
  const f = c.features || {};
  const pos = Object.entries(f).filter(([k, v]) => k !== "filler" && v >= 0.6).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k]) => REASONS[k] || k);
  if (f.filler >= 0.5) pos.push("⚠ filler words");
  if (f.llm_rank === 0) pos.unshift("LLM top pick");
  return pos;
}
async function viewClips(pid) {
  const p = await api.get(`/api/projects/${pid}`);
  crumbs([["Projects", "#/projects"], [p.name, `#/p/${pid}`], ["Clips"]]);
  $("#view").innerHTML = `${projectTabs(p, "clips")}<div id="clips"></div>`;
  let compare = new Set();
  const render = async () => {
    const proj = await api.get(`/api/projects/${pid}`);
    const clips = proj.clips; const live = ["analyzing", "processing"].includes(proj.status);
    const sel = clips.filter((c) => c.selected);
    $("#clips").innerHTML = `
      <div class="section-title"><div><h1>AI clip discovery</h1><div class="muted small">${clips.length} non-overlapping, sentence-aligned moments ranked by the ${clips[0]?.scorer || proj.settings.scorer} scorer · ${sel.length} selected</div></div>
        <div class="actions">
          <button class="btn" id="rerun" ${live ? "disabled" : ""}>↻ Re-run selection</button>
          <button class="btn" id="compare" ${compare.size < 2 ? "disabled" : ""}>⇄ Compare (${compare.size})</button>
          <button class="btn btn-primary" id="process" ${live || !sel.length ? "disabled" : ""}>▶ Process &amp; export ${sel.length} selected</button>
        </div></div>
      ${live ? `<div class="banner info" style="margin-bottom:16px"><span class="spin pill accent" style="border:0"></span> A job is running — <a href="#/p/${pid}/processing" style="text-decoration:underline">open the processing view</a>.</div>` : ""}
      ${clips.length ? `<div class="clip-grid">${clips.map((c) => clipCard(pid, c, compare.has(c.id))).join("")}</div>` : `<div class="empty"><div class="big">✦</div>No moments yet. ${proj.status === "ready" ? `<button class="btn btn-primary" id="analyze0" style="margin-top:10px">Start AI analysis</button>` : `<a href="#/p/${pid}/upload" style="color:var(--accent-2)">Upload a video and transcript first.</a>`}</div>`}`;
    $$("[data-sel]").forEach((cb) => (cb.onchange = async () => { await api.patch(`/api/projects/${pid}/clips/${cb.dataset.sel}`, { selected: cb.checked }); render(); }));
    $$("[data-cmp]").forEach((cb) => (cb.onchange = () => { cb.checked ? compare.add(cb.dataset.cmp) : compare.delete(cb.dataset.cmp); $("#compare").disabled = compare.size < 2; $("#compare").textContent = `⇄ Compare (${compare.size})`; }));
    $$("[data-preview]").forEach((b) => (b.onclick = () => previewModal(pid, clips.find((c) => c.id === b.dataset.preview))));
    $("#rerun").onclick = async () => { if (await confirmModal("Re-run moment selection?", "Current candidates and their rendered outputs will be replaced with a fresh selection using the project settings.", "Re-run")) { try { await api.post(`/api/projects/${pid}/analyze`); location.hash = `#/p/${pid}/processing`; } catch (e) { toast(e.message, "err"); } } };
    $("#process").onclick = async () => { try { await api.post(`/api/projects/${pid}/process`, {}); location.hash = `#/p/${pid}/processing`; } catch (e) { toast(e.message, "err"); } };
    $("#compare").onclick = () => compareModal(clips.filter((c) => compare.has(c.id)));
    $("#analyze0") && ($("#analyze0").onclick = async () => { await api.post(`/api/projects/${pid}/analyze`); location.hash = `#/p/${pid}/processing`; });
    return live;
  };
  if (await render()) poll(async () => { if (!(await render())) stopPollers(); }, 3000);
}
function clipCard(pid, c, cmp) {
  const sp = c.speaker_faces ? `${c.speaker_faces.length} face${c.speaker_faces.length !== 1 ? "s" : ""} · ${c.speaker_segments} speaker segment${c.speaker_segments !== 1 ? "s" : ""}` : "speaker tracking after processing";
  return `<div class="clip ${c.selected ? "selected" : ""}">
    <div class="thumb" style="background-image:url(/api/projects/${pid}/clips/${c.id}/thumbnail)">
      <span class="rank pill accent">#${c.rank + 1}</span><span class="dur">${(c.duration_ms / 1000).toFixed(1)}s</span>
      <a class="play" data-preview="${c.id}" title="Preview"><b>▶</b></a>
    </div>
    <div class="body">
      <div class="times"><span>${fmtTime(c.start_ms)} → ${fmtTime(c.end_ms)}</span>${statusPill(c.status)}</div>
      <div class="excerpt">“${esc(c.text)}”</div>
      <div class="reasons">${reasonsFor(c).map((r) => `<span class="pill">${esc(r)}</span>`).join("") || '<span class="dim small">—</span>'}</div>
      <div class="dim small">${esc(sp)}</div>
      <div class="foot">
        <label class="switch small"><input type="checkbox" class="checkbox" data-sel="${c.id}" ${c.selected ? "checked" : ""}> Selected</label>
        <span class="score">${c.score.toFixed(2)}<small>score</small></span>
      </div>
      <div class="foot">
        <label class="switch small dim"><input type="checkbox" data-cmp="${c.id}" ${cmp ? "checked" : ""}> compare</label>
        <div class="actions">
          ${c.files.output ? `<a class="btn btn-sm" href="#/p/${pid}/edit/${c.id}">✎ Editor</a><a class="btn btn-sm btn-primary" href="/api/projects/${pid}/clips/${c.id}/media/output?download=1">⤓ MP4</a>` : `<button class="btn btn-sm" data-preview="${c.id}">▶ Preview</button>${c.files.phase2 ? `<a class="btn btn-sm" href="#/p/${pid}/edit/${c.id}">✎ Editor</a>` : ""}`}
        </div>
      </div>
    </div></div>`;
}
function previewModal(pid, c) {
  modal(`<h2>Clip ${c.rank + 1} preview <span class="dim small mono">${fmtTime(c.start_ms)}–${fmtTime(c.end_ms)}</span></h2>
    <video id="pv" controls style="width:100%;border-radius:10px;background:#000" src="/api/projects/${pid}/source#t=${(c.start_ms / 1000).toFixed(3)},${(c.end_ms / 1000).toFixed(3)}"></video>
    <p class="muted small" style="margin-top:10px">“${esc(c.text)}”</p>
    <div class="feature-bars" style="margin-top:10px">${Object.entries(c.features || {}).map(([k, v]) => `<span class="dim">${esc(k)}</span><div class="fb"><i class="${k === "filler" ? "neg" : ""}" style="width:${Math.round(100 * Math.min(1, v))}%"></i></div><span class="mono dim">${v.toFixed(2)}</span>`).join("")}</div>
    <div class="foot"><a class="btn" href="#/p/${pid}/edit/${c.id}" ${c.files.phase2 ? "" : 'title="Process the clip first for crop + tracking" style="opacity:.6"'}>Open in editor</a><button class="btn btn-primary" data-x>Close</button></div>`,
    (root, close) => {
      const v = $("#pv", root); const end = c.end_ms / 1000;
      v.addEventListener("timeupdate", () => { if (v.currentTime >= end) { v.pause(); v.currentTime = c.start_ms / 1000; } });
      v.addEventListener("loadedmetadata", () => { v.currentTime = c.start_ms / 1000; v.play().catch(() => {}); });
      $("[data-x]", root).onclick = close;
    });
}
function compareModal(clips) {
  const keys = Object.keys(clips[0].features || {});
  modal(`<h2>Compare candidates</h2><div style="overflow:auto"><table class="table"><thead><tr><th>Feature</th>${clips.map((c) => `<th>#${c.rank + 1} · ${(c.duration_ms / 1000).toFixed(0)}s</th>`).join("")}</tr></thead>
    <tr><td><b>Score</b></td>${clips.map((c) => `<td class="mono"><b>${c.score.toFixed(2)}</b></td>`).join("")}</tr>
    <tr><td>Range</td>${clips.map((c) => `<td class="mono small">${fmtTime(c.start_ms)}–${fmtTime(c.end_ms)}</td>`).join("")}</tr>
    ${keys.map((k) => { const best = Math.max(...clips.map((c) => c.features[k] ?? 0)); return `<tr><td class="muted">${esc(k)}</td>${clips.map((c) => `<td class="mono small" style="${(c.features[k] ?? 0) === best && k !== "filler" ? "color:var(--ok)" : ""}">${(c.features[k] ?? 0).toFixed(2)}</td>`).join("")}</tr>`; }).join("")}
    <tr><td>Excerpt</td>${clips.map((c) => `<td class="small muted">${esc(c.text.slice(0, 160))}…</td>`).join("")}</tr></table></div><div class="foot"><button class="btn btn-primary" data-x>Close</button></div>`,
    (root, close) => ($("[data-x]", root).onclick = close));
}

// ---------------------------------------------------------------- editor
async function viewEditor(pid, cid) {
  const [p, c] = await Promise.all([api.get(`/api/projects/${pid}`), api.get(`/api/projects/${pid}/clips/${cid}`)]);
  crumbs([["Projects", "#/projects"], [p.name, `#/p/${pid}`], ["Clips", `#/p/${pid}/clips`], [`Clip ${c.rank + 1}`]]);
  if (!c.files.clip) {
    $("#view").innerHTML = `${projectTabs(p, "clips")}<div class="empty"><div class="big">✎</div>This clip has not been processed yet, so there is no crop plan or tracking data to show.<div style="margin-top:12px"><button class="btn btn-primary" id="go">▶ Process this clip</button></div></div>`;
    $("#go").onclick = async () => { try { await api.post(`/api/projects/${pid}/process`, { clip_ids: [cid] }); location.hash = `#/p/${pid}/processing`; } catch (e) { toast(e.message, "err"); } };
    return;
  }
  let tracks = null; try { tracks = await api.get(`/api/projects/${pid}/clips/${cid}/tracks`); } catch {}
  const srcKind = c.files.output ? "clip" : "clip";
  $("#view").innerHTML = `${projectTabs(p, "clips")}
    <div class="section-title"><div><h1>Clip ${c.rank + 1} <span class="muted" style="font-weight:400;font-size:14px">${fmtTime(c.start_ms)} → ${fmtTime(c.end_ms)} · ${(c.duration_ms / 1000).toFixed(1)}s</span></h1><div class="muted small">${statusPill(c.status)} ${c.asd ? `· speaker model <b>${esc(c.asd.backend)}</b> · ${c.segments.length} segment(s)` : ""}</div></div>
      <div class="actions">${c.files.output ? `<a class="btn btn-primary" href="/api/projects/${pid}/clips/${cid}/media/output?download=1">⤓ Download MP4</a><button class="btn" id="rerender">↻ Re-render</button>` : `<button class="btn btn-primary" id="process">▶ Render</button>`}</div></div>
    ${!tracks ? `<div class="banner warn" style="margin-bottom:14px">Tracking data is not available yet (detection/speaker stages have not finished).</div>` : ""}
    <div class="editor">
      <div>
        <div class="stage" id="stage"><span class="badge pill" id="badge">source clip</span><video id="v" src="/api/projects/${pid}/clips/${cid}/media/${srcKind}" playsinline></video><canvas id="ov"></canvas></div>
        <div class="transport">
          <button class="btn btn-icon" id="play">▶</button>
          <span class="time" id="time">0:00.0 / ${fmtTime(c.duration_ms)}</span>
          <div class="timeline" id="tl"><div class="head" id="head"></div></div>
        </div>
        <div class="toggle-row" style="margin-top:10px">
          <button class="toggle on" data-layer="boxes">Face boxes</button>
          <button class="toggle on" data-layer="active">Active speaker</button>
          <button class="toggle on" data-layer="crop">Crop window</button>
          <button class="toggle" data-layer="scores">Scores</button>
          <span style="flex:1"></span>
          <button class="toggle" id="dbg" ${c.files.debug ? "" : "disabled title='debug.mp4 not rendered'"}>Debug overlay video</button>
          <button class="toggle" id="out" ${c.files.output ? "" : "disabled"}>Rendered output</button>
        </div>
        <div class="legend" style="margin-top:10px">${(tracks?.segments || []).map((s) => s.speaker_face_id).filter((v, i, a) => v !== null && a.indexOf(v) === i).map((id) => `<span><i style="background:${faceColor(id)}"></i>face ${id}</span>`).join("")}<span><i style="background:#2bff6a"></i>crop window</span></div>
      </div>
      <div>
        <div class="card"><div class="card-head"><h2>Vertical preview</h2><span class="pill">${c.plan ? c.plan.output.aspect_ratio : "9:16"}</span></div><div class="vertical"><canvas id="vert"></canvas></div><p class="help" style="margin-top:8px">Live crop of the source using the exact per-frame rectangle from <code>crop_plan.json</code> — what the renderer produced.</p></div>
        <div class="card" style="margin-top:14px"><div class="card-head"><h2>Clip info</h2></div>
          <dl class="kv"><dt>In source</dt><dd>${fmtTime(c.start_ms)} → ${fmtTime(c.end_ms)}</dd><dt>Sentences</dt><dd>${c.first_sentence}–${c.last_sentence}</dd><dt>Score</dt><dd>${c.score.toFixed(3)}</dd>${c.plan ? `<dt>Output</dt><dd>${c.plan.output.width}×${c.plan.output.height}</dd><dt>Source</dt><dd>${c.plan.source.width}×${c.plan.source.height} @ ${c.plan.source.fps} fps</dd>` : ""}${c.asd ? `<dt>Threshold</dt><dd>${c.asd.speaking_threshold}</dd>` : ""}</dl>
          <div style="margin-top:12px" class="small muted">“${esc(c.text)}”</div>
          ${c.segments ? `<div class="card-head" style="margin-top:14px"><h3>Speaker segments</h3></div><div class="list">${c.segments.map((s) => `<div class="list-item small"><i style="width:10px;height:10px;border-radius:2px;background:${s.speaker_face_id === null ? "#666" : faceColor(s.speaker_face_id)}"></i><span class="mono">${fmtTime(s.start_ms)}–${fmtTime(s.end_ms)}</span><span class="muted">${s.speaker_face_id === null ? "no face" : "face " + s.speaker_face_id}</span></div>`).join("")}</div>` : ""}
          <div class="actions" style="margin-top:14px">${["moment", "speakers", "crop_plan", "manifest"].map((n) => `<a class="btn btn-sm btn-ghost" href="/api/projects/${pid}/clips/${cid}/artifact/${n}">${n}.json</a>`).join("")}</div>
        </div>
      </div>
    </div>`;

  const v = $("#v"), ov = $("#ov"), vert = $("#vert"), tl = $("#tl"), head = $("#head");
  const layers = { boxes: true, active: true, crop: true, scores: false };
  let mode = "clip";
  $$("[data-layer]").forEach((b) => (b.onclick = () => { layers[b.dataset.layer] = !layers[b.dataset.layer]; b.classList.toggle("on", layers[b.dataset.layer]); draw(); }));
  const setMode = (m) => { const t = v.currentTime, playing = !v.paused; mode = m; v.src = `/api/projects/${pid}/clips/${cid}/media/${m === "clip" ? "clip" : m}`; v.currentTime = t; if (playing) v.play().catch(() => {}); $("#badge").textContent = { clip: "source clip + live overlay", debug: "debug.mp4 (rendered overlay)", output: "output.mp4 (rendered vertical)" }[m]; $("#dbg").classList.toggle("on", m === "debug"); $("#out").classList.toggle("on", m === "output"); ov.style.display = m === "clip" ? "" : "none"; };
  $("#dbg").onclick = () => setMode(mode === "debug" ? "clip" : "debug");
  $("#out").onclick = () => setMode(mode === "output" ? "clip" : "output");
  $("#play").onclick = () => (v.paused ? v.play() : v.pause());
  v.onplay = () => ($("#play").textContent = "❚❚"); v.onpause = () => ($("#play").textContent = "▶");
  tl.onclick = (e) => { const r = tl.getBoundingClientRect(); v.currentTime = ((e.clientX - r.left) / r.width) * (v.duration || c.duration_ms / 1000); };
  if (tracks) { const dur = c.duration_ms; tl.insertAdjacentHTML("afterbegin", tracks.segments.map((s) => `<div class="seg" style="left:${100 * s.start_ms / dur}%;width:${100 * (s.end_ms - s.start_ms) / dur}%;background:${s.speaker_face_id === null ? "#666" : faceColor(s.speaker_face_id)}"></div>`).join("")); }
  tl.insertAdjacentHTML("beforeend", `<span class="label">${fmtTime(c.start_ms)} in source</span>`);
  $("#rerender") && ($("#rerender").onclick = async () => { try { await api.post(`/api/projects/${pid}/rerender`, { clip_ids: [cid] }); location.hash = `#/p/${pid}/processing`; } catch (e) { toast(e.message, "err"); } });
  $("#process") && ($("#process").onclick = async () => { try { await api.post(`/api/projects/${pid}/process`, { clip_ids: [cid] }); location.hash = `#/p/${pid}/processing`; } catch (e) { toast(e.message, "err"); } });

  const fps = tracks?.video.fps || c.plan?.source.fps || 30;
  const frameAt = () => tracks ? tracks.frames[Math.min(tracks.frames.length - 1, Math.max(0, Math.round(v.currentTime * fps)))] : null;
  function draw() {
    const W = v.videoWidth, H = v.videoHeight; if (!W) return;
    if (ov.width !== W || ov.height !== H) { ov.width = W; ov.height = H; }
    const g = ov.getContext("2d"); g.clearRect(0, 0, W, H);
    const f = frameAt();
    if (f && mode === "clip") {
      for (const face of f.faces) {
        const [x0, y0, x1, y1] = face.b; const X = x0 * W, Y = y0 * H, w = (x1 - x0) * W, h = (y1 - y0) * H; const col = faceColor(face.id); const isActive = f.a === face.id;
        if (layers.boxes || (layers.active && isActive)) { g.lineWidth = isActive && layers.active ? 5 : 1.5; g.strokeStyle = col; g.strokeRect(X, Y, w, h); }
        if (layers.active && isActive) { g.fillStyle = col; g.font = `bold ${Math.max(14, W / 60)}px Inter`; const ty = Y - 10 > 16 ? Y - 10 : Y + h + 20; g.fillText(`face ${face.id} · SPEAKING`, X, ty); }
        if (layers.scores && face.s != null) { const bw = Math.max(60, w), frac = Math.min(1, Math.max(0, 0.4 + (face.s - tracks.threshold) * (0.4 / tracks.threshold))); g.fillStyle = "#333"; g.fillRect(X, Y + h + 6, bw, 8); g.fillStyle = col; g.fillRect(X, Y + h + 6, bw * frac, 8); g.fillStyle = "#fff"; g.fillRect(X + bw * 0.4, Y + h + 4, 1.5, 12); g.font = `${Math.max(11, W / 90)}px JetBrains Mono`; g.fillText(face.s.toFixed(3), X + bw + 6, Y + h + 14); }
      }
      if (layers.crop) { const [cx, cy, cw, ch] = f.c; g.lineWidth = 3; g.strokeStyle = "#2bff6a"; g.strokeRect(cx, cy, cw, ch); g.fillStyle = "rgba(0,0,0,.45)"; g.fillRect(0, 0, cx, H); g.fillRect(cx + cw, 0, W - cx - cw, H); g.fillRect(cx, 0, cw, cy); g.fillRect(cx, cy + ch, cw, H - cy - ch); }
    }
    // vertical preview: crop of the current video frame via crop_plan
    const vg = vert.getContext("2d"); const ow = tracks?.output.width || c.plan?.output.width || 1080, oh = tracks?.output.height || c.plan?.output.height || 1920;
    if (vert.width !== ow / 2 || vert.height !== oh / 2) { vert.width = ow / 2; vert.height = oh / 2; }
    if (mode === "output") { vg.drawImage(v, 0, 0, vert.width, vert.height); }
    else if (f) { const [cx, cy, cw, ch] = f.c; vg.drawImage(v, cx, cy, cw, ch, 0, 0, vert.width, vert.height); }
    $("#time").textContent = `${fmtTime(v.currentTime * 1000)} / ${fmtTime(c.duration_ms)}`;
    head.style.left = `${100 * v.currentTime / (v.duration || c.duration_ms / 1000)}%`;
  }
  let raf; const loop = () => { draw(); raf = requestAnimationFrame(loop); }; loop();
  state.pollers.add(setInterval(() => { if (!document.body.contains(v)) cancelAnimationFrame(raf); }, 1000));
  v.addEventListener("loadeddata", draw);
}

// ---------------------------------------------------------------- settings
async function viewSettings(pid) {
  const p = await api.get(`/api/projects/${pid}`);
  crumbs([["Projects", "#/projects"], [p.name, `#/p/${pid}`], ["Settings"]]);
  const s = p.settings; const h = state.health;
  $("#view").innerHTML = `${projectTabs(p, "settings")}
    <div class="grid cols-2">
      <div class="card"><div class="card-head"><h2>Moment selection</h2></div>
        <div class="form-grid">
          <div class="field"><label>Min clip length (s)</label><input class="input" type="number" name="min_clip_s" value="${s.min_clip_s}" min="5" max="600"></div>
          <div class="field"><label>Max clip length (s)</label><input class="input" type="number" name="max_clip_s" value="${s.max_clip_s}" min="6" max="600"></div>
          <div class="field"><label>Moments to discover</label><input class="input" type="number" name="max_clips" value="${s.max_clips}" min="1" max="20"></div>
          <div class="field"><label>Scorer</label><select class="input" name="scorer"><option value="heuristic" ${s.scorer === "heuristic" ? "selected" : ""}>Heuristic (local, default)</option><option value="llm" ${s.scorer === "llm" ? "selected" : ""}>LLM re-ranker (local endpoint)</option></select></div>
          <div class="field"><label>LLM endpoint</label><input class="input" name="llm_endpoint" value="${esc(s.llm_endpoint)}"></div>
          <div class="field"><label>LLM model</label><input class="input" name="llm_model" value="${esc(s.llm_model)}"></div>
        </div>
        <label class="switch" style="margin-top:12px"><input type="checkbox" name="allow_hosted_llm" ${s.allow_hosted_llm ? "checked" : ""}> <span>Allow a <b>hosted</b> LLM endpoint <span class="dim small">(off by default — non-local endpoints are refused)</span></span></label>
      </div>
      <div class="card"><div class="card-head"><h2>Speaker tracking &amp; output</h2></div>
        <div class="form-grid">
          <div class="field"><label>Active-speaker model</label><select class="input" name="asd_backend"><option value="heuristic" ${s.asd_backend === "heuristic" ? "selected" : ""}>Lip-motion heuristic (local)</option><option value="lrasd" ${s.asd_backend === "lrasd" ? "selected" : ""} ${h?.lrasd_available ? "" : "disabled"}>LR-ASD neural (audio-visual)${h?.lrasd_available ? "" : " — install extras"}</option></select></div>
          <div class="field"><label>Min speaker segment (ms)</label><input class="input" type="number" name="min_segment_ms" value="${s.min_segment_ms}" min="200" step="100"></div>
          <div class="field"><label>Output width</label><input class="input" type="number" name="output_width" value="${s.output_width}" step="2"></div>
          <div class="field"><label>Output height</label><input class="input" type="number" name="output_height" value="${s.output_height}" step="2"></div>
          <div class="field"><label>Quality (CRF, lower = better)</label><input class="input" type="number" name="render_crf" value="${s.render_crf}" min="0" max="51"></div>
          <div class="field"><label>Encoder speed</label><select class="input" name="render_preset">${["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"].map((x) => `<option ${s.render_preset === x ? "selected" : ""}>${x}</option>`).join("")}</select></div>
        </div>
        <label class="switch" style="margin-top:12px"><input type="checkbox" name="debug_overlay" ${s.debug_overlay ? "checked" : ""}> Render a debug overlay video (debug.mp4) alongside each clip</label>
      </div>
    </div>
    <div class="actions" style="margin-top:16px"><button class="btn btn-primary" id="save">Save settings</button><span class="help">Changes apply to the next analysis / processing run. Re-rendering with new output size requires “Re-render”.</span></div>`;
  $("#save").onclick = async () => {
    const patch = {}; $$(".view [name]").forEach((el) => { patch[el.name] = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value; });
    try { await api.patch(`/api/projects/${pid}`, { settings: patch }); toast("Settings saved", "ok"); } catch (e) { toast(e.message, "err", 7000); }
  };
}

// ---------------------------------------------------------------- export center
async function viewExports() {
  crumbs([["Export center"]]);
  const render = async () => {
    const ps = await api.get("/api/projects");
    const full = await Promise.all(ps.map((p) => api.get(`/api/projects/${p.id}`)));
    const rows = full.flatMap((p) => p.clips.filter((c) => c.files.output || c.status === "processing" || c.status === "error").map((c) => ({ p, c })));
    const jobs = await api.get("/api/jobs");
    const active = jobs.filter((j) => ["queued", "running"].includes(j.status));
    $("#view").innerHTML = `
      <div class="section-title"><div><h1>Export center</h1><div class="muted small">${rows.filter((r) => r.c.files.output).length} rendered clip(s) across ${full.filter((p) => p.clip_counts.rendered).length} project(s)</div></div></div>
      ${active.length ? `<div class="card" style="margin-bottom:16px"><div class="card-head"><h2>Export status</h2></div>${active.map((j) => jobSummary(j)).join("")}</div>` : ""}
      ${full.filter((p) => p.clips.some((c) => c.files.output)).map((p) => `
        <div class="card pad-0" style="margin-bottom:16px">
          <div class="card-head" style="padding:16px 18px 0"><h2><a href="#/p/${p.id}/clips">${esc(p.name)}</a></h2><div class="actions"><button class="btn btn-sm" data-zip="${p.id}">⤓ Download all as ZIP</button><button class="btn btn-sm" data-zip-sel="${p.id}" disabled>⤓ Download checked</button></div></div>
          <table class="table"><thead><tr><th></th><th>Clip</th><th>Range</th><th>Status</th><th>Size</th><th>Rendered</th><th></th></tr></thead>
          ${p.clips.filter((c) => c.files.output || c.status === "processing" || c.status === "error").map((c) => `<tr><td><input type="checkbox" class="checkbox" data-pick="${p.id}" value="${c.id}" ${c.files.output ? "" : "disabled"}></td><td><div style="display:flex;gap:10px;align-items:center"><div style="width:56px;height:32px;border-radius:6px;background:#000 url(/api/projects/${p.id}/clips/${c.id}/thumbnail) center/cover"></div><div><b>Clip ${c.rank + 1}</b><div class="dim small" style="max-width:260px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(c.text)}</div></div></div></td><td class="mono small">${fmtTime(c.start_ms)}–${fmtTime(c.end_ms)}</td><td>${statusPill(c.status)}${c.error ? `<div class="small" style="color:var(--err)">${esc(c.error.slice(0, 80))}</div>` : ""}</td><td class="mono small">${fmtBytes(c.output_size_bytes)}</td><td class="small dim">${c.rendered_at ? ago(c.rendered_at) : "—"}</td><td><div class="actions">${c.files.output ? `<a class="btn btn-sm btn-primary" href="/api/projects/${p.id}/clips/${c.id}/media/output?download=1">⤓ MP4</a>` : ""}${c.files.debug ? `<a class="btn btn-sm" href="/api/projects/${p.id}/clips/${c.id}/media/debug?download=1">debug</a>` : ""}<a class="btn btn-sm btn-ghost" href="#/p/${p.id}/edit/${c.id}">✎</a><button class="btn btn-sm btn-ghost" data-rr="${p.id}:${c.id}" title="Re-render (reuses detection)">↻</button></div></td></tr>`).join("")}</table></div>`).join("") || `<div class="empty"><div class="big">⤓</div>No rendered clips yet. Process selected moments from a project's <b>Clips</b> tab.</div>`}`;
    $$("[data-zip]").forEach((b) => (b.onclick = () => downloadZip(b.dataset.zip, null)));
    $$("[data-pick]").forEach((cb) => (cb.onchange = () => { const pid = cb.dataset.pick; const n = $$(`[data-pick="${pid}"]:checked`).length; const btn = $(`[data-zip-sel="${pid}"]`); btn.disabled = n === 0; btn.textContent = `⤓ Download checked (${n})`; }));
    $$("[data-zip-sel]").forEach((b) => (b.onclick = () => downloadZip(b.dataset.zipSel, $$(`[data-pick="${b.dataset.zipSel}"]:checked`).map((x) => x.value))));
    $$("[data-rr]").forEach((b) => (b.onclick = async () => { const [pid, cid] = b.dataset.rr.split(":"); try { await api.post(`/api/projects/${pid}/rerender`, { clip_ids: [cid] }); toast("Re-render queued (detection reused)", "ok"); render(); } catch (e) { toast(e.message, "err"); } }));
  };
  await render(); poll(render, 5000);
}
async function downloadZip(pid, ids) {
  try {
    const r = await fetch(`/api/projects/${pid}/export`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ clip_ids: ids }) });
    if (!r.ok) throw new Error((await r.json()).detail);
    const blob = await r.blob(); const name = (r.headers.get("content-disposition") || "").match(/filename="(.+)"/)?.[1] || "clips.zip";
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } catch (e) { toast(e.message, "err"); }
}

// ---------------------------------------------------------------- boot
async function boot() {
  try {
    state.health = await api.get("/api/health"); state.steps = state.health.steps;
    const es = $("#engine-status"); es.className = "engine-status ok"; es.innerHTML = `<span class="dot"></span> engine ready · ffmpeg ${state.health.ffprobe ? "+ ffprobe" : "(bundled)"}${state.health.lrasd_available ? " · LR-ASD" : ""}`;
  } catch (e) { const es = $("#engine-status"); es.className = "engine-status err"; es.innerHTML = `<span class="dot"></span> backend unreachable`; }
  window.addEventListener("hashchange", route); route();
}
boot();
