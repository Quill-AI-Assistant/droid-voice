/* ============================================================================
   droid voice — demo UI
   Vanilla JS, no framework, no build step. Wires the single page to the
   stdlib HTTP server's /api/* endpoints.

   Two views: Home (static explainer, the default) and Demo (hear a fresh,
   never-repeating reaction). DEMO-ONLY — training/judging happen in the CLI.
   This page never writes; it only calls:
     GET  /api/profiles
     GET  /api/cases?profile=
     GET  /api/state?profile=
     POST /api/demo  {profile, case, text, valence, arousal, temperature}
     GET  /api/wav?token=

   STRICT contract: the `case` field POSTed to /api/demo is NEVER free user
   text. It is only ever a validated dropdown value or a validated [tag].

   Sections:
     1. DOM helpers
     2. API (fetch wrappers, error normalization)
     3. Audio playback
     4. App state + case validation
     5. Header (profiles, /api/state)
     6. UI status helpers
     7. Cases (loader, select builder)
     8. Demo view
     9. Example chips
    10. Tabs (router + keyboard)
    11. Theme + font controls
    12. Boot
   ============================================================================ */

"use strict";

/* ---------------------------------------------------------------------------
   1. DOM HELPERS
   --------------------------------------------------------------------------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/* Tiny element builder. attrs: class/text/html special-cased; on* => listener;
   dataset => data-* attrs; everything else => setAttribute. Children flattened. */
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (k === "dataset") {
      for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
    } else {
      node.setAttribute(k, v === true ? "" : String(v));
    }
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

const fmt2 = (n) => (n ?? 0).toFixed(2);

/* ---------------------------------------------------------------------------
   2. API — fetch wrappers + normalized errors
   --------------------------------------------------------------------------- */
async function apiGet(path) {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
  if (!r.ok) throw await apiError(r);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw await apiError(r);
  return r.json();
}

async function apiError(r) {
  let detail = "";
  try {
    const payload = await r.json();
    detail = payload.error || payload.reason || JSON.stringify(payload);
  } catch (_) {
    try { detail = await r.text(); } catch (_) { detail = ""; }
  }
  const e = new Error(`HTTP ${r.status}${detail ? ": " + detail : ""}`);
  e.status = r.status;
  return e;
}

/* ---------------------------------------------------------------------------
   3. AUDIO — in-memory WAV via /api/wav?token=
   Resolves when the clip FINISHES (ended/error), not when it starts, so a
   sequential caller (Play 5, tour, multi-tag) plays one-after-another. A fresh
   play() interrupts any prior clip. Optional `meta` surfaces a now-playing
   region and toggles .is-playing on a control.
   --------------------------------------------------------------------------- */
let currentAudio = null;

function playToken(token, meta = {}) {
  if (!token) return Promise.resolve();
  try {
    // Only interrupt a clip that's still playing (barge-in). A clip that already
    // ended in a sequence must NOT be re-paused/seeked — that produced an audible
    // click at every tour/Play-5 boundary.
    if (currentAudio && !currentAudio.ended) { currentAudio.pause(); currentAudio.currentTime = 0; }
  } catch (_) {}

  const np = meta.nowPlaying ? $("#demo-now-playing") : null;
  const ctrl = meta.control || null;
  const showPlaying = () => {
    if (np) { np.textContent = `▶ playing${meta.label ? " — " + meta.label : ""}`; np.hidden = false; }
    if (ctrl) ctrl.classList.add("is-playing");
  };
  const hidePlaying = () => {
    if (np) np.hidden = true;
    if (ctrl) ctrl.classList.remove("is-playing");
  };

  const audio = new Audio(`/api/wav?token=${encodeURIComponent(token)}`);
  currentAudio = audio;
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (done) return; done = true; hidePlaying(); resolve(); };
    audio.addEventListener("ended", finish, { once: true });
    audio.addEventListener("error", finish, { once: true });
    const p = audio.play();
    if (p && p.then) {
      p.then(showPlaying).catch((err) => {
        // Autoplay can be blocked until a user gesture; the button click is one,
        // so this is rare. Surface quietly and don't hang the sequence.
        console.warn("playback blocked:", err);
        finish();
      });
    } else {
      showPlaying();
    }
  });
}

/* ---------------------------------------------------------------------------
   4. APP STATE + STRICT case validation
   The `case` sent to the API is only ever a name present in state.cases.
   --------------------------------------------------------------------------- */
const state = {
  profile: "untrained-johndoe",
  defaultProfile: "untrained-johndoe",
  cases: [],
  activeVersion: null,
  labels: {},
};

const isValidCase = (name) => !!name && state.cases.some((c) => c.name === name);
const caseByName = (name) => state.cases.find((c) => c.name === name) || null;

/* ---------------------------------------------------------------------------
   5. HEADER — profile (voice) select + /api/state line
   --------------------------------------------------------------------------- */
async function loadProfiles() {
  const data = await apiGet("/api/profiles");
  state.defaultProfile = data.default || "untrained-johndoe";
  state.labels = data.labels || {};
  if (!state.profile) state.profile = state.defaultProfile;

  const sel = $("#profile-select");
  if (!sel) return;
  sel.innerHTML = "";
  const label = (p) => state.labels[p] || p;
  const profiles = data.profiles || [state.defaultProfile];
  for (const p of profiles) {
    sel.append(el("option", { value: p, selected: p === state.profile }, label(p)));
  }
  if (!profiles.includes(state.profile)) {
    sel.append(el("option", { value: state.profile, selected: true }, label(state.profile)));
  }
}

async function onProfileChange(profile) {
  state.profile = profile;
  state.activeVersion = null;
  await Promise.all([refreshState(), reloadActiveView()]);
}

async function refreshState() {
  const line = $("#state-line");
  try {
    const s = await apiGet(`/api/state?profile=${encodeURIComponent(state.profile)}`);
    state.activeVersion = s.active_version;
    if (line) {
      const judge = s.has_judge ? `judge v${s.active_version}` : "no judge — analytic floor";
      const votes = `${s.vote_total} vote${s.vote_total === 1 ? "" : "s"}`;
      line.textContent = `${s.active_dataset} · ${judge} · ${votes}`;
      line.classList.remove("err");
    }
    return s;
  } catch (e) {
    showError(e, line);
    return null;
  }
}

/* ---------------------------------------------------------------------------
   6. UI STATUS HELPERS
   --------------------------------------------------------------------------- */
function setBusy(node, busy) {
  if (!node) return;
  node.classList.toggle("busy", !!busy);
  if (node.tagName === "BUTTON") node.disabled = !!busy;
}

function showError(err, target) {
  const msg = err && err.message ? err.message : String(err);
  const node = typeof target === "string" ? $(target) : target;
  if (node) {
    node.textContent = msg;
    node.classList.add("err");
  } else {
    console.error(msg);
  }
}

function setStatus(text) {
  const node = $("#demo-status");
  if (node) { node.textContent = text; node.classList.remove("err"); }
}

/* ---------------------------------------------------------------------------
   7. CASES — loader + select builder
   --------------------------------------------------------------------------- */
async function loadCases() {
  const data = await apiGet(`/api/cases?profile=${encodeURIComponent(state.profile)}`);
  state.cases = data.cases || [];
  return data;
}

/* Build <optgroup>s by type into a <select>; known types first, in order. */
function fillCaseSelect(sel, cases) {
  if (!sel) return;
  sel.innerHTML = "";
  const order = ["emotion", "expression", "vocalization"];
  const groups = {};
  for (const c of cases) (groups[c.type] = groups[c.type] || []).push(c);

  const addGroup = (type, list, labelled) => {
    const og = el("optgroup", { label: type });
    for (const c of list) {
      const text = labelled ? `${c.name}  (v ${fmt2(c.valence)}, a ${fmt2(c.arousal)})` : c.name;
      og.append(el("option", { value: c.name }, text));
    }
    sel.append(og);
  };

  for (const type of order) {
    if (groups[type] && groups[type].length) addGroup(type, groups[type], true);
  }
  for (const type of Object.keys(groups)) {
    if (!order.includes(type)) addGroup(type, groups[type], false);
  }
}

/* ---------------------------------------------------------------------------
   8. DEMO VIEW
   --------------------------------------------------------------------------- */
const Demo = {
  last: null,

  async load() {
    try {
      setStatus("loading cases…");
      const data = await loadCases();
      fillCaseSelect($("#demo-case"), data.cases);
      this.updateCaseMeta();
      buildExamples();
      setStatus("");
      await refreshState();
    } catch (e) {
      showError(e, "#demo-status");
    }
  },

  /* Read-only display of the selected case's type + (valence, arousal). */
  updateCaseMeta() {
    const meta = $("#demo-case-meta");
    if (!meta) return;
    const sel = $("#demo-case");
    const c = sel ? caseByName(sel.value) : null;
    meta.textContent = c ? `${c.type} · v ${fmt2(c.valence)}, a ${fmt2(c.arousal)}` : "";
  },

  /* STRICT request resolution:
     - case is ALWAYS a validated dropdown value, or a validated single [tag].
     - any other typed text travels as plain `text`, never as a case.
     - ad-hoc ON sends both valence + arousal (the server's ad-hoc branch needs both). */
  resolveRequest() {
    const sel = $("#demo-case");
    const value = sel ? sel.value : "";
    if (!isValidCase(value)) return { error: "Pick a valid case." };

    let caseName = value;
    const phraseEl = $("#demo-phrase");
    let text = phraseEl ? phraseEl.value.trim() : "";
    // Honor the FIRST VALID [tag] anywhere in the phrase as the case (not just the
    // first bracket — an invalid leading tag must not shadow a later valid one), and
    // ALWAYS strip every bracket group so invalid tags never leak through as literals.
    const validTag = [...text.matchAll(/\[([^\]]+)\]/g)].map((m) => m[1].trim()).find(isValidCase);
    if (validTag) caseName = validTag;
    text = text.replace(/\[[^\]]+\]/g, "").trim();

    const tempOn = $("#demo-temp-on");
    const tempSlider = $("#demo-temp");
    const temperature = tempOn && tempOn.checked && tempSlider ? Number(tempSlider.value) : null;

    let valence = null;
    let arousal = null;
    const adhocOn = $("#demo-adhoc-on");
    if (adhocOn && adhocOn.checked) {
      const vEl = $("#demo-valence");
      const aEl = $("#demo-arousal");
      valence = vEl ? Number(vEl.value) : null;
      arousal = aEl ? Number(aEl.value) : null;
    }
    return { case: caseName, text, temperature, valence, arousal, tagged: !!validTag };
  },

  buildBody(req) {
    return {
      profile: state.profile,
      case: req.case,
      text: req.text || "",
      valence: req.valence,
      arousal: req.arousal,
      temperature: req.temperature,
      // free text (no explicit [tag], no ad-hoc v/a) -> let the WORDS set the affect
      map_affect: !!(req.text && !req.tagged && req.valence == null),
    };
  },

  /* Every valid [tag] in the phrase, in order — used to detect a multi-tag sequence. */
  tagsInPhrase() {
    const phraseEl = $("#demo-phrase");
    const text = phraseEl ? phraseEl.value : "";
    return [...text.matchAll(/\[([^\]]+)\]/g)].map((m) => m[1].trim()).filter(isValidCase);
  },

  async react() {
    const tags = this.tagsInPhrase();
    if (tags.length >= 2) return this.playSequence(tags, "phrase");  // multi-tag → sequence

    const req = this.resolveRequest();
    if (req.error) { setStatus(req.error); return; }

    const react = $("#demo-react");
    const again = $("#demo-again");
    setBusy(react, true);
    setBusy(again, true);
    setStatus("generating…");
    try {
      const r = await apiPost("/api/demo", this.buildBody(req));
      this.last = r;
      this.renderResult(r);
      setStatus("");
      if (again) again.hidden = false;
      await playToken(r.wav_token, { nowPlaying: true, label: r.case, control: react });
    } catch (e) {
      this.hideResult();
      showError(e, "#demo-status");
    } finally {
      setBusy(react, false);
      setBusy(again, false);
    }
  },

  /* "Play 5, all different": render the same resolved case five times and play
     them back-to-back. seed=None server-side guarantees each is fresh. */
  async reactFive() {
    const req = this.resolveRequest();
    if (req.error) { setStatus(req.error); return; }

    const ctrls = ["#demo-react", "#demo-five", "#demo-again"].map((s) => $(s));
    const again = $("#demo-again");
    ctrls.forEach((c) => setBusy(c, true));
    try {
      for (let i = 0; i < 5; i++) {
        setStatus(`generating ${i + 1} / 5…`);
        const r = await apiPost("/api/demo", this.buildBody(req));
        this.last = r;
        this.renderResult(r);
        if (again) again.hidden = false;
        await playToken(r.wav_token, { nowPlaying: true, label: r.case, control: $("#demo-five") });
      }
      setStatus("");
    } catch (e) {
      this.hideResult();
      showError(e, "#demo-status");
    } finally {
      ctrls.forEach((c) => setBusy(c, false));
    }
  },

  /* Render + play a list of case names back-to-back. Powers multi-tag phrases + tour. */
  async playSequence(caseNames, kind) {
    const names = caseNames.filter(isValidCase);
    if (!names.length) { setStatus("(no valid cases to play)"); return; }

    const ctrls = ["#demo-react", "#demo-five", "#demo-tour", "#demo-again"].map((s) => $(s));
    const again = $("#demo-again");
    ctrls.forEach((c) => setBusy(c, true));
    try {
      for (let i = 0; i < names.length; i++) {
        setStatus(`${kind} · ${i + 1} / ${names.length} — ${names[i]}`);
        const r = await apiPost("/api/demo", { profile: state.profile, case: names[i] });
        this.last = r;
        this.renderResult(r);
        if (again) again.hidden = false;
        await playToken(r.wav_token, { nowPlaying: true, label: r.case, control: $("#demo-react") });
      }
      setStatus("");
    } catch (e) {
      this.hideResult();
      showError(e, "#demo-status");
    } finally {
      ctrls.forEach((c) => setBusy(c, false));
    }
  },

  /* A SHOWCASE REEL as a CLI-style list. Per case: play it PLAIN, then SPOKEN WITH
     its example phrase (so the text→sound difference is hearable — the phrase shapes
     length + terminal). Then a Combinations section: multi-[tag] phrases that play
     each segment's text on its emotion in sequence. Rows are clickable to replay. */
  async tour() {
    const cases = (state.cases || []).filter((c) => isValidCase(c.name));
    const list = $("#demo-tour-list");
    if (!cases.length || !list) { setStatus("(no cases to tour)"); return; }
    const myRun = (this.tourRun = (this.tourRun || 0) + 1);   // a fresh run cancels the prior loop
    this.hideResult();
    list.hidden = false;
    list.innerHTML = "";

    // 1) build the entry list (plain + phrased per case, then valid combos)
    const PAIRS = "Each case — plain, then spoken with its phrase";
    const COMBOS = "Combinations — text + tags in sequence";
    const entries = [];
    for (const c of cases) {
      entries.push({ kind: "plain", case: c.name, label: c.name, sub: "plain", text: "", group: PAIRS });
      if (c.example) {
        entries.push({ kind: "phrased", case: c.name, label: c.name, sub: "+ phrase",
                       text: c.example, display: c.example, group: PAIRS });
      }
    }
    for (const phrase of tourCombos()) {
      const segs = comboSegments(phrase);
      if (segs.length >= 2) {
        entries.push({ kind: "combo", label: phrase, sub: `${segs.length} parts`,
                       segments: segs, display: phrase, group: COMBOS });
      }
    }

    // 2) render rows with group headers; stash the row element on each entry
    let lastGroup = null;
    for (const e of entries) {
      if (e.group !== lastGroup) { list.append(el("div", { class: "tour-group" }, e.group)); lastGroup = e.group; }
      const row = el("button", {
        class: `tour-row${e.kind === "phrased" ? " is-phrased" : ""}${e.kind === "combo" ? " is-combo" : ""}`,
        type: "button",
        onClick: () => this.playEntry(e, row).catch((err) => showError(err, "#demo-status")),
      },
        el("span", { class: "tour-name" }, e.label),
        el("span", { class: "tour-sub" }, e.sub || ""),
        el("span", { class: "tour-mark", "aria-hidden": "true" }, "♪"),
        el("span", { class: "tour-tr" }, e.display || ""),
      );
      e.row = row;
      list.append(row);
    }

    // 3) play through in order
    const ctrls = ["#demo-react", "#demo-five", "#demo-tour"].map((s) => $(s));
    ctrls.forEach((c) => setBusy(c, true));
    $$(".tour-row", list).forEach((r) => (r.disabled = true));  // no concurrent row clicks mid-run
    try {
      for (let i = 0; i < entries.length; i++) {
        if (myRun !== this.tourRun) return;                    // superseded by a newer run → stop quietly
        setStatus(`tour · ${i + 1} / ${entries.length} — ${entries[i].label}`);
        await this.playEntry(entries[i], entries[i].row);
        if (myRun !== this.tourRun) return;
        if (i < entries.length - 1) await new Promise((res) => setTimeout(res, 200));  // breath between items
      }
      setStatus("");
    } catch (e) {
      showError(e, "#demo-status");
    } finally {
      if (myRun === this.tourRun) {                            // only the owning run cleans up
        ctrls.forEach((c) => setBusy(c, false));
        $$(".tour-row", list).forEach((r) => (r.disabled = false));  // re-enable click-to-replay
        $$(".tour-row.is-current").forEach((r) => r.classList.remove("is-current"));
        const np = $("#demo-now-playing"); if (np) np.hidden = true;
      }
    }
  },

  /* Render + play one tour entry into its row. The row's BACKGROUND highlights only
     while its clip is actually playing — the mark happens AFTER the render (synced to
     the audio, not the fetch) and clears when the clip ends, so switching to the next
     item never flickers a moving border. plain/phrased = one /api/demo (text shapes
     the sound); combo = each segment's text spoken on its emotion, in order. */
  async playEntry(entry, row) {
    const mark = () => {
      $$(".tour-row.is-current").forEach((r) => r.classList.remove("is-current"));
      if (row) row.classList.add("is-current");
    };
    const clear = () => { if (row) row.classList.remove("is-current"); };
    try {
      if (entry.kind === "combo") {
        for (const seg of entry.segments) {
          const r = await apiPost("/api/demo", { profile: state.profile, case: seg.case, text: seg.text || "" });
          this.last = r;
          mark();
          await playToken(r.wav_token, { nowPlaying: true, label: seg.case });
        }
        return;
      }
      const r = await apiPost("/api/demo", { profile: state.profile, case: entry.case, text: entry.text || "" });
      this.last = r;
      if (row && entry.kind === "plain") {                     // plain rows show the glyph transcript
        const tr = row.querySelector(".tour-tr");
        if (tr) tr.textContent = r.transcript || "";
      }
      mark();
      await playToken(r.wav_token, { nowPlaying: true, label: entry.label });
    } finally {
      clear();                                                 // background highlight lives only while playing
    }
  },

  hideResult() {
    const out = $("#demo-result");
    if (out) out.hidden = true;
  },

  renderResult(r) {
    const out = $("#demo-result");
    if (!out) return;
    const tl = $("#demo-tour-list");
    if (tl) tl.hidden = true;                 // single-result view hides the tour list
    out.hidden = false;
    out.innerHTML = "";
    const replay = el("button", {
      class: "btn btn-secondary btn-mini btn-play",
      type: "button",
      "aria-label": "replay clip",
      onClick: () => {
        if (this.last && this.last.wav_token) {
          playToken(this.last.wav_token, { nowPlaying: true, label: r.case, control: replay });
        }
      },
    }, "Replay");
    out.append(
      el("div", { class: "result-head" },
        el("div", { class: "result-field" },
          el("span", { class: "result-label" }, "Case"),
          el("span", { class: "pill pill-case" }, r.case),
          r.type ? el("span", { class: "pill pill-type" }, r.type) : null,
        ),
        el("div", { class: "result-field" },
          el("span", { class: "result-label" }, "Feeling"),
          el("span", { class: "result-va" }, `v ${fmt2(r.valence)}  ·  a ${fmt2(r.arousal)}`),
        ),
        replay,
      ),
      el("div", { class: "result-transcript", "aria-label": "phrase transcript" }, r.transcript || ""),
      el("div", { class: "result-field result-source-field" },
        el("span", { class: "result-label" }, "Source"),
        el("span", { class: "result-source" }, r.source || ""),
      ),
    );
  },
};

/* ---------------------------------------------------------------------------
   9. EXAMPLE CHIPS — fill the phrase (or set the case) and react, to showcase
   the range + multi-tag. Phrase examples self-filter unknown tags.
   --------------------------------------------------------------------------- */
/* Pick up to k distinct case names that EXIST on the active profile, preferring the
   listed ones and then filling from whatever the profile actually has. Used to build
   examples + tour combos that are always valid (case sets differ per profile). */
function pickCases(prefer, k) {
  const names = (state.cases || []).map((c) => c.name).filter(isValidCase);
  const out = prefer.filter((n) => names.includes(n));
  for (const n of names) { if (out.length >= k) break; if (!out.includes(n)) out.push(n); }
  return out.slice(0, k);
}

/* Combination phrases for the tour's "Combinations" section, built from THIS profile's
   real cases so every [tag] resolves (case sets can differ between qd and the
   untrained-johndoe demo default). */
function tourCombos() {
  const combos = [];
  const a = pickCases(["curious", "excited", "proud", "worried", "alarmed"], 2);
  const b = pickCases(["worried", "content", "proud", "sad", "done", "found"], 3);
  if (a.length === 2) combos.push(`huh? [${a[0]}] ... yes! [${a[1]}]`);
  if (b.length === 3) combos.push(`[${b[0]}] ... [${b[1]}] [${b[2]}]`);
  return combos;
}

/* Split a phrase into [{case, text}] segments: each valid [tag] paired with the
   words since the previous tag (brackets stripped). Powers the tour's combos. */
function comboSegments(phrase) {
  const segs = [];
  const re = /\[([^\]]+)\]/g;
  let last = 0, m;
  while ((m = re.exec(phrase)) !== null) {
    const tag = m[1].trim();
    if (!isValidCase(tag)) continue;
    const text = phrase.slice(last, m.index).replace(/\[[^\]]+\]/g, "").trim();
    segs.push({ case: tag, text });
    last = re.lastIndex;
  }
  return segs;
}

/* Build the example chips FROM the active profile's real cases, so every example is
   valid on the current voice (the old hard-coded [error]/[online]/… only existed on
   the qd voice and did nothing on the untrained-johndoe demo default). A few
   single-case chips + several [tag] combinations that actually play. */
function buildExamples() {
  const box = $("#demo-examples");
  if (!box) return;
  box.innerHTML = "";
  const names = (state.cases || []).map((c) => c.name).filter(isValidCase);
  if (!names.length) return;

  const examples = [];
  for (const c of pickCases(["proud", "curious", "alarmed", "done", "laugh"], 3)) examples.push({ label: c, case: c });
  const seq = pickCases(["curious", "worried", "proud", "done", "found", "thinking", "sad", "excited"], 4);
  if (seq.length >= 2) examples.push({ label: `[${seq[0]}] [${seq[1]}]`, phrase: `[${seq[0]}] [${seq[1]}]` });
  if (seq.length >= 3) examples.push({ label: `[${seq[0]}] [${seq[1]}] [${seq[2]}]`, phrase: `[${seq[0]}] [${seq[1]}] [${seq[2]}]` });
  if (seq.length >= 4) examples.push({ label: `${seq.length}-tag sequence`, phrase: seq.map((n) => `[${n}]`).join(" ") });

  for (const ex of examples) {
    box.append(el("button", {
      class: "btn btn-secondary btn-mini chip",
      type: "button",
      onClick: () => {
        const phraseEl = $("#demo-phrase");
        const caseSel = $("#demo-case");
        if (ex.case && caseSel && isValidCase(ex.case)) {
          caseSel.value = ex.case;
          Demo.updateCaseMeta();
          if (phraseEl) phraseEl.value = "";
        } else if (ex.phrase && phraseEl) {
          phraseEl.value = ex.phrase;
        }
        Demo.react();
      },
    }, ex.label));
  }
}

/* ---------------------------------------------------------------------------
   10. TABS — router + ARIA keyboard pattern
   Home is static HTML, so its view object is a no-op load.
   --------------------------------------------------------------------------- */
const Home = { load() {} };
const VIEWS = { home: Home, demo: Demo };
let activeTab = "home";

function switchTab(name) {
  if (!VIEWS[name]) return;
  activeTab = name;
  // roving tabindex: the selected tab is the only tab-stop.
  $$(".tab").forEach((b) => {
    const sel = b.dataset.tab === name;
    b.setAttribute("aria-selected", sel ? "true" : "false");
    b.tabIndex = sel ? 0 : -1;
  });
  $$(".view").forEach((v) => (v.hidden = v.id !== `view-${name}`));
  reloadActiveView();
}

async function reloadActiveView() {
  const view = VIEWS[activeTab];
  if (view) return view.load();
}

function wireTablistKeys() {
  const list = $("#tablist");
  if (!list) return;
  list.addEventListener("keydown", (ev) => {
    const tabs = $$(".tab", list);
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    let next = -1;
    switch (ev.key) {
      case "ArrowRight": next = (i + 1) % tabs.length; break;
      case "ArrowLeft": next = (i - 1 + tabs.length) % tabs.length; break;
      case "Home": next = 0; break;
      case "End": next = tabs.length - 1; break;
      default: return;
    }
    ev.preventDefault();
    const btn = tabs[next];
    if (btn) { switchTab(btn.dataset.tab); btn.focus(); }
  });
}

/* ---------------------------------------------------------------------------
   11. THEME + FONT CONTROLS (dark/light + adjustable font size, persisted)
   --------------------------------------------------------------------------- */
const FONT_SIZES = ["xsmall", "small", "medium", "normal", "large", "xlarge", "xxlarge", "xxxlarge"];
let fontIdx = FONT_SIZES.indexOf("normal");

function toggleTheme() {
  const html = document.documentElement;
  const next = html.getAttribute("data-theme") === "light" ? "dark" : "light";
  html.setAttribute("data-theme", next);
  try { localStorage.setItem("droid-voice-theme", next); } catch (_) {}
  const btn = $("#theme-toggle");
  if (btn) btn.innerHTML = next === "dark" ? "☽" : "☀";
}

function applyFont(size) {
  const html = document.documentElement;
  if (size === "normal") html.removeAttribute("data-font");
  else html.setAttribute("data-font", size);
  try { localStorage.setItem("droid-voice-font", size); } catch (_) {}
}

function stepFont(delta) {
  const next = fontIdx + delta;
  if (next < 0 || next >= FONT_SIZES.length) return;
  fontIdx = next;
  applyFont(FONT_SIZES[fontIdx]);
}

function restorePrefs() {
  let savedTheme = null;
  let savedFont = null;
  try {
    savedTheme = localStorage.getItem("droid-voice-theme");
    savedFont = localStorage.getItem("droid-voice-font");
  } catch (_) {}
  document.documentElement.setAttribute("data-theme", savedTheme || "dark");
  if (savedFont && savedFont !== "normal") {
    document.documentElement.setAttribute("data-font", savedFont);
    const i = FONT_SIZES.indexOf(savedFont);
    fontIdx = i < 0 ? 0 : i;
  }
}

/* ---------------------------------------------------------------------------
   12. BOOT — every handler attach is null-guarded so one missing node can't
   abort boot before loadProfiles / refreshState run.
   --------------------------------------------------------------------------- */
function on(sel, ev, fn) {
  const node = $(sel);
  if (node) node.addEventListener(ev, fn);
}

function wireDemoControls() {
  on("#demo-react", "click", () => Demo.react());
  on("#demo-again", "click", () => Demo.react());
  on("#demo-five", "click", () => Demo.reactFive());
  on("#demo-tour", "click", () => Demo.tour());
  on("#demo-case", "change", () => Demo.updateCaseMeta());
  on("#demo-phrase", "keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); Demo.react(); }
  });

  // Temperature override: the slider only applies when "override" is checked.
  const tempOn = $("#demo-temp-on");
  const tempSlider = $("#demo-temp");
  const tempVal = $("#demo-temp-val");
  const syncTemp = () => {
    if (!tempOn || !tempSlider) return;
    tempSlider.disabled = !tempOn.checked;
    if (tempVal) tempVal.textContent = tempOn.checked ? fmt2(Number(tempSlider.value)) : "per-type";
  };
  on("#demo-temp-on", "change", syncTemp);
  on("#demo-temp", "input", syncTemp);
  syncTemp();

  // Ad-hoc affect: toggle reveals two -1..1 sliders; disabling them when off
  // removes them from the tab order. Each value span mirrors its live value.
  const adhocOn = $("#demo-adhoc-on");
  const adhocFields = $("#demo-adhoc-fields");
  const vSlider = $("#demo-valence");
  const aSlider = $("#demo-arousal");
  const syncAdhoc = () => {
    if (!adhocOn) return;
    const checked = adhocOn.checked;
    if (adhocFields) adhocFields.hidden = !checked;
    if (vSlider) vSlider.disabled = !checked;
    if (aSlider) aSlider.disabled = !checked;
  };
  const mirror = (slider, outSel) => {
    const out = $(outSel);
    if (slider && out) out.textContent = fmt2(Number(slider.value));
  };
  on("#demo-adhoc-on", "change", syncAdhoc);
  on("#demo-valence", "input", () => mirror(vSlider, "#demo-valence-val"));
  on("#demo-arousal", "input", () => mirror(aSlider, "#demo-arousal-val"));
  syncAdhoc();
  mirror(vSlider, "#demo-valence-val");
  mirror(aSlider, "#demo-arousal-val");
}

async function boot() {
  restorePrefs();

  // Tabs
  $$(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  wireTablistKeys();

  // "Try the demo →" affordances jump to the Demo tab and focus it.
  $$("[data-go-demo]").forEach((b) =>
    b.addEventListener("click", () => {
      switchTab("demo");
      const tab = $("#tab-demo");
      if (tab) tab.focus();
    })
  );

  // Theme / font controls
  const themeBtn = $("#theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", toggleTheme);
    themeBtn.innerHTML =
      document.documentElement.getAttribute("data-theme") === "light" ? "☀" : "☽";
  }
  on("#font-inc", "click", () => stepFont(+1));
  on("#font-dec", "click", () => stepFont(-1));

  // Profile (voice) select
  on("#profile-select", "change", (e) => onProfileChange(e.target.value));

  wireDemoControls();

  try {
    await loadProfiles();
  } catch (e) {
    showError(e, "#state-line");
  }
  await refreshState();
  switchTab("home");  // Home is the default landing view.
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
