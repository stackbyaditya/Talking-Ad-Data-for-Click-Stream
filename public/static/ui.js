/* Panel logic. Renders /api/predict results, live session vitals, the demo scripts and the optional save flow.
   Every string that comes from the server is written with textContent, never innerHTML. */
(function () {
  "use strict";

  var T = window.ClickstreamTracker;
  var TRACE = window.ClickstreamTrace;
  var BOTS = window.ClickstreamBots;
  var boot = JSON.parse(document.getElementById("boot").textContent);
  var gate = boot.gate || { seconds: 5, mouse_events: 20, clicks: 3 };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  var CLASSES = {
    human:        { label: "Human",        key: "human" },
    moderate_bot: { label: "Moderate bot", key: "mod" },
    advanced_bot: { label: "Advanced bot", key: "adv" }
  };

  /* ---- plain-language names and units for the model's features ---- */
  function compact(v) { return v >= 10000 ? Math.round(v / 1000) + "k" : v >= 1000 ? (v / 1000).toFixed(1) + "k" : String(Math.round(v)); }
  function clock(s) { s = Math.max(0, Math.round(s)); var m = Math.floor(s / 60), r = s % 60; return m + ":" + (r < 10 ? "0" : "") + r; }
  function num(v) { var a = Math.abs(v); return a >= 100 ? String(Math.round(v)) : a >= 10 ? v.toFixed(1) : v.toFixed(2); }
  var FEATURES = {
    mouse_speed_mean:        ["Average cursor speed",          function (v) { return compact(v) + " px/s"; }],
    mouse_speed_std:         ["Speed variation",               function (v) { return compact(v) + " px/s"; }],
    mouse_path_length:       ["Total cursor travel",           function (v) { return compact(v) + " px"; }],
    direction_change_count:  ["Sharp turns",                   function (v) { return String(Math.round(v)); }],
    movement_std:            ["Step size variation",           function (v) { return num(v) + " px"; }],
    coordinate_entropy:      ["Screen area covered",           function (v) { return num(v) + " bits"; }],
    session_duration_sec:    ["Session length",                function (v) { return clock(v); }],
    request_interval_mean:   ["Average gap between clicks",   function (v) { return num(v) + " s"; }],
    request_interval_std:    ["Click timing variation",        function (v) { return num(v) + " s"; }],
    clicks_per_minute:       ["Click rate",                    function (v) { return num(v) + " per min"; }],
    burstiness:              ["Click burstiness",              function (v) { return num(v); }],
    click_interval_entropy:  ["Click rhythm variety",          function (v) { return num(v) + " bits"; }],
    movement_acceleration:   ["Speed swings",                  function (v) { return num(v); }],
    mouse_direction_entropy: ["Turns per pixel travelled",     function (v) { return v.toFixed(4); }],
    click_burst_score:       ["Click rate against spacing",    function (v) { return num(v); }],
    session_idle_ratio:      ["Click timing spread vs length", function (v) { return v.toFixed(3); }],
    trajectory_smoothness:   ["Pixels per turn",               function (v) { return compact(v) + " px"; }],
    interaction_variability: ["Overall variability",           function (v) { return num(v); }],
    behavioral_complexity:   ["Overall complexity",            function (v) { return num(v); }]
  };
  function describe(name, value) {
    var f = FEATURES[name];
    return f ? { label: f[0], text: f[1](value) } : { label: name.replace(/_/g, " "), text: num(value) };
  }

  var ui = { running: false, finished: false, lastKey: "" };

  /* ---------- verdict ---------- */
  function setOdds(probs) {
    var bar = $("odds"), parts = [];
    ["human", "moderate_bot", "advanced_bot"].forEach(function (c) {
      var p = probs ? probs[c] : 0;
      bar.querySelector('[data-c="' + c + '"]').style.width = (p * 100).toFixed(1) + "%";
      $("legend").querySelector('[data-c="' + c + '"] b').textContent = Math.round(p * 100) + "%";
      parts.push(CLASSES[c].label + " " + Math.round(p * 100) + " percent");
    });
    bar.setAttribute("aria-label", probs ? parts.join(", ") : "No odds yet");
  }

  function setState(state, headline, sub, pen) {
    $("verdict").setAttribute("data-state", state);
    document.querySelector(".why").setAttribute("data-state", state);
    $("headline").textContent = headline;
    $("subline").textContent = sub;
    if (TRACE) TRACE.setPen(pen);
    var key = state + "|" + headline;
    if (key !== ui.lastKey) { ui.lastKey = key; $("sr").textContent = headline + ". " + sub; }
  }

  function renderDrivers(items, predicted) {
    var list = $("drivers");
    list.textContent = "";
    if (!items || !items.length) { $("why-hint").hidden = false; return; }
    var max = 0.5;
    items.forEach(function (it) { max = Math.max(max, Math.abs(it.contribution)); });
    var label = CLASSES[predicted] ? CLASSES[predicted].label.toLowerCase() : predicted;
    $("why-hint").hidden = false;
    $("why-hint").textContent = "Bars to the right pushed the model toward " + label + ". Bars to the left pushed against it.";
    items.forEach(function (it) {
      var d = describe(it.feature, it.value);
      var li = el("li", "driver");
      var top = el("div", "driver-top");
      top.appendChild(el("span", "", d.label));
      top.appendChild(el("span", "", d.text));
      var bar = el("div", "driver-bar");
      var fill = el("i", it.contribution >= 0 ? "toward" : "away");
      fill.style.width = (Math.min(Math.abs(it.contribution) / max, 1) * 50).toFixed(1) + "%";
      bar.appendChild(fill);
      li.appendChild(top); li.appendChild(bar);
      list.appendChild(li);
    });
  }

  function renderWaiting(reasons) {
    var touch = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
    setState("wait", "Not enough yet",
      touch ? "Touch screens leave no cursor path, so use a mouse or trackpad for a meaningful reading."
            : "Move the cursor and scroll the page. A verdict appears once there is enough to measure.", "ink");
    setOdds(null);
    renderDrivers(null);
    $("why-hint").textContent = "The five measurements that moved the verdict most will appear here.";
  }

  function render(r) {
    if (!r.ready) { renderWaiting(r.reasons); return; }
    var pred = CLASSES[r.predicted] || CLASSES.human;
    var pct = Math.round(r.confidence * 100);
    if (r.uncertain) {
      setState("unsure", "Hard to call",
        "Leaning toward " + pred.label.toLowerCase() + " at " + pct + "%. More activity will sharpen this.", "ink");
    } else if (r.predicted === "human") {
      setState("human", "Looks human", "Human at " + pct + "% confidence.", "human");
    } else {
      setState(pred.key, "Looks automated", "Closest match: " + pred.label.toLowerCase() + " at " + pct + "%.", pred.key);
    }
    setOdds(r.probabilities);
    renderDrivers(r.explanation, r.predicted);
    $("latency").textContent = "Last reading scored in " + r.latency_ms + " ms.";
  }

  window.addEventListener("clickstream:result", function (e) {
    if (e.detail.ok) render(e.detail.body);
    else $("subline").textContent = (e.detail.body && e.detail.body.error) || "The server rejected that request.";
  });
  window.addEventListener("clickstream:error", function () {
    $("subline").textContent = "The server cannot be reached. Retrying.";
  });
  window.addEventListener("clickstream:reset", function () { renderWaiting(); updateVitals(); });

  /* ---------- live vitals and evidence meters ---------- */
  function updateVitals() {
    var s = T.stats();
    $("v-time").textContent = clock(s.seconds);
    $("v-moves").textContent = String(s.moves);
    $("v-clicks").textContent = String(s.clicks);
    $("v-path").textContent = compact(s.path) + " px";

    var tp = Math.min(s.seconds / gate.seconds, 1);
    var ap = Math.max(Math.min(s.moves / gate.mouse_events, 1), Math.min(s.clicks / gate.clicks, 1));
    $("m-time").style.width = (tp * 100) + "%";
    $("m-act").style.width = (ap * 100) + "%";
    $("m-time-v").textContent = Math.min(Math.floor(s.seconds), gate.seconds) + " of " + gate.seconds + " s";
    $("m-act-v").textContent = s.clicks > s.moves / gate.mouse_events * gate.clicks
      ? Math.min(s.clicks, gate.clicks) + " of " + gate.clicks + " clicks"
      : Math.min(s.moves, gate.mouse_events) + " of " + gate.mouse_events + " moves";
  }
  setInterval(updateVitals, 250);

  /* ---------- sample ads ---------- */
  document.querySelectorAll("[data-ad]").forEach(function (btn) {
    var n = 0;
    btn.addEventListener("click", function () {
      n += 1;
      btn.querySelector(".ad-count").textContent = n === 1 ? "1 click" : n + " clicks";
      btn.classList.remove("stamp"); void btn.offsetWidth; btn.classList.add("stamp");
    });
  });

  /* ---------- scripts ---------- */
  var scriptButtons = Array.prototype.slice.call(document.querySelectorAll("[data-script]"));
  var statusEl = $("script-status");
  var IDLE_TEXT = statusEl.textContent;
  var countdown = null;

  function lockScripts(on, activeName) {
    scriptButtons.forEach(function (b) {
      b.disabled = on;
      b.setAttribute("aria-pressed", on && b.getAttribute("data-script") === activeName ? "true" : "false");
    });
  }

  function previewScript(b) {
    if (ui.running) return;
    statusEl.textContent = b.textContent + ": " + b.getAttribute("data-desc");
  }
  function clearPreview() { if (!ui.running && !ui.finished) statusEl.textContent = IDLE_TEXT; }

  scriptButtons.forEach(function (b) {
    b.addEventListener("mouseenter", function () { previewScript(b); });
    b.addEventListener("focus", function () { previewScript(b); });
    b.addEventListener("mouseleave", clearPreview);
    b.addEventListener("blur", clearPreview);
  });

  function startScript(name) {
    if (ui.running) return;
    ui.running = true; ui.finished = false;
    lockScripts(true, name);
    if (TRACE) TRACE.setScriptCursor(true);
    BOTS.run(name, {
      start: function (label, ms) {
        var end = Date.now() + ms;
        clearInterval(countdown);
        countdown = setInterval(function () {
          var left = Math.max(0, Math.ceil((end - Date.now()) / 1000));
          statusEl.textContent = label + " is running, " + left + " s left. Your own mouse is ignored until it finishes.";
        }, 250);
      },
      done: function (label) {
        clearInterval(countdown);
        ui.running = false; ui.finished = true;
        lockScripts(false);
        statusEl.textContent = label + " finished. The verdict describes the script, not you. Press Start over to read your own session.";
      }
    });
  }

  scriptButtons.forEach(function (b) {
    b.addEventListener("click", function (e) {
      if (!e.isTrusted) return;                    // scripts are started by a person, never by another script
      startScript(b.getAttribute("data-script"));
    });
  });

  $("restart").addEventListener("click", function () {
    BOTS.stop();
    clearInterval(countdown);
    ui.running = false; ui.finished = false;
    lockScripts(false);
    if (TRACE) TRACE.setScriptCursor(false);
    statusEl.textContent = IDLE_TEXT;
    var saveMsg = $("save-msg"); if (saveMsg) saveMsg.textContent = "";
    T.setMode("human");
  });

  /* ---------- optional: contribute a session ---------- */
  var consent = $("consent"), save = $("save");
  if (consent && save) {
    consent.addEventListener("change", function () { save.disabled = !consent.checked; });
    save.addEventListener("click", function () {
      var msg = $("save-msg");
      if (T.mode === "script") { msg.textContent = "This session was scripted. Press Start over, then try again."; return; }
      T.save("human", consent.checked).then(function (res) {
        msg.textContent = res.ok ? "Saved. " + res.body.total + " sessions stored." : "Not saved: " + (res.body.error || res.status);
      });
    });
  }

  renderWaiting();
  updateVitals();
  T.predict();
})();
