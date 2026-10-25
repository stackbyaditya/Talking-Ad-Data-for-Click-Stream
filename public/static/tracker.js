/* Session tracker.
   Records the same event types the original collector recorded (mousemove, click, scroll, visibility),
   asks the server for a verdict every couple of seconds, and re-broadcasts every accepted event so the
   ink trail (trace.js) and the panel (ui.js) can react.

   Two modes keep sessions clean:
     "human"  only real input counts (event.isTrusted)   -> what a visitor or Playwright produces
     "script" only synthetic input counts                -> what the in-page demo scripts (bots.js) produce
   No cookies, no IP address or user-agent is sent; only coordinates and timestamps. */
(function () {
  "use strict";

  var MAX_EVENTS = 20000;
  var POLL_MS = 2000;
  var IDLE_STOP_MS = 120000;   // stop polling after two minutes without input

  var mode = "human";
  var sessionStart = Date.now();
  var frozenEnd = null;
  var events = [];
  var lastResult = null;
  var lastMove = null;
  var path = 0, moves = 0, clicks = 0;
  var epoch = 0;
  var inFlight = false;
  var soon = null;

  function accepts(e) { return mode === "script" ? !e.isTrusted : e.isTrusted; }
  function live() { return frozenEnd === null; }
  function emit(name, detail) { window.dispatchEvent(new CustomEvent(name, { detail: detail })); }
  function push(ev) {
    if (events.length < MAX_EVENTS) events.push(ev);
    emit("clickstream:event", ev);
  }

  document.addEventListener("mousemove", function (e) {
    if (!live() || !accepts(e)) return;
    var ev = { type: "mousemove", ts: Date.now(), x: e.clientX, y: e.clientY };
    if (lastMove) path += Math.hypot(ev.x - lastMove.x, ev.y - lastMove.y);
    lastMove = ev;
    moves += 1;
    push(ev);
  }, { passive: true });

  document.addEventListener("click", function (e) {
    // detail 0 means a keyboard-activated click, which has no cursor position
    if (!live() || !accepts(e) || e.detail === 0) return;
    clicks += 1;
    push({ type: "click", ts: Date.now(), x: e.clientX, y: e.clientY });
    clearTimeout(soon);
    soon = setTimeout(predict, 500);
  });

  document.addEventListener("scroll", function (e) {
    if (!live() || !accepts(e)) return;
    push({ type: "scroll", ts: Date.now() });
  }, { passive: true });

  document.addEventListener("visibilitychange", function () {
    if (!live()) return;
    push({ type: "visibility", ts: Date.now() });
  });

  function snapshot() {
    return {
      sessionStart: sessionStart,
      sessionEnd: frozenEnd !== null ? frozenEnd : Date.now(),
      events: events,
      viewport: { w: window.innerWidth, h: window.innerHeight }
    };
  }

  function post(url, body, headers) {
    return fetch(url, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, headers || {}),
      body: JSON.stringify(body)
    }).then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, status: r.status, body: j }; });
    });
  }

  function predict() {
    if (inFlight) return Promise.resolve(lastResult);
    inFlight = true;
    var myEpoch = epoch;
    return post("/api/predict", snapshot()).then(function (res) {
      inFlight = false;
      if (myEpoch !== epoch) return lastResult;      // session was reset while the request was in flight
      lastResult = res.body;
      emit("clickstream:result", res);
      return res.body;
    }).catch(function (err) {
      inFlight = false;
      emit("clickstream:error", err);
    });
  }

  function save(label, consent, token) {
    var body = snapshot();
    body.label = label;
    body.consent = consent === true;
    return post("/api/save", body, token ? { "X-Collect-Token": token } : {});
  }

  function reset() {
    epoch += 1;
    events = []; lastResult = null; lastMove = null;
    path = 0; moves = 0; clicks = 0;
    frozenEnd = null;
    sessionStart = Date.now();
    emit("clickstream:reset", { mode: mode });
  }

  function setMode(m) { mode = m; reset(); }

  /* Stop the clock: used when a scripted run ends so the verdict does not drift while the page sits idle. */
  function freeze() {
    frozenEnd = Date.now();
    clearTimeout(soon);
    inFlight = false;
    return predict();
  }

  function stats() {
    var end = frozenEnd !== null ? frozenEnd : Date.now();
    return { seconds: (end - sessionStart) / 1000, moves: moves, clicks: clicks, path: path };
  }

  /* Poll only while someone is looking and doing something: a forgotten background tab should not
     keep invoking the server. */
  var lastActivity = Date.now();
  window.addEventListener("clickstream:event", function () { lastActivity = Date.now(); });
  setInterval(function () {
    if (live() && !document.hidden && Date.now() - lastActivity < IDLE_STOP_MS) predict();
  }, POLL_MS);

  window.ClickstreamTracker = {
    snapshot: snapshot, predict: predict, save: save, reset: reset,
    setMode: setMode, freeze: freeze, stats: stats,
    get mode() { return mode; },
    get lastResult() { return lastResult; },
    get eventCount() { return events.length; }
  };
})();
