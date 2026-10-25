/* Demo scripts. Each one drives the page with synthetic mouse events for 14 to 24 seconds so a visitor can
   watch the model react to automated behaviour without installing anything.
   Synthetic events are only counted while the tracker is in "script" mode, and real input is ignored then.
   The scripts make no promise about the verdict: whatever the model says is what it says. */
(function () {
  "use strict";

  var T = window.ClickstreamTracker;
  var token = 0;

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function frame() { return new Promise(function (r) { requestAnimationFrame(r); }); }

  function fire(type, x, y) {
    var el = document.elementFromPoint(x, y) || document.body;
    el.dispatchEvent(new MouseEvent(type, {
      clientX: x, clientY: y, bubbles: true, cancelable: true, view: window,
      detail: type === "click" ? 1 : 0
    }));
  }

  function visibleTiles() {
    var out = [];
    document.querySelectorAll("[data-ad]").forEach(function (el) {
      var r = el.getBoundingClientRect();
      if (r.top > 24 && r.bottom < window.innerHeight - 24 && r.left > 0 && r.right < window.innerWidth) {
        out.push({ x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height });
      }
    });
    return out;
  }

  function prepare() {
    var first = document.querySelector("[data-ad]");
    if (first) first.scrollIntoView({ block: "center" });
  }

  function pick() {
    var t = visibleTiles();
    if (!t.length) {
      var reader = document.getElementById("reader");
      var rr = reader ? reader.getBoundingClientRect() : { left: 40, width: 500 };
      return { x: rr.left + rr.width * (0.2 + Math.random() * 0.6), y: window.innerHeight * (0.3 + Math.random() * 0.4) };
    }
    var p = t[Math.floor(Math.random() * t.length)];
    return { x: p.x + (Math.random() - 0.5) * p.w * 0.4, y: p.y + (Math.random() - 0.5) * p.h * 0.4 };
  }

  /* Stays put and clicks twice a second. */
  async function spammer(alive) {
    var p = pick();
    fire("mousemove", p.x, p.y);
    var end = Date.now() + 14000;
    while (Date.now() < end && alive()) { fire("click", p.x, p.y); await sleep(470); }
  }

  /* Jumps straight onto an ad and clicks; there is no path between targets. */
  async function teleporter(alive) {
    var end = Date.now() + 15000;
    while (Date.now() < end && alive()) {
      var p = pick();
      fire("mousemove", p.x, p.y);
      await sleep(60);
      fire("click", p.x, p.y);
      await sleep(1100 + Math.random() * 300);
    }
  }

  /* Smooth eased curves between ads and a click on a fixed rhythm. */
  async function patient(alive) {
    var end = Date.now() + 24000;
    var cur = pick();
    fire("mousemove", cur.x, cur.y);
    while (Date.now() < end && alive()) {
      var to = pick();
      var dx = to.x - cur.x, dy = to.y - cur.y, dist = Math.hypot(dx, dy) || 1;
      var cx = (cur.x + to.x) / 2 - (dy / dist) * dist * 0.25;
      var cy = (cur.y + to.y) / 2 + (dx / dist) * dist * 0.25;
      var dur = 1400, t0 = performance.now(), u = 0;
      while (u < 1 && alive()) {
        await frame();
        u = Math.min((performance.now() - t0) / dur, 1);
        var s = u * u * (3 - 2 * u), q = 1 - s;
        fire("mousemove", q * q * cur.x + 2 * q * s * cx + s * s * to.x, q * q * cur.y + 2 * q * s * cy + s * s * to.y);
      }
      if (!alive()) break;
      fire("click", to.x, to.y);
      cur = to;
      await sleep(2400);
    }
  }

  var SCRIPTS = {
    spammer:    { label: "Click spammer", ms: 14000, fn: spammer },
    teleporter: { label: "Teleporter",    ms: 15000, fn: teleporter },
    patient:    { label: "Patient bot",   ms: 24000, fn: patient }
  };

  async function run(name, hooks) {
    var script = SCRIPTS[name];
    if (!script) return;
    var id = ++token;
    var alive = function () { return id === token; };
    prepare();
    await sleep(120);                      // let the scroll settle so targets are where we think they are
    if (!alive()) return;
    T.setMode("script");
    hooks.start(script.label, script.ms);
    await script.fn(alive);
    if (alive()) { await T.freeze(); hooks.done(script.label); }
  }

  function stop() { token += 1; }

  window.ClickstreamBots = { run: run, stop: stop, scripts: SCRIPTS };
})();
