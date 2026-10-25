/* Ink trail. A full-viewport canvas draws the cursor path as it is recorded, and drops a ring where
   you click. The pen takes the colour of the model's current verdict, so you can watch it change its mind.
   Purely visual: it never affects what is measured. */
(function () {
  "use strict";

  var canvas = document.getElementById("trace");
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext("2d");
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  var FADE_MS = 4500;      // how long a stretch of trail stays visible
  var GAP_MS = 350;        // a pause longer than this lifts the pen
  var RIPPLE_MS = 750;

  var pts = [];
  var ripples = [];
  var pen = "ink";
  var scriptCursor = false;
  var palette = {};
  var w = 0, h = 0;
  var running = false;

  function readPalette() {
    var s = getComputedStyle(document.documentElement);
    ["ink", "human", "mod", "adv"].forEach(function (k) { palette[k] = s.getPropertyValue("--" + k).trim() || "#15233d"; });
  }

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function arrow(x, y, color) {
    ctx.save();
    ctx.translate(x, y);
    ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(0, 17); ctx.lineTo(4.5, 13); ctx.lineTo(8, 20.5);
    ctx.lineTo(10.6, 19.4); ctx.lineTo(7.2, 12); ctx.lineTo(12.5, 12); ctx.closePath();
    ctx.fillStyle = color; ctx.strokeStyle = palette.ink; ctx.lineWidth = 1.2;
    ctx.globalAlpha = 1;
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  function frame() {
    var now = Date.now();
    ctx.clearRect(0, 0, w, h);

    while (pts.length && now - pts[0].t > FADE_MS) pts.shift();
    ripples = ripples.filter(function (r) { return now - r.t < RIPPLE_MS; });

    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (var i = 0; i < pts.length; i++) {
      var b = pts[i], a = pts[i - 1];
      var life = 1 - (now - b.t) / FADE_MS;
      if (life <= 0) continue;
      var connected = a && b.t - a.t <= GAP_MS;
      ctx.strokeStyle = ctx.fillStyle = palette[b.c] || palette.ink;
      if (connected) {
        ctx.globalAlpha = life * 0.9;
        ctx.lineWidth = 1.3 + 2.4 * life;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      } else {
        ctx.globalAlpha = life * 0.9;                       // pen lands: a small dot
        ctx.beginPath(); ctx.arc(b.x, b.y, 2.4, 0, 6.2832); ctx.fill();
      }
    }

    for (var j = 0; j < ripples.length; j++) {
      var r = ripples[j], p = (now - r.t) / RIPPLE_MS, e = 1 - Math.pow(1 - p, 3);
      ctx.strokeStyle = ctx.fillStyle = palette[r.c] || palette.ink;
      ctx.globalAlpha = (1 - p) * 0.85;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(r.x, r.y, 7 + 42 * e, 0, 6.2832); ctx.stroke();
      ctx.beginPath(); ctx.arc(r.x, r.y, 3.2 * (1 - p) + 1, 0, 6.2832); ctx.fill();
    }

    var head = pts[pts.length - 1];
    if (head && now - head.t < 2500) {
      var col = palette[head.c] || palette.ink;
      if (scriptCursor) {
        arrow(head.x, head.y, col);
      } else {
        ctx.globalAlpha = Math.max(0, 1 - (now - head.t) / 2500) * 0.9;
        ctx.strokeStyle = col; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(head.x, head.y, 6, 0, 6.2832); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    if (pts.length || ripples.length) requestAnimationFrame(frame); else running = false;
  }

  function kick() { if (!running) { running = true; requestAnimationFrame(frame); } }

  window.addEventListener("clickstream:event", function (e) {
    var ev = e.detail;
    if (ev.type === "mousemove") {
      pts.push({ x: ev.x, y: ev.y, t: ev.ts, c: pen });
    } else if (ev.type === "click") {
      if (!reduceMotion.matches) ripples.push({ x: ev.x, y: ev.y, t: ev.ts, c: pen });
    } else {
      return;
    }
    kick();
  });
  window.addEventListener("clickstream:reset", function () { pts.length = 0; ripples.length = 0; kick(); });
  window.addEventListener("resize", resize);
  var scheme = window.matchMedia("(prefers-color-scheme: dark)");
  if (scheme.addEventListener) scheme.addEventListener("change", readPalette);

  readPalette(); resize();

  window.ClickstreamTrace = {
    setPen: function (name) { pen = palette[name] ? name : "ink"; },
    setScriptCursor: function (on) { scriptCursor = !!on; }
  };
})();
