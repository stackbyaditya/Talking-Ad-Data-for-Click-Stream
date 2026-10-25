"""Drive the demo page with a scripted browser, print the live verdict, optionally save the
session as a labelled bot example for retraining.

Setup (once):
    pip install playwright && python -m playwright install chromium

Usage (app running on http://127.0.0.1:5000):
    python webapp/tools/bot_demo.py --mode linear
    python webapp/tools/bot_demo.py --mode curved --save --token $COLLECT_TOKEN --repeat 20

Modes
    linear   straight mouse paths at constant speed, click every N seconds (classic naive bot)
    curved   human-like curved paths with jitter, but metronomic click timing
    clicker  rapid clicks at one spot, no mouse movement
    random   random mouse waypoints and random click gaps (a "tries to look human" bot)
"""

from __future__ import annotations

import argparse
import math
import random
import time

from playwright.sync_api import sync_playwright

LABELS = {"linear": "bot_linear", "curved": "bot_curved", "clicker": "bot_clicker", "random": "bot_naive"}


def move(page, x0, y0, x1, y1, steps, curved=False, jitter=0.0, step_delay=0.01):
    cx = (x0 + x1) / 2 + (random.gauss(0, 70) if curved else 0)
    cy = (y0 + y1) / 2 + (random.gauss(0, 70) if curved else 0)
    for i in range(1, steps + 1):
        u = i / steps
        s = u * u * (3 - 2 * u) if curved else u
        x = (1 - s) ** 2 * x0 + 2 * (1 - s) * s * cx + s ** 2 * x1 + random.gauss(0, jitter)
        y = (1 - s) ** 2 * y0 + 2 * (1 - s) * s * cy + s ** 2 * y1 + random.gauss(0, jitter)
        page.mouse.move(x, y)
        time.sleep(step_delay * (random.uniform(0.7, 1.3) if curved else 1.0))


def run_once(browser, args):
    ctx = browser.new_context(viewport={"width": 1100, "height": 760})
    page = ctx.new_page()
    page.goto(args.url)
    page.wait_for_selector(".ad")
    x, y = 200.0, 200.0
    t_end = time.time() + args.seconds

    if args.mode == "clicker":
        page.mouse.move(300, 300)
        while time.time() < t_end:
            page.mouse.click(300, 300)
            time.sleep(args.gap)
    else:
        while time.time() < t_end:
            tx, ty = random.uniform(80, 1000), random.uniform(120, 650)
            if args.mode == "linear":
                move(page, x, y, tx, ty, steps=25, curved=False)
                gap = args.gap
            elif args.mode == "curved":
                move(page, x, y, tx, ty, steps=int(random.uniform(40, 90)), curved=True, jitter=1.2)
                gap = args.gap + random.gauss(0, 0.03)
            else:  # random
                move(page, x, y, tx, ty, steps=int(random.uniform(10, 60)), curved=random.random() < 0.5, jitter=2.0)
                gap = random.uniform(0.3, 6.0)
            page.mouse.click(tx, ty)
            x, y = tx, ty
            time.sleep(max(gap, 0.05))

    result = page.evaluate("() => window.ClickstreamTracker.predict()")
    if result and result.get("ready"):
        p = result["probabilities"]
        print(f"[{args.mode}] verdict={result['verdict']:5s} conf={result['confidence']:.2f}  "
              f"human={p['human']:.3f} mod={p['moderate_bot']:.3f} adv={p['advanced_bot']:.3f}  "
              f"path={result['features']['mouse_path_length']:.0f}px  cpm={result['features']['clicks_per_minute']:.1f}")
    else:
        print(f"[{args.mode}] not enough evidence:", (result or {}).get("reasons"))

    if args.save:
        res = page.evaluate(
            "([label, token]) => window.ClickstreamTracker.save(label, false, token)",
            [LABELS[args.mode], args.token],
        )
        print("  save ->", res["status"], res["body"])
    ctx.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000/")
    ap.add_argument("--mode", choices=list(LABELS), default="linear")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--gap", type=float, default=2.0, help="seconds between clicks")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--save", action="store_true", help="store the session as a labelled bot example")
    ap.add_argument("--token", default="", help="COLLECT_TOKEN of the server")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        for _ in range(args.repeat):
            run_once(browser, args)
        browser.close()


if __name__ == "__main__":
    main()
