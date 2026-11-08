"""Quick regression: navigate every view of an already-processed project and assert zero JS errors."""

from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"


def main() -> int:
    errors: list[str] = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path="/usr/local/bin/google-chrome", headless=True, args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"])
        page = b.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        page.goto(f"{BASE}/#/", wait_until="networkidle")
        pid = json.loads(page.evaluate(f"fetch('{BASE}/api/projects').then(r => r.text())"))[0]["id"]
        clips = json.loads(page.evaluate(f"fetch('{BASE}/api/projects/{pid}/clips').then(r => r.text())"))
        cid = [c for c in clips if c["files"]["output"]][0]["id"]
        for h in ["#/", "#/projects", f"#/p/{pid}", f"#/p/{pid}/upload", f"#/p/{pid}/processing", f"#/p/{pid}/clips", f"#/p/{pid}/edit/{cid}", f"#/p/{pid}/settings", "#/exports", f"#/p/{pid}/edit/{cid}", "#/"]:
            page.goto(f"{BASE}/{h}", wait_until="networkidle")
            page.wait_for_timeout(900)
            if "/edit/" in h:
                page.wait_for_function("document.querySelector('#v') && document.querySelector('#v').readyState >= 2", timeout=60_000)
                page.click("#play"); page.wait_for_timeout(1200)
                page.click("#dbg"); page.wait_for_timeout(800)
                page.click("#out"); page.wait_for_timeout(800)
        page.screenshot(path="output/ui2/11_dashboard_after.png")
        b.close()
    print("JS errors:", errors or "none")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
