"""Browser check: project name + video upload alone must enable "Start AI analysis" (or show why not).

Usage: python scripts/ui_upload_check.py [--base http://127.0.0.1:8765]
"""

from __future__ import annotations

import argparse
import sys

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--video", default="assets/podcast/source.mp4")
    ap.add_argument("--chrome", default="/usr/local/bin/google-chrome")
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=args.chrome, headless=True, args=["--no-sandbox"])
        page = b.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.base + "/#/upload", wait_until="networkidle")
        health = page.evaluate("fetch('/api/health').then(r => r.json())")
        check("button disabled before any upload, with guidance shown", page.locator("#go").is_disabled() and "upload a video" in page.locator("#readiness").inner_text().lower())
        page.fill("#pname", "Video-only project")
        with page.expect_response(lambda r: "/video" in r.url and r.request.method == "POST", timeout=180_000) as resp:
            page.set_input_files("#f-video", args.video)
        check("video upload accepted", resp.value.status == 200)
        page.wait_for_selector("#dz-video.done", timeout=30_000)
        banner = page.locator("#readiness").inner_text()
        if health["transcription_available"]:
            check("button ENABLED with name + video only (transcription installed)", page.locator("#go").is_enabled(), banner)
            check("banner explains automatic local transcription", "transcribed locally" in banner)
        else:
            check("button disabled but with an explicit transcript-needed message", page.locator("#go").is_disabled() and "transcript is required" in banner, banner)
        check("transcript card is marked optional", "optional" in page.locator("#dz-tr").locator("xpath=..").inner_text().lower())
        page.screenshot(path="output/ui2/12_upload_video_only.png")
        check("no JS errors", not errors, "; ".join(errors[:2]))
        # clean up the project created by this check
        pid = page.url.split("#/p/")[1].split("/")[0]
        page.evaluate(f"fetch('/api/projects/{pid}', {{method: 'DELETE'}})")
        b.close()
    print("UPLOAD CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
