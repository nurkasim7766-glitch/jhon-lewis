"""Drive the real UI end to end with a headless Chrome: upload -> analyze -> process -> editor -> download.

Usage: python scripts/ui_walkthrough.py --base http://127.0.0.1:8765 --out output/ui
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="output/ui")
    ap.add_argument("--video", default="assets/podcast/source.mp4")
    ap.add_argument("--transcript", default="assets/podcast/transcript.json")
    ap.add_argument("--chrome", default="/usr/local/bin/google-chrome")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome, headless=True, args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        # console "Failed to load resource" lines are HTTP statuses (expected 400/404s), not script errors
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)

        page.goto(args.base + "/#/", wait_until="networkidle")
        page.wait_for_selector(".stat .value")
        check("dashboard loads", page.locator(".stat").count() == 4, f"engine status: {page.locator('#engine-status').inner_text()}")
        page.screenshot(path=str(out / "01_dashboard.png"))

        # upload page: create project + upload video + transcript
        page.goto(args.base + "/#/upload", wait_until="networkidle")
        page.fill("#pname", "UI walkthrough — synthetic podcast")
        with page.expect_response(lambda r: "/video" in r.url and r.request.method == "POST", timeout=180_000) as resp:
            page.set_input_files("#f-video", args.video)
        check("video upload accepted", resp.value.status == 200, f"status {resp.value.status}")
        page.wait_for_selector("#dz-video.done", timeout=30_000)
        with page.expect_response(lambda r: "/transcript" in r.url and r.request.method == "POST", timeout=60_000) as resp:
            page.set_input_files("#f-tr", args.transcript)
        check("transcript upload accepted", resp.value.status == 200, f"status {resp.value.status}")
        page.wait_for_selector("#dz-tr.done", timeout=30_000)
        page.screenshot(path=str(out / "02_upload.png"))
        pid = page.url.split("#/p/")[1].split("/")[0]

        # bad upload shows a clear error
        bad = out / "notavideo.mp4"
        bad.write_bytes(b"definitely not a video")
        with page.expect_response(lambda r: "/video" in r.url and r.request.method == "POST", timeout=60_000):
            page.set_input_files("#f-video", str(bad))
        page.wait_for_selector("#video-error .banner", timeout=15_000)
        check("invalid video rejected with visible error", "could not be decoded" in page.locator("#video-error").inner_text())

        # start analysis
        page.click("#go")
        page.wait_for_url(f"**/#/p/{pid}/processing", timeout=15_000)
        page.wait_for_selector(".step", timeout=15_000)
        t0 = time.time()
        while time.time() - t0 < 120:
            pills = page.locator(".step.done, .step.reused").count()
            if pills >= 3 and page.locator("#process").count():
                break
            time.sleep(1)
        page.screenshot(path=str(out / "03_processing_analysis.png"))
        check("analysis job completed (3 steps done, process button available)", page.locator("#process").count() == 1,
              f"{time.time() - t0:.0f}s")

        # discovery
        page.goto(f"{args.base}/#/p/{pid}/clips", wait_until="networkidle")
        page.wait_for_selector(".clip", timeout=15_000)
        n_clips = page.locator(".clip").count()
        check("AI-selected clips appear with thumbnails and scores", n_clips >= 2 and page.locator(".clip .score").count() == n_clips, f"{n_clips} clips")
        page.screenshot(path=str(out / "04_discovery.png"), full_page=True)
        # deselect all but the first two to keep processing short, then preview one
        boxes = page.locator("[data-sel]")
        for i in range(boxes.count()):
            if i >= 2 and boxes.nth(i).is_checked():
                boxes.nth(i).click()
                page.wait_for_timeout(400)
        page.locator("[data-preview]").first.click()
        page.wait_for_selector("#pv", timeout=10_000)
        page.wait_for_function("document.querySelector('#pv') && document.querySelector('#pv').readyState >= 2", timeout=30_000)
        ct = page.evaluate("document.querySelector('#pv').currentTime")
        check("candidate preview seeks into the source at the clip start", ct > 1.0, f"currentTime={ct:.1f}s")
        page.screenshot(path=str(out / "05_preview.png"))
        page.locator(".modal [data-x]").click()
        page.locator("[data-cmp]").nth(0).click()
        page.locator("[data-cmp]").nth(1).click()
        page.click("#compare")
        page.wait_for_selector(".modal table")
        check("compare candidates modal renders", page.locator(".modal table tr").count() > 5)
        page.locator(".modal [data-x]").click()

        # process selected clips
        page.click("#process")
        page.wait_for_url(f"**/#/p/{pid}/processing", timeout=15_000)
        t0 = time.time()
        seen_running = set()
        mid_shot = False
        while time.time() - t0 < 900:
            for s in page.locator(".step.running").all_inner_texts():
                seen_running.add(s.split("\n")[0])
            if not mid_shot and page.locator(".step.running .bar").count():
                page.screenshot(path=str(out / "06_processing_live.png"))
                mid_shot = True
            txt = page.locator("#proc").inner_text()
            if "Done" in txt and "Cancel job" not in txt and page.locator(".pill.err").count() == 0 and "Rendered" in txt:
                break
            if "Failed" in txt:
                break
            time.sleep(2)
        page.screenshot(path=str(out / "07_processing_done.png"), full_page=True)
        proc_txt = page.locator("#proc").inner_text()
        check("processing job finished without failure", "Failed" not in proc_txt and "Rendered" in proc_txt, f"{time.time() - t0:.0f}s, steps seen running: {sorted(seen_running)}")
        check("real frame-level progress was displayed during processing", mid_shot)

        # editor
        clips = json.loads(page.evaluate(f"fetch('/api/projects/{pid}/clips').then(r => r.text())"))
        rendered = [c for c in clips if c["files"]["output"]]
        check("rendered clips have output.mp4 + debug.mp4 + crop plan", bool(rendered) and all(c["files"]["debug"] and c["files"]["phase2"] for c in rendered), f"{len(rendered)} rendered")
        cid = rendered[0]["id"]
        page.goto(f"{args.base}/#/p/{pid}/edit/{cid}", wait_until="networkidle")
        page.wait_for_selector("#v", timeout=15_000)
        page.wait_for_function("document.querySelector('#v') && document.querySelector('#v').readyState >= 2", timeout=60_000)
        page.click("#play")
        page.wait_for_timeout(2500)
        page.click("#play")
        has_overlay = page.evaluate("(() => { const c = document.querySelector('#ov'); const g = c.getContext('2d'); const d = g.getImageData(0,0,c.width,c.height).data; let n=0; for (let i=3;i<d.length;i+=4) if (d[i]>0) n++; return n; })()")
        vert_px = page.evaluate("(() => { const c = document.querySelector('#vert'); const g = c.getContext('2d'); const d = g.getImageData(0,0,c.width,c.height).data; let n=0; for (let i=0;i<d.length;i+=4) if (d[i]+d[i+1]+d[i+2]>30) n++; return n; })()")
        check("editor plays the clip and draws tracking + crop overlay from pipeline data", has_overlay > 1000, f"{has_overlay} overlay px")
        check("vertical 9:16 preview renders the crop_plan crop", vert_px > 1000, f"{vert_px} lit px")
        page.screenshot(path=str(out / "08_editor.png"))
        page.click("#dbg")
        page.wait_for_function("document.querySelector('#v').src.includes('/media/debug')")
        page.wait_for_function("document.querySelector('#v').readyState >= 2", timeout=60_000)
        check("debug overlay toggle switches to the rendered debug.mp4", True)
        page.screenshot(path=str(out / "09_editor_debug.png"))
        page.click("#out")
        page.wait_for_function("document.querySelector('#v').src.includes('/media/output')")
        page.wait_for_function("document.querySelector('#v').readyState >= 1", timeout=60_000)
        dims = page.evaluate("[document.querySelector('#v').videoWidth, document.querySelector('#v').videoHeight]")
        check("rendered output plays in the editor at 1080x1920", dims == [1080, 1920], f"{dims}")

        # export center + download
        page.goto(args.base + "/#/exports", wait_until="networkidle")
        page.wait_for_selector("table", timeout=15_000)
        with page.expect_download(timeout=60_000) as dl:
            page.locator("a.btn-primary[href*='media/output']").first.click()
        f = dl.value
        dest = out / "downloaded_clip.mp4"
        f.save_as(str(dest))
        check("download of a rendered MP4 works", dest.stat().st_size > 100_000, f"{dest.stat().st_size} bytes, name {f.suggested_filename}")
        page.locator("[data-pick]").first.check()
        with page.expect_download(timeout=60_000) as dl:
            page.locator("[data-zip-sel]").first.click()
        zpath = out / "clips.zip"
        dl.value.save_as(str(zpath))
        import zipfile
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
        check("ZIP export of checked clips works", len(names) == 1 and names[0].endswith(".mp4"), f"{names}")
        page.screenshot(path=str(out / "10_exports.png"))

        # re-render reuses detection: queue, wait, inspect run.json
        page.locator("[data-rr]").first.click()
        page.wait_for_timeout(1000)
        t0 = time.time()
        while time.time() - t0 < 300:
            jobs = json.loads(page.evaluate("fetch('/api/jobs').then(r => r.text())"))
            if jobs and jobs[0]["kind"] == "render" and jobs[0]["status"] in ("done", "failed"):
                break
            time.sleep(2)
        run = json.loads(page.evaluate(f"fetch('/api/projects/{pid}/clips/{cid}/artifact/run').then(r => r.text())"))
        st = {s["name"]: s["status"] for s in run["stages"]}
        check("re-render reused detect/speakers and re-ran render", st.get("detect") == "reused" and st.get("render") == "ran", f"{st}")

        check("no uncaught JS errors", not errors, "; ".join(errors[:3]))
        browser.close()

    ok = all(r[1] for r in results)
    print("UI WALKTHROUGH:", "PASS" if ok else "FAIL")
    (out / "results.json").write_text(json.dumps([{"check": n, "ok": o, "detail": d} for n, o, d in results], indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
