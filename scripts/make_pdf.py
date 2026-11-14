"""Render docs/technical_report.md (with its figures) to docs/technical_report.pdf.

Markdown -> HTML (python-markdown, tables extension) -> PDF with a headless Chromium browser
(Microsoft Edge or Google Chrome, whichever is installed; override with --browser).

Usage:
    python scripts/make_pdf.py [--input docs/technical_report.md] [--output docs/technical_report.pdf]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
# Chrome first: Edge's launcher can return before the PDF is written (handled by polling below)
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "chrome", "msedge",
]

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.2pt; line-height: 1.42;
       color: #1a1a1a; background: #fff; }
h1 { font-size: 19pt; margin: 0 0 6pt; color: #0b0b0b; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1.5px solid #2a78d6;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul { padding-left: 18pt; margin: 4pt 0; }
li { margin: 2pt 0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; background: #f2f2ef;
       padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.4pt;
        break-inside: avoid; }
th, td { border: 1px solid #d6d5d0; padding: 3pt 4pt; text-align: left; vertical-align: top; }
th { background: #eef3fb; font-weight: 600; }
tr:nth-child(even) td { background: #fafaf8; }
img { display: block; max-width: 100%; margin: 6pt auto 10pt; break-inside: avoid; }
img[src*="accuracy_vs_size"] { max-width: 72%; }
hr { border: 0; border-top: 1px solid #d6d5d0; margin: 12pt 0; }
a { color: #1f5fae; text-decoration: none; }
"""


def find_browser(explicit: str | None) -> str:
    for c in ([explicit] if explicit else CANDIDATES):
        if c and (Path(c).exists() or shutil.which(c)):
            return c if Path(c).exists() else shutil.which(c)
    raise SystemExit("no Chromium-based browser found (pass --browser PATH)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(ROOT / "docs" / "technical_report.md"))
    p.add_argument("--output", default=str(ROOT / "docs" / "technical_report.pdf"))
    p.add_argument("--browser", default=None)
    a = p.parse_args(argv)
    src, out = Path(a.input).resolve(), Path(a.output).resolve()
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "sane_lists"])
    title = src.stem.replace("_", " ").title()
    html = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<base href="{src.parent.as_uri()}/"><style>{CSS}</style></head><body>{body}</body></html>')
    browser = find_browser(a.browser)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        if out.exists():
            out.unlink()
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
               f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={out}", page.as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        for _ in range(60):  # wait until the browser has finished writing the file
            if out.exists() and out.stat().st_size > 0:
                time.sleep(1)
                break
            time.sleep(0.5)
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit("PDF was not produced")
    print(f"[pdf] wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
