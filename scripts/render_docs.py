"""
Render README.md and docs/methods.md to standalone HTML files next to them
(README.html, docs/methods.html), for reading in a browser. The Markdown files
are the source; the HTML files are not tracked (.gitignore).

    python scripts/render_docs.py [project directory]

The project directory defaults to the one containing this script's folder.
Equations (```math blocks and inline $...$) are converted to MathML, which
browsers display without a network connection. Needs the "docs" extra: pip install -e ".[docs]".
"""
import base64
import re
import sys
from pathlib import Path

import markdown
from latex2mathml.converter import convert as latex_to_mathml


STYLE = """
body { max-width: 52rem; margin: 2rem auto; padding: 0 1rem; font: 16px/1.55 Georgia, serif; color: #1d1d1f; background: #fff; }
h1, h2, h3, h4 { font-family: system-ui, sans-serif; line-height: 1.25; }
h2 { font-size: 1.55rem; border-bottom: 2px solid #bbb; padding-bottom: .25rem; margin-top: 2.6rem; }
h3 { font-size: 1.22rem; margin-top: 2rem; }
h4 { font-size: 1.02rem; margin-top: 1.6rem; font-style: italic; }
.secnum { color: #666; margin-right: .45em; font-weight: 600; }
blockquote { margin: 1rem 0 1rem 1.5rem; padding: .6rem 1rem; border-left: 3px solid #b8bec4; background: #f3f4f5; }
.tablewrap { overflow-x: auto; margin: 1rem 0; }
table { border-collapse: collapse; font-size: .9rem; }
table.cols { width: 100%; table-layout: fixed; }
td code, th code { overflow-wrap: break-word; }
th, td { border: 1px solid #ccc; padding: .35rem .5rem; vertical-align: top; text-align: left; }
th { background: #f3f3f3; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .88em; background: #f5f5f5; padding: 0 .2em; }
pre { background: #f5f5f5; padding: .75rem; overflow-x: auto; }
pre code { padding: 0; }
li > p { margin: .35rem 0; }
img { width: min(96vw, 1250px); max-width: none; height: auto; display: block; margin: 1rem 0; position: relative; left: 50%; transform: translateX(-50%); }
li > ul, li > ol { margin: .35rem 0; }
.toc { font-family: system-ui, sans-serif; font-size: .9rem; border: 1px solid #ddd; padding: .5rem 1rem; margin-bottom: 2rem; }
@media (prefers-color-scheme: dark) {
  body { background: #161618; color: #e6e6e6; }
  th { background: #26262a; } th, td { border-color: #444; }
  code, pre { background: #232327; } h2 { border-color: #555; } .toc { border-color: #444; } .secnum { color: #aaa; } blockquote { background: #232327; border-color: #555; }
  a { color: #8ab4f8; }
}
"""


UNNUMBERED = {"References"}


# Column widths (percent) for the document's recurring table layouts, keyed by
# their header cells. Other tables are laid out by the browser.
COLUMN_WIDTHS = {
    ("Setting", "Default", "Basis", "Configuration"): (15, 31, 34, 20),
    ("Symbol", "Meaning"): (14, 86),
    ("", "Linear tier", "Nonlinear tier", "Free-form tier"): (16, 28, 28, 28),
    ("", "Linear tier", "Nonlinear tier"): (18, 41, 41),
    ("C adds beyond W", "W adds beyond C", "Relation"): (16, 16, 68),
    ("Diagnostic", "Tiers", "What it indicates"): (38, 14, 48),
    ("Tier", "Assumption", "Plot", "Formal test"): (12, 26, 40, 22),
    ("Code", "Raised when"): (26, 74),
    ("Component", "Series whose block length is estimated", "How it is used"): (26, 46, 28),
    ("Component", "How the lag is used", "Basis"): (24, 36, 40),
}


def set_column_widths(body: str) -> str:
    """Wrap every table in a scrolling box and give the recurring layouts fixed
    column widths, so wide text columns are not squeezed by long code names."""
    def one(m):
        table = m.group(0)
        head = re.search(r"<thead>(.*?)</thead>", table, re.S)
        cells = re.findall(r"<th[^>]*>(.*?)</th>", head.group(1), re.S) if head else []
        key = tuple(" ".join(re.sub(r"<[^>]+>", " ", c).split()) for c in cells)
        widths = COLUMN_WIDTHS.get(key)
        if widths:
            cols = "".join(f'<col style="width:{w}%">' for w in widths)
            table = table.replace("<table>", f'<table class="cols"><colgroup>{cols}</colgroup>', 1)
        # Long code names break only after a dot or an underscore.
        table = re.sub(r"<code>(.*?)</code>",
                       lambda c: "<code>" + re.sub(r"([._])", r"\1<wbr>", c.group(1)) + "</code>", table)
        return f'<div class="tablewrap">{table}</div>'
    return re.sub(r"<table>.*?</table>", one, body, flags=re.S)


def number_headings(body: str, toc_html: str) -> tuple[str, str]:
    """Number the h2-h4 headings of the rendered body (1, 1.1, 1.1.1; appendices
    "Appendix A: ..." as A, A.1, A.1.1) and the matching table-of-contents
    entries. The Markdown keeps unnumbered headings, so references by name are
    unaffected."""
    counters = [0, 0, 0]
    letter = None
    numbers = {}

    def label(m):
        nonlocal letter
        level, attrs, text = int(m.group(1)), m.group(2), m.group(3)
        plain = re.sub(r"<[^>]+>", "", text)
        if level == 2:
            if plain in UNNUMBERED:
                letter = None
                return m.group(0)
            app = re.match(r"Appendix ([A-Z]):\s*(.*)", plain)
            counters[1:] = [0, 0]
            if app:
                letter = app.group(1)
                num = letter
                text = app.group(2)
            else:
                counters[0] += 1
                num = str(counters[0])
        elif level == 3:
            counters[1] += 1
            counters[2] = 0
            num = f"{letter or counters[0]}.{counters[1]}"
        else:
            counters[2] += 1
            num = f"{letter or counters[0]}.{counters[1]}.{counters[2]}"
        prefix = f"Appendix {num}:" if level == 2 and letter else num
        ident = re.search(r'id="([^"]+)"', attrs)
        if ident:
            numbers[ident.group(1)] = prefix
        return f'<h{level}{attrs}><span class="secnum">{prefix}</span>{text}</h{level}>'

    body = re.sub(r"<h([234])([^>]*)>(.*?)</h\1>", label, body)

    def toc_label(m):
        ident, text = m.group(1), m.group(2)
        if ident in numbers:
            plain = re.sub(r"^Appendix [A-Z]:\s*", "", text)
            return f'<a href="#{ident}"><span class="secnum">{numbers[ident]}</span>{plain}</a>'
        return m.group(0)

    toc_html = re.sub(r'<a href="#([^"]+)">(.*?)</a>', toc_label, toc_html)
    return body, toc_html


def render(src: Path, dst: Path, title: str, toc: bool) -> None:
    text = src.read_text(encoding="utf-8")
    # Display equations (```math blocks) and inline math ($...$) become MathML, rendered by the browser
    # without a network connection. They are inserted after the Markdown conversion, so that Markdown
    # does not alter the LaTeX.
    equations = []
    inline = []

    def hold(m):
        equations.append(latex_to_mathml(m.group(1).strip(), display="block"))
        return f"\n\nEQUATION{len(equations) - 1}PLACEHOLDER\n\n"

    def hold_inline(m):
        inline.append(latex_to_mathml(m.group(1).strip()))
        return f"INLINEMATH{len(inline) - 1}X"

    text = re.sub(r"```(?:math|latex)\n(.*?)\n```", hold, text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", hold_inline, text)
    # Links between the rendered documents point at the HTML versions.
    text = re.sub(r"\]\((docs/)?methods\.md\)", lambda m: "](" + (m.group(1) or "") + "methods.html)", text)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = md.convert(text)
    for i, eq in enumerate(equations):
        body = body.replace(f"<p>EQUATION{i}PLACEHOLDER</p>", eq)
    for i, eq in reversed(list(enumerate(inline))):
        body = body.replace(f"INLINEMATH{i}X", eq)
    # Local SVG figures are embedded, so the HTML file is self-contained.
    def embed(m):
        figure = src.parent / m.group(1)
        if not figure.exists():
            return m.group(0)
        data = base64.b64encode(figure.read_bytes()).decode("ascii")
        return f'src="data:image/svg+xml;base64,{data}"'
    body = re.sub(r'src="([^":]+\.svg)"', embed, body)
    body = set_column_widths(body)
    toc_html = md.toc
    if toc:
        body, toc_html = number_headings(body, toc_html)
    contents = f'<nav class="toc"><strong>Contents</strong>{toc_html}</nav>' if toc else ""
    dst.write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title}</title><style>{STYLE}</style></head>\n"
        f"<body>\n<p><em>Rendered from {src.name}; the Markdown file is the source.</em></p>\n{contents}\n{body}\n</body></html>\n",
        encoding="utf-8")


def main(root: Path) -> list[Path]:
    """Render both documents under `root`; returns the HTML paths written."""
    targets = [(root / "README.md", root / "README.html", "model_selection", False),
               (root / "docs" / "methods.md", root / "docs" / "methods.html", "model_selection methods", True)]
    for src, dst, title, toc in targets:
        render(src, dst, title, toc)
    return [dst for _, dst, _, _ in targets]


if __name__ == "__main__":
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    for path in main(project):
        print(f"wrote {path}")
