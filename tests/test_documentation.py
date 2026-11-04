"""Every reference to a section of docs/methods.md, from the README and the
package source, names a section that exists."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
METHODS = ROOT / "docs" / "methods.md"


def _headings() -> set[str]:
    return {m.group(1).strip() for m in re.finditer(r"^#{2,4} (.+)$", METHODS.read_text(encoding="utf-8"), re.M)}


def _references() -> list[tuple[str, str]]:
    found = []
    files = [ROOT / "README.md", *(ROOT / "model_selection").rglob("*.py")]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"docs/methods\.md,\s+([^)\n]+?)(?:\)|\.\s|\.$|\.\n|;)", text):
            found.append((str(path.relative_to(ROOT)), " ".join(m.group(1).split())))
    return found


def test_methods_document_exists_and_is_linked_from_readme():
    assert METHODS.exists()
    assert "docs/methods.md" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_section_reference_resolves():
    headings = _headings()
    refs = _references()
    assert refs, "expected at least one section reference"
    missing = [(f, s) for f, s in refs if s not in headings]
    assert not missing, f"references to missing sections: {missing}"


def _anchors() -> set[str]:
    """The methods document's heading anchors as GitHub makes them (lowercase,
    punctuation removed, each space a hyphen; repeats numbered -1, -2, ...)."""
    seen, anchors = {}, set()
    in_code = False
    for line in METHODS.read_text(encoding="utf-8").split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^#{1,4} (.+)$", line)
        if m:
            base = re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-")
            k = seen.get(base, 0)
            seen[base] = k + 1
            anchors.add(base if k == 0 else f"{base}-{k}")
    return anchors


def test_every_link_to_a_section_resolves():
    """Links within the methods document, and from the README into it, point at
    existing headings, so a renamed section fails here rather than leaving a dead link."""
    anchors = _anchors()
    internal = re.findall(r"\]\(#([^)]+)\)", METHODS.read_text(encoding="utf-8"))
    external = re.findall(r"methods\.md#([^)\s]+)\)", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert internal, "expected links between sections"
    missing = sorted({a for a in internal + external if a not in anchors})
    assert not missing, f"links to missing sections: {missing}"
