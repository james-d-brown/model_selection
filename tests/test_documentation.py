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
