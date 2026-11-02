"""Test of the HTML rendering of the documentation (scripts/render_docs.py)."""
import pytest

pytest.importorskip("markdown")
pytest.importorskip("latex2mathml")

from scripts.render_docs import main  # noqa: E402


def test_renders_both_documents_with_equations_tables_and_html_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Title\n\nSee [the methods](docs/methods.md).\n")
    (tmp_path / "docs" / "methods.md").write_text(
        "# Methods\n\n## Section\n\n```math\nH_0:\\; C \\perp O \\mid Z\n```\n\n"
        "Inline $R_\\phi$ here.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n")
    readme, methods = main(tmp_path)
    assert 'href="docs/methods.html"' in readme.read_text()
    html = methods.read_text()
    assert html.count("<math") == 2 and "PLACEHOLDER" not in html and "INLINEMATH" not in html and "<table>" in html and 'class="toc"' in html
