from __future__ import annotations

import inspect
from pathlib import Path
import re

import numpy as np

import geojax.learning as learning
from docs.audit_html import audit_site, tex_syntax_errors


DOCS = Path(__file__).resolve().parents[1] / "docs"
LEGACY_MATH_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")
SCIENTIFIC_GUIDES = {
    Path("guide/foundations.md"),
    Path("guide/geometry.md"),
    Path("guide/learning.md"),
    Path("guide/optimization.md"),
    Path("development/geometry_protocol.md"),
    Path("development/learning_protocol.md"),
    Path("development/optimization_protocol.md"),
}


def test_learning_api_documents_every_public_symbol_once():
    text = (DOCS / "api" / "learning.md").read_text(encoding="utf-8")
    documented_functions = re.findall(
        r"\.\. autofunction::\s+([A-Za-z_][A-Za-z0-9_]*)", text
    )
    documented_classes = re.findall(
        r"\.\. (?:autoclass|autoexception)::\s+([A-Za-z_][A-Za-z0-9_]*)", text
    )
    public_functions = {
        name for name in learning.__all__ if inspect.isfunction(getattr(learning, name))
    }
    public_classes = {
        name for name in learning.__all__ if inspect.isclass(getattr(learning, name))
    }

    assert set(documented_functions) == public_functions
    assert set(documented_classes) == public_classes
    assert len(documented_functions) == len(set(documented_functions))
    assert len(documented_classes) == len(set(documented_classes))


def markdown_prose_blocks(path: Path):
    """Yield non-code Markdown paragraphs with their source line numbers."""
    in_fence = False
    block: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        without_inline_code = re.sub(r"`[^`]*`", "", line)
        if without_inline_code.strip():
            block.append((line_number, without_inline_code))
        elif block:
            yield block
            block = []

    if block:
        yield block


def markdown_math_fragments(path: Path):
    """Yield TeX bodies from MyST math delimiters outside code blocks."""
    for block in markdown_prose_blocks(path):
        paragraph = "\n".join(line for _, line in block)
        index = 0
        while index < len(paragraph):
            if paragraph[index] != "$" or (index > 0 and paragraph[index - 1] == "\\"):
                index += 1
                continue

            delimiter = "$$" if paragraph.startswith("$$", index) else "$"
            start = index + len(delimiter)
            end = start
            while end < len(paragraph):
                if paragraph.startswith(delimiter, end) and (
                    end == 0 or paragraph[end - 1] != "\\"
                ):
                    line_number = block[0][0] + paragraph[:index].count("\n")
                    yield line_number, paragraph[start:end]
                    index = end + len(delimiter)
                    break
                end += 1
            else:
                index = len(paragraph)


def test_markdown_uses_myst_math_delimiters():
    offenders: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for delimiter in LEGACY_MATH_DELIMITERS:
            if delimiter in text:
                offenders.append(f"{path.relative_to(DOCS)}: {delimiter}")

    assert not offenders, (
        "Use $...$ or $$...$$ for MyST math; legacy delimiters leak as text:\n"
        + "\n".join(offenders)
    )


def test_markdown_math_delimiters_are_balanced_per_paragraph():
    offenders: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        for block in markdown_prose_blocks(path):
            paragraph = "\n".join(line for _, line in block)
            unescaped_dollars = re.findall(r"(?<!\\)\$", paragraph)
            if len(unescaped_dollars) % 2:
                relative_path = path.relative_to(DOCS)
                offenders.append(
                    f"{relative_path}:{block[0][0]}-{block[-1][0]} "
                    f"contains {len(unescaped_dollars)} dollar delimiters"
                )

    assert not offenders, "Unbalanced MyST math delimiters:\n" + "\n".join(offenders)


def test_markdown_math_has_safe_tex_structure():
    offenders: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        for line_number, tex in markdown_math_fragments(path):
            for error in tex_syntax_errors(tex):
                offenders.append(f"{path.relative_to(DOCS)}:{line_number}: {error}")

    assert not offenders, "Unsafe TeX in Markdown math:\n" + "\n".join(offenders)


def test_tex_syntax_audit_detects_text_mode_specials():
    assert tex_syntax_errors(r"\texttt{exp_batch}(x)")
    assert tex_syntax_errors(r"\text{R&D}")
    assert tex_syntax_errors(r"\frac{x}{y")
    assert tex_syntax_errors(r"\begin{aligned}x\end{split}")
    assert tex_syntax_errors(r"x $ y")
    assert not tex_syntax_errors(r"\mathtt{exp\_batch}(x)")
    assert not tex_syntax_errors(r"\begin{aligned}x&=y\end{aligned}")


def test_rendered_audit_detects_embedded_notebook_tracebacks(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(
        '<html><body><div class="output traceback highlight-ipythontb">'
        "ValueError: failed</div></body></html>",
        encoding="utf-8",
    )

    errors = audit_site(tmp_path)
    assert any("notebook traceback" in error for error in errors)


def test_kendall_hand_tutorial_data_is_complete():
    data_directory = DOCS / "_static" / "data" / "hands"
    hands = np.loadtxt(data_directory / "hands.txt", skiprows=1)
    labels = np.loadtxt(data_directory / "labels.txt", skiprows=1, dtype=int)

    assert hands.shape == (52, 22 * 3)
    assert labels.shape == (52,)
    assert np.array_equal(np.unique(labels, return_counts=True)[0], np.array([0, 1]))
    assert np.array_equal(np.unique(labels, return_counts=True)[1], np.array([25, 27]))


def test_physionet_eeg_tutorial_data_is_complete():
    data_path = DOCS / "_static" / "data" / "eeg" / "physionet_motor_imagery.npz"
    with np.load(data_path) as data:
        epochs = data["epochs"]
        labels = data["labels"]
        runs = data["runs"]
        subjects = data["subjects"]
        channels = data["channels"]

    assert epochs.shape == (225, 8, 480)
    assert epochs.dtype == np.float32
    assert np.isfinite(epochs).all()
    assert np.array_equal(np.unique(subjects), np.arange(1, 6))
    assert np.array_equal(np.unique(runs, return_counts=True)[0], np.array([4, 8, 12]))
    assert np.array_equal(np.unique(runs, return_counts=True)[1], np.array([75, 75, 75]))
    assert np.array_equal(np.unique(labels, return_counts=True)[0], np.array([0, 1]))
    assert np.array_equal(np.unique(labels, return_counts=True)[1], np.array([113, 112]))
    assert channels.tolist() == [
        "Fc3.",
        "Fc4.",
        "C3..",
        "C1..",
        "C2..",
        "C4..",
        "Cp3.",
        "Cp4.",
    ]


def test_every_scientific_page_has_citations_and_a_local_bibliography():
    tutorial_pages = {
        path.relative_to(DOCS)
        for path in (DOCS / "tutorials").glob("*.md")
        if path.name != "index.md"
    }
    offenders: list[str] = []
    for relative_path in sorted(SCIENTIFIC_GUIDES | tutorial_pages):
        text = (DOCS / relative_path).read_text(encoding="utf-8")
        if "{cite:" not in text:
            offenders.append(f"{relative_path}: no citation role")
        if "```{bibliography}" not in text:
            offenders.append(f"{relative_path}: no local bibliography")

    assert not offenders, "Scientific documentation needs traceable sources:\n" + "\n".join(
        offenders
    )


def test_all_documentation_citations_resolve_to_unique_bibtex_entries():
    bibliography = (DOCS / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibliography)
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})

    cited_keys: set[str] = set()
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for payload in re.findall(r"\{cite(?::\w+)?\}`([^`]+)`", text):
            cited_keys.update(key.strip() for key in payload.split(","))

    missing_keys = sorted(cited_keys - set(keys))
    assert not duplicate_keys, f"Duplicate BibTeX keys: {duplicate_keys}"
    assert not missing_keys, f"Missing BibTeX entries: {missing_keys}"


def test_bibliography_does_not_leak_math_delimiters_into_prose():
    bibliography = (DOCS / "references.bib").read_text(encoding="utf-8")
    assert "$" not in bibliography, (
        "BibTeX math delimiters render literally in citation titles; use plain "
        "bibliographic text instead."
    )
