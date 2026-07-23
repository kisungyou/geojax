from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from docs.audit_html import tex_syntax_errors


DOCS = Path(__file__).resolve().parents[1] / "docs"
LEGACY_MATH_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")


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
            if paragraph[index] != "$" or (
                index > 0 and paragraph[index - 1] == "\\"
            ):
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


def test_kendall_hand_tutorial_data_is_complete():
    data_directory = DOCS / "_static" / "data" / "hands"
    hands = np.loadtxt(data_directory / "hands.txt", skiprows=1)
    labels = np.loadtxt(data_directory / "labels.txt", skiprows=1, dtype=int)

    assert hands.shape == (52, 22 * 3)
    assert labels.shape == (52,)
    assert np.array_equal(np.unique(labels, return_counts=True)[0], np.array([0, 1]))
    assert np.array_equal(np.unique(labels, return_counts=True)[1], np.array([25, 27]))
