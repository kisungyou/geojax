"""Audit rendered GeoJAX documentation for broken math and local references."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


IGNORED_TEXT_TAGS = {"code", "pre", "script", "style"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}
TEX_LEAK = re.compile(
    r"\\(?:begin|end|frac|lVert|mathbb|mathcal|operatorname|pi|rVert|sum|Theta|top)\b"
)
PROSE_IN_MATH = re.compile(
    r"[.`]\s+(?:A|An|For|It|Its|The|This|When|where|which|is|requires|to)\b"
)


class RenderedPageParser(HTMLParser):
    """Collect rendered prose, math nodes, references, and element IDs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []
        self.ignored_depth = 0
        self.math_depth = 0
        self.current_math: list[str] = []
        self.math_nodes: list[str] = []
        self.prose: list[str] = []
        self.references: list[tuple[str, str]] = []
        self.ids: set[str] = set()

    def _collect_attributes(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        element_id = attrs.get("id")
        if element_id:
            self.ids.add(element_id)
        if tag in {"a", "link"} and attrs.get("href"):
            self.references.append(("href", attrs["href"]))
        if tag in {"img", "script", "source"} and attrs.get("src"):
            self.references.append(("src", attrs["src"]))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._collect_attributes(tag, attrs)
        if tag in VOID_TAGS:
            return

        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        starts_ignored = tag in IGNORED_TEXT_TAGS
        starts_math = "math" in classes and "nohighlight" in classes
        self.stack.append((tag, starts_ignored, starts_math))

        if starts_ignored:
            self.ignored_depth += 1
        if starts_math:
            if self.math_depth == 0:
                self.current_math = []
            self.math_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._collect_attributes(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return
        open_tag, ends_ignored, ends_math = self.stack.pop()
        if open_tag != tag:
            return
        if ends_ignored:
            self.ignored_depth -= 1
        if ends_math:
            self.math_depth -= 1
            if self.math_depth == 0:
                self.math_nodes.append("".join(self.current_math).strip())
                self.current_math = []

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self.math_depth:
            self.current_math.append(data)
        else:
            self.prose.append(data)


def parse_page(path: Path) -> RenderedPageParser:
    parser = RenderedPageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def resolve_local_reference(site: Path, page: Path, reference: str) -> tuple[Path, str] | None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or reference.startswith("//"):
        return None
    if parsed.scheme in {"data", "javascript", "mailto"}:
        return None

    path_text = unquote(parsed.path)
    if not path_text:
        target = page
    elif path_text.startswith("/"):
        target = site / path_text.lstrip("/")
    else:
        target = page.parent / path_text
    if path_text.endswith("/"):
        target = target / "index.html"
    return target.resolve(), unquote(parsed.fragment)


def audit_site(site: Path) -> list[str]:
    pages = sorted(path for path in site.rglob("*.html") if "_static" not in path.parts)
    if not pages:
        return [f"No HTML pages found under {site}."]

    parsed_pages = {page.resolve(): parse_page(page) for page in pages}
    errors: list[str] = []
    math_count = 0
    local_reference_count = 0

    for page, parser in parsed_pages.items():
        relative_page = page.relative_to(site)
        prose = " ".join(parser.prose)
        if TEX_LEAK.search(prose):
            errors.append(f"{relative_page}: raw TeX command leaked into rendered prose")
        if re.search(r"(?<!\\)\$", prose):
            errors.append(f"{relative_page}: dollar delimiter leaked into rendered prose")

        for math in parser.math_nodes:
            math_count += 1
            is_inline = math.startswith(r"\(")
            expected_end = r"\)" if is_inline else r"\]"
            if not math.endswith(expected_end):
                errors.append(f"{relative_page}: unterminated rendered math node {math[:100]!r}")
            if is_inline and len(math) > 180:
                errors.append(f"{relative_page}: suspiciously long inline math node {math[:100]!r}")
            if "`" in math or PROSE_IN_MATH.search(math):
                errors.append(f"{relative_page}: prose appears inside math node {math[:100]!r}")

        for kind, reference in parser.references:
            resolved = resolve_local_reference(site, page, reference)
            if resolved is None:
                continue
            local_reference_count += 1
            target, fragment = resolved
            if not target.exists():
                errors.append(f"{relative_page}: broken {kind} target {reference!r}")
                continue
            if fragment and target.suffix == ".html":
                target_parser = parsed_pages.get(target)
                if target_parser is None:
                    target_parser = parse_page(target)
                    parsed_pages[target] = target_parser
                if fragment not in target_parser.ids:
                    errors.append(f"{relative_page}: missing fragment {reference!r}")

    if not errors:
        print(
            f"Audited {len(pages)} HTML pages, {math_count} math nodes, "
            f"and {local_reference_count} local references."
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="Rendered Sphinx HTML directory")
    args = parser.parse_args()
    site = args.site.resolve()
    errors = audit_site(site)
    if errors:
        raise SystemExit("Rendered documentation audit failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
