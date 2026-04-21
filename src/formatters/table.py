"""Generic plain-text table renderer.

Separates visual layout from domain formatting logic in results.py.
"""

from __future__ import annotations


def render_table(
    headers: list[str],
    rows: list[list[str]],
    title: str = "",
    footer: str = "",
) -> str:
    """Render a plain-text table with auto-fitted column widths.

    First column is left-aligned; all others are right-aligned.
    Optional title is printed above the header, footer below the last row.
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        parts = [
            cell.ljust(col_widths[i]) if i == 0 else cell.rjust(col_widths[i])
            for i, cell in enumerate(cells)
        ]
        return "  ".join(parts)

    header_line = _fmt_row(headers)
    separator = "-" * len(header_line)

    lines: list[str] = []
    if title:
        lines.append(title)
    lines.extend([header_line, separator])
    lines.extend(_fmt_row(r) for r in rows)
    if footer:
        lines.append(footer)

    return "\n".join(lines)
