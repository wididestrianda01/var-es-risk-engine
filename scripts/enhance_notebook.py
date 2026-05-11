#!/usr/bin/env python3
"""Helper script for inserting markdown cells into Jupyter notebooks."""
import nbformat as nbf


def load_notebook(path):
    """Load a notebook from path."""
    with open(path, 'r', encoding='utf-8') as f:
        return nbf.read(f, as_version=4)


def save_notebook(nb, path):
    """Save notebook to path."""
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def make_md_cell(source):
    """Create a markdown cell from source string."""
    return nbf.v4.new_markdown_cell(source)


def insert_cells_before_code(nb, cell_indices, sources):
    """
    Insert markdown cells before specified code cell indices.

    nb: notebook object
    cell_indices: list of int — code cell indices to prepend to
    sources: list of str — markdown source for each context cell

    Inserts in reverse order to preserve indices.
    """
    for idx, src in sorted(zip(cell_indices, sources), reverse=True):
        nb.cells.insert(idx, make_md_cell(src))
    return nb


def insert_cells_after_code(nb, cell_indices, sources):
    """
    Insert markdown cells after specified code cell indices.

    Inserts in reverse order to preserve indices.
    """
    for idx, src in sorted(zip(cell_indices, sources), reverse=True):
        nb.cells.insert(idx + 1, make_md_cell(src))
    return nb


def find_code_cells(nb):
    """Return list of (index, cell) for all code cells."""
    return [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'code']


def find_md_cells(nb):
    """Return list of (index, cell) for all markdown cells."""
    return [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'markdown']


def add_logging_suppression(nb, code_cell_index=0):
    """
    Add arch logger suppression to the first code cell (imports cell).
    Inserts after the last import statement, before other code.
    """
    cell = nb.cells[code_cell_index]
    if cell.cell_type != 'code':
        return nb
    source = cell.source
    lines = source.split('\n')
    # Find last import line
    last_import = -1
    for j, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import = j
    if last_import >= 0:
        insertion = 'import logging\nlogging.getLogger("arch").setLevel(logging.ERROR)'
        lines.insert(last_import + 1, insertion)
        cell.source = '\n'.join(lines)
    return nb


def replace_title_cell(nb, new_source):
    """Replace the first markdown cell (title) with new header."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            cell.source = new_source
            return nb
    return nb


def append_md_cell(nb, source):
    """Append a markdown cell to the end of the notebook."""
    nb.cells.append(make_md_cell(source))
    return nb


def notebook_summary(nb):
    """Return a summary string for the notebook."""
    md_count = sum(1 for c in nb.cells if c.cell_type == 'markdown')
    code_count = sum(1 for c in nb.cells if c.cell_type == 'code')
    return f"{len(nb.cells)} cells ({md_count} md, {code_count} code)"


if __name__ == '__main__':
    print("Notebook enhancement helper. Import and use, or run with:")
    print("  python enhance_notebook.py <notebook_path>")
