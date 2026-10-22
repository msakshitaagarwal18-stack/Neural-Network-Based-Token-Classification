"""Graphviz rendering of the project's recursive-descent AST.

Academic mode hides source coordinates; detailed mode can show them.
"""
import subprocess
from typing import Optional

PALETTE = {
    'TranslationUnit': ('#DBEAFE', '#1D4ED8'),
    'FunctionDefinition': ('#D1FAE5', '#047857'),
    'Declaration': ('#EDE9FE', '#6D28D9'),
    'BinaryExpression': ('#FEF3C7', '#B45309'),
    'Identifier': ('#E0F2FE', '#0369A1'),
    'Literal': ('#FCE7F3', '#BE185D'),
}


def _escape(value):
    return str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')


def _label(node, academic):
    name = str(node.node_type)
    detail = []
    if getattr(node, 'name', None):
        detail.append(str(node.name))
    if getattr(node, 'value', None) is not None:
        detail.append(str(node.value))
    if academic:
        # Make operators and leaf values legible in conventional academic diagrams.
        if name in ('BinaryOp', 'BinaryExpression', 'UnaryOp', 'UnaryExpression') and detail:
            return f'{name} ({" · ".join(detail)})'
        if name in ('Identifier', 'Literal') and detail:
            return f'{name} ({" · ".join(detail)})'
    return name + ('\n' + ' · '.join(detail) if detail else '')


def ast_to_dot(root, direction='TB', show_locations=False, academic=True):
    """Convert ASTNode into a parent-child DOT graph.

    Existing calls using `show_locations=` remain compatible.
    """
    if direction not in ('TB', 'LR'):
        raise ValueError('direction must be TB or LR')
    lines = [
        'digraph AST {',
        f'  graph [rankdir={direction}, bgcolor="transparent", pad="0.3", nodesep="0.35", ranksep="0.65", splines=polyline];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.16,0.10", penwidth=1.5];',
        '  edge [color="#94A3B8", arrowsize=0.55, penwidth=1.4];',
    ]
    stack = [(root, None)]
    counter = 0
    while stack:
        node, parent = stack.pop()
        node_id = f'n{counter}'
        counter += 1
        name = str(node.node_type)
        label = _label(node, academic)
        if show_locations:
            line = getattr(node, 'line', None)
            col = getattr(node, 'col', None)
            if line is not None and col is not None:
                label += f'\nL{line}:C{col}'
        fill, border = PALETTE.get(name, ('#F1F5F9', '#64748B'))
        if 'Function' in name:
            fill, border = '#D1FAE5', '#047857'
        elif 'Expression' in name or name in ('BinaryOp', 'UnaryOp'):
            fill, border = '#FEF3C7', '#B45309'
        elif 'Statement' in name or name.endswith('Stmt'):
            fill, border = '#EDE9FE', '#6D28D9'
        lines.append(f'  {node_id} [label="{_escape(label)}", fillcolor="{fill}", color="{border}", fontcolor="#0F172A"];')
        if parent is not None:
            lines.append(f'  {parent} -> {node_id};')
        for child in reversed(getattr(node, 'children', [])):
            stack.append((child, node_id))
    lines.append('}')
    return '\n'.join(lines)


def dot_to_svg(dot: str) -> Optional[bytes]:
    """Render downloadable SVG when Graphviz `dot` is installed."""
    try:
        result = subprocess.run(['dot', '-Tsvg'], input=dot.encode('utf-8'),
                                capture_output=True, timeout=20, check=True)
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
