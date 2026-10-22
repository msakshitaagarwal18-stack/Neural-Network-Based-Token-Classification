"""One-time safe patch of the graphical AST controls in app.py.
Run from project root: python update_app_ast.py
"""
from pathlib import Path
import shutil

app = Path('app.py')
if not app.is_file():
    raise SystemExit('ERROR: Run this script from the project root (beside app.py).')
source = app.read_text(encoding='utf-8')
old = '''            show_locations = st.checkbox("Show source line and column", value=True,
                                         key="sec4_tree_locations")
            dot_source = ast_to_dot(ast_root,
                                    direction="TB" if direction == "Top to bottom" else "LR",
                                    show_locations=show_locations)'''
new = '''            ast_mode = st.radio(
                "Visualization mode",
                ["Academic AST", "Detailed AST"],
                horizontal=True,
                key="sec4_ast_mode",
                help="Academic mode emphasizes syntax structure; Detailed mode includes source metadata."
            )
            show_locations = st.checkbox(
                "Show Source Locations (Line & Column)",
                value=(ast_mode == "Detailed AST"),
                key="sec4_tree_locations"
            )
            dot_source = ast_to_dot(
                ast_root,
                direction="TB" if direction == "Top to bottom" else "LR",
                show_locations=show_locations,
                academic=(ast_mode == "Academic AST")
            )'''
if new in source:
    print('Already updated; no changes needed.')
elif old not in source:
    raise SystemExit('ERROR: AST section differs from the saved version. app.py was NOT changed.')
else:
    shutil.copy2(app, app.with_suffix('.py.bak'))
    app.write_text(source.replace(old, new, 1), encoding='utf-8')
    print('Updated app.py. Original saved as app.py.bak.')
