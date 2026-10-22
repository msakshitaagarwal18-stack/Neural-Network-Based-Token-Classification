"""Unit tests for C AST parser and syntax error detection."""
import pytest
from src.c_parser import parse_c_code, ASTNode, SyntaxErrorInfo

def test_parse_simple_c_function():
    code = """
    int calculate(int x) {
        int y = x + 10;
        return y;
    }
    """
    ast, errors = parse_c_code(code)
    assert len(errors) == 0
    assert ast.node_type == "Program"
    
    # Check function def exists
    fn_nodes = [c for c in ast.children if c.node_type == "FunctionDef"]
    assert len(fn_nodes) == 1
    fn = fn_nodes[0]
    assert fn.name == "calculate"
    assert fn.line >= 2
    assert fn.col >= 1
    assert fn.attributes["return_type"] == "int"

def test_parse_ast_node_locations():
    code = "int a = 42;"
    ast, errors = parse_c_code(code)
    assert len(errors) == 0
    assert len(ast.children) == 1
    decl = ast.children[0]
    assert decl.node_type == "VarDecl"
    assert decl.name == "a"
    assert decl.line == 1
    assert decl.col == 5

def test_parse_syntax_error_missing_semicolon():
    bad_code = """
    int main() {
        int x = 10
        return x;
    }
    """
    ast, errors = parse_c_code(bad_code)
    assert len(errors) > 0
    err = errors[0]
    assert err.line >= 2
    assert "Expected ';'" in err.message or "Unexpected" in err.message

def test_parse_c_control_flow():
    code = """
    int loop_check(int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (i > 5) {
                sum += i;
            } else {
                sum += 1;
            }
        }
        return sum;
    }
    """
    ast, errors = parse_c_code(code)
    assert len(errors) == 0
    fn = [c for c in ast.children if c.node_type == "FunctionDef"][0]
    assert fn.name == "loop_check"
    tree_lines = ast.format_tree()
    assert len(tree_lines) > 10

def test_hierarchical_ascii_and_html_tree_rendering():
    code = """
    int calculate(int x) {
        int y = x + 10;
        return y;
    }
    """
    ast, errors = parse_c_code(code)
    assert len(errors) == 0

    # 1. Test ASCII indented tree
    ascii_lines = ast.format_ascii_tree()
    assert len(ascii_lines) >= 4
    assert any("+-- " in line or "\\-- " in line for line in ascii_lines)
    assert any("[L" in line and ":C" in line for line in ascii_lines)

    # 2. Test HTML interactive tree
    html_tree = ast.render_html_tree()
    assert "<details" in html_tree
    assert "ast-type-" in html_tree
    assert "ast-loc-pill" in html_tree
    assert "calculate" in html_tree

    # 3. Test tree metrics
    total_nodes, max_depth = ast.get_tree_metrics()
    assert total_nodes >= 6
    assert max_depth >= 3

