import pytest
from src.lexer_utils import tokenize_c_code, CLASSES, CLASS_TO_ID

def test_tokenize_basic_c():
    code = """
    int main() {
        float x = 3.14;
        char *str = "hello";
        if (x > 0) {
            return 1;
        }
        return 0;
    }
    """
    tokens = tokenize_c_code(code)
    assert len(tokens) > 0
    
    # Check that key tokens got expected labels
    token_dict = {t.value: t.label for t in tokens}
    
    assert token_dict.get("int") == "KEYWORD"
    assert token_dict.get("float") == "KEYWORD"
    assert token_dict.get("if") == "KEYWORD"
    assert token_dict.get("return") == "KEYWORD"
    assert token_dict.get("main") == "IDENTIFIER"
    assert token_dict.get("x") == "IDENTIFIER"
    assert token_dict.get("=") == "OPERATOR"
    assert token_dict.get(">") == "OPERATOR"
    assert token_dict.get("3.14") == "LITERAL"
    assert token_dict.get('"hello"') == "LITERAL"
    assert token_dict.get(";") == "DELIMITER"
    assert token_dict.get("{") == "DELIMITER"
    assert token_dict.get("(") == "DELIMITER"

def test_tokenize_empty():
    assert tokenize_c_code("") == []
    assert tokenize_c_code("   \n\t  ") == []

def test_tokenize_all_classes_present():
    code = "int a = 10;"
    tokens = tokenize_c_code(code)
    labels = [t.label for t in tokens]
    assert "KEYWORD" in labels
    assert "IDENTIFIER" in labels
    assert "OPERATOR" in labels
    assert "LITERAL" in labels
    assert "DELIMITER" in labels
