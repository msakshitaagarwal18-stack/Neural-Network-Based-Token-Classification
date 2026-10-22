"""Lexer utilities for C tokenization and ground truth labeling using Pygments.

Maps Pygments tokens to 5 canonical compiler classes:
- KEYWORD
- IDENTIFIER
- OPERATOR
- LITERAL
- DELIMITER
"""
from typing import List, Tuple, Optional, Dict, Any
import bisect
import pygments
from pygments.lexers.c_cpp import CLexer
from pygments.token import Token

CLASSES = ["KEYWORD", "IDENTIFIER", "OPERATOR", "LITERAL", "DELIMITER"]
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}
ID_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Multi-character C operators
COMPOUND_OPERATORS = {
    "==", "!=", "<=", ">=", "&&", "||", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "<<", ">>", "->", "::", "<<=", ">>="
}

# Standard delimiters in C
DELIMITER_CHARS = {";", ",", "(", ")", "{", "}", "[", "]", ":", "?"}

def map_pygments_type(tok_type) -> Optional[str]:
    """Map a Pygments token type to one of the 5 canonical compiler classes."""
    if tok_type in Token.Keyword or tok_type in (
        Token.Keyword.Type, Token.Keyword.Constant,
        Token.Keyword.Declaration, Token.Keyword.Reserved,
        Token.Keyword.Namespace
    ):
        return "KEYWORD"

    if tok_type in Token.Operator:
        return "OPERATOR"

    if tok_type in Token.Literal or tok_type in Token.Number or tok_type in Token.String:
        return "LITERAL"

    if tok_type in Token.Punctuation:
        return "DELIMITER"

    if tok_type in Token.Name or tok_type in (
        Token.Name.Function, Token.Name.Variable,
        Token.Name.Constant, Token.Name.Builtin,
        Token.Name.Class, Token.Name.Label,
        Token.Name.Namespace, Token.Name.Other
    ):
        return "IDENTIFIER"

    if tok_type in Token.Comment.Preproc:
        return "KEYWORD"
    if tok_type in Token.Comment.PreprocFile:
        return "LITERAL"

    return None

class TokenItem:
    def __init__(self, value: str, line: int, col: int, label: Optional[str] = None):
        self.value = value
        self.line = line
        self.col = col
        self.label = label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "line": self.line,
            "col": self.col,
            "label": self.label
        }

    def __repr__(self):
        return f"TokenItem({self.value!r}, L{self.line}:C{self.col}, label={self.label})"


def tokenize_c_code(source_code: str, require_labels: bool = False) -> List[TokenItem]:
    """
    Tokenizes C source code using Pygments CLexer.
    Calculates line and column numbers.
    Merges split string literals and compound operators into full tokens.
    Filters out comments and whitespace.
    """
    if not source_code or not source_code.strip():
        return []

    lexer = CLexer()
    raw_tokens: List[Tuple[int, Any, str]] = list(lexer.get_tokens_unprocessed(source_code))

    # Precalculate line starts for line:col indexing
    line_starts = [0]
    for idx, ch in enumerate(source_code):
        if ch == '\n':
            line_starts.append(idx + 1)

    def get_line_col(index: int) -> Tuple[int, int]:
        line_idx = bisect.bisect_right(line_starts, index) - 1
        line_num = line_idx + 1
        col_num = (index - line_starts[line_idx]) + 1
        return line_num, col_num

    # Step 1: Filter comments and pure whitespace, keeping (index, type, value, end_index)
    filtered = []
    for idx, tok_type, val in raw_tokens:
        if tok_type in Token.Text or not val.strip():
            continue
        # Skip pure comments
        if tok_type in Token.Comment and tok_type not in (Token.Comment.Preproc, Token.Comment.PreprocFile):
            continue
        
        # Check if preprocessor directive
        if tok_type in Token.Comment.Preproc:
            # e.g. #include, #define
            val_clean = val.strip()
            if val_clean:
                filtered.append((idx, tok_type, val_clean, idx + len(val_clean)))
            continue

        val_clean = val.strip()
        if val_clean:
            filtered.append((idx, tok_type, val_clean, idx + len(val_clean)))

    # Step 2: Merge contiguous tokens (string literals, compound operators)
    merged_tokens: List[TokenItem] = []
    n = len(filtered)
    i = 0

    while i < n:
        idx, tok_type, val, end_idx = filtered[i]
        label = map_pygments_type(tok_type)

        # 2a. String / Character literals merging: e.g. '"', 'hello', '"'
        if (tok_type in Token.Literal.String or tok_type in Token.Literal.Character or val.startswith('"') or val.startswith("'")) and (val.startswith('"') or val.startswith("'")):
            quote_char = val[0]
            # If the literal is already self-contained (e.g. "abc" or 'a')
            if len(val) > 1 and val.endswith(quote_char) and not val.endswith(f"\\{quote_char}"):
                line, col = get_line_col(idx)
                merged_tokens.append(TokenItem(val, line, col, "LITERAL"))
                i += 1
                continue

            # Otherwise gather parts until closing quote
            combined_val = val
            j = i + 1
            while j < n:
                next_idx, next_type, next_val, next_end = filtered[j]
                # If there's an immediate continuation or string chunk
                combined_val += next_val
                if next_val.endswith(quote_char) and not next_val.endswith(f"\\{quote_char}"):
                    j += 1
                    break
                j += 1
            line, col = get_line_col(idx)
            merged_tokens.append(TokenItem(combined_val, line, col, "LITERAL"))
            i = j
            continue

        # 2b. Compound operator merging: e.g. '+', '=' -> '+='
        if tok_type in Token.Operator and i + 1 < n:
            next_idx, next_type, next_val, _ = filtered[i + 1]
            # If adjacent in source code (no space between) and next is also operator
            if next_idx == end_idx and next_type in Token.Operator:
                candidate = val + next_val
                # Check for 3-character operators (e.g. <<=, >>=)
                if i + 2 < n:
                    n3_idx, n3_type, n3_val, _ = filtered[i + 2]
                    if n3_idx == filtered[i + 1][3] and candidate + n3_val in COMPOUND_OPERATORS:
                        line, col = get_line_col(idx)
                        merged_tokens.append(TokenItem(candidate + n3_val, line, col, "OPERATOR"))
                        i += 3
                        continue

                if candidate in COMPOUND_OPERATORS:
                    line, col = get_line_col(idx)
                    merged_tokens.append(TokenItem(candidate, line, col, "OPERATOR"))
                    i += 2
                    continue

        # 2c. Delimiters vs Operators refinement
        # Pygments sometimes classifies [ or ] or ; as Punctuation, which we mapped to DELIMITER.
        # But if it classified a delimiter as Operator (e.g. ? or :) or vice versa:
        if val in DELIMITER_CHARS:
            label = "DELIMITER"

        line, col = get_line_col(idx)
        if require_labels and label is None:
            i += 1
            continue

        merged_tokens.append(TokenItem(val, line, col, label or "IDENTIFIER"))
        i += 1

    return merged_tokens
