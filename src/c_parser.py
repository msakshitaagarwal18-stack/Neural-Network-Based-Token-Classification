"""C syntax parser and Abstract Syntax Tree (AST) visualizer engine.

Parses ANSI C / C99 code constructs into a hierarchical AST, annotating each node
with exact source locations (line, column). Detects and reports syntax errors
with precise diagnostics and recovery mechanisms.
"""
from typing import List, Dict, Any, Optional, Tuple
import html
from src.lexer_utils import tokenize_c_code, TokenItem

# ---------------------------------------------------------------------------
# AST Node Hierarchy
# ---------------------------------------------------------------------------
class ASTNode:
    """Base class for all Abstract Syntax Tree nodes."""

    def __init__(
        self,
        node_type: str,
        name: str = "",
        value: Any = None,
        line: int = 1,
        col: int = 1,
        children: Optional[List["ASTNode"]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.node_type = node_type
        self.name = name
        self.value = value
        self.line = line
        self.col = col
        self.children = children if children is not None else []
        self.attributes = attributes if attributes is not None else {}

    def add_child(self, child: Optional["ASTNode"]):
        if child is not None:
            self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Converts AST node and descendants to a JSON-serializable dictionary."""
        return {
            "type": self.node_type,
            "name": self.name,
            "value": self.value,
            "line": self.line,
            "col": self.col,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children]
        }

    def format_tree(self, indent: int = 0) -> List[str]:
        """Formats the AST node and its subtrees into an indented hierarchical string list."""
        pad = "  " * indent
        loc_str = f"[L{self.line}:C{self.col}]"
        
        detail_parts = []
        if self.name:
            detail_parts.append(f"name='{self.name}'")
        if self.value is not None:
            detail_parts.append(f"val='{self.value}'")
        for k, v in self.attributes.items():
            if v is not None:
                detail_parts.append(f"{k}='{v}'")

        details = f" ({', '.join(detail_parts)})" if detail_parts else ""
        lines = [f"{pad}|-- {self.node_type} {loc_str}{details}"]

        for child in self.children:
            lines.extend(child.format_tree(indent + 1))
        return lines

    def format_ascii_tree(self, prefix: str = "", is_last: bool = True, is_root: bool = True) -> List[str]:
        """Formats the AST into a hierarchical ASCII tree with clear parent-child connectors."""
        loc_str = f"[L{self.line}:C{self.col}]"
        detail_parts = []
        if self.name:
            detail_parts.append(f"name='{self.name}'")
        if self.value is not None:
            detail_parts.append(f"val='{self.value}'")
        for k, v in self.attributes.items():
            if v is not None:
                detail_parts.append(f"{k}='{v}'")

        details = f" ({', '.join(detail_parts)})" if detail_parts else ""

        lines = []
        if is_root:
            lines.append(f"{self.node_type} {loc_str}{details}")
            next_prefix = ""
        else:
            connector = "\\-- " if is_last else "+-- "
            lines.append(f"{prefix}{connector}{self.node_type} {loc_str}{details}")
            next_prefix = prefix + ("    " if is_last else "|   ")

        for i, child in enumerate(self.children):
            last = (i == len(self.children) - 1)
            lines.extend(child.format_ascii_tree(next_prefix, last, is_root=False))
        return lines

    def render_html_tree(self) -> str:
        """Renders an interactive expandable HTML tree with clear node badges and source locations."""
        detail_parts = []
        if self.name:
            detail_parts.append(f'<span class="ast-meta-key">name=</span><span class="ast-meta-val">&quot;{html.escape(str(self.name))}&quot;</span>')
        if self.value is not None:
            detail_parts.append(f'<span class="ast-meta-key">val=</span><span class="ast-meta-val">&quot;{html.escape(str(self.value))}&quot;</span>')
        for k, v in self.attributes.items():
            if v is not None:
                detail_parts.append(f'<span class="ast-meta-key">{html.escape(str(k))}=</span><span class="ast-meta-val">&quot;{html.escape(str(v))}&quot;</span>')

        details_html = f'<span class="ast-details">({", ".join(detail_parts)})</span>' if detail_parts else ""
        loc_badge = f'<span class="ast-loc-pill">L{self.line}:C{self.col}</span>'
        type_badge = f'<span class="ast-type-badge ast-type-{html.escape(self.node_type.lower())}">{html.escape(self.node_type)}</span>'

        if self.children:
            child_html = "".join(child.render_html_tree() for child in self.children)
            return (
                f'<details open class="ast-branch">\n'
                f'  <summary class="ast-summary"><span class="ast-toggle">▶</span> {type_badge} {loc_badge} {details_html}</summary>\n'
                f'  <div class="ast-children-container">\n{child_html}  </div>\n'
                f'</details>\n'
            )
        else:
            return (
                f'<div class="ast-leaf">\n'
                f'  <span class="ast-leaf-dot">•</span> {type_badge} {loc_badge} {details_html}\n'
                f'</div>\n'
            )

    def get_tree_metrics(self) -> Tuple[int, int]:
        """Calculates (total_nodes, max_depth) across the entire subtree."""
        total_nodes = 1
        max_depth = 1
        for child in self.children:
            c_nodes, c_depth = child.get_tree_metrics()
            total_nodes += c_nodes
            if c_depth + 1 > max_depth:
                max_depth = c_depth + 1
        return total_nodes, max_depth



class SyntaxErrorInfo:
    """Represents a syntax error detected during parsing."""

    def __init__(self, message: str, line: int, col: int, token: str, expected: str = ""):
        self.message = message
        self.line = line
        self.col = col
        self.token = token
        self.expected = expected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "line": self.line,
            "col": self.col,
            "token": self.token,
            "expected": self.expected
        }

    def __repr__(self) -> str:
        return f"SyntaxError(L{self.line}:C{self.col}: {self.message} [found '{self.token}'])"


# ---------------------------------------------------------------------------
# Recursive Descent C Parser
# ---------------------------------------------------------------------------
C_BASE_TYPES = {
    "int", "float", "double", "char", "void", "short", "long", "unsigned",
    "signed", "struct", "union", "enum", "size_t", "bool", "int8_t", "int16_t",
    "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t"
}

ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
LOGICAL_OPS = {"||", "&&"}
EQUALITY_OPS = {"==", "!="}
RELATIONAL_OPS = {"<", ">", "<=", ">="}
ADDITIVE_OPS = {"+", "-"}
MULTIPLICATIVE_OPS = {"*", "/", "%"}

class CParser:
    """
    Lightweight recursive descent parser for C source code.
    Generates AST with node locations and captures syntax errors.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tokens: List[TokenItem] = tokenize_c_code(source_code, require_labels=False)
        self.pos: int = 0
        self.errors: List[SyntaxErrorInfo] = []

    def current_token(self) -> Optional[TokenItem]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def peek_token(self, offset: int = 1) -> Optional[TokenItem]:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def match(self, *expected_values: str) -> Optional[TokenItem]:
        cur = self.current_token()
        if cur and cur.value in expected_values:
            self.pos += 1
            return cur
        return None

    def consume(self, expected_value: str, error_message: str = "") -> Optional[TokenItem]:
        cur = self.current_token()
        if cur and cur.value == expected_value:
            self.pos += 1
            return cur
        
        # Record syntax error
        line = cur.line if cur else (self.tokens[-1].line if self.tokens else 1)
        col = cur.col if cur else (self.tokens[-1].col if self.tokens else 1)
        tok_val = cur.value if cur else "EOF"
        msg = error_message or f"Expected '{expected_value}', found '{tok_val}'"
        
        self.errors.append(SyntaxErrorInfo(msg, line, col, tok_val, expected=expected_value))
        return None

    def synchronize(self, sync_tokens: Tuple[str, ...] = (";", "}", ")")):
        """Error recovery: advances tokens until hitting a synchronization boundary."""
        while self.pos < len(self.tokens):
            tok = self.current_token()
            if tok and tok.value in sync_tokens:
                self.pos += 1
                break
            self.pos += 1

    def parse(self) -> Tuple[ASTNode, List[SyntaxErrorInfo]]:
        """Parses the source code into a top-level Program AST node."""
        root = ASTNode("Program", name="TranslationUnit", line=1, col=1)
        
        while self.pos < len(self.tokens):
            decl = self._parse_external_declaration()
            if decl:
                root.add_child(decl)
            else:
                # If cannot parse, advance to recover
                cur = self.current_token()
                if cur:
                    self.errors.append(SyntaxErrorInfo(
                        f"Unexpected token '{cur.value}' in top-level declaration",
                        cur.line, cur.col, cur.value
                    ))
                    self.pos += 1

        return root, self.errors

    def _parse_external_declaration(self) -> Optional[ASTNode]:
        """Parses a top-level construct: PreprocessorDirective, StructDecl, FunctionDef, or GlobalVar."""
        cur = self.current_token()
        if not cur:
            return None

        # 1. Preprocessor Directive
        if cur.value == "#":
            line, col = cur.line, cur.col
            self.pos += 1
            dir_parts = ["#"]
            while self.pos < len(self.tokens):
                t = self.current_token()
                if t and t.line == line:
                    dir_parts.append(t.value)
                    self.pos += 1
                else:
                    break
            dir_str = " ".join(dir_parts)
            return ASTNode("PreprocessorDirective", value=dir_str, line=line, col=col)

        # 2. Struct Declaration
        if cur.value == "struct" and self.peek_token(1) and self.peek_token(2) and self.peek_token(2).value == "{":
            return self._parse_struct_definition()

        # 3. Type specification (Function or Variable)
        type_str, type_line, type_col = self._parse_type_specifier()
        if not type_str:
            return None

        # Look for identifier name
        name_tok = self.current_token()
        if not name_tok:
            return None

        name = name_tok.value
        name_line, name_col = name_tok.line, name_tok.col
        self.pos += 1

        # Check if Function Definition: `type name ( ... ) { ... }`
        if self.match("("):
            return self._parse_function_definition(type_str, name, name_line, name_col)
        
        # Variable Declaration: `type name [= expr] ;`
        return self._parse_var_declaration_rest(type_str, name, name_line, name_col)

    def _parse_type_specifier(self) -> Tuple[str, int, int]:
        """Reads type keywords and pointer stars (e.g. 'const int *', 'struct Node *')."""
        cur = self.current_token()
        if not cur:
            return "", 1, 1

        parts = []
        start_line, start_col = cur.line, cur.col

        # Modifiers: const, static, volatile
        while cur and cur.value in {"const", "static", "volatile", "unsigned", "signed"}:
            parts.append(cur.value)
            self.pos += 1
            cur = self.current_token()

        if cur and (cur.value in C_BASE_TYPES or cur.value == "struct"):
            parts.append(cur.value)
            self.pos += 1
            cur = self.current_token()
            # If `struct Node`
            if parts[-1] == "struct" and cur:
                parts.append(cur.value)
                self.pos += 1
                cur = self.current_token()
        elif cur and cur.label in ("KEYWORD", "IDENTIFIER"):
            # Custom typedef name
            parts.append(cur.value)
            self.pos += 1
            cur = self.current_token()
        else:
            return "", 1, 1

        # Pointer stars: *
        while cur and cur.value == "*":
            parts.append("*")
            self.pos += 1
            cur = self.current_token()

        return " ".join(parts), start_line, start_col

    def _parse_struct_definition(self) -> ASTNode:
        struct_tok = self.consume("struct")
        line = struct_tok.line if struct_tok else 1
        col = struct_tok.col if struct_tok else 1

        name_tok = self.current_token()
        name = name_tok.value if name_tok and name_tok.value != "{" else "<anonymous>"
        if name != "<anonymous>":
            self.pos += 1

        node = ASTNode("StructDecl", name=name, line=line, col=col)
        self.consume("{", "Expected '{' to start struct body")

        while self.pos < len(self.tokens):
            if self.match("}"):
                break
            m_type, m_l, m_c = self._parse_type_specifier()
            m_name = self.current_token()
            if m_name:
                self.pos += 1
                member_node = ASTNode("StructMember", name=m_name.value, line=m_l, col=m_c, attributes={"type": m_type})
                node.add_child(member_node)
            self.consume(";", "Expected ';' after struct member declaration")

        self.consume(";", "Expected ';' after struct definition")
        return node

    def _parse_function_definition(self, return_type: str, func_name: str, line: int, col: int) -> ASTNode:
        fn_node = ASTNode(
            "FunctionDef",
            name=func_name,
            line=line,
            col=col,
            attributes={"return_type": return_type}
        )

        params_node = ASTNode("ParameterList", line=line, col=col)
        # Parse parameter list
        while self.pos < len(self.tokens):
            if self.match(")"):
                break
            
            p_type, p_l, p_c = self._parse_type_specifier()
            if p_type:
                p_name_tok = self.current_token()
                p_name = p_name_tok.value if p_name_tok and p_name_tok.value not in (",", ")") else ""
                if p_name:
                    self.pos += 1
                # Array parameter e.g. int arr[]
                if self.match("["):
                    self.match("]")
                    p_type += "[]"
                
                param_node = ASTNode("Param", name=p_name, line=p_l, col=p_c, attributes={"type": p_type})
                params_node.add_child(param_node)

            if not self.match(","):
                self.consume(")", "Expected ')' after parameter list")
                break

        fn_node.add_child(params_node)

        # Function body: { ... } or prototype ;
        if self.match(";"):
            fn_node.node_type = "FunctionPrototype"
            return fn_node

        if self.match("{"):
            body = self._parse_compound_statement(open_brace_consumed=True)
            fn_node.add_child(body)
        else:
            cur = self.current_token()
            self.errors.append(SyntaxErrorInfo(
                f"Expected function body '{{' or ';' for function '{func_name}'",
                cur.line if cur else line, cur.col if cur else col, cur.value if cur else "EOF"
            ))
            self.synchronize(("}", ";"))

        return fn_node

    def _parse_var_declaration_rest(self, type_str: str, var_name: str, line: int, col: int) -> ASTNode:
        var_node = ASTNode("VarDecl", name=var_name, line=line, col=col, attributes={"type": type_str})
        
        # Array specifier: e.g. int arr[10];
        if self.match("["):
            size_expr = self._parse_expression()
            if size_expr:
                var_node.attributes["array_size"] = size_expr.value or size_expr.name
            self.consume("]", "Expected ']' in array declaration")

        if self.match("="):
            init_expr = self._parse_expression()
            var_node.add_child(init_expr)

        self.consume(";", f"Expected ';' after declaration of '{var_name}'")
        return var_node

    def _parse_compound_statement(self, open_brace_consumed: bool = False) -> ASTNode:
        cur = self.current_token()
        line = cur.line if cur else 1
        col = cur.col if cur else 1

        if not open_brace_consumed:
            self.consume("{", "Expected '{' to start compound block")

        block_node = ASTNode("CompoundStmt", line=line, col=col)

        while self.pos < len(self.tokens):
            if self.match("}"):
                break
            stmt = self._parse_statement()
            if stmt:
                block_node.add_child(stmt)
            else:
                # Synchronize to prevent infinite loops on malformed code
                cur_t = self.current_token()
                if cur_t and cur_t.value != "}":
                    self.errors.append(SyntaxErrorInfo(
                        f"Unexpected statement syntax at '{cur_t.value}'",
                        cur_t.line, cur_t.col, cur_t.value
                    ))
                    self.synchronize((";", "}"))
                else:
                    self.match("}")
                    break

        return block_node

    def _parse_statement(self) -> Optional[ASTNode]:
        cur = self.current_token()
        if not cur or cur.value == "}":
            return None

        # 1. Nested Block { ... }
        if cur.value == "{":
            self.pos += 1
            return self._parse_compound_statement(open_brace_consumed=True)

        # 2. Return Statement
        if cur.value == "return":
            line, col = cur.line, cur.col
            self.pos += 1
            ret_node = ASTNode("ReturnStmt", line=line, col=col)
            if not self.match(";"):
                expr = self._parse_expression()
                ret_node.add_child(expr)
                self.consume(";", "Expected ';' after return statement")
            return ret_node

        # 3. If Statement
        if cur.value == "if":
            line, col = cur.line, cur.col
            self.pos += 1
            if_node = ASTNode("IfStmt", line=line, col=col)
            self.consume("(", "Expected '(' after 'if'")
            cond = self._parse_expression()
            if_node.add_child(cond)
            self.consume(")", "Expected ')' after 'if' condition")
            then_stmt = self._parse_statement()
            if_node.add_child(then_stmt)
            if self.match("else"):
                else_stmt = self._parse_statement()
                if_node.add_child(else_stmt)
            return if_node

        # 4. While Statement
        if cur.value == "while":
            line, col = cur.line, cur.col
            self.pos += 1
            while_node = ASTNode("WhileStmt", line=line, col=col)
            self.consume("(", "Expected '(' after 'while'")
            cond = self._parse_expression()
            while_node.add_child(cond)
            self.consume(")", "Expected ')' after 'while' condition")
            body = self._parse_statement()
            while_node.add_child(body)
            return while_node

        # 5. For Statement
        if cur.value == "for":
            line, col = cur.line, cur.col
            self.pos += 1
            for_node = ASTNode("ForStmt", line=line, col=col)
            self.consume("(", "Expected '(' after 'for'")
            # init
            if not self.match(";"):
                # Could be `int i = 0` or `i = 0`
                t_str, _, _ = self._parse_type_specifier()
                if t_str:
                    name_tok = self.current_token()
                    if name_tok:
                        self.pos += 1
                        init_node = self._parse_var_declaration_rest(t_str, name_tok.value, line, col)
                        for_node.add_child(init_node)
                else:
                    init_expr = self._parse_expression()
                    for_node.add_child(init_expr)
                    self.consume(";", "Expected ';' after for-init")
            # cond
            if not self.match(";"):
                cond_expr = self._parse_expression()
                for_node.add_child(cond_expr)
                self.consume(";", "Expected ';' after for-condition")
            # update
            if not self.match(")"):
                update_expr = self._parse_expression()
                for_node.add_child(update_expr)
                self.consume(")", "Expected ')' after for-update")
            body = self._parse_statement()
            for_node.add_child(body)
            return for_node

        # 6. Variable Declaration inside block
        # Check if current token is type keyword
        if cur.value in C_BASE_TYPES or (cur.label == "KEYWORD" and cur.value in ("struct", "const", "static", "unsigned")):
            t_str, t_l, t_c = self._parse_type_specifier()
            if t_str:
                n_tok = self.current_token()
                if n_tok and n_tok.value not in (";", "(", ")"):
                    self.pos += 1
                    return self._parse_var_declaration_rest(t_str, n_tok.value, t_l, t_c)

        # 7. Expression Statement (Assignment, function call, increment, etc.)
        expr = self._parse_expression()
        if expr:
            self.consume(";", "Expected ';' after expression")
            stmt_node = ASTNode("ExprStmt", line=expr.line, col=expr.col)
            stmt_node.add_child(expr)
            return stmt_node

        return None

    # -----------------------------------------------------------------------
    # Expression Parsing (Precedence Hierarchy)
    # -----------------------------------------------------------------------
    def _parse_expression(self) -> Optional[ASTNode]:
        return self._parse_assignment()

    def _parse_assignment(self) -> Optional[ASTNode]:
        left = self._parse_logical_or()
        if not left:
            return None

        cur = self.current_token()
        if cur and cur.value in ASSIGN_OPS:
            op_tok = cur
            self.pos += 1
            right = self._parse_assignment()
            node = ASTNode("BinaryOp", name=op_tok.value, line=op_tok.line, col=op_tok.col)
            node.add_child(left)
            node.add_child(right)
            return node
        return left

    def _parse_logical_or(self) -> Optional[ASTNode]:
        left = self._parse_logical_and()
        while True:
            cur = self.current_token()
            if cur and cur.value == "||":
                self.pos += 1
                right = self._parse_logical_and()
                node = ASTNode("BinaryOp", name="||", line=cur.line, col=cur.col)
                node.add_child(left)
                node.add_child(right)
                left = node
            else:
                break
        return left

    def _parse_logical_and(self) -> Optional[ASTNode]:
        left = self._parse_equality()
        while True:
            cur = self.current_token()
            if cur and cur.value == "&&":
                self.pos += 1
                right = self._parse_equality()
                node = ASTNode("BinaryOp", name="&&", line=cur.line, col=cur.col)
                node.add_child(left)
                node.add_child(right)
                left = node
            else:
                break
        return left

    def _parse_equality(self) -> Optional[ASTNode]:
        left = self._parse_relational()
        while True:
            cur = self.current_token()
            if cur and cur.value in EQUALITY_OPS:
                op = cur.value
                self.pos += 1
                right = self._parse_relational()
                node = ASTNode("BinaryOp", name=op, line=cur.line, col=cur.col)
                node.add_child(left)
                node.add_child(right)
                left = node
            else:
                break
        return left

    def _parse_relational(self) -> Optional[ASTNode]:
        left = self._parse_additive()
        while True:
            cur = self.current_token()
            if cur and cur.value in RELATIONAL_OPS:
                op = cur.value
                self.pos += 1
                right = self._parse_additive()
                node = ASTNode("BinaryOp", name=op, line=cur.line, col=cur.col)
                node.add_child(left)
                node.add_child(right)
                left = node
            else:
                break
        return left

    def _parse_additive(self) -> Optional[ASTNode]:
        left = self._parse_multiplicative()
        while True:
            cur = self.current_token()
            if cur and cur.value in ADDITIVE_OPS:
                op = cur.value
                self.pos += 1
                right = self._parse_multiplicative()
                node = ASTNode("BinaryOp", name=op, line=cur.line, col=cur.col)
                node.add_child(left)
                node.add_child(right)
                left = node
            else:
                break
        return left

    def _parse_multiplicative(self) -> Optional[ASTNode]:
        left = self._parse_unary()
        while True:
            cur = self.current_token()
            if cur and cur.value in MULTIPLICATIVE_OPS:
                op = cur.value
                self.pos += 1
                right = self._parse_unary()
                node = ASTNode("BinaryOp", name=op, line=cur.line, col=cur.col)
                node.add_child(left)
                node.add_child(right)
                left = node
            else:
                break
        return left

    def _parse_unary(self) -> Optional[ASTNode]:
        cur = self.current_token()
        if cur and cur.value in ("-", "+", "!", "~", "*", "&", "++", "--", "sizeof"):
            op = cur.value
            line, col = cur.line, cur.col
            self.pos += 1
            operand = self._parse_unary()
            node = ASTNode("UnaryOp", name=op, line=line, col=col)
            node.add_child(operand)
            return node
        return self._parse_postfix()

    def _parse_postfix(self) -> Optional[ASTNode]:
        left = self._parse_primary()
        if not left:
            return None

        while True:
            cur = self.current_token()
            if not cur:
                break

            # Function Call: expr(arg1, arg2)
            if cur.value == "(":
                line, col = cur.line, cur.col
                self.pos += 1
                call_node = ASTNode("FuncCall", name=left.name or "<callee>", line=line, col=col)
                call_node.add_child(left)

                while self.pos < len(self.tokens):
                    if self.match(")"):
                        break
                    arg_expr = self._parse_expression()
                    if arg_expr:
                        call_node.add_child(arg_expr)
                    if not self.match(","):
                        self.consume(")", "Expected ')' after function call arguments")
                        break
                left = call_node

            # Array Subscript: expr[index]
            elif cur.value == "[":
                line, col = cur.line, cur.col
                self.pos += 1
                index_expr = self._parse_expression()
                self.consume("]", "Expected ']' in array subscript")
                sub_node = ASTNode("ArrayAccess", line=line, col=col)
                sub_node.add_child(left)
                sub_node.add_child(index_expr)
                left = sub_node

            # Member Access: expr.member or expr->member
            elif cur.value in (".", "->"):
                op = cur.value
                line, col = cur.line, cur.col
                self.pos += 1
                member_tok = self.current_token()
                member_name = member_tok.value if member_tok else ""
                if member_tok:
                    self.pos += 1
                mem_node = ASTNode("MemberAccess", name=member_name, line=line, col=col, attributes={"op": op})
                mem_node.add_child(left)
                left = mem_node

            # Postfix ++ or --
            elif cur.value in ("++", "--"):
                op = cur.value
                line, col = cur.line, cur.col
                self.pos += 1
                post_node = ASTNode("PostfixOp", name=op, line=line, col=col)
                post_node.add_child(left)
                left = post_node
            else:
                break

        return left

    def _parse_primary(self) -> Optional[ASTNode]:
        cur = self.current_token()
        if not cur:
            return None

        # Parenthesized expression: ( expr )
        if cur.value == "(":
            self.pos += 1
            # Check for Type Cast: e.g. (struct Record*) or (int)
            peek_t = self.current_token()
            if peek_t and (peek_t.value in C_BASE_TYPES or peek_t.value == "struct"):
                cast_type, c_line, c_col = self._parse_type_specifier()
                self.consume(")", "Expected ')' after type cast")
                cast_expr = self._parse_unary()
                node = ASTNode("TypeCast", line=c_line, col=c_col, attributes={"target_type": cast_type})
                node.add_child(cast_expr)
                return node

            expr = self._parse_expression()
            self.consume(")", "Expected ')' to close grouped expression")
            return expr

        # Identifier
        if cur.label == "IDENTIFIER" or (cur.value.isidentifier() and cur.value not in C_BASE_TYPES):
            node = ASTNode("Identifier", name=cur.value, line=cur.line, col=cur.col)
            self.pos += 1
            return node

        # Literal (Number, String, Char)
        if cur.label == "LITERAL" or cur.value.isdigit() or (cur.value.startswith('"') and cur.value.endswith('"')) or (cur.value.startswith("'") and cur.value.endswith("'")):
            lit_type = "string" if cur.value.startswith('"') else ("char" if cur.value.startswith("'") else "number")
            node = ASTNode("Literal", value=cur.value, line=cur.line, col=cur.col, attributes={"literal_type": lit_type})
            self.pos += 1
            return node

        # Special keywords as primary
        if cur.value in ("NULL", "true", "false"):
            node = ASTNode("Literal", value=cur.value, line=cur.line, col=cur.col, attributes={"literal_type": "constant"})
            self.pos += 1
            return node

        return None


def parse_c_code(source_code: str) -> Tuple[ASTNode, List[SyntaxErrorInfo]]:
    """Convenience helper to parse arbitrary C source code and extract AST + syntax errors."""
    parser = CParser(source_code)
    return parser.parse()
