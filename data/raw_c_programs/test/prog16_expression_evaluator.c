/* Program 16: Simple Arithmetic Expression Evaluator (Test Program) */
int apply_operator(int operand_a, int operand_b, char op) {
    switch (op) {
        case '+':
            return operand_a + operand_b;
        case '-':
            return operand_a - operand_b;
        case '*':
            return operand_a * operand_b;
        case '/':
            if (operand_b != 0) {
                return operand_a / operand_b;
            }
            return 0;
        default:
            return 0;
    }
}

int evaluate_simple_pair(int left_val, int right_val, char operator_sym) {
    if (operator_sym == '+' || operator_sym == '-' ||
        operator_sym == '*' || operator_sym == '/') {
        return apply_operator(left_val, right_val, operator_sym);
    }
    return -1;
}
