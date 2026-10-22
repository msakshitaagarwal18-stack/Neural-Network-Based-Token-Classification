/* Program 03: Matrix Multiplication and Transpose */
void multiply_matrices(int r1, int c1, int first[][10], int r2, int c2, int second[][10], int result[][10]) {
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c2; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < c1; ++k) {
                result[i][j] += first[i][k] * second[k][j];
            }
        }
    }
}

void transpose_matrix(int row, int col, int matrix[][10], int trans[][10]) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            trans[j][i] = matrix[i][j];
        }
    }
}
