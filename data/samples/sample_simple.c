int calculate_sum(int a, int b) {
    int total = a + b;
    if (total > 100) {
        return total * 2;
    }
    return total;
}
