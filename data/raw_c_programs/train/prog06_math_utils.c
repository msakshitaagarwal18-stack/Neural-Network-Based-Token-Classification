/* Program 06: Math Utilities and Prime Sieve */
int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    }
    return 1;
}

double power(double base, int exp) {
    double res = 1.0;
    int p = exp;
    if (p < 0) p = -p;
    while (p > 0) {
        if (p % 2 == 1) res *= base;
        base *= base;
        p /= 2;
    }
    return (exp < 0) ? (1.0 / res) : res;
}

long long factorial(int n) {
    if (n <= 1) return 1;
    long long ans = 1;
    for (int i = 2; i <= n; i++) {
        ans *= i;
    }
    return ans;
}
