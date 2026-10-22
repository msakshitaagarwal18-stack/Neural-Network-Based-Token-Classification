/* Program 15: Complex Number Arithmetic (Test Program) */
struct Complex {
    double real;
    double imag;
};

struct Complex add_complex(struct Complex a, struct Complex b) {
    struct Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

struct Complex multiply_complex(struct Complex a, struct Complex b) {
    struct Complex res;
    res.real = (a.real * b.real) - (a.imag * b.imag);
    res.imag = (a.real * b.imag) + (a.imag * b.real);
    return res;
}

double complex_magnitude_sq(struct Complex c) {
    return (c.real * c.real) + (c.imag * c.imag);
}
