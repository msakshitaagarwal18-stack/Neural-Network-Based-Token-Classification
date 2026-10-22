/* Program 09: Bitwise Manipulation and Flag Registers */
unsigned int count_set_bits(unsigned int n) {
    unsigned int count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

int is_power_of_two(unsigned int x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

unsigned int reverse_bits(unsigned int n) {
    unsigned int rev = 0;
    for (int i = 0; i < 32; i++) {
        rev = (rev << 1) | (n & 1);
        n >>= 1;
    }
    return rev;
}

unsigned int set_bit(unsigned int num, int pos) {
    return num | (1U << pos);
}

unsigned int clear_bit(unsigned int num, int pos) {
    return num & ~(1U << pos);
}
