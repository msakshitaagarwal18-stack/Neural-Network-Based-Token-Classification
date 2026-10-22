/* Program 10: Buffer and Character Frequency Statistics */
struct CharFreq {
    char character;
    int frequency;
};

void compute_frequencies(const char *buffer, int length, int freqs[256]) {
    for (int i = 0; i < 256; i++) {
        freqs[i] = 0;
    }
    for (int i = 0; i < length; i++) {
        unsigned char c = (unsigned char)buffer[i];
        freqs[c]++;
    }
}

int find_most_frequent(const int freqs[256]) {
    int max_idx = 0;
    int max_val = freqs[0];
    for (int i = 1; i < 256; i++) {
        if (freqs[i] > max_val) {
            max_val = freqs[i];
            max_idx = i;
        }
    }
    return max_idx;
}
