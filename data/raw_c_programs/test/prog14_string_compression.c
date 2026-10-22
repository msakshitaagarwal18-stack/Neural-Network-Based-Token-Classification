/* Program 14: Run-Length String Compression (Test Program) */
int compress_run_length(const char *source, char *destination) {
    int src_len = 0;
    while (source[src_len] != '\0') src_len++;
    
    int write_idx = 0;
    for (int i = 0; i < src_len; i++) {
        destination[write_idx++] = source[i];
        int count = 1;
        while (i + 1 < src_len && source[i] == source[i + 1]) {
            count++;
            i++;
        }
        if (count > 1) {
            destination[write_idx++] = (char)('0' + count);
        }
    }
    destination[write_idx] = '\0';
    return write_idx;
}
