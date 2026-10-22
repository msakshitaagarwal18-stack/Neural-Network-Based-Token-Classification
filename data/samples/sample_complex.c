struct DatabaseRecord {
    int record_id;
    char code_name[32];
    double score;
};

int update_record(struct DatabaseRecord *record, double bonus) {
    if (record == 0) {
        return -1;
    }
    record->score += bonus;
    if (record->score >= 100.0) {
        return 1;
    }
    return 0;
}
