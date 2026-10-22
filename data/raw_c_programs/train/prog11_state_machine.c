/* Program 11: Finite State Machine Lexical Scanner */
enum State {
    STATE_START,
    STATE_IN_ID,
    STATE_IN_NUM,
    STATE_ERROR
};

enum State handle_transition(enum State current, char input) {
    switch (current) {
        case STATE_START:
            if ((input >= 'a' && input <= 'z') || (input >= 'A' && input <= 'Z') || input == '_') {
                return STATE_IN_ID;
            } else if (input >= '0' && input <= '9') {
                return STATE_IN_NUM;
            }
            return STATE_ERROR;
        case STATE_IN_ID:
            if ((input >= 'a' && input <= 'z') || (input >= '0' && input <= '9') || input == '_') {
                return STATE_IN_ID;
            }
            return STATE_START;
        case STATE_IN_NUM:
            if (input >= '0' && input <= '9') {
                return STATE_IN_NUM;
            }
            return STATE_START;
        default:
            return STATE_ERROR;
    }
}
