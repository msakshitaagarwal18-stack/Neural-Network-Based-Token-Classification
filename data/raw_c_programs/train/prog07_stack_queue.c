/* Program 07: Stack and Queue Implementation */
#define CAPACITY 100

struct Stack {
    int top;
    int items[CAPACITY];
};

void init_stack(struct Stack *s) {
    s->top = -1;
}

int is_full(struct Stack *s) {
    return s->top == CAPACITY - 1;
}

int is_empty(struct Stack *s) {
    return s->top == -1;
}

void push(struct Stack *s, int val) {
    if (!is_full(s)) {
        s->items[++(s->top)] = val;
    }
}

int pop(struct Stack *s) {
    if (!is_empty(s)) {
        return s->items[(s->top)--];
    }
    return -1;
}
