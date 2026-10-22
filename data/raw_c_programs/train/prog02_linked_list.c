/* Program 02: Singly Linked List Implementation */
struct Node {
    int data;
    struct Node* next;
};

struct Node* create_node(int value) {
    struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
    if (new_node == NULL) {
        return NULL;
    }
    new_node->data = value;
    new_node->next = NULL;
    return new_node;
}

void insert_head(struct Node** head_ref, int new_data) {
    struct Node* node = create_node(new_data);
    node->next = *head_ref;
    *head_ref = node;
}

int delete_value(struct Node** head_ref, int key) {
    struct Node* temp = *head_ref;
    struct Node* prev = NULL;
    if (temp != NULL && temp->data == key) {
        *head_ref = temp->next;
        free(temp);
        return 1;
    }
    while (temp != NULL && temp->data != key) {
        prev = temp;
        temp = temp->next;
    }
    if (temp == NULL) return 0;
    prev->next = temp->next;
    free(temp);
    return 1;
}
