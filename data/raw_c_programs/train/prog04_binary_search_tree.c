/* Program 04: Binary Search Tree */
struct BSTNode {
    int key;
    struct BSTNode *left;
    struct BSTNode *right;
};

struct BSTNode* new_bst_node(int item) {
    struct BSTNode* temp = (struct BSTNode*)malloc(sizeof(struct BSTNode));
    temp->key = item;
    temp->left = temp->right = NULL;
    return temp;
}

struct BSTNode* insert(struct BSTNode* node, int key) {
    if (node == NULL) return new_bst_node(key);
    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    return node;
}

struct BSTNode* search(struct BSTNode* root, int key) {
    if (root == NULL || root->key == key)
        return root;
    if (root->key < key)
        return search(root->right, key);
    return search(root->left, key);
}
