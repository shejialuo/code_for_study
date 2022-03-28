#include <iostream>

struct binarySearchTree {
  binarySearchTree* parent;
  binarySearchTree* left;
  binarySearchTree* right;
  int key;
};

binarySearchTree* treeSearch(binarySearchTree* node, int value) {
  while(node != nullptr) {
    if(node->key == value) {
      return node;
    } else if (node->key > value) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  return node;
}

binarySearchTree* treeMinimum(binarySearchTree* node) {
  while(node->left != nullptr) {
    node = node->left;
  }
  return node;
}

binarySearchTree* treeMaximum(binarySearchTree* node) {
  while(node->right != nullptr) {
    node = node -> right;
  }
  return node;
}

/*
  * If we want to find a node'successor in a binary search tree,
  * if the node has right subtree, just find the smallest value
  * of the subtree.

  * If there is no right subtree, just find its outermost left
  * subtree.
*/
binarySearchTree* treeSuccessor(binarySearchTree* node) {
  if(node -> right != nullptr) {
    return treeMinimum(node->right);
  }
  binarySearchTree* parent = node -> parent;
  while(parent != nullptr && parent->right == node) {
    node = parent;
    parent = node->parent;
  }
  return parent;
}

void treeInsert(binarySearchTree* root, binarySearchTree* node) {
  binarySearchTree* parent = nullptr;
  while(root != nullptr) {
    parent = root;
    if(node->key < root -> key) {
      root = root -> left;
    } else {
      root = root -> right;
    }
  }
  node->parent = parent;
  if(parent == nullptr) {
    // the tree is empty
  } else if (node -> key > parent->key) {
    parent -> right = node;
  } else {
    parent -> left = node;
  }
}

