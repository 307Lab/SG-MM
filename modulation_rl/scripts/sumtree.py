import numpy as np
import random
import torch
from collections import deque
from sklearn.neighbors import LocalOutlierFactor 


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries_count = 0 
        

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries_count < self.capacity: 
            self.n_entries_count += 1

        return tree_idx

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

    def min_priority(self):
        if self.n_entries_count == 0:
            return 0.0 

        min_val = float('inf')
        for i in range(self.n_entries_count):
            leaf_idx = i + self.capacity - 1
            min_val = min(min_val, self.tree[leaf_idx])
        return min_val if min_val != float('inf') else 0.0


    def sample(self, s):
        return self.get(s)

    def n_entries(self):
        return self.n_entries_count 