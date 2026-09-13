# File: examples/week3_demo.py
"""
Week 3 demo: the data structures doing practical work.

Run from the project root:

    python examples/week3_demo.py

Shows (1) a priority queue scheduling hospital triage, (2) an AVL tree
staying balanced under sorted input that would flatten a naive BST,
and (3) hash collisions being handled by both strategies.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.structures import (
    MinHeap, PriorityQueue, AVLTree, ChainingHashTable,
    LinearProbingHashTable,
)


def demo_priority_queue():
    print("=" * 64)
    print("1. PriorityQueue: emergency-room triage")
    print("=" * 64)
    pq = PriorityQueue()
    arrivals = [
        ("sprained ankle", 4),
        ("chest pain", 1),
        ("broken arm", 3),
        ("severe bleeding", 1),   # same priority as chest pain
        ("mild fever", 5),
    ]
    for patient, severity in arrivals:
        pq.enqueue(patient, severity)
        print(f"  arrived: {patient} (severity {severity})")
    print("\n  Treatment order (severity first, FIFO on ties):")
    order = 1
    while not pq.is_empty():
        print(f"    {order}. {pq.dequeue()}")
        order += 1


def demo_avl_balance():
    print()
    print("=" * 64)
    print("2. AVLTree: sorted inserts that would break a naive BST")
    print("=" * 64)
    n = 1023
    tree = AVLTree()
    for k in range(n):          # worst-case insertion order: ascending
        tree.insert(k)
    print(f"  Inserted keys 0..{n - 1} in sorted order")
    print(f"  Naive BST height would be: {n} (a linked list)")
    print(f"  AVL height:                {tree.height()} "
          f"(perfect binary tree height for {n} nodes is 10)")
    print(f"  First five keys in order:  "
          f"{[k for k, _ in tree.in_order_traversal()[:5]]}")
    tree.delete(500)
    print(f"  After delete(500): height {tree.height()}, "
          f"500 in tree -> {500 in tree}")


def demo_hash_collisions():
    print()
    print("=" * 64)
    print("3. Hash tables: same data, two collision strategies")
    print("=" * 64)
    random.seed(42)
    inventory = {f"sku_{i:04d}": random.randint(1, 500) for i in range(1000)}

    chaining = ChainingHashTable()
    probing = LinearProbingHashTable()
    for sku, stock in inventory.items():
        chaining.insert(sku, stock)
        probing.insert(sku, stock)

    for name, table in [("chaining", chaining), ("probing ", probing)]:
        print(f"  {name}: {len(table)} items, capacity {table.capacity}, "
              f"load factor {table.load_factor:.2f}")

    sku = "sku_0042"
    print(f"\n  Lookup {sku}: chaining={chaining.get(sku)}, "
          f"probing={probing.get(sku)}, dict oracle={inventory[sku]}")
    chaining.delete(sku)
    probing.delete(sku)
    print(f"  After delete: present in chaining -> {sku in chaining}, "
          f"probing -> {sku in probing}")


def demo_heap_top_k():
    print()
    print("=" * 64)
    print("4. Bonus — MinHeap: streaming top-3 without sorting everything")
    print("=" * 64)
    random.seed(7)
    scores = [random.randint(0, 1000) for _ in range(20)]
    heap = MinHeap()
    k = 3
    for s in scores:            # keep only the k largest seen so far
        heap.insert(s)
        if len(heap) > k:
            heap.extract_min()
    top = sorted([heap.extract_min() for _ in range(k)], reverse=True)
    print(f"  Stream: {scores}")
    print(f"  Top {k}: {top} (oracle: {sorted(scores, reverse=True)[:k]})")


if __name__ == "__main__":
    demo_priority_queue()
    demo_avl_balance()
    demo_hash_collisions()
    demo_heap_top_k()
