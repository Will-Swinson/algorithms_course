# File: src/structures/heap.py
"""
Array-based binary heaps and a heap-backed priority queue.

A binary heap is a complete binary tree stored flat in a Python list:
for the node at index i, its children live at 2i + 1 and 2i + 2 and
its parent at (i - 1) // 2. Completeness means the array has no holes,
so the tree's height is always floor(log2 n) — which is what makes
insert and extract O(log n).

Heap property:
    MinHeap: every parent <= its children (root is the minimum)
    MaxHeap: every parent >= its children (root is the maximum)

Complexity summary:
    Operation       Time            Space
    insert          O(log n)        O(1)
    extract_*       O(log n)        O(1)
    peek            O(1)            O(1)
    heapify         O(n)            O(1) extra
    is_empty        O(1)            O(1)
"""
from typing import Any, Iterable, List, Optional


class _BinaryHeap:
    """
    Shared machinery for MinHeap and MaxHeap.

    Subclasses override _comes_first(a, b), which returns True when `a`
    belongs closer to the root than `b` — the only place the min/max
    ordering differs.
    """

    def __init__(self, items: Optional[Iterable] = None):
        """
        Create a heap, optionally bulk-loading initial items.

        Args:
            items: Optional iterable of initial values; loaded with the
                O(n) heapify rather than n O(log n) inserts.
        """
        self._items: List = []
        if items is not None:
            self.heapify(items)

    # -- ordering hook -----------------------------------------------------

    def _comes_first(self, a: Any, b: Any) -> bool:
        """Return True if `a` must sit above `b` in the heap."""
        raise NotImplementedError

    # -- public interface --------------------------------------------------

    def __len__(self) -> int:
        """Number of items currently stored."""
        return len(self._items)

    def is_empty(self) -> bool:
        """Return True if the heap holds no items."""
        return not self._items

    def peek(self) -> Any:
        """
        Return the root item (min or max) without removing it.

        Raises:
            IndexError: If the heap is empty.

        Complexity: O(1) — the root is always at index 0.
        """
        if not self._items:
            raise IndexError("peek from an empty heap")
        return self._items[0]

    def insert(self, value: Any) -> None:
        """
        Add a value to the heap.

        The value is appended at the first free slot (keeping the tree
        complete) and then sifted UP: repeatedly swapped with its
        parent while it belongs above it. At most one swap per tree
        level, so O(log n).

        Args:
            value: Item comparable with the heap's existing items.
        """
        self._items.append(value)
        self._sift_up(len(self._items) - 1)

    def heapify(self, items: Iterable) -> None:
        """
        Replace the heap's contents with `items`, restoring the heap
        property in O(n).

        Bottom-up construction: starting from the last internal node
        and walking toward the root, sift each node down. Most nodes
        are near the bottom and barely move, which is why the total
        work is O(n) rather than the O(n log n) of repeated inserts.

        Args:
            items: Iterable of mutually comparable values.
        """
        self._items = list(items)
        for i in range(len(self._items) // 2 - 1, -1, -1):
            self._sift_down(i)

    def _extract_root(self) -> Any:
        """
        Remove and return the root in O(log n).

        The last array element replaces the root (keeping the tree
        complete), then sifts DOWN: repeatedly swapped with whichever
        child belongs highest until the heap property holds again.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._items:
            raise IndexError("extract from an empty heap")
        root = self._items[0]
        last = self._items.pop()
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return root

    # -- internal sifting --------------------------------------------------

    def _sift_up(self, index: int) -> None:
        """Move the item at `index` toward the root until placed."""
        item = self._items[index]
        while index > 0:
            parent = (index - 1) // 2
            if self._comes_first(item, self._items[parent]):
                self._items[index] = self._items[parent]
                index = parent
            else:
                break
        self._items[index] = item

    def _sift_down(self, index: int) -> None:
        """Move the item at `index` toward the leaves until placed."""
        size = len(self._items)
        item = self._items[index]
        while True:
            child = 2 * index + 1
            if child >= size:
                break
            # Pick whichever child belongs higher in the heap
            right = child + 1
            if right < size and self._comes_first(self._items[right],
                                                  self._items[child]):
                child = right
            if self._comes_first(self._items[child], item):
                self._items[index] = self._items[child]
                index = child
            else:
                break
        self._items[index] = item


class MinHeap(_BinaryHeap):
    """Binary min-heap: peek() and extract_min() return the smallest item."""

    def _comes_first(self, a: Any, b: Any) -> bool:
        return a < b

    def extract_min(self) -> Any:
        """
        Remove and return the smallest item in O(log n).

        Raises:
            IndexError: If the heap is empty.
        """
        return self._extract_root()


class MaxHeap(_BinaryHeap):
    """Binary max-heap: peek() and extract_max() return the largest item."""

    def _comes_first(self, a: Any, b: Any) -> bool:
        return a > b

    def extract_max(self) -> Any:
        """
        Remove and return the largest item in O(log n).

        Raises:
            IndexError: If the heap is empty.
        """
        return self._extract_root()


class PriorityQueue:
    """
    Priority queue backed by a MinHeap: lower priority number = served
    first, and items with EQUAL priority come out in insertion (FIFO)
    order.

    Entries are stored as (priority, sequence, item) tuples. The
    monotonically increasing sequence number both breaks priority ties
    fairly and guarantees the heap never has to compare two `item`
    payloads directly — so items themselves need not be comparable.
    """

    def __init__(self):
        self._heap: MinHeap = MinHeap()
        self._sequence: int = 0

    def __len__(self) -> int:
        """Number of queued items."""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Return True if no items are queued."""
        return self._heap.is_empty()

    def enqueue(self, item: Any, priority: float) -> None:
        """
        Add an item with the given priority in O(log n).

        Args:
            item: Any payload (need not be comparable).
            priority: Numeric priority; lower is served sooner.
        """
        self._heap.insert((priority, self._sequence, item))
        self._sequence += 1

    def dequeue(self) -> Any:
        """
        Remove and return the highest-priority (lowest number) item in
        O(log n); FIFO among equal priorities.

        Raises:
            IndexError: If the queue is empty.
        """
        return self._heap.extract_min()[2]

    def peek(self) -> Any:
        """
        Return the next item to be dequeued without removing it, O(1).

        Raises:
            IndexError: If the queue is empty.
        """
        return self._heap.peek()[2]
