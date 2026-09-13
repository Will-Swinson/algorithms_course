# File: tests/test_heap.py
"""Test suite for MinHeap, MaxHeap, and PriorityQueue."""

import random

import pytest

from src.structures.heap import MinHeap, MaxHeap, PriorityQueue

HEAPS = [MinHeap, MaxHeap]


def _drain(heap):
    """Extract every element, returning them in extraction order."""
    extract = heap.extract_min if isinstance(heap, MinHeap) else heap.extract_max
    out = []
    while not heap.is_empty():
        out.append(extract())
    return out


def _expected_order(heap_cls, values):
    """The order a correct heap must produce for these values."""
    return sorted(values, reverse=(heap_cls is MaxHeap))


@pytest.mark.parametrize("heap_cls", HEAPS, ids=lambda c: c.__name__)
class TestHeapCorrectness:
    """Insert-then-extract must always produce fully ordered output —
    which is only possible if the heap property held throughout."""

    @pytest.mark.parametrize("values", [
        [],
        [1],
        [3, 1, 2],
        [5, 5, 5, 5],
        [2, 1, 2, 1, 2],
        [-10, 0, 10, -20, 20],
        [7, 6, 5, 4, 3, 2, 1],  # adversarial: reverse order inserts
    ], ids=["empty", "single", "small", "all_equal", "duplicates",
            "negatives", "reverse"])
    def test_insert_then_extract_sorted(self, heap_cls, values):
        heap = heap_cls()
        for v in values:
            heap.insert(v)
        assert _drain(heap) == _expected_order(heap_cls, values)

    def test_large_random(self, heap_cls):
        random.seed(42)
        values = [random.randint(-1000, 1000) for _ in range(1000)]
        heap = heap_cls()
        for v in values:
            heap.insert(v)
        assert _drain(heap) == _expected_order(heap_cls, values)

    def test_heapify_bulk_load(self, heap_cls):
        random.seed(7)
        values = [random.randint(0, 100) for _ in range(500)]
        heap = heap_cls()
        heap.heapify(values)
        assert len(heap) == len(values)
        assert _drain(heap) == _expected_order(heap_cls, values)

    def test_constructor_accepts_items(self, heap_cls):
        heap = heap_cls([4, 2, 8, 6])
        assert _drain(heap) == _expected_order(heap_cls, [4, 2, 8, 6])

    def test_heapify_replaces_existing_contents(self, heap_cls):
        heap = heap_cls([1, 2, 3])
        heap.heapify([9, 8])
        assert len(heap) == 2
        assert _drain(heap) == _expected_order(heap_cls, [9, 8])

    def test_interleaved_insert_extract(self, heap_cls):
        """The heap must stay valid under mixed operations, not just
        build-then-drain."""
        random.seed(13)
        heap = heap_cls()
        mirror = []  # reference list kept in sync
        extract = "extract_min" if heap_cls is MinHeap else "extract_max"
        pick = min if heap_cls is MinHeap else max
        for _ in range(2000):
            if mirror and random.random() < 0.4:
                expected = pick(mirror)
                mirror.remove(expected)
                assert getattr(heap, extract)() == expected
            else:
                v = random.randint(0, 100)
                mirror.append(v)
                heap.insert(v)
        assert _drain(heap) == _expected_order(heap_cls, mirror)


@pytest.mark.parametrize("heap_cls", HEAPS, ids=lambda c: c.__name__)
class TestHeapBasics:

    def test_peek_matches_extract_and_is_nondestructive(self, heap_cls):
        heap = heap_cls([5, 1, 9])
        top = heap.peek()
        assert len(heap) == 3            # peek removed nothing
        extract = heap.extract_min if heap_cls is MinHeap else heap.extract_max
        assert extract() == top

    def test_is_empty_transitions(self, heap_cls):
        heap = heap_cls()
        assert heap.is_empty()
        heap.insert(1)
        assert not heap.is_empty()
        _drain(heap)
        assert heap.is_empty()

    def test_len_tracks_operations(self, heap_cls):
        heap = heap_cls()
        assert len(heap) == 0
        for i in range(5):
            heap.insert(i)
        assert len(heap) == 5
        _drain(heap)
        assert len(heap) == 0

    def test_peek_empty_raises(self, heap_cls):
        with pytest.raises(IndexError):
            heap_cls().peek()

    def test_extract_empty_raises(self, heap_cls):
        heap = heap_cls()
        extract = heap.extract_min if heap_cls is MinHeap else heap.extract_max
        with pytest.raises(IndexError):
            extract()


class TestHeapProperty:
    """Verify the array invariant directly, not just via extraction."""

    @pytest.mark.parametrize("heap_cls", HEAPS, ids=lambda c: c.__name__)
    def test_array_invariant_after_operations(self, heap_cls):
        random.seed(99)
        heap = heap_cls([random.randint(0, 1000) for _ in range(300)])
        extract = heap.extract_min if heap_cls is MinHeap else heap.extract_max
        for _ in range(100):
            extract()
        for _ in range(50):
            heap.insert(random.randint(0, 1000))

        items = heap._items
        for i in range(1, len(items)):
            parent = items[(i - 1) // 2]
            if heap_cls is MinHeap:
                assert parent <= items[i], f"parent > child at index {i}"
            else:
                assert parent >= items[i], f"parent < child at index {i}"


class TestPriorityQueue:

    def test_dequeues_in_priority_order(self):
        pq = PriorityQueue()
        pq.enqueue("low", 30)
        pq.enqueue("urgent", 1)
        pq.enqueue("normal", 10)
        assert [pq.dequeue() for _ in range(3)] == ["urgent", "normal", "low"]

    def test_fifo_among_equal_priorities(self):
        pq = PriorityQueue()
        for name in ["first", "second", "third"]:
            pq.enqueue(name, 5)
        assert [pq.dequeue() for _ in range(3)] == ["first", "second", "third"]

    def test_items_need_not_be_comparable(self):
        """Ties are broken by sequence number, so payloads (here dicts,
        which don't support <) must never be compared."""
        pq = PriorityQueue()
        pq.enqueue({"task": "a"}, 1)
        pq.enqueue({"task": "b"}, 1)
        assert pq.dequeue() == {"task": "a"}
        assert pq.dequeue() == {"task": "b"}

    def test_peek_is_nondestructive(self):
        pq = PriorityQueue()
        pq.enqueue("only", 1)
        assert pq.peek() == "only"
        assert len(pq) == 1

    def test_empty_behavior(self):
        pq = PriorityQueue()
        assert pq.is_empty()
        assert len(pq) == 0
        with pytest.raises(IndexError):
            pq.dequeue()
        with pytest.raises(IndexError):
            pq.peek()

    def test_mixed_priorities_large(self):
        random.seed(21)
        pq = PriorityQueue()
        entries = [(random.randint(0, 9), i) for i in range(500)]
        for priority, ident in entries:
            pq.enqueue(ident, priority)
        drained = [pq.dequeue() for _ in range(len(entries))]
        # stable sort by priority == priority order with FIFO ties
        expected = [ident for _, ident in
                    sorted(entries, key=lambda e: e[0])]
        assert drained == expected
