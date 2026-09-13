# File: src/structures/hash_table.py
"""
Hash tables with the two classic collision-handling strategies.

A hash table turns a key into an array index (hash(key) % capacity)
so insert/get/delete are O(1) on average. Two keys can land on the
same index — a COLLISION — and the two standard remedies are:

    ChainingHashTable      each slot holds a linked list of entries;
                           colliding keys simply share the list
    LinearProbingHashTable each slot holds at most one entry; on
                           collision, scan forward (wrapping around)
                           to the next free slot ("open addressing")

Both track their LOAD FACTOR (stored entries / capacity) and REHASH —
allocate a bigger array and re-insert everything — when it crosses a
threshold. Rehashing is O(n), but doubling capacity each time makes
the AMORTIZED cost per insert O(1): an expensive rehash at size n is
"paid for" by the n cheap inserts that preceded it.

Complexity summary:
    Operation   Average     Worst case (all keys colliding)
    insert      O(1)*       O(n)
    get         O(1)        O(n)
    delete      O(1)        O(n)
    * amortized; an individual insert that triggers a rehash is O(n)
"""
from typing import Any, Iterator, List, Optional, Tuple


class _ChainNode:
    """One link in a separate-chaining bucket's linked list."""

    __slots__ = ("key", "value", "next")

    def __init__(self, key: Any, value: Any,
                 next_node: Optional["_ChainNode"] = None):
        self.key = key
        self.value = value
        self.next = next_node


class ChainingHashTable:
    """
    Hash table using separate chaining: each bucket is a singly linked
    list of _ChainNode entries.

    Collisions only lengthen one bucket's chain, so performance
    degrades gracefully — with a good hash and load factor L, the
    average chain has L entries. Rehashing doubles capacity when the
    load factor exceeds `max_load_factor` (default 0.75).
    """

    _INITIAL_CAPACITY = 8

    def __init__(self, initial_capacity: int = _INITIAL_CAPACITY,
                 max_load_factor: float = 0.75):
        """
        Args:
            initial_capacity: Starting number of buckets.
            max_load_factor: Rehash when size/capacity exceeds this.
                Pass float("inf") to disable rehashing (used by the
                worst-case benchmark).
        """
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be >= 1")
        self._buckets: List[Optional[_ChainNode]] = [None] * initial_capacity
        self._size: int = 0
        self._max_load_factor = max_load_factor

    # -- inspection ------------------------------------------------------

    def __len__(self) -> int:
        """Number of key/value pairs stored."""
        return self._size

    @property
    def capacity(self) -> int:
        """Current number of buckets."""
        return len(self._buckets)

    @property
    def load_factor(self) -> float:
        """size / capacity — average entries per bucket."""
        return self._size / len(self._buckets)

    def __contains__(self, key: Any) -> bool:
        node = self._buckets[hash(key) % len(self._buckets)]
        while node is not None:
            if node.key == key:
                return True
            node = node.next
        return False

    # -- core operations -------------------------------------------------

    def insert(self, key: Any, value: Any) -> None:
        """
        Store value under key (updating in place if the key exists),
        O(1) amortized.

        Raises:
            TypeError: If the key is unhashable (raised by hash()).
        """
        index = hash(key) % len(self._buckets)
        node = self._buckets[index]
        while node is not None:
            if node.key == key:
                node.value = value
                return
            node = node.next
        # Prepend: O(1) and no tail pointer needed
        self._buckets[index] = _ChainNode(key, value, self._buckets[index])
        self._size += 1
        if self.load_factor > self._max_load_factor:
            self._rehash()

    def get(self, key: Any) -> Any:
        """
        Return the value stored under key, O(1) average.

        Raises:
            KeyError: If the key is not present.
        """
        node = self._buckets[hash(key) % len(self._buckets)]
        while node is not None:
            if node.key == key:
                return node.value
            node = node.next
        raise KeyError(key)

    def delete(self, key: Any) -> None:
        """
        Remove a key, O(1) average — unlink its node from the chain.

        Raises:
            KeyError: If the key is not present.
        """
        index = hash(key) % len(self._buckets)
        node = self._buckets[index]
        previous: Optional[_ChainNode] = None
        while node is not None:
            if node.key == key:
                if previous is None:
                    self._buckets[index] = node.next
                else:
                    previous.next = node.next
                self._size -= 1
                return
            previous, node = node, node.next
        raise KeyError(key)

    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Yield every (key, value) pair (arbitrary order), O(n)."""
        for node in self._buckets:
            while node is not None:
                yield node.key, node.value
                node = node.next

    # -- resizing ------------------------------------------------------------

    def _rehash(self) -> None:
        """Double the bucket array and re-insert every entry (O(n)).
        Every key's index changes because the modulus changed."""
        old_buckets = self._buckets
        self._buckets = [None] * (len(old_buckets) * 2)
        self._size = 0
        for node in old_buckets:
            while node is not None:
                self.insert(node.key, node.value)
                node = node.next


class LinearProbingHashTable:
    """
    Hash table using open addressing with linear probing: all entries
    live directly in one flat array. A collision walks forward one
    slot at a time (wrapping at the end) until a free slot is found.

    Deletion needs TOMBSTONES: simply emptying a slot would break the
    probe chain for any key that had probed past it, so deleted slots
    are marked with a sentinel that lookups walk through but inserts
    may reuse.

    Probing clusters as the table fills, so the rehash threshold is
    kept lower than chaining's (default 0.6), and tombstones count
    toward it.
    """

    _INITIAL_CAPACITY = 8
    _EMPTY = None
    _TOMBSTONE = object()  # sentinel marking a deleted slot

    def __init__(self, initial_capacity: int = _INITIAL_CAPACITY,
                 max_load_factor: float = 0.6):
        """
        Args:
            initial_capacity: Starting number of slots.
            max_load_factor: Rehash when (entries + tombstones) /
                capacity exceeds this. Pass a value >= 1 at your own
                risk: a full table with no free slot cannot be probed.
        """
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be >= 1")
        self._slots: List[Any] = [self._EMPTY] * initial_capacity
        self._values: List[Any] = [None] * initial_capacity
        self._size: int = 0        # live entries
        self._occupied: int = 0    # live entries + tombstones
        self._max_load_factor = max_load_factor

    # -- inspection --------------------------------------------------------

    def __len__(self) -> int:
        """Number of live key/value pairs."""
        return self._size

    @property
    def capacity(self) -> int:
        """Current number of slots."""
        return len(self._slots)

    @property
    def load_factor(self) -> float:
        """Live entries / capacity (tombstones excluded)."""
        return self._size / len(self._slots)

    def __contains__(self, key: Any) -> bool:
        return self._probe(key)[1] is not None

    # -- probing core ------------------------------------------------------

    def _probe(self, key: Any) -> Tuple[int, Optional[int]]:
        """
        Walk the probe sequence for `key`.

        Returns:
            (insert_index, found_index): found_index is the slot
            holding the key (None if absent); insert_index is where an
            insert should place it — the key's own slot if present,
            else the first tombstone or empty slot encountered.
        """
        slots = self._slots
        capacity = len(slots)
        index = hash(key) % capacity
        first_tombstone = -1
        for _ in range(capacity):
            slot = slots[index]
            if slot is self._EMPTY:
                insert_at = first_tombstone if first_tombstone >= 0 else index
                return insert_at, None
            if slot is self._TOMBSTONE:
                if first_tombstone < 0:
                    first_tombstone = index
            elif slot == key:
                return index, index
            index = (index + 1) % capacity
        # Probed every slot (all full/tombstones) without finding key
        return (first_tombstone if first_tombstone >= 0 else -1), None

    # -- core operations ----------------------------------------------------

    def insert(self, key: Any, value: Any) -> None:
        """
        Store value under key (updating in place if present), O(1)
        amortized while the load factor stays below the threshold.

        Raises:
            TypeError: If the key is unhashable (raised by hash()).
        """
        insert_at, found = self._probe(key)
        if found is not None:
            self._values[found] = value
            return
        reusing_tombstone = self._slots[insert_at] is self._TOMBSTONE
        self._slots[insert_at] = key
        self._values[insert_at] = value
        self._size += 1
        if not reusing_tombstone:
            self._occupied += 1
        if self._occupied / len(self._slots) > self._max_load_factor:
            self._rehash()

    def get(self, key: Any) -> Any:
        """
        Return the value stored under key, O(1) average.

        Raises:
            KeyError: If the key is not present.
        """
        _, found = self._probe(key)
        if found is None:
            raise KeyError(key)
        return self._values[found]

    def delete(self, key: Any) -> None:
        """
        Remove a key by replacing its slot with a tombstone, O(1)
        average. The tombstone keeps later keys in the same probe
        chain reachable.

        Raises:
            KeyError: If the key is not present.
        """
        _, found = self._probe(key)
        if found is None:
            raise KeyError(key)
        self._slots[found] = self._TOMBSTONE
        self._values[found] = None
        self._size -= 1
        # _occupied unchanged: the tombstone still occupies the slot

    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Yield every live (key, value) pair (arbitrary order), O(n)."""
        for index, slot in enumerate(self._slots):
            if slot is not self._EMPTY and slot is not self._TOMBSTONE:
                yield slot, self._values[index]

    # -- resizing --------------------------------------------------------------

    def _rehash(self) -> None:
        """Double the slot array and re-insert live entries (O(n)).
        Tombstones are dropped here — rehashing is also how the table
        reclaims deleted-slot space."""
        old_items = list(self.items())
        new_capacity = len(self._slots) * 2
        self._slots = [self._EMPTY] * new_capacity
        self._values = [None] * new_capacity
        self._size = 0
        self._occupied = 0
        for key, value in old_items:
            self.insert(key, value)
