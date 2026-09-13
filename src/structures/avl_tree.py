# File: src/structures/avl_tree.py
"""
AVL tree — a self-balancing binary search tree.

A binary search tree keeps keys in order (left subtree < node < right
subtree), which makes search a walk down one root-to-leaf path. The
catch: inserting sorted data into a plain BST produces a linked list —
height n, O(n) operations. The AVL invariant repairs this: every
node's BALANCE FACTOR (left subtree height minus right subtree height)
must stay in {-1, 0, +1}. Whenever an insert or delete pushes a factor
to ±2, one or two ROTATIONS — local O(1) pointer rearrangements that
preserve BST order — restore the invariant on the way back up.

The invariant bounds the height at ~1.44·log2(n + 2), so every
operation is O(log n) in the WORST case, not just on average.

Complexity summary:
    Operation           Time            Space
    insert              O(log n)        O(log n) recursion
    delete              O(log n)        O(log n) recursion
    search              O(log n)        O(1)
    in_order_traversal  O(n)            O(n) output
    height              O(1)            O(1) (stored per node)
"""
from typing import Any, Iterator, List, Optional, Tuple


class AVLNode:
    """
    One node of an AVL tree.

    Attributes:
        key: Comparable key used for ordering.
        value: Payload associated with the key.
        left, right: Child nodes (None for missing children).
        height: Height of the subtree rooted here (leaf = 1). Stored,
            not recomputed, so balance factors cost O(1).
    """

    __slots__ = ("key", "value", "left", "right", "height")

    def __init__(self, key: Any, value: Any = None):
        self.key = key
        self.value = value
        self.left: Optional["AVLNode"] = None
        self.right: Optional["AVLNode"] = None
        self.height: int = 1

    @property
    def balance_factor(self) -> int:
        """height(left subtree) - height(right subtree); AVL keeps this
        in {-1, 0, +1} at every node."""
        left = self.left.height if self.left else 0
        right = self.right.height if self.right else 0
        return left - right


class AVLTree:
    """
    Self-balancing binary search tree with a dict-like key/value API.

    Duplicate keys are not stored twice: inserting an existing key
    updates its value (matching dict semantics, and keeping the
    AVL-vs-dict benchmark comparison fair).
    """

    def __init__(self):
        self._root: Optional[AVLNode] = None
        self._size: int = 0

    # -- basic queries -------------------------------------------------

    def __len__(self) -> int:
        """Number of keys stored."""
        return self._size

    def is_empty(self) -> bool:
        """Return True if the tree holds no keys."""
        return self._root is None

    def height(self) -> int:
        """
        Height of the tree (0 for an empty tree, 1 for a single node).

        Useful for verifying balance: an AVL tree with n nodes must
        satisfy height <= 1.44 * log2(n + 2).
        """
        return self._root.height if self._root else 0

    def __contains__(self, key: Any) -> bool:
        """Return True if `key` is in the tree (O(log n))."""
        return self._find_node(key) is not None

    def search(self, key: Any) -> Any:
        """
        Return the value stored under `key` in O(log n).

        Walks one path from the root: go left when the key is smaller
        than the current node, right when larger, stop on equal.

        Raises:
            KeyError: If the key is not present.
        """
        node = self._find_node(key)
        if node is None:
            raise KeyError(key)
        return node.value

    def _find_node(self, key: Any) -> Optional[AVLNode]:
        node = self._root
        while node is not None:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node
        return None

    # -- insertion -------------------------------------------------------

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Insert a key (with optional value) in O(log n), rebalancing on
        the way back up. Inserting an existing key updates its value.

        Args:
            key: Comparable key.
            value: Payload to associate with the key.
        """
        self._root = self._insert(self._root, key, value)

    def _insert(self, node: Optional[AVLNode], key: Any,
                value: Any) -> AVLNode:
        # Standard BST descent...
        if node is None:
            self._size += 1
            return AVLNode(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value  # duplicate key: update in place
            return node
        # ...then repair height and balance while the recursion unwinds
        self._update_height(node)
        return self._rebalance(node)

    # -- deletion ---------------------------------------------------------

    def delete(self, key: Any) -> None:
        """
        Remove a key in O(log n), rebalancing on the way back up.

        The three classic BST cases apply — leaf (unlink), one child
        (splice the child in), two children (replace with the in-order
        successor, then delete the successor from the right subtree) —
        followed by AVL rebalancing at every node on the path.

        Raises:
            KeyError: If the key is not present.
        """
        if key not in self:
            raise KeyError(key)
        self._root = self._delete(self._root, key)
        self._size -= 1

    def _delete(self, node: Optional[AVLNode], key: Any) -> Optional[AVLNode]:
        if node is None:  # unreachable thanks to the membership check
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Found it. Zero or one child: splice the other side in.
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # Two children: swap in the in-order successor (smallest
            # key of the right subtree), then delete that key below.
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key, node.value = successor.key, successor.value
            node.right = self._delete(node.right, successor.key)
        self._update_height(node)
        return self._rebalance(node)

    # -- traversal ----------------------------------------------------------

    def in_order_traversal(self) -> List[Tuple[Any, Any]]:
        """
        Return all (key, value) pairs in ascending key order, O(n).

        Iterative (explicit stack) so a large tree cannot hit Python's
        recursion limit during a full walk.
        """
        result: List[Tuple[Any, Any]] = []
        stack: List[AVLNode] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            result.append((node.key, node.value))
            node = node.right
        return result

    def keys(self) -> Iterator[Any]:
        """Yield keys in ascending order."""
        return (key for key, _ in self.in_order_traversal())

    # -- balancing machinery -----------------------------------------------

    @staticmethod
    def _update_height(node: AVLNode) -> None:
        left = node.left.height if node.left else 0
        right = node.right.height if node.right else 0
        node.height = 1 + (left if left > right else right)

    def _rebalance(self, node: AVLNode) -> AVLNode:
        """
        Restore the AVL invariant at `node` if its balance factor hit
        ±2. Four cases, named for where the extra weight sits:

            LL (left-left):   single right rotation
            RR (right-right): single left rotation
            LR (left-right):  left-rotate the left child, then right
            RL (right-left):  right-rotate the right child, then left

        The double rotations exist because a single rotation on a
        "zig-zag" shape just flips the zig-zag around without fixing
        the height imbalance.
        """
        balance = node.balance_factor
        if balance > 1:
            if node.left.balance_factor < 0:      # LR case
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)        # LL case
        if balance < -1:
            if node.right.balance_factor > 0:      # RL case
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)         # RR case
        return node

    def _rotate_left(self, node: AVLNode) -> AVLNode:
        """
        Left rotation: the right child becomes the local root.

              node                pivot
             /    \\              /     \\
            A    pivot   ==>   node      C
                /     \\        /   \\
               B       C      A     B

        BST order (A < node < B < pivot < C) is preserved; only three
        pointers change, so the rotation is O(1).
        """
        pivot = node.right
        node.right = pivot.left
        pivot.left = node
        self._update_height(node)
        self._update_height(pivot)
        return pivot

    def _rotate_right(self, node: AVLNode) -> AVLNode:
        """Mirror image of _rotate_left: the left child becomes the
        local root."""
        pivot = node.left
        node.left = pivot.right
        pivot.right = node
        self._update_height(node)
        self._update_height(pivot)
        return pivot
