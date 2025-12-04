import bisect
import hashlib
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Optional, Set


def default_hash(value: str) -> int:
    """
    64‑bit hash using SHA‑1 truncated.
    """
    h = hashlib.sha1(value.encode("utf-8")).digest()
    # take first 8 bytes => 64 bits
    return int.from_bytes(h[:8], byteorder="big", signed=False)


class ConsistentHashRing:
    """
    Consistent hashing ring with virtual nodes and simple replication.

    - positions: sorted list of ring positions (ints)
    - vnode_to_node: ring position -> physical node id
    - node_to_vnodes: physical node id -> list of ring positions
    """

    def __init__(
        self,
        replicas_per_node: int = 100,
        hash_func=default_hash,
        replication_factor: int = 1,
    ):
        assert replicas_per_node > 0
        assert replication_factor > 0

        self.replicas_per_node = replicas_per_node
        self.hash = hash_func
        self.replication_factor = replication_factor

        self.positions: List[int] = []
        self.vnode_to_node: Dict[int, str] = {}
        self.node_to_vnodes: Dict[str, List[int]] = defaultdict(list)

    # ---------- node management ----------

    def add_node(self, node_id: str) -> None:
        """
        Add a physical node with its virtual nodes to the ring.
        """
        if node_id in self.node_to_vnodes:
            raise ValueError(f"Node {node_id} already exists")

        vnodes: List[int] = []
        for replica_idx in range(self.replicas_per_node):
            key = f"{node_id}#{replica_idx}"
            pos = self.hash(key)
            # handle rare collision by linear probing
            while pos in self.vnode_to_node:
                pos = (pos + 1) & ((1 << 64) - 1)

            bisect.insort(self.positions, pos)
            self.vnode_to_node[pos] = node_id
            vnodes.append(pos)

        self.node_to_vnodes[node_id] = vnodes

    def remove_node(self, node_id: str) -> None:
        """
        Remove a physical node and all its virtual nodes from the ring.
        """
        vnodes = self.node_to_vnodes.pop(node_id, None)
        if vnodes is None:
            raise ValueError(f"Node {node_id} not found")

        for pos in vnodes:
            idx = bisect.bisect_left(self.positions, pos)
            if idx < len(self.positions) and self.positions[idx] == pos:
                self.positions.pop(idx)
            self.vnode_to_node.pop(pos, None)

    def list_nodes(self) -> List[str]:
        return list(self.node_to_vnodes.keys())

    # ---------- key placement ----------

    def _find_vnode_index(self, key_hash: int) -> int:
        """
        Locate the index of the first vnode clockwise from key_hash.
        If we wrap around, return index 0.
        """
        if not self.positions:
            raise RuntimeError("Ring is empty")

        idx = bisect.bisect_left(self.positions, key_hash)
        if idx == len(self.positions):
            idx = 0
        return idx

    def get_nodes_for_key(self, key: str) -> List[str]:
        """
        Return a list of distinct physical nodes that should store this key.
        Length is at most replication_factor (may be smaller if ring has fewer nodes).
        """
        if not self.positions:
            raise RuntimeError("Ring is empty")

        key_hash = self.hash(key)
        idx = self._find_vnode_index(key_hash)

        chosen: List[str] = []
        seen: Set[str] = set()

        while len(chosen) < self.replication_factor and len(seen) < len(
            self.node_to_vnodes
        ):
            pos = self.positions[idx]
            node_id = self.vnode_to_node[pos]
            if node_id not in seen:
                chosen.append(node_id)
                seen.add(node_id)

            idx = (idx + 1) % len(self.positions)

        return chosen

    def get_primary_node(self, key: str) -> str:
        """
        Convenience: first node in get_nodes_for_key.
        """
        return self.get_nodes_for_key(key)[0]

    # ---------- simulation helpers ----------

    def assign_keys(self, keys: Iterable[str]) -> Dict[str, List[str]]:
        """
        Assign each key to its primary node. Returns node_id -> list of keys.
        """
        assignments: Dict[str, List[str]] = defaultdict(list)
        for k in keys:
            node = self.get_primary_node(k)
            assignments[node].append(k)
        return assignments

    def key_distribution(self, keys: Iterable[str]) -> Counter:
        """
        Count how many keys each node owns as primary.
        """
        dist: Counter = Counter()
        for k in keys:
            node = self.get_primary_node(k)
            dist[node] += 1
        return dist

    def moved_keys_ratio(
        self,
        keys: Iterable[str],
        old_mapping: Dict[str, str],
    ) -> float:
        """
        Given a previous mapping key -> node_id, compute fraction of keys whose
        primary node changed under the current ring.
        """
        moved = 0
        total = 0
        for k in keys:
            total += 1
            new_node = self.get_primary_node(k)
            if old_mapping.get(k) != new_node:
                moved += 1
        return moved / total if total else 0.0


# ---------- example usage & test harness ----------

def demo():
    """
    Simple demonstration of load distribution and minimal key movement
    when adding/removing nodes.
    """
    ring = ConsistentHashRing(replicas_per_node=100, replication_factor=2)

    # initial cluster
    for node in ["node-A", "node-B", "node-C"]:
        ring.add_node(node)

    # sample keys
    keys = [f"key-{i}" for i in range(10000)]

    # baseline distribution
    base_dist = ring.key_distribution(keys)
    base_owner = {k: ring.get_primary_node(k) for k in keys}

    print("Initial distribution:")
    for node, count in base_dist.items():
        print(f"{node}: {count}")

    # add a node
    ring.add_node("node-D")
    dist_after_add = ring.key_distribution(keys)
    moved_after_add = ring.moved_keys_ratio(keys, base_owner)

    print("\nAfter adding node-D:")
    for node, count in dist_after_add.items():
        print(f"{node}: {count}")
    print(f"Fraction of keys moved: {moved_after_add:.4f}")

    # remove a node
    ring.remove_node("node-B")
    dist_after_remove = ring.key_distribution(keys)
    moved_after_remove = ring.moved_keys_ratio(keys, base_owner)

    print("\nAfter removing node-B:")
    for node, count in dist_after_remove.items():
        print(f"{node}: {count}")
    print(f"Fraction of keys moved since baseline: {moved_after_remove:.4f}")


if __name__ == "__main__":
    demo()
