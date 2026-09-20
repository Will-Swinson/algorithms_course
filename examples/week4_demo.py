# File: examples/week4_demo.py
"""
Week 4 demo: graphs doing practical work.

Run from the project root:

    python examples/week4_demo.py

Shows (1) a tiny social network in both representations, (2) BFS vs
DFS visit orders on the same graph, (3) BFS as "degrees of
separation," and (4) Dijkstra routing on a weighted flight map.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs import (
    Graph, ADJACENCY_MATRIX, bfs, bfs_shortest_paths, dfs_iterative,
    dijkstra, reconstruct_path,
)

FRIENDSHIPS = [
    ("Alice", "Bob"), ("Alice", "Carol"), ("Bob", "Dave"),
    ("Carol", "Dave"), ("Dave", "Erin"), ("Erin", "Frank"),
]


def demo_representations():
    print("=" * 64)
    print("1. One social network, two representations")
    print("=" * 64)
    g = Graph()  # adjacency list (default)
    for u, v in FRIENDSHIPS:
        g.add_edge(u, v)
    print(g)

    m = Graph(representation=ADJACENCY_MATRIX)
    for u, v in FRIENDSHIPS:
        m.add_edge(u, v)
    print(f"\n  Same queries on the matrix version: "
          f"Alice-Bob edge: {m.has_edge('Alice', 'Bob')}, "
          f"Alice's friends: {[n for n, _ in m.get_neighbors('Alice')]}")


def demo_traversals():
    print()
    print("=" * 64)
    print("2. BFS vs DFS from Alice — same graph, different orders")
    print("=" * 64)
    g = Graph()
    for u, v in FRIENDSHIPS:
        g.add_edge(u, v)
    print(f"  BFS (rippling outward):  {bfs(g, 'Alice')}")
    print(f"  DFS (one path at a time): {dfs_iterative(g, 'Alice')}")


def demo_degrees_of_separation():
    print()
    print("=" * 64)
    print("3. BFS predecessors = degrees of separation")
    print("=" * 64)
    g = Graph()
    for u, v in FRIENDSHIPS:
        g.add_edge(u, v)
    preds = bfs_shortest_paths(g, "Alice")
    chain = ["Frank"]
    while preds[chain[-1]] is not None:
        chain.append(preds[chain[-1]])
    chain.reverse()
    print(f"  Alice -> Frank introduction chain: {' -> '.join(chain)}")
    print(f"  ({len(chain) - 1} hops — guaranteed the fewest possible)")


def demo_dijkstra():
    print()
    print("=" * 64)
    print("4. Dijkstra: cheapest flights from Austin")
    print("=" * 64)
    flights = Graph(directed=True)
    for u, v, price in [
        ("AUS", "DFW", 80), ("AUS", "IAH", 60), ("DFW", "ORD", 190),
        ("IAH", "ORD", 240), ("DFW", "JFK", 260), ("ORD", "JFK", 120),
        ("IAH", "MIA", 150), ("MIA", "JFK", 180),
    ]:
        flights.add_edge(u, v, price)
    flights.add_node("HNL")  # no route from Austin

    distances, preds = dijkstra(flights, "AUS")
    for city in ["JFK", "ORD", "MIA", "HNL"]:
        path = reconstruct_path(preds, "AUS", city)
        if path is None:
            print(f"  AUS -> {city}: unreachable "
                  f"(distance = {distances[city]})")
        else:
            print(f"  AUS -> {city}: ${distances[city]:g} "
                  f"via {' -> '.join(path)}")
    print("\n  Two candidate routes to JFK: AUS->DFW->JFK ($80+$260=$340)"
          "\n  vs AUS->DFW->ORD->JFK ($80+$190+$120=$390). Dijkstra "
          "settles it\n  by proof, not guesswork — and unreachable HNL "
          "comes back as\n  distance infinity instead of crashing.")


if __name__ == "__main__":
    demo_representations()
    demo_traversals()
    demo_degrees_of_separation()
    demo_dijkstra()
