import heapq
from typing import Any, List, Set


def manhattan_distance(a, b):
    """
    Calculates the manhattan distance between location a and b given that both elements are [x, y]
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def minimum_cost_connecting_coordinates(coords: List[List[int]], return_edges: bool = False) -> Any:
    """
    Calculates the minimal cost of connecting coordinates
    return_edges: Toggles whether the sum is returned or the MST itself
    """
    n = len(coords)
    if n < 2:
        return 0

    ms_tree: Set[int] = set()
    mst_graph = []

    # Packaged as (weight, (edge_intree_index, edge_outoftree_index))
    edges = [(0, (0, 0))]

    while len(ms_tree) < n:
        weight, (edge_tree_index, edge_new_index) = heapq.heappop(edges)

        if edge_new_index in ms_tree:
            continue

        mst_graph.append((weight, (coords[edge_tree_index], coords[edge_new_index])))
        ms_tree.add(edge_new_index)

        for i in range(n):
            if i not in ms_tree:
                heapq.heappush(edges, (manhattan_distance(coords[edge_new_index], coords[i]), (edge_new_index, i)))

    if return_edges:
        return mst_graph[1:]
    else:
        return sum([x[0] for x in mst_graph])


if __name__ == "__main__":
    print(minimum_cost_connecting_coordinates([[0, 0], [1, 10], [2, 4], [3, 8]]))
    print(minimum_cost_connecting_coordinates([[0, 0], [1, 10], [2, 4], [3, 8]], return_edges=True))
