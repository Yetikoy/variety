from collections import defaultdict

def connected_components(edges, nodes): # ([[1,2]], [1,2,3]) -> [[1,2], [3]]
    allegiances = {}
    for a,b in edges:
        if a not in allegiances and b not in allegiances:
            allegiances[a] = a
            allegiances[b] = a
        if a in allegiances and b in allegiances:
            if allegiances[a] == allegiances[b]:
                pass
            else:
                full_alliance = [x for x, xa in allegiances.items() if xa == allegiances[a] or xa == allegiances[b]]
                new_allegiance = allegiances[a]
                for x in full_alliance:
                    allegiances[x] = new_allegiance
        if a in allegiances and b not in allegiances:
            allegiances[b] = allegiances[a]
        if a not in allegiances and b in allegiances:
            allegiances[a] = allegiances[b]
    for node in nodes:
        if node not in allegiances:
            allegiances[node] = node
    interm = defaultdict(set)
    for member, alleg in allegiances.items():
        interm[alleg].add(member)
    return list(interm.values())

from util import flatten
def test_nodes_are_the_same(e, n):
    assert set(n) == set(flatten(connected_components(e,n)))

def test_components_are_disjoint(e, n):
    new_nodes = list(flatten(connected_components(e,n)))
    assert len(new_nodes) == len(set(new_nodes))

def test_components_are_traversible(e,n):
    components_by_node = {}
    for component in connected_components(e,n):
        for node in component:
            components_by_node[node] = component
    for node1, node2 in e:
        assert components_by_node[node1] is components_by_node[node2]

def test_unreachable_nodes_are_in_different_components(e,n):
    known_nodes = n
    first_nodes = [node1 for node1, node2 in e]
    second_nodes = [node2 for node1, node2 in e]
    def replace_reachable(n1, n2):
        nonlocal first_nodes, second_nodes
        first_nodes = [n if n is not n1 else n2 for n in first_nodes]
        second_nodes = [n if n is not n1 else n2 for n in second_nodes]

    def trim():
        nonlocal first_nodes, second_nodes
        first_nodes = first_nodes[1:]
        second_nodes = second_nodes[1:]

    while first_nodes:
        n1, n2 = first_nodes[0], second_nodes[0]
        if n1 == n2:
            trim()
        else:
            replace_reachable(n1, n2)
            known_nodes.remove(n1)

    components_by_node = {}
    for index, component in enumerate(connected_components(e,n)):
        for node in component:
            components_by_node[node] = index

    known_nodes_components = [components_by_node[node] for node in known_nodes]
    assert len(set(known_nodes)) == len(set(known_nodes_components))


if __name__ == "__main__":
    test_nodes_are_the_same([[1,2]], [1,2,3])
    test_components_are_disjoint([[1,2]], [1,2,3])
    test_components_are_traversible([[1,2]], [1,2,3])
    test_unreachable_nodes_are_in_different_components([[1,2]], [1,2,3])
