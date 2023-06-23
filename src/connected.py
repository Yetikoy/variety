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

