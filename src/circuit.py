from collections import OrderedDict, Counter, defaultdict
from dataclasses import dataclass, field
from utils import group_by
from connected import connected_components
# TODO: name is bad. It is sort of like array - keeps elements in series - but with a quick .index

class UniqCounter(OrderedDict):
    def __init__(self):
        super().__init__()
    def __setitem__(self, key, value):
        raise TypeError(f'Cannot set item number directly for "{key}"')
    def append(self, key):
        if key in self:
            raise ValueError(f'Key "{key}" already exists')
        super().__setitem__(key, len(self)) 

@dataclass
class Circuit:
    name: str
    element_counts: dict = field(default_factory=Counter)
    elements: dict = field(default_factory=dict)
    element_connections: dict = field(default_factory=dict) # connector: list[float] -> v1,v2
    currents: dict = field(default_factory=dict)
    voltage_mismatches: dict = field(default_factory=dict)
    commits: dict = field(default_factory=dict)
    nodes: dict = field(default_factory=UniqCounter)
    inflow: dict = field(default_factory=lambda:defaultdict(set))
    outflow: dict = field(default_factory=lambda:defaultdict(set))
    snode_inflow: dict = field(default_factory=lambda:defaultdict(set))
    snode_outflow: dict = field(default_factory=lambda:defaultdict(set))
    supernode_edges: list = field(default_factory=list)
    lenses: list = field(default_factory=list)
    def add_node(self, name):
        self.nodes.append(name)
        index = self.nodes[name]
        self.lenses.append([f'Potential at {name}', lambda xs, state, err: xs[index]])

    def attach(self, element_factory, node1, node2, *args, **kwargs):
        element = element_factory(self.element_counts, *args, **kwargs)
        self.elements[element.name] = element
        index1 = self.nodes[node1]
        index2 = self.nodes[node2]
        self.currents[element.name] = lambda voltages, state: element.get_current(voltages[index2] - voltages[index1], state[element.name])
        self.voltage_mismatches[element.name] = lambda voltages, state: element.get_mismatch(voltages[index2] - voltages[index1], state[element.name])
        self.commits[element.name] = lambda voltages, currents, state, dt: element.commit(voltages[index2] - voltages[index1], currents[element.name], state[element.name], dt)

        if element.is_supernode:
            self.supernode_edges.append([element.name, [node1, node2]])
            self.snode_inflow[node2].add(element.name)
            self.snode_outflow[node1].add(element.name)
        else:
            self.inflow[node2].add(element.name)
            self.outflow[node1].add(element.name)

        for lens_name, lens in element.lenses:
            adapted_lens = lambda xs, state, err: lens(xs[index2]-xs[index1], state[element.name])
            self.lenses.append([f'{lens_name} at {element.name}', adapted_lens])

    def compile_current_funcs(self):
        #TODO: add grounding
        elements_by_super = group_by(self.elements, key=lambda e:e.is_supernode)
        simple_elements = elements_by_super[False]
        snodes = elements_by_super[True]

        def get_snode_current_calculators():
            updaters = []
            supernode_edges = dict(self.supernode_edges)

            def get_superdeps(edge, node):
                inflow = [e for e in self.snode_inflow[node] if e != edge]
                outflow = [e for e in self.snode_outflow[node] if e != edge]
                return inflow, outflow

            def get_edge_superdeps(edge):
                outflow_node, inflow_node = supernode_edges[edge]
                return [get_superdeps(edge, inflow_node), get_superdeps(edge, outflow_node)]

            snode_deps = {name: get_edge_superdeps(name) for name in snodes}
            calculated_edges = set(simple_elements)

            def mk_updater(edge, is_inflow=True):
                outflow_node, inflow_node = supernode_edges[edge]
                reference_node = inflow_node if is_inflow else outflow_node

                def updater(edge_currents, node_currents):
                    current = node_currents[reference_node]
                    edge_currents[edge] = current
                    node_currents[inflow_node] -= current
                    node_currents[outflow_node] += current
                return updater

            def is_calculated(deps): return all(dep in calculated_edges for dep in deps)

            while True:
                progress = None
                for _ in snodes:
                    name = _
                    if name in calculated_edges:
                        continue

                    deps_in, deps_out = snode_deps[name]
                    if all(is_calculated(deps) for deps in deps_in):
                        updaters.append(mk_updater(name, is_inflow=True))
                        calculated_edges.add(name)
                        progress = True
                        continue

                    if all(is_calculated(deps) for deps in deps_out):
                        updaters.append(mk_updater(name, is_inflow=False))
                        calculated_edges.add(name)
                        progress = True
                        continue

                if progress is None:
                    break

            for element_name in self.elements:
                if element_name not in calculated_edges:
                    print(f"Unable to calculate current for {element_name}, stubbing with None and hoping for the best")
                    updaters.append(lambda edge_currents, node_currents: edge_currents.__setitem__(element_name, None))

            def result(edge_currents, node_currents):
                for u in updaters:
                    u(edge_currents, node_currents)

            return result

        account_for_snode_currents = get_snode_current_calculators()

        def get_currents_simple(voltages, state):
            edge_currents = {name:self.currents[name](voltages, state) for name in simple_elements}
            node_currents = {}
            for node in self.nodes:
                inflow = sum(edge_currents[element] for element in self.inflow[node])
                outflow = sum(edge_currents[element] for element in self.outflow[node])
                node_currents[node] = inflow - outflow
            return edge_currents, node_currents

        def get_currents_full(voltages, state):
            edge_currents, node_currents = get_currents_simple(voltages, state)
            account_for_snode_currents(edge_currents, node_currents)
            return edge_currents, node_currents
        return get_currents_simple, get_currents_full

    def compile_error_func(self, get_currents):
        supernode_edges = [edge for name, edge in self.supernode_edges]
        conn_components = connected_components(supernode_edges, list(self.nodes.keys()))
        snodes = [name for name, elem in self.elements.items() if elem.is_supernode]

        def get_errors(voltages, state):
            edge_currents, node_currents = get_currents(voltages, state)
            net_currents = [sum(node_currents[node] for node in cluster) for cluster in conn_components]
            net_currents.append(sum(node_currents.values()))
            snode_mismatches = [self.voltage_mismatches[name](voltages, state) for name in snodes]
            return net_currents + snode_mismatches

        return get_errors

    def compile_state_func(self, get_currents_full):
        initial_state = {name: element.get_initial_state() for name, element in self.elements.items()}
        elements_by_state_status = group_by(initial_state, key=lambda state: state is not None)
        stateful = elements_by_state_status[True]
        stateless = elements_by_state_status[False]
        stub_state = {elem: None for elem in stateless}

        def fun(voltages, state, dt):
            edge_currents, node_currents = get_currents_full(voltages, state)
            result = dict(stub_state)
            for elem in stateful:
                result[elem] = self.commits[elem](voltages, edge_currents, state, dt)
            return result

        return initial_state, fun

    def compile_funcs(self):
        get_currents_simple, get_currents_full = self.compile_current_funcs()
        error_func = self.compile_error_func(get_currents_simple)
        initial_state, fun = self.compile_state_func(get_currents_full)
        return initial_state, fun, error_func

