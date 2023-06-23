

import numpy as np

from scipy import optimize

from dataclasses import dataclass, field

from collections.abc import Callable

from copy import deepcopy


@dataclass

class Simulation:

    name: str

    fun: Callable[[list[float], dict, float],[list[float], dict, float]] = lambda xs, state, dt: [list(xs), dict(state), 0]

    xs: list[float] = field(default_factory=list)

    state: dict = field(default_factory=dict)

    steps: int = 0

    # np array is better for querying, but with vanilla array we don't have to know the shape beforehand

    # Thus, we accumulate in a vanilla array and then reformat to np

    recorded_data: list[list] = field(default_factory=list)

    

    def snapshot(self, name): return Simulation(name, self.fun, list(self.xs), deepcopy(self.state))

    def slice_data(self, xs, state, error): return xs + [error]


    def step(self, tol, dt):

        xs, state, error = self.fun(self.xs, self.state, dt)

        if tol is not None and error > tol:

            raise ValueError(f'Error value {error} must be lower than {tol}')

        # we only get here if all above went smoothly

        self.recorded_data.append(self.slice_data(xs, state, error))

        self.xs = xs

        self.state = state

        self.steps += 1

    

    def _get_sim_limit(self, steps, dt, total_time):

        if total_time is not None:

            if dt is not None:

                if steps is not None:

                    print('Overspecified time steps, ignoring the exact value')

                    steps = int(total_time / dt)

                return steps, dt, total_time

            if steps is not None:

                dt = total_time / steps

                return steps, dt, total_time

            raise ValueError('Underspecified time steps, please provide either two of: steps, dt, total_time')

        else:

            return steps, dt, steps * dt

    def pre_sim_routine(self, steps, dt, total_time):

        pass

    def post_sim_routine(self, steps, dt, total_time):

        self.recorded_data = np.array(self.recorded_data)

        

    def simulate(self, steps=None, dt=None, total_time=None, tol=None):

        steps, dt, total_time = self._get_sim_limit(steps, dt, total_time)

        self.pre_sim_routine(steps, dt, total_time)

        for i in range(steps):

            self.step(tol, dt)

        self.post_sim_routine(steps, dt, total_time)

        self.total_time = total_time


from math import log


def make_exponent(bases):

    state = {'expo_coeffs': [log(x) for x in bases]}

    def exponent(xs, state, dt):

        xs = [x * (1 + dt * ln) for x, ln in zip(xs, state['expo_coeffs'])]

        error = dt * dt

        return xs, state, error

    return exponent, bases, state


s = Simulation('exp', *make_exponent([1,2,3,4]))

s.simulate(steps=100, total_time=2)


print(s.xs)

for x, x_t in zip(s.xs, [1,8,27,64]):

    print((x-x_t) / x_t)


[1.0, 7.924192127014331, 26.365220918758766, 61.629634948515765]
0.0
-0.009475984123208603
-0.023510336342267932
-0.037036953929441174

import matplotlib.pyplot as plt

from numpy import linspace

from collections import Counter


class GraphSimulation(Simulation):

    def __init__(self, *args, **kwargs):

        Simulation.__init__(self, *args, **kwargs)

        self._lenses = OrderedDict()

        self._lens_index = {}

        self._lens_locked = False

    def pre_sim_routine(self, *args):

            super().pre_sim_routine(*args)

            self._lens_locked = True

    def add_lens(self, name, extractor):

        if self._lens_locked:

            raise ValueError('Locked, cannot add any more lenses now')

        if name in self._lenses:

            raise ValueError(f'Lens "{name}" already registered')

        self._lenses[name] = extractor

        self._lens_index[name] = len(self._lens_index)

        

    def add_lenses(self, lenses):

        for name, extractor in lenses:

            self.add_lens(name, extractor)

    def slice_data(self, xs, state, error): return [extr(xs, state, error) for extr in self._lenses.values()]

    

    def extract_dataline(self, name):

        if name not in self._lenses:

            raise KeyError(f'Unknown lens "{name}"')

        return self.recorded_data[:, self._lens_index[name]]

    def plot(self, *names, start=0):

        xs = linspace(start, start + self.total_time, num=self.steps)

        for name in names:

            dataline = self.extract_dataline(name)

            label_name = name

            # Force TeX if elements are used and it is not invoked

            if '^' in name or '\\' in name or '_' in name:

                if '$' not in name:

                    label_name = f'${name}$'

            plt.plot(xs, dataline, label=label_name)

        plt.legend()


err_ext = lambda xs, state, err: err

xs_ext = lambda index: lambda xs, state, err: xs[index]

state_ext = lambda key: lambda xs, state, err: state[key]


exponents = [1,2,3,4]

s = GraphSimulation('exp', *make_exponent(exponents))

s.add_lenses([f'{e}^x', xs_ext(i)] for i, e in enumerate(exponents))

s.simulate(steps=100, total_time=2)

s.plot('1^x', '2^x', '3^x', '4^x', start=1)

from numbers import Number

from collections import defaultdict, OrderedDict, Counter


class UniqSet(set):

    def add(self, key):

        if key in self:

            raise ValueError(f'Key "{key}" already exists')

        super().add(key)

        

class NamedWithPrefix:

    def __init__(self, prefix, counts, name=''):

        assert isinstance(counts[prefix], int), "Number of previous instances is not an integer"

        self.name = f'{prefix}{counts[prefix]}{name}'

        counts[prefix] += 1


class Element(NamedWithPrefix):

    def get_prefix(self): return 'Undefined'

    def __init__(self, counts, value, name='', is_supernode=False):

        super().__init__(self.get_prefix(), counts, name)

        self.value = value

        self.is_supernode = is_supernode

        self.lenses = [['Voltage', lambda v, state: v]]

    def get_initial_state(self): return None

    def get_current(self, v, state):

        raise NotImplementedError()

    def get_mismatch(self, v, state):

        raise NotImplementedError()

    def commit(self, v, i, state, dt):

        raise NotImplementedError()


class Resistor(Element):

    def get_prefix(self): return 'R'

    def get_current(self, v, state):

        return v / self.value


class Source(Element):

    def get_prefix(self): return 'V'

    def __init__(self, counts, voltage, name=''):

        if isinstance(voltage, Number):

            constant_voltage = voltage

            voltage = lambda t: constant_voltage

        super().__init__(counts, voltage, name, is_supernode=True)

    def get_initial_state(self): return 0 # tracks time

    def get_mismatch(self, v, state):

        expected_voltage = self.value(state)

        return expected_voltage - v

    def commit(self, v, i, state, dt): return state+dt


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


        

def group_by(items, key=lambda x:x):

    if isinstance(items, dict):

        items = items.items()

    result = defaultdict(list)

    for k,v in items:

        result[key(v)].append(k)

    return result

        


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

            


from math import sin

c = Circuit('test1')

c.add_node('a')

c.add_node('b')

c.attach(Resistor, 'a', 'b', 50)

c.attach(Source, 'a', 'b', lambda t: sin(t))

print(c.lenses)

[['Potential at a', <function Circuit.add_node.<locals>.<lambda> at 0x7fefc310c9d0>], ['Potential at b', <function Circuit.add_node.<locals>.<lambda> at 0x7fefc2fe9790>], ['Voltage at R0', <function Circuit.attach.<locals>.<lambda> at 0x7fefc2fe91f0>], ['Voltage at V0', <function Circuit.attach.<locals>.<lambda> at 0x7fefc2fe9820>]]

def sum_squares(arr): return sum(x*x for x in arr)


from scipy.optimize import minimize

def solve_minimize(xs, err_fun):

    res = minimize(err_fun, xs)

    if not res.success:

        raise ValueError(res.message)

    return res.x, res.fun


class CircuitSimulation(GraphSimulation):

    def compile_fun(self, get_errors, get_state):

        get_error = lambda state: lambda voltages: self.loss(get_errors(voltages, state))

        def get_next_state(voltages, state, dt):

            new_voltages, error = self.solve(voltages, get_error(state))

            new_state = get_state(new_voltages, state, dt)

            return new_voltages, new_state, error

        return get_next_state

    

    def __init__(self, circuit, loss=sum_squares, solve=solve_minimize, default_voltages=None):

        self.loss = loss

        self.solve = solve

        name = f'Simulation of circuit "{circuit.name}"'

        initial_state, state_fun, sim_errors = circuit.compile_funcs()

        fun = self.compile_fun(sim_errors, state_fun)

        voltages = default_voltages or [0] * len(circuit.nodes)

        super().__init__(name, fun, voltages, initial_state)

        self.add_lenses(circuit.lenses)


c = Circuit('test1')

c.add_node('a')

c.add_node('b')

c.attach(Resistor, 'a', 'b', 50)

c.attach(Source, 'a', 'b', lambda t: sin(t))

print(c.lenses)

cs = CircuitSimulation(c)

cs.simulate(steps=100, total_time=2)

cs.plot('Voltage at R0', 'Voltage at V0')

[['Potential at a', <function Circuit.add_node.<locals>.<lambda> at 0x7fefc310c9d0>], ['Potential at b', <function Circuit.add_node.<locals>.<lambda> at 0x7fefc2fe9790>], ['Voltage at R0', <function Circuit.attach.<locals>.<lambda> at 0x7fefc2fe94c0>], ['Voltage at V0', <function Circuit.attach.<locals>.<lambda> at 0x7fefc2fe98b0>]]

class Inductor(Element):

    def get_prefix(self): return 'L'

    def get_initial_state(self): return [0, 0.000000000001]

    def get_current(self, v, state):

        I, dt = state

        dI = dt * v / self.value

        return I + dI

    def commit(self, v, i, state, dt): return [i, dt]


c = Circuit('test2')

c.add_node('1')

c.add_node('2')

c.add_node('3')

c.attach(Resistor, '1', '2', 20)

c.attach(Inductor, '2', '3', 10)

c.attach(Source, '1', '3', 5)

cs = CircuitSimulation(c)

cs.simulate(steps=100, total_time=10)

cs.plot('Potential at 1', 'Potential at 2')

class Diode(Element):

    def __init__(self, counts, value=1, name=''):

        super().__init__(counts, value, name)

    def get_prefix(self): return 'D'

    def get_current(self, v, state):

        if v > 0:

            return v / self.value

        else:

            return 0


class Capacitor(Element):

    def __init__(self, counts, value, name=''):

        super().__init__(counts, value, name, is_supernode=True)

    def get_prefix(self): return 'C'

    def get_initial_state(self): return 0

    def commit(self, v, i, state, dt): return state + i * dt

    def get_mismatch(self, v, state): return self.value * state - v


    

c = Circuit('Cap test')

c.add_node('1')

c.add_node('2')

c.add_node('3')

c.attach(Resistor, '1', '2', 20)

c.attach(Capacitor, '2', '3', 10)

c.attach(Source, '1', '3', 5)

cs = CircuitSimulation(c)

cs.simulate(steps=100, total_time=10)

cs.plot('Potential at 1', 'Potential at 2', 'Voltage at C0')

def spike(height=5, duration=1, delay=0): return lambda t:max(0, height * (1 - abs(t - delay) / duration))


c = Circuit('Oscillation test')

c.add_node('1')

c.add_node('2')

c.add_node('3')

c.attach(Inductor, '1', '2', 20)

c.attach(Capacitor, '2', '3', 10)

c.attach(Source, '1', '3', spike(duration=0.5, delay=2))

cs = CircuitSimulation(c)

cs.simulate(steps=100, total_time=10)


cs.plot('Voltage at V0', 'Voltage at L0', 'Voltage at C0')



