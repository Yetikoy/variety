from numbers import Number
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



class Inductor(Element):
    def get_prefix(self): return 'L'
    def get_initial_state(self): return [0, 0.000000000001]
    def get_current(self, v, state):
        I, dt = state
        dI = dt * v / self.value
        return I + dI
    def commit(self, v, i, state, dt): return [i, dt]

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
