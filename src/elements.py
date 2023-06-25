from numbers import Number
class NamedWithPrefix:
    def __init__(self, prefix, counts, name=''):
        assert isinstance(counts[prefix], int), "Number of previous instances is not an integer"
        self.name = f'{prefix}{counts[prefix]}{name}'
        counts[prefix] += 1


class Element(NamedWithPrefix):
    def get_prefix(self): return 'Undefined'
    def __init__(self, counts, value, name='', is_supernode=False, lenses=tuple()):
        super().__init__(self.get_prefix(), counts, name)
        self.value = value
        self.is_supernode = is_supernode
        self.lenses = [['Voltage', lambda v, state: v]] + list(lenses)

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
    def __init__(self, counts, value, name='', initial_current=0):
        super().__init__(counts, value, name)
        self.lenses.append(['Current', lambda v, state: state[0]])
        self.initial_current = initial_current
    def get_prefix(self): return 'L'
    def get_initial_state(self): return [self.initial_current, 0]
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
    def __init__(self, counts, value, name='', initial_charge=0):
        super().__init__(counts, value, name, is_supernode=True)
        self.lenses.append(['Charge', lambda v, state: state])
        self.get_initial_state = lambda: initial_charge
    def get_prefix(self): return 'C'
    def commit(self, v, i, state, dt):
        #print('C0:', v, i, state)
        return state + i * dt
    def get_mismatch(self, v, state): return state / self.value - v    
    
class Jumper(Element):
    def __init__(self, counts, value, name=''):
        super().__init__(counts, value, name, is_supernode=True)
    def get_prefix(self): return 'J'
    def get_mismatch(self, v, state): return v
