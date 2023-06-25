from simulation import Simulation
from collections import OrderedDict
import matplotlib.pyplot as plt
from numpy import linspace

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
        return plt

err_ext = lambda xs, state, err: err
xs_ext = lambda index: lambda xs, state, err: xs[index]
state_ext = lambda key: lambda xs, state, err: state[key]

# Example:
#s = GraphSimulation('exp', *make_exponent(exponents))
#s.add_lenses([f'{e}^x', xs_ext(i)] for i, e in enumerate(exponents))
#s.simulate(steps=100, total_time=2)
#s.plot('1^x', '2^x', '3^x', '4^x', start=1)



