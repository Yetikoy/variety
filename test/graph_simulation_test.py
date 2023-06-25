from graph_simulation import GraphSimulation
from util import auc

import numpy as np

def make_lc_simple(L,C,charge=10):
    state={'L':L,'C':C}
    base = [charge,0]
    def calc_next(xs, state, dt):
        charge, current = xs
        voltage = charge / state['C']
        new_charge = charge + current * dt
        new_current = current - dt * voltage / state['L']
        return [new_charge, new_current], state, dt*dt
    return calc_next, base, state

def test_simple_lc():
    L = 20
    C = 10
    time = 50
    s = GraphSimulation('LC_simple', *make_lc_simple(L,C))
    s.add_lens('charge', lambda xs,state,err:xs[0])
    s.simulate(steps=100, total_time=time)
    
    initial_charge = 10
    cap_voltage = s.extract_dataline('charge')
    times = np.linspace(0, s.total_time, s.steps)
        
    omega = 1 / np.sqrt(L * C)
    expected = (initial_charge) * np.cos(times * omega)
    errors = cap_voltage - expected
    
    rel_error = auc(times, abs(errors)) / auc(times, abs(cap_voltage))
    assert rel_error < 0.1
    
    #p = s.plot('charge')
    #p.plot(times, expected, label='expected')
