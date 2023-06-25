from util import auc
import numpy as np

from circuit_simulation import CircuitSimulation
from circuit import Circuit
from elements import Inductor, Capacitor

def run_LC(L, C, initial_charge=10):
    c = Circuit('Oscillation test')
    c.add_node('1')
    c.add_node('2')
    c.attach(Inductor, '1', '2', L)
    c.attach(Capacitor, '2', '1', C, initial_charge=initial_charge)
    cs = CircuitSimulation(c)
    cs.simulate(steps=100, total_time=100)

    cap_voltage = cs.extract_dataline('Voltage at C0')
    times = np.linspace(0, cs.total_time, cs.steps)

    omega = 1 / np.sqrt(L * C)
    expected = (initial_charge / C) * np.cos(times * omega)
    errors = cap_voltage - expected

    rel_error = auc(times, abs(errors)) / auc(times, abs(cap_voltage))
    assert rel_error < 0.1

    #p = cs.plot('Voltage at L0', 'Voltage at C0')
    #p.plot(times, expected, label='expected')

def test_lc():
    run_LC(20,10)
