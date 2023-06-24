

import numpy as np

from scipy import optimize

from dataclasses import dataclass, field

from collections.abc import Callable

from copy import deepcopy

from math import log


import matplotlib.pyplot as plt

from numpy import linspace

from collections import Counter





from numbers import Number

from collections import defaultdict, OrderedDict, Counter


from math import sin

c = Circuit('test1')

c.add_node('a')

c.add_node('b')

c.attach(Resistor, 'a', 'b', 50)

c.attach(Source, 'a', 'b', lambda t: sin(t))

print(c.lenses)

[['Potential at a', <function Circuit.add_node.<locals>.<lambda> at 0x7fefc310c9d0>], ['Potential at b', <function Circuit.add_node.<locals>.<lambda> at 0x7fefc2fe9790>], ['Voltage at R0', <function Circuit.attach.<locals>.<lambda> at 0x7fefc2fe91f0>], ['Voltage at V0', <function Circuit.attach.<locals>.<lambda> at 0x7fefc2fe9820>]]

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



