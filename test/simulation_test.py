from simulation import Simulation
from math import log

from hypothesis import given, strategies as st


def make_exponent(bases):
    state = {'expo_coeffs': [log(x) for x in bases]}
    def exponent(xs, state, dt):
        xs = [x * (1 + dt * ln) for x, ln in zip(xs, state['expo_coeffs'])]
        error = dt * dt
        return xs, state, error
    return exponent, bases, state

# we don't expect precision here because of nonlinearity
# TODO: use this example to properly gauge precision in nonlinear processes
@given(st.lists(st.integers(1, 10), min_size=1))
def test_exp_growth(exponents):
    time = 2
    #print(exponents, time)
    s = Simulation('exp', *make_exponent(exponents))
    s.simulate(steps=100, total_time=time)
    expected = [x**(time+1) for x in exponents]
    actual = s.xs
    for a,e in zip(actual, expected):
        assert abs(e-a) / e < 0.1

def make_linear(ks):
    state = {'ks':ks}
    def linear(xs, state, dt):
        xs = [x + k * dt for x,k in zip(xs, state['ks'])]
        error = dt * dt
        return xs, state, error
    return linear, [0]*len(ks), state

@given(st.lists(st.integers(1,10), min_size=1))
def test_linear_growth(ks):
    time = 2
    s = Simulation('lin', *make_linear(ks))
    s.simulate(steps=100, total_time=time)
    expected = [k * time for k in ks]
    actual = s.xs
    for a,e in zip(actual, expected):
        assert abs(e-a) / e < 0.0001


