from graph_simulation import GraphSimulation
from scipy.optimize import minimize
from functools import reduce
from numbers import Number

import numpy as np
def sum_squares(arr): return sum(x*x for x in arr)

def minmany(xs, err):
    alpha = 0.01
    for meth, opt in [
            ['Nelder-Mead', {'xatol':alpha, 'fatol':alpha}],
            ['Powell', None],
            ['CG', None],
            ['BFGS', None],
            #['Newton-CG', None],
            ['L-BFGS-B', { 'gtol':alpha, 'ftol':alpha}],
            ['TNC', None],
            ['COBYLA', None],
            ['SLSQP', None],
            ['trust-constr', None],
            #['dogleg', None],
            #['trust-ncg', None],
            #['trust-exact', None],
            #['trust-krylov', None],
        ]:
        res = minimize(err, xs, method=meth, options=opt or {})
        if res.success or err(res.x) > alpha:
            return res.x, res.fun
    print(xs)
    raise ValueError(res.message)



def solve_minimize(xs, err_fun):
    res = minimize(err_fun, xs)
    if not res.success:
        if err_fun(res.x) > 0.01:
            print(err_fun(res.x), res.x)
            raise ValueError(res.message)
    return res.x, res.fun
#solve_minimize = minmany


class CircuitSimulation(GraphSimulation):
    def compile_fun(self, get_errors, get_state):
        get_error = lambda state, dt: lambda voltages: self.loss(get_errors(voltages, state))
        def get_next_state(voltages, state, dt):
            #print(voltages, state, dt)
            new_voltages, error = self.solve(voltages, get_error(state, dt))
            if any(v > 1000 for v in new_voltages):
                print("Outbreak!", voltages)
                print(new_voltages)

            v0 = list(new_voltages)[0]
            new_voltages = [v - v0 for v in list(new_voltages)]
            new_state = get_state(new_voltages, state, dt)
            return new_voltages, new_state, error

        def combine(Gs, weights):
            def weight_merge(a,b):
                #print('w merge', a, b)
                e1, w1 = a
                e2, w2 = b
                def rec_merge(e1, e2):
                    #print('rec merge', e1, e2)
                    if isinstance(e1, Number):
                        return (e1 * w1 + e2 * w2)
                    if isinstance(e1, list) or isinstance(e1, tuple) or isinstance(e1, np.ndarray):
                        return [rec_merge(el1,el2) for el1, el2 in zip(e1, e2)]
                    if isinstance(e1, dict):
                        return {key:rec_merge(e1[key],e2[key]) for key in e1}
                    print(e1)
                    raise ValueError('unknown type in merging states')
                return rec_merge(e1,e2),1
            return reduce(weight_merge, zip(Gs, weights))[0]

            return Gs[0]
#TODO: these all don't work because elements remember previous dt. weave dt through get_current and try again
        def get_next_state_rk2_diff(voltages, state, dt):
            if self.first_time:
                self.first_time = False
                return get_next_state(voltages, state, dt)
            error = 0
            def get_G(voltages, state, dt):
                v, s, e = get_next_state(voltages, state, dt)
                nonlocal error
                if e > error:
                    error = e
                return v,s
            def f(s):
                alpha = 0.0001
                s1 = get_G(*s, alpha * dt)
                return combine([s1, s],[-1/alpha, 1/alpha])
            y0 = [voltages, state]
            get_G(*y0, dt/10000)
            print('this works')
            f0 = f(y0)
            y1 = combine([y0, f0],[1, dt/2])
            f1 = f(y1)
            y2 = combine([y0, f1],[1, dt])
            new_voltages, new_state = y2
            return new_voltages, new_state, error

        def get_next_state_rk2(voltages, state, dt):
            error = 0
            def get_G(voltages, state, dt):
                v, s, e = get_next_state(voltages, state, dt)
                nonlocal error
                if e > error:
                    error = e
                return v,s
            G0 = [voltages, state]
            print('G0', G0)
            G1 = get_G(*G0, dt/10000)
            print('G1', G1)
            G2 = get_G(*combine([G1, G0], [0.5, 0.5]), dt/2)
            print('G2', G2)
            new_voltages, new_state = combine([G2, G1, G0],[1, -0.5, 0.5])
            return new_voltages, new_state, error


        def get_next_state_rk4(voltages, state, dt):
            error = 0
            def get_G(voltages, state, dt):
                v, s, e = get_next_state(voltages, state, dt)
                nonlocal error
                if e > error:
                    error = e
                return v,s
            G0 = [voltages, state]
            print('G0', G0)
            G1 = get_G(*G0, dt/10000)
            print('G1', G1)
            G2 = get_G(*combine([G1, G0], [0.5, 0.5]), dt/2)
            print('G2', G2)
            G3 = get_G(*combine([G2, G1, G0], [0.5, -0.25, 0.75]), dt/2)
            print('G3', G3)
            G4 = get_G(*combine([G3, G2, G1, G0], [1, -0.5, 0.25, 0.25]), dt)
            print('G4', G4)
            hks = combine([G4, G3, G2, G1, G0],[1,1,1.5,0.25,-3.75])
            new_voltages, new_state = combine([G0, hks], [1, 1/6])
            return new_voltages, new_state, error



        #return get_next_state_rk2_diff
        #return get_next_state_rk2
        #return get_next_state_rk4
        return get_next_state

    def __init__(self, circuit, loss=sum_squares, solve=solve_minimize, default_voltages=None):
        self.first_time = True
        self.loss = loss
        self.solve = solve
        name = f'Simulation of circuit "{circuit.name}"'
        initial_state, state_fun, sim_errors = circuit.compile_funcs()
        fun = self.compile_fun(sim_errors, state_fun)
        voltages = default_voltages or [0] * len(circuit.nodes)
        super().__init__(name, fun, voltages, initial_state)
        self.add_lenses(circuit.lenses)

