from graph_simulation import GraphSimulation
from scipy.optimize import minimize
def sum_squares(arr): return sum(x*x for x in arr)

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

