import numpy as np
from dataclasses import dataclass, field
from collections.abc import Callable
from copy import deepcopy

# TODO: move lens business here and according tests too

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
        #TODO: rewrite using match
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
        #TODO: maybe actually keep track in total_time?
        steps, dt, total_time = self._get_sim_limit(steps, dt, total_time)
        self.pre_sim_routine(steps, dt, total_time)
        for i in range(steps):
            self.step(tol, dt)
        self.post_sim_routine(steps, dt, total_time)
        self.total_time = total_time


