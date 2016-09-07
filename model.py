"""In Jovanic & Schneider-Mizell et. al. (2016), behaviour selection was modelled. This is that model."""

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import pickle
from itertools import product
import multiprocessing as mp
from functools import wraps

# SOLVER = 'SimpleRungeKutta'  # 'SimpleEuler', 'Odeint', 'ode', 'SimpleRungeKutta'
# SOLVER = 'SimpleEuler'
SOLVER = 'Odeint'

NUM_STEPS = 1000

HUNCH = 1
BEND = 2

R2_MAX = 12.500  # r[:, 1].max() when w_iLNb = w_iLNa = 1
R3_MAX = 7.998  # r[:, 2].max() when w_iLNb = w_iLNa = 1

element_names = [
    "Mechano-ch",
    "Basin-1",
    "Basin-2",
    "iLNb",
    "iLNa",
    "H"  # fbLNs
]

# sensory input into lateral inhibitory neurons
w_iLNb_DEFAULT = 1
w_iLNa_DEFAULT = 1

tau = [0.1, 35, 35, 35, 35, 35]  # time step
V_0 = [0, 10, 10, 10, 10, 10]  # activation threshold
s = [0, 0, 0, 0, 0, 0]  # stimulus input: s[0] = 2 during stimulus
r_max = [20, 20, 20, 20, 20, 20]  # maximum firing rate
r = [0, 0, 0, 0, 0, 0]  # current rate


def constrain(constraints):
    """
    Constrains the decorated ODE of the form f(t, y, *args, **kwargs) such that the dependent variable y does not
    significantly exceed the given arguments.

    :param constraints: tuple of (lower, upper) constraint for all dependent variables
    """
    if all(constraint is not None for constraint in constraints):
        assert constraints[0] < constraints[1]

    def wrap(f):

        @wraps(f)
        def wrapper(t, y, *args, **kwargs):
            lower, upper = constraints
            if lower is None:
                lower = -np.inf
            if upper is None:
                upper = np.inf

            too_low = y <= lower
            too_high = y >= upper

            y = np.maximum(y, np.ones(np.shape(y)) * lower)
            y = np.minimum(y, np.ones(np.shape(y)) * upper)

            result = f(t, y, *args, **kwargs)

            result[too_low] = np.maximum(result[too_low], np.ones(too_low.sum()) * lower)
            result[too_high] = np.minimum(result[too_high], np.ones(too_high.sum()) * upper)

            return result

        return wrapper

    return wrap


@constrain((0, None))
def f(t, r, *f_args):
    """
    Return dr/dt

    :param t: int, the current time
    :param r: array, the current activity levels
    :param f_args: varargs, any other arguments which need to be included
    :return: vector, dr/dt at the current time
    """
    dr_dt = np.empty(np.shape(r))
    dr_dt[:] = np.nan

    s = [0, 0, 0, 0, 0, 0]  # stimulus input: s[0] = 2 during stimulus

    if len(f_args) != 2:
        w_iLNb = w_iLNb_DEFAULT
        w_iLNa = w_iLNa_DEFAULT
    else:
        w_iLNb, w_iLNa = f_args

    # Connectivity matrices
    A_ex = np.array([
        [0, 0, 0, 0, 0, 0],
        [1.5, 0, 0, 0, 0, 0],
        [0.75, 0, 0, 0, 0, 0],
        [w_iLNb, 0, 0, 0, 0, 0],
        [w_iLNa, 0, 0, 0, 0, 0],
        [0.1, 0.3, 0.3, 0, 0, 0]
    ])

    A_in = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 5, 0],
        [0, 0, 0, 0, 4, 2],
        [0, 0, 0, 4, 0, 2],
        [0, 0, 0, 2, 2, 0]
    ])

    if t > 50 and not s[0]:
        s[0] = 2

    for i, r_i in enumerate(r):
        dr_dt[i] = (
                       -V_0[i] - r_i + s[i]
                       + (r_max[i] - r_i) * np.sum(A_ex[i, :] * r)
                       - np.sum(A_in[i, :] * r)
                   ) / tau[i]

    return dr_dt


def f_(r, t, *f_args):
    """
    Invert r and t's argument positions for use with scipy.integrate.odeint
    """
    return f(t, r, *f_args)


class SimpleEulerSolver:
    """
    A simple implementation of a first-order Euler solver
    """
    def __init__(self, f, init, t, *f_args):
        """

        :param f: function, dy/dt of form f(t, y, *f_args)
        :param init: sequence, initial y values
        :param t: vector, time values to be evaluated
        :param f_args:
        """
        self.f = f
        self.init = init
        self.t = t
        self.f_args = f_args

        self.y = np.empty((len(t), len(r)))
        self.y[:] = np.nan
        self.y[0, :] = self.init

    def evaluate(self):
        for i, this_t in enumerate(self.t[1:], start=1):
            tstep = this_t - self.t[i - 1]
            dr_dt = self.f(this_t, self.y[i - 1], *self.f_args)
            this_y = self.y[i - 1, :] + dr_dt * tstep

            self.y[i, :] = this_y

        return self.y


class OdeintSolver:
    """
    A wrapper for scipy.integrate.odeint
    """
    def __init__(self, f, init, t, *f_args, **kwargs):
        """

        :param f: function, dy/dt of form f(t, y, *f_args)
        :param init: sequence, initial y values
        :param t: vector, time values to be evaluated
        :param f_args:
        """
        self.f = f
        self.init = init
        self.t = t
        self.f_args = f_args

        self.kwargs = kwargs

        self.y = np.empty((len(t), len(r)))
        self.y[:] = np.nan

    def evaluate(self):
        soln = integrate.odeint(self.f, self.init, self.t, args=self.f_args, **self.kwargs)
        self.y = soln
        return self.y


class SimpleRungeKuttaSolver:
    """
    A simple implementation of a 4th-order Runge-Kutta solver
    """
    def __init__(self, f, init, t, *f_args):
        """

        :param f: function, dy/dt of form f(t, y, *f_args)
        :param init: sequence, initial y values
        :param t: vector, time values to be evaluated
        :param f_args:
        """
        self.f = f
        self.init = init
        self.t = t
        self.f_args = f_args

        self.y = np.empty((len(t), len(r)))
        self.y[:] = np.nan
        self.y[0, :] = self.init

    def evaluate(self):
        for i, this_t in enumerate(self.t[1:], start=1):
            tstep = this_t - self.t[i - 1]

            k1 = f(self.t[i - 1], self.y[i - 1], *self.f_args)
            k2 = f(self.t[i - 1] + tstep / 2, self.y[i - 1] + (tstep / 2) * k1, *self.f_args)
            k3 = f(self.t[i - 1] + tstep / 2, self.y[i - 1] + (tstep / 2) * k2, *self.f_args)
            k4 = f(self.t[i - 1] + tstep, self.y[i - 1] + tstep * k3, *self.f_args)

            this_y = self.y[i - 1, :] + (tstep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            self.y[i, :] = this_y

        return self.y


def get_response(r):
    """
    Find out what behaviour the basin activity maps to

    :param r: vector of neuron activities
    :return: 0 for no response
             1 for hunch
             2 for bend
    """
    r2_max = R2_MAX
    r3_max = R3_MAX

    r2_hat = r[1] / r2_max
    r3_hat = r[2] / r3_max
    if r2_hat <= 0.5 and r3_hat <= 0.5:
        return 0  # no response
    if r3_hat < 0.3 and (r2_hat > 0.5 or r3_hat > 0.5):  # 3rd term is not possible given 1st?
        return HUNCH  # hunch
    if r3_hat >= 0.3 and (r2_hat > 0.5 or r3_hat > 0.5):
        return BEND  # bend


def single_solution(w_iLNb=w_iLNb_DEFAULT, w_iLNa=w_iLNa_DEFAULT):
    """
    Given values of the inhibitory Local Interneuron's weights (i.e. response to sensory input), calculate the evolution
    of neuron activities in the circuit.

    :param w_iLNb: float
    :param w_iLNa: float
    :return: tuple(1D array, 2D array), the time vector and array of activities at each time step
    """
    t_0 = 0
    t_final = 200
    num_steps = NUM_STEPS
    r_init = [0, 0, 0, 0, 0, 0]
    t = np.linspace(t_0, t_final, num_steps)

    if SOLVER == 'SimpleEuler':
        solver = SimpleEulerSolver(f, r_init, t, w_iLNb, w_iLNa)
    elif SOLVER == 'SimpleRungeKutta':
        solver = SimpleRungeKuttaSolver(f, r_init, t, w_iLNb, w_iLNa)
    elif SOLVER == 'Odeint':
        solver = OdeintSolver(f_, r_init, t, w_iLNb, w_iLNa)
    else:
        solver = 'SimpleEuler'

    soln = solver.evaluate()
    soln[soln < 0] = 0

    return t, soln


def plot_single_solution(t, r, show=True):
    """
    Plot neuron activities and behavioural response

    :param t: 1D array, time values
    :param r: 2D array, neuron activity at each time value
    :param show: bool, whether to show plot
    :return: fig
    """
    responses = np.array([get_response(r_row) for r_row in r])
    hunches = responses == HUNCH
    bends = responses == BEND

    fig, ax_arr = plt.subplots(2, sharex=True)
    ax_arr[0].plot(t, r)
    ax_arr[0].legend(element_names)
    ax_arr[0].set_ylabel('activity')

    ax_arr[1].fill_between(t, 0, hunches, color='r')
    ax_arr[1].fill_between(t, 0, bends, color='b')
    ax_arr[1].legend(['hunch', 'bend'])

    plt.xlabel('time')

    if show:
        plt.show()

    return fig


def get_trajectory(r):
    """
    From all neuron activities, get coincident activity of Basins 1 and 2

    :param r: 2D array, neuron activities
    :return: 2D array, Basin-1 activity vs Basin-2 activity
    """
    return np.array([r[:, 1], r[:, 2]]).T


def single_soln_tup(args):
    """
    Wrap single_solution to accept a tuple of arguments for parallelisation

    :param args: tuple of arguments to single_solution
    :return:
    """
    return single_solution(*args)


def get_trajectories():
    """
    For a range of inhibitory Local interNeuron weights, determine the trajectory of Basin-1 and Basin-2 activities.

    :return: list of trajectories
    """
    ranges = (0.5, 1.5)
    n = (15, 15)

    w_iLNs = list(product(np.linspace(ranges[0], ranges[1], n[0]), np.linspace(ranges[0], ranges[1], n[1])))

    with mp.Pool(mp.cpu_count()) as p:
        solns = p.map(single_soln_tup, w_iLNs, chunksize=int(len(w_iLNs) / mp.cpu_count()))

    return [get_trajectory(r) for _, r in solns]


def plot_trajectories(trajectories, show=True):
    """
    Plot the trajectories of Basin activity in state space

    :param trajectories: list of trajectories
    :param show: whether to show plot
    :return: fig
    """
    plt.figure()
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], color='k', alpha=0.1)

    plt.xlabel('Basin-1 activity')
    plt.ylabel('Basin-2 activity')
    if show:
        plt.show()


def save_trajs(trajs, fname='traj.gpickle'):
    """
    Pickle things
    """
    with open(fname, 'wb') as f:
        pickle.dump(trajs, f)


if __name__ == '__main__':
    t, r = single_solution()
    plot_single_solution(t, r)

    plot_trajectories(get_trajectories())
