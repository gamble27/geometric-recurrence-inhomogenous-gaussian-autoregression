from typing import Iterable, Callable

import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt


# helpers

def save_plot(
        name,
        directory = "plots"
):
    """
    save current plot
    :param directory:
    :param name:
    :return:
    """
    plt.savefig(
        f'{directory}/{name}.jpg',
        dpi=800
    )

# steps 1 and 2

def simulate_trajectory(X_0, N, gen_alpha, gen_sigma):
    """
    simulate Gaussian autoregression process
    up to step N
    :param X_0: starting point
    :param N: max step
    :param gen_alpha: generator function of alphas
    :param gen_sigma: generator function of sigmas
    :return: generated trajectory
    """
    X = np.zeros([N])
    X[0] = X_0

    for i in range(1,N):
        X[i] = X[i-1]*gen_alpha(i) + rnd.randn(1)*gen_sigma(i)

    return X


def plot_trajectory(X, caption=""):
    """
    plots given trajectory
    :param X:
    :param caption:
    :return:
    """
    n = np.array(range(X.shape[0]))
    fig, ax = plt.subplots()
    ax.plot(n, X)
    ax.set_xlabel("n")
    ax.set_ylabel("Xn")
    ax.set_title(caption)

# steps 3, 4

def get_return_time(X:np.array, c: float, return_n: int = 1):
    """
    get n-th return time for given set & trajectory
    :param X: trajectory of interest
    :param c: set [-c;c] to return to
    :param return_n: number of return (0 or 1)
    :return: return time
    """
    assert c > 0, "wrong arg"

    return np.array((-c <= X[return_n:]) * (X[return_n:] <= c)).argmax()

def simulate_return_times_sample(
        n_samples,
        gen_alpha,
        gen_sigma,
        x_0,
        c,
        max_steps,
        return_n: int = 1
):
    """
    simulate n_samples of trajectories and get their return times to set [-c;c]
    :param n_samples: number of trajectories to generate
    :param gen_alpha: alpha generator
    :param gen_sigma: sigma generator
    :param x_0: starting point
    :param c: set of interest
    :param max_steps: max steps for each trajectory
    :param return_n: number of return (0 or 1)
    :return: list of return times
    """
    t = []
    for i in range(n_samples):
        X = simulate_trajectory(
            N = max_steps,
            X_0 = x_0,

            gen_alpha=gen_alpha,
            gen_sigma=gen_sigma,
        )
        t.append(get_return_time(X, c, return_n=return_n))

    return t

def ind(x,c):
    """
    indicator function
    :param x: function argument
    :param c: control of set C=[-c;c]
    :return: 1 if x in C else 0
    """
    return int(-c <= x <= c)

def plot_return_times(
        t: Iterable,
        n_ticks = 4
):
    """
    plot return times distribution
    :param n_ticks: number of ticks/columns-1 for plot
    :param t: return times
    :return:
    """
    fig, ax = plt.subplots()
    ax.hist(t, bins=range(n_ticks))
    ax.set_xticks(range(n_ticks))
    ax.set_title("Distribution of the first return time to C")
    ax.set_xlabel("t")
    ax.set_ylabel("N observations")

# step 5

def simulate_return_times(
        gen_alpha: Callable,
        gen_sigma: Callable,
        x_0: Iterable[float],
        n_samples: int = 10**4,
        c: float = 1,
        max_steps: int = 50,
        return_n: int = 1
):
    """
    generate exponential moment prerequisites
    :param gen_alpha:
    :param gen_sigma:
    :param x_0:
    :param n_samples:
    :param c:
    :param max_steps:
    :param return_n:
    :return:
    """
    t = []
    for x_i in x_0:
        t.append(simulate_return_times_sample(
            n_samples,
            gen_alpha,
            gen_sigma,
            x_0=x_i,
            c=c,
            max_steps=max_steps,
            return_n = return_n
        ))

    return t


def calculate_em(
        x: Iterable[float],
        t: [[int]],
        eps: float,
        sigma: float,
        c: float = 1
):
    """
    calculate expectation of exp moment and its bound
    :param x:
    :param t:
    :param eps:
    :param sigma:
    :param c:
    :return:
    """
    K = 2*sigma/np.sqrt(2*np.pi) + 1 - (1-eps)/2/2
    phi = 2 / (eps+1)

    exp_moment = []
    upper_bound = []
    for i, x_i in enumerate(list(x)):
        exp_moment.append(np.average(
            list(map(
                lambda tau: phi**tau,
                t[i]
            ))
        ))
        upper_bound.append(1 + np.abs(x_i) + K*ind(x_i,c)/2)

    return exp_moment, upper_bound


def plot_em(
        x,
        exp_moment,
        upper_bound
):
    """
    plot expectation of exp moment and its bound
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(x, exp_moment, label="exp moment")
    ax.plot(x, upper_bound, label="upper bound")
    ax.set_title("Demo of Corollary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.show()

# encapsulate steps above to a single function

def perform_experiment(
        name,
        gen_alpha,
        gen_sigma,
        starting_point_single,
        starting_point_multiple,
        c,
        eps,
        sigma,
        n_steps,
        n_samples,
        n_ticks=7
):
    """
    1. generate & show 1 full trajectory
    2. show its path to C

    3. generate n_samples trajectories
    4. show exponential moment distribution

    5. show theorem statement:
        5.1. for each starting point generate n_samples trajectories
        5.2. calculate their E[exp moment] and bound values
        5.3. plot results
    :return:
    """
    # 1
    single_trajectory = simulate_trajectory(
        starting_point_single,
        n_steps,
        gen_alpha,
        gen_sigma
    )
    single_return_time = get_return_time(
        single_trajectory,
        c,
        return_n=1
    )

    # 2
    plot_trajectory(
        single_trajectory,
        caption=f"Markov chain trajectory, {n_steps} steps"
    )
    save_plot(f"{name}_single_trajectory_full")

    plot_trajectory(
        single_trajectory[0:single_return_time+1],
        caption=f"Markov chain trajectory up to the hitting of [-{c};{c}]"
    )
    save_plot(f"{name}_single_trajectory_up_to_hitting")

    # 3
    return_times_sample = simulate_return_times_sample(
        n_samples,
        gen_alpha,
        gen_sigma,
        starting_point_single,
        c,
        n_steps,
        return_n=1
    )

    # 4
    plot_return_times(
        return_times_sample,
        n_ticks
    )
    save_plot(f"{name}_return_time_distribution")

    # 5
    return_times = simulate_return_times(
        gen_alpha,
        gen_sigma,
        starting_point_multiple,
        n_samples,
        c,
        n_steps,
        return_n=1
    )
    exp_moment, upper_bound = calculate_em(
        starting_point_multiple,
        return_times,
        eps,
        sigma,
        c
    )
    plot_em(
        starting_point_multiple,
        exp_moment,
        upper_bound
    )
    save_plot(f"{name}_theorem_demo")
