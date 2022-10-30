from state import State
import numpy as np
import sympy as sm


def gaussian_hill(domain, h_mean, k_x=50.0, k_y=50.0):
    state = State.zeros(domain.nx, domain.ny)
    dx = (domain.xx - 0.8 * np.mean(domain.x)) / domain.x
    dy = (domain.yy - 0.8 * np.mean(domain.y)) / domain.y
    state.h = h_mean + 0.1 * h_mean * np.exp(-k_x * dx ** 2 - k_y * dy ** 2)
    return state

