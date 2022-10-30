import operators as op
from state import State

class SweAdvectiveFormOperator:

    def __init__(self, g, pcori, diff_method):
        self.g = g
        self.pcori = pcori
        self.diff_method = diff_method

    def calc_rhs(self, state, domain):
        gx, gy = op.calc_grad(state.h, domain, self.diff_method)
        div = op.calc_div(state.h * state.u, state.h * state.v, domain, self.diff_method)

        du_dx, du_dy = op.calc_grad(state.u, domain, self.diff_method)
        dv_dx, dv_dy = op.calc_grad(state.v, domain, self.diff_method)

        return State(- self.g * gx + self.pcori * state.v - state.u * du_dx - state.v * du_dy,
                     - self.g * gy - self.pcori * state.u - state.u * dv_dx - state.v * dv_dy,
                     - div)
