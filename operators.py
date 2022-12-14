import numpy as np


def calc_grad(f, domain, diff_method):
    return diff_method(f, 'x', domain), diff_method(f, 'y', domain)


def calc_div(u, v, domain, diff_method):
    return diff_method(u, 'x', domain) + diff_method(v, 'y', domain)


def calc_curl(u, v, domain, diff_method):
    return -diff_method(u, 'y', domain) + diff_method(v, 'x', domain)

    
def central_diff4(f, direction, domain):
    diff_f = np.empty_like(f)
    if direction == 'y':
        for i in range(2, (domain.ny - 1)):
            diff_f[i, :] = (f[i-2, :] - 8*f[i-1, :]+8*f[i+1, :]-f[i+2, :])/(12*domain.dy)
        diff_f[0, :] = (f[-3, :] - 8*f[-2, :]+8*f[1, :]-f[2, :])/(12*domain.dy)
        diff_f[1, :] = (f[-2, :] - 8*f[0, :]+8*f[2, :]-f[3, :])/(12*domain.dy)
        diff_f[-2, :] = (f[-4, :] - 8*f[-3, :]+8*f[0, :]-f[1, :])/(12*domain.dy)
        diff_f[-1, :] = diff_f[0, :]
    elif direction == 'x':
        for i in range(2, (domain.nx - 1)):
            diff_f[:, i] = (f[:,i-2] - 8*f[:,i-1]+8*f[:,i+1]-f[:,i+2])/(12*domain.dx)
        diff_f[:,0] = (f[:,-3] - 8*f[:,-2]+8*f[:,1]-f[:,2])/(12*domain.dx)
        diff_f[:,1] = (f[:,-2] - 8*f[:,0]+8*f[:,2]-f[:,3])/(12*domain.dx)
        diff_f[:,-2] = (f[:,-4] - 8*f[:,-3]+8*f[:,0]-f[:,1])/(12*domain.dx)
        diff_f[:,-1] = diff_f[:,0]
    else:
        raise Exception(f"Error in central_diff4. Wrong direction value {direction}!")
    return diff_f
