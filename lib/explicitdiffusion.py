#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numba

#compute dq/dt = div q grad q
# and returns q(t+dt), dt
# where dt = min(stable dt, max_dt)
def solve_step(d, q, dx, max_dt):
    """
    d: local diffusion
    q: diffused quantity
    """
    dxp = np.c_[.5*(d[:,:-1] + d[:,1:]), d[:,-1] ]
    dxm = np.c_[d[:,0], .5*(d[:,:-1] + d[:,1:]) ]
    dyp = np.r_[.5*(d[:-1] + d[1:]), [d[-1]] ]
    dym = np.r_[[d[0]], .5*(d[:-1] + d[1:]) ]

    gxp = np.c_[(q[:,1:] - q[:,:-1]), q[:,-1] - q[:,-2] ]
    gxm = np.c_[q[:,1] - q[:,0], (q[:,1:] - q[:,:-1]) ]
    gyp = np.r_[(q[1:] - q[:-1]), [q[-1] - q[-2]] ]
    gym = np.r_[[q[1] - q[0]], (q[1:] - q[:-1]) ]

    max_d = max(max(np.max(dxp) , np.max(dxm)), max(np.max(dyp), np.max(dym)))
    if max_d == 0:
        max_d = 1.0
    dt = min(max_dt, dx*dx/4 /max_d)

    return q + dt * (dxp * gxp - dxm * gxm + dyp * gyp - dym * gym) / dx / dx, dt
