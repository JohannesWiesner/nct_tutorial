# -*- coding: utf-8 -*-
"""
Extensions to https://github.com/BassettLab/control_package that might be 
offered as pull requests to that package
@author: Johannes.Wiesner
"""

import numpy as np

def state_trajectory(A,xi,T):
    """This function caclulates the trajectory for the network given our model
     if there are no constraints, and the target state is unknown, using the
     control equation precess x(t+1) = Ax(t). x(t) is the state vector, A is
     the adjacency matrix.
    
    Args:
     A             : NxN state matrix (numpy array), where N is the number of nodes in your
                   network (for example, a structural connectivity matrix 
                   constructed from DTI). A should be stable to prevent
                   uncontrolled trajectories.
         
     xi            : Nx1 initial state (numpy array) of your system where N is the number of
                   nodes. xi MUST have N rows. 
    
    T              : Number of time points, int
        
      Returns:
     x             : x is the NxT trajectory (numpy array) that results from simulating
                   x(t+1) = Ax(t)
    
     @author JohannesWiesner
     June 2021
    """

    # Simulate trajectory
    N = np.size(A,0)

    # initialize x
    x = np.zeros((N,T))
    xt = xi
    for t in range(T):
        x[:,t] = np.reshape(xt, N) # annoying python 1d array thing
        xt_1 = np.matmul(A,xt)
        xt = xt_1
    return x