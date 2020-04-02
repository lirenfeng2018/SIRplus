import numpy as np
from numpy.random import random_sample as rand
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cycler import cycler


def plain_SIR(ic, beta, gamma, duration):
    ''' Solves the SIR model with given initial conditions 'ic' and parameters beta, gamma for given time duration
    '''
    N = np.sum(ic)

    def deriv(t, y):
        dS = - beta * y[0] * y[1] / N
        dI = beta * y[0] * y[1] / N - gamma * y[1]
        dR = gamma * y[1]
        return np.array([dS, dI, dR])

    t_eval = np.linspace(0, duration, 1000)
    sol = solve_ivp(deriv, (0, duration), ic, t_eval=t_eval)
    
    return sol.t, sol.y


def discrete_SIR(ic, beta, gamma, duration, tstep=1e-4):
    ''' Implements the discrete Markov Chain model with given parameters
    '''
    N = np.sum(ic)
    p_inf = beta * tstep / N
    p_rec = gamma * tstep

    if p_inf * N * N + p_rec * N > 1:
        print("Please specify smaller time step or rescale the parameterse beta and gamma")
        return

    t = np.arange(0, duration, tstep)
    y = np.zeros((3, t.shape[0]))
    y[:,0] = ic

    def step(S, I):
        r = rand()
        if r < p_inf * S * I:
            return S-1, I+1
        elif r > 1 - p_rec * I:
            return S, I-1
        else:
            return S, I

    S, I = ic[0], ic[1]
    for i in range(y.shape[1] - 1):
        S, I = step(S, I)
        y[:,i+1] = S, I, N-S-I

    return t, y


def stratified_SIR(ic, beta, gamma, duration):
    ''' Solves the SIR model with multiple sub-populations, or strata.
        ic: initial condition. Array of length 3*n_strata
        beta: pairwaise infection rate. Matrix of size n_strata X n_strata
        (Note: beta_ij = (infection rate per contact)
                * (average # of contacts a susceptible one from stratum i makes with someone from j per time)
         Therefore, usually beta_ij * N_i = beta_ji * N_j (no sum))
        gamma: recovery rate. Array of length n_strata
    '''
    n_str = ic.shape[0] // 3
    N = np.sum(ic)
    Ns = np.zeros(n_str)    
    for i in range(n_str):
        # Polulation within each stratum
        Ns[i] = np.sum(ic[i::n_str])
        print("Stratum #", i+1, "has N =", Ns[i])

    # It is sufficient to solve the ODE for S and I only
    def deriv(t, y):
        S = y[:n_str]
        I = y[n_str:]
        tmp = np.matmul(beta, I / Ns) 
        dS = - tmp * S
        dI = tmp * S - gamma * I
        return np.concatenate((dS, dI))

    t = np.linspace(0, duration, 1000)
    sol = solve_ivp(deriv, (0, duration), ic[:2*n_str], t_eval=t)

    y = np.zeros((3 * n_str, t.shape[0]))
    y[:2*n_str, :] = sol.y
    for i in range(n_str):
        y[2*n_str+i,:] = Ns[i] - y[i,:] - y[n_str+i,:]

    return t, y

def plot_SIR(t, y):
    ''' Plots results from the SIR model
    '''
    fig = plt.figure()
    ax = fig.subplots()
    N = np.sum(y[:,0])
    ax.plot(t, y[0,:]/N, 'y', label='Susceptible')
    ax.plot(t, y[1,:]/N, 'r', label='Infected')
    ax.plot(t, y[2,:]/N, 'g', label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Ratio')
    ax.legend()

    return

def plot_stratified_SIR(t, y):
    ''' Plots results from the stratified SIR model
    '''

    n_str = y.shape[0] // 3
    N = np.sum(y[:,0])
    Ns = np.zeros(n_str)
    for i in range(n_str):
        Ns[i] = np.sum(y[i::n_str, 0])

    custom_cycler = (cycler(linestyle=['-', '--', ':']) * cycler(color=['y', 'r', 'g']))

    plt.rc('axes', prop_cycle=custom_cycler)
    fig = plt.figure()
    ax = fig.subplots()

    for i in range(n_str):
        ax.plot(t, y[i,:]/Ns[i], label='Susceptible '+str(i+1))
        ax.plot(t, y[i+n_str,:]/Ns[i], label='Infected '+str(i+1))
        ax.plot(t, y[i+2*n_str,:]/Ns[i], label='Recovered '+str(i+1))
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Ratio')
    #ax.set_yscale('log')
    ax.legend()
    

''' Test SIR model - continuous and discrete
BETA = 1.0
GAMMA = 0.2
IC = np.array([990, 10, 0])
DURATION = 50

t, y = plain_SIR(IC, BETA, GAMMA, DURATION)
plot_SIR(t, y)

t, y = discrete_SIR(IC, BETA, GAMMA, DURATION)
plot_SIR(t, y)
'''

#''' Test stratified SIR model
BETA = np.array([[1.0, 0.01], [0.02, 1.5]])
GAMMA = np.array([0.2, 0.2])
Ns = np.array([1000, 2000])
IC = np.array([990, 2000, 10, 0, 0, 0])
DURATION = 50

t, y = stratified_SIR(IC, BETA, GAMMA, DURATION)
plot_stratified_SIR(t, y)
#'''

plt.show()
