import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot2(y, x, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

tol = 1e-4  # define a tolerance level 
epsilon_start = 0.1  # beginning value of epsilon
L = 4
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
A = 1; phi0 = [1, np.sqrt(L**2 - epsilon_start)]; xp = [-L, L] 
xshoot =  np.arange(xp[0], xp[1]+0.1, 0.1)

A1 = np.zeros((len(xshoot), 5)) # for eigenfunctions values
A2 = np.array([]) # for eigenvalues


for modes in range(1, 6):  # begin mode loop
    epsilon = epsilon_start  # initial value of eigenvalue epsilon
    depsilon = 20 / 100  # default step size in epsilon
    for _ in range(1000):  # begin convergence loop for epsilon
        phi0 = [1, np.sqrt(L**2 - epsilon_start)]
        y = odeint(shoot2, phi0, xshoot, args=(epsilon,))  

        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon)*y[-1, 0]) < tol:  # check for convergence
            print(epsilon)  # write out eigenvalue
            A2 = np.append(A2, epsilon) # add eigenvalue before breaking out
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon)*y[-1, 0]) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2

    epsilon_start = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot)  # calculate the normalization
    eigenfunc_normalized = y[:, 0] / np.sqrt(norm) # get normalized eigenfunc values
    A1[:, modes - 1] = np.abs(eigenfunc_normalized) # add them to matrix

    plt.plot(xshoot, y[:, 0] / np.sqrt(norm), col[modes - 1], label=f'Eig func {modes}')  # plot modes
    
plt.legend(loc='lower left')
plt.show()  # end mode loop

print(A2)