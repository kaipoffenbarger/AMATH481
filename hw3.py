import numpy as np
from scipy.integrate import odeint, simpson, solve_ivp
from scipy.sparse import spdiags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import math

## Part a

def shoot21(x, y, epsilon):
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
        answer = solve_ivp(shoot21, [xshoot[0], xshoot[-1]], phi0, t_eval=xshoot, args=(epsilon,))
        y = answer.y.T

        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon)*y[-1, 0]) < tol:  # check for convergence
            # print(epsilon)  # write out eigenvalue
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

    # plt.plot(xshoot, (y[:, 0] / np.sqrt(norm)), col[modes - 1], label=f'Eig func {modes}')  # plot modes
    
# plt.legend(loc='lower left')
# plt.show()  # end mode loop





## Part b


L = 4
xp = [-L, L]
dx = 0.1
N = 79
xspan =  np.arange(xp[0], xp[1] + 0.1, 0.1)
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors

A = np.zeros((xspan.size - 2, xspan.size - 2))  # Exclude boundary points in matrix

# Fill the main diagonal with -2
for j in range(1, 78):
    A[j, j] = -2 - dx**2 * xspan[j + 1]**2

# Fill the diagonals above and below the main diagonal with 1
np.fill_diagonal(A[1:], 1)
np.fill_diagonal(A[:, 1:], 1)

A[0, 0] = -2/3 - dx**2 * xspan[1]**2
A[0, 1] = 2/3
A[-1, -1] = -2/3 - dx**2 * xspan[-2]**2
A[-1, -2] = 2/3

A = A / (dx**2)

# Compute linear operator
linL = -A

# Get eigenvalues and eigenvectors
eigValues, eigVectors = eigs(linL, k=5, which='SM')

# Calcualte boundary points
temp = eigVectors.real
phi0 = []
phiN = []
for i in range(1, 6):
    start = (4 * temp[0, i - 1] - temp[1, i - 1]) / 3
    end = (4 * temp[-1, i - 1] - temp[-2, i - 1]) / 3
    phi0 = np.append(phi0, start)
    phiN = np.append(phiN, end)

# Convert phi0 and phiN to arrays and stack them onto temp
phi0 = np.array(phi0)
phiN = np.array(phiN)
temp = np.vstack([phi0, temp, phiN])

# Normalize and take the absolute values of the eigenvectors
for i in range(temp.shape[1]):
    norm = np.trapz(temp[:, i] * temp[:, i], xspan)  # Calculate the normalization factor
    eigenVectorNormalized = temp[:, i] / np.sqrt(norm) # Normalize and take the absolute value
    temp[:, i] = np.abs(temp[:, i] / np.sqrt(norm))  # Normalize and take the absolute value
    plt.plot(xspan, np.abs(eigenVectorNormalized), col[i], label = f'Eig func {i + 1}')


plt.show()

# Part b answers
A3 = temp
A4 = eigValues.real
print(A4)




## Part c


#### gamma = 0.05

tol = 1e-6
L = 2
dx = 0.1
xspan = np.arange(-L, L+dx, dx)

def shoot1(x, phi, epsilon):
    # phi[0] = phi_n; phi[1] = phi_n'
    return [phi[1], (0.05 * np.abs(phi[0])**2 + x**2 - epsilon) * phi[0]]


colors = ['r', 'b', 'g', 'c', 'm']
A5 = np.zeros((len(xspan), 2))
A6 = np.zeros(2)
epsilon_start = 0.1
A_start = 0.01 
A = A_start

for modes in range(2):  # loop to find the first 2 eigenfunctions
    
   
    
    dA = 0.01
   
    for k in range(1000):  # Area loop  
        epsilon = epsilon_start
        depsilon = 0.5
        print(str(modes) + " " + "area loop " + str(k))
        
        for j in range(1000):  # Epsilon loop

            x0 = [A, np.sqrt(L**2 - epsilon) * A]
           
            answer = solve_ivp(shoot1, [xspan[0], xspan[-1]], x0, t_eval=xspan, args=(epsilon,))
            phi = answer.y.T
            x_used = answer.t

            # check for convergence
            if abs(phi[-1, 0] * np.sqrt(L**2 - epsilon) + phi[-1, 1] - 0) < tol:
                break
   
            # raise/lower epsilon based on whether we're under/over shooting
            if (-1) ** (modes) * (phi[-1, 0] * np.sqrt(L**2 - epsilon) + phi[-1, 1]) > 0:
                epsilon = epsilon + depsilon
               
            else:
                epsilon = epsilon - depsilon / 2
                depsilon = depsilon / 2
       
        # Check area under curve = 1
        print(len(xspan))
        print(len(phi[:,0]**2))
        area = simpson(phi[:,0]**2, x=x_used)

        # Adjust A
        if np.abs(area - 1) < tol:
            print(f"Eigenvalue for mode {modes}: {epsilon}")
            A6[modes] = epsilon

            epsilon_start = epsilon + 0.5

            break

        if area < 1:
            A = A + dA
        else:
            A = A - dA / 2
            dA = dA/2
        print(A)

    # next initial guess for epsilon (after convergence of previous mode)
    A5[:,modes] = np.abs(phi[:,0])
    plt.plot(xspan, A5[:, modes], colors[modes], label=f"n = {modes}") # plot abs value
    

# Plot eigenfunctions
plt.title('Normalized Eigenfunctions 1 and 2')
plt.xlabel('x')
plt.ylabel(r'$\phi_n$')
plt.legend()
plt.grid(True)
plt.show()



#### gamma = -0.05

tol = 1e-6
L = 2
dx = 0.1
xspan = np.arange(-L, L+dx, dx)

def shoot1(x, phi, epsilon):
    # phi[0] = phi_n; phi[1] = phi_n'
    return [phi[1], (-0.05 * np.abs(phi[0])**2 + x**2 - epsilon) * phi[0]]


colors = ['r', 'b', 'g', 'c', 'm']
A7 = np.zeros((len(xspan), 2))
A8 = np.zeros(2)
epsilon_start = 0.1
A_start = 0.01  # or 1e-6. should be really small ????
A = A_start

for modes in range(2):  # loop to find the first 5 eigenfunctions
    
   
    
    dA = 0.01
   
    for k in range(1000):  # Area loop  
        epsilon = epsilon_start
        depsilon = 0.5
        print(str(modes) + " " + "area loop " + str(k))
        
        for j in range(1000):  # Epsilon loop

            x0 = [A, np.sqrt(L**2 - epsilon) * A]
           
            answer = solve_ivp(shoot1, [xspan[0], xspan[-1]], x0, t_eval=xspan, args=(epsilon,))
            phi = answer.y.T
            x_used = answer.t

            # check for convergence
            if abs(phi[-1, 0] * np.sqrt(L**2 - epsilon) + phi[-1, 1] - 0) < tol:
                break
   
            # raise/lower epsilon based on whether we're under/over shooting
            if (-1) ** (modes) * (phi[-1, 0] * np.sqrt(L**2 - epsilon) + phi[-1, 1]) > 0:
                epsilon = epsilon + depsilon
               
            else:
                epsilon = epsilon - depsilon / 2
                depsilon = depsilon / 2
       
        # Check area under curve = 1
        print(len(xspan))
        print(len(phi[:,0]**2))
        area = simpson(phi[:,0]**2, x=x_used)

        # Adjust A
        if np.abs(area - 1) < tol:
            print(f"Eigenvalue for mode {modes}: {epsilon}")
            A8[modes] = epsilon

            epsilon_start = epsilon + 0.5

            break

        if area < 1:
            A = A + dA
        else:
            A = A - dA / 2
            dA = dA/2
        print(A)

    # next initial guess for epsilon (after convergence of previous mode)
    A7[:,modes] = np.abs(phi[:,0])
    plt.plot(xspan, A5[:, modes], colors[modes], label=f"n = {modes}") # plot abs value
    

# Plot eigenfunctions
plt.title('Normalized Eigenfunctions 1 and 2')
plt.xlabel('x')
plt.ylabel(r'$\phi_n$')
plt.legend()
plt.grid(True)
plt.show()



## Part d



# Given parameters
epsilon = 1
gamma = 0
phi0 = [1, np.sqrt(3)]
xp = [-2, 2]  # x in [-L, L] with L = 2
xshoot = np.arange(xp[0], xp[1] + 0.1, 0.1)

# Tolerance levels for convergence study
TOL = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# Define the differential equation function
def shoot21(x, y, epsilon):
    phi = y[0]
    dphi_dx = y[1]
    K = 1  # Assuming K=1 as before
    return [dphi_dx, (K * x**2 - epsilon) * phi]

# Arrays to store average step sizes for each method
avg_step_sizes_RK45 = []
avg_step_sizes_RK23 = []
avg_step_sizes_Radau = []
avg_step_sizes_BDF = []

# Run convergence study for each method
for tolerance in TOL:
    options = {'rtol': tolerance, 'atol': tolerance}
    
    # Solve with RK45
    sol_RK45 = solve_ivp(shoot21, xp, phi0, method='RK45', args=(epsilon,), **options)
    step_sizes_RK45 = np.diff(sol_RK45.t)
    avg_step_sizes_RK45.append(np.mean(step_sizes_RK45))
    
    # Solve with RK23
    sol_RK23 = solve_ivp(shoot21, xp, phi0, method='RK23', args=(epsilon,), **options)
    step_sizes_RK23 = np.diff(sol_RK23.t)
    avg_step_sizes_RK23.append(np.mean(step_sizes_RK23))
    
    # Solve with Radau
    sol_Radau = solve_ivp(shoot21, xp, phi0, method='Radau', args=(epsilon,), **options)
    step_sizes_Radau = np.diff(sol_Radau.t)
    avg_step_sizes_Radau.append(np.mean(step_sizes_Radau))
    
    # Solve with BDF
    sol_BDF = solve_ivp(shoot21, xp, phi0, method='BDF', args=(epsilon,), **options)
    step_sizes_BDF = np.diff(sol_BDF.t)
    avg_step_sizes_BDF.append(np.mean(step_sizes_BDF))

# Log-log fit to find the slopes
log_TOL = np.log(TOL)

log_avg_step_sizes_RK45 = np.log(avg_step_sizes_RK45)
log_avg_step_sizes_RK23 = np.log(avg_step_sizes_RK23)
log_avg_step_sizes_Radau = np.log(avg_step_sizes_Radau)
log_avg_step_sizes_BDF = np.log(avg_step_sizes_BDF)

# Calculate slopes using polyfit
slope_RK45, _ = np.polyfit(log_avg_step_sizes_RK45, log_TOL, 1)
slope_RK23, _ = np.polyfit(log_avg_step_sizes_RK23, log_TOL, 1)
slope_Radau, _ = np.polyfit(log_avg_step_sizes_Radau, log_TOL, 1)
slope_BDF, _ = np.polyfit(log_avg_step_sizes_BDF, log_TOL, 1)

# Store the results in A9 as a 4x1 vector with the slopes
A9 = np.array([slope_RK45, slope_RK23, slope_Radau, slope_BDF])

# Plot results on a log-log scale for visual verification
plt.figure(figsize=(10, 6))
plt.loglog(avg_step_sizes_RK45, TOL, 'o-', label='RK45')
plt.loglog(avg_step_sizes_RK23, TOL, 's-', label='RK23')
plt.loglog(avg_step_sizes_Radau, TOL, 'd-', label='Radau')
plt.loglog(avg_step_sizes_BDF, TOL, 'x-', label='BDF')
plt.xlabel("Average Step Size")
plt.ylabel("Tolerance")
plt.title("Convergence Study of RK45, RK23, Radau, and BDF")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()





## Part e

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

L = 4
xspan = np.linspace(-L, L, 81)
dx = xspan[1] - xspan[0]

# Exact eigenvalues
exact_eigs = [1, 3, 5, 7, 9]

# Gauss-Hermite polynomial solutions
h = np.array([np.ones_like(xspan),
     2 * xspan,    
     4 * xspan**2 - 2,            
     8 * xspan**3 - 12 * xspan,    
     16 * xspan**4 - 48 * xspan**2 + 12,
     ])

# Eigenvector errors
# exact_efs = [
#     (np.exp(-xspan/2) * h_j) / np.sqrt(math.factorial(index) * 2**index * np.sqrt(np.pi))
#     for index, h_j in enumerate(h)
# ]

exact_efs = np.zeros((len(xspan), 5))
for j in range(5):
    exact_efs[:,j] = np.exp(-xspan**2/2) * h[j,:] / np.sqrt(factorial(j) * 2**j * np.sqrt(np.pi))
   

ef_errors_a = np.zeros(5)
ef_errors_b = np.zeros(5)


for j in range(len(h)):
    abs_diff_a = abs(A1[:,j]) - abs(exact_efs[:,j])
    ef_errors_a[j] = simpson(abs_diff_a**2, x=xspan)
   
    abs_diff_b = abs(A3[:,j]) - abs(exact_efs[:,j])
    ef_errors_b[j] = simpson(abs_diff_b**2, x=xspan)

A10 = ef_errors_a
A12 = ef_errors_b


A11 = np.zeros(5)
A13 = np.zeros(5)

for eig_num in range(len(exact_eigs)):
    A11[eig_num] = 100 * abs(A2[eig_num] - exact_eigs[eig_num]) / exact_eigs[eig_num]
    A13[eig_num] = 100 * abs(A4[eig_num] - exact_eigs[eig_num]) / exact_eigs[eig_num]

print("A2: ", A2)
print("A4: ", A4)
print("A6: ", A6)
print("A8: ", A8)

print("A9: ", A9)

print("A10: ", A10)
print("A11: ", A11)
print("A12: ", A12)
print("A13: ", A13)