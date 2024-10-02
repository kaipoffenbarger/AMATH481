import numpy as np

### Problem 1

import numpy as np 

# Define the function f(x)
def f(x):
    return x * np.sin(3 * x) - np.exp(x)

# Define the derivative f'(x)
def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

# Newton-Raphson method
def newton_raphson(x0, tol=1e-6, max_iter=1000):
    x = x0
    x_values = [x]  # List to store x values
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-10:  # To avoid division by zero
            print("Derivative is too small. Stopping.")
            return None, x_values, i
        
        # Newton-Raphson formula
        x_new = x - fx / dfx
        x_values.append(x_new)  # Store the new x value
        
        # Check for convergence
        if abs(x_new - x) < tol:
            break
        
        x = x_new
    
    return x_values, len(x_values) - 1

# Initial guess
x0 = -1.6

# Run the Newton-Raphson method
xNR, iterationsNR = newton_raphson(x0)


## Bisection Method
xr = -0.4; xl = -0.7 # initial endpoints
xmid = np.array([])
iterationsBis = 0
for j in range(0, 100):
    xc = (xr + xl)/2
    fc = xc*np.sin(3*xc) - np.exp(xc)
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc
    xmid = np.append(xmid, xc)
    iterationsBis = iterationsBis + 1
    if ( abs(fc) < 1e-6 ):
        break

# create array for iterations of both methods
iterationArray = np.array([iterationsNR, iterationsBis])

# save answers
A1 = xNR
A2 = xmid
A3 = iterationArray

### Problem 2

## Define variables
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

## Part a
answerA = A + B
A4 = answerA

## Part b
answerB = 3*x - 4*y
A5 = answerB

## Part c
answerC = np.dot(A, x)
A6 = answerC

## Part d
answerD = np.dot(B, (x-y))
A7 = answerD

## Part e
answerE = np.dot(D, x)
A8 = answerE


## Part f
answerF = np.dot(D, y) + z
A9 = answerF

## Part g
answerG = np.dot(A, B)
A10 = answerG

## Part h
answerH = np.dot(B, C)
A11 = answerH

## Part i
answerI = np.dot(C, D)
A12 = answerI
