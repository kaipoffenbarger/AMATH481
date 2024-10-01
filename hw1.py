import numpy as np

### Problem 1

## Newton-Raphson Method
xNR = np.array([-1.6]) # initial guess
iterationsNR = 0
for j in range(1000):
    xNR = np.append(xNR, xNR[j]-( (xNR[j]*np.sin(3*xNR[j])-np.exp(xNR[j])) / (np.sin(3*xNR[j])+3*xNR[j]*np.cos(3*xNR[j])-np.exp(xNR[j])) ))
    fc = xNR[j + 1]*np.sin(3*xNR[j + 1]) - np.exp(xNR[j + 1])
    iterationsNR = iterationsNR + 1
    if abs(fc) < 1e-6:
        break

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
np.save('A1.npy', xNR)
np.save('A2.npy', xmid)
np.save('A3.npy', iterationArray)



### Problem 2

## Define variables
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

## Part a
answerA = A + B
np.save('A4.npy', answerA)

## Part b
answerB = 3*x - 4*y
np.save('A5.npy', answerB)

## Part c
answerC = np.dot(A, x)
np.save('A6.npy', answerC)

## Part d
answerD = np.dot(B, (x-y))
np.save('A7.npy', answerD)

## Part e
answerE = np.dot(D, x)
np.save('A8.npy', answerE)

## Part f
answerF = np.dot(D, y) + z
np.save('A9.npy', answerF)

## Part g
answerG = np.dot(A, B)
np.save('A10.npy', answerG)

## Part h
answerH = np.dot(B, C)
np.save('A11.npy', answerH)

## Part i
answerI = np.dot(C, D)
np.save('A12.npy', answerI)