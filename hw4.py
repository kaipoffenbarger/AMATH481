import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

dx = 2.5
m = 8    # N value in x and y directions
n = m * m  # total size of matrix

##### A matrix #####

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()
matA = matA / (dx**2)

A1 = matA

# Plot matrix structure
plt.figure(5)
plt.spy(matA)
plt.title('Matrix Structure')
plt.show()

##### B matrix #####

# Initialize vectors
e1 = np.ones((n, 1))   # vector of ones

# Construct the central difference matrix B (derivative with respect to x)
diagonals = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(n-m), -m, m, (n-m)]

# Construct the sparse matrix
matB = spdiags(diagonals, offsets, n, n).toarray()
matB = matB / (2 * dx)  # Second-order central difference

A2 = matB

# Plot matrix structure for B
plt.figure()
plt.spy(matB)
plt.title('Matrix Structure for ∂x')
plt.show()


##### C matrix #####

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [-m+1, -1, 1, m-1]

matC = spdiags(diagonals, offsets, n, n).toarray()
matC = matC / (2 * dx)

A3 = matC

# Plot matrix structure for B
plt.figure()
plt.spy(matC)
plt.title('Matrix Structure for ∂y')
plt.show()

