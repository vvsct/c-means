import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2.5, 3.0, 3.0, 3.5, 5.5, 6.0, 6.0, 6.5],
              [3.5, 3.0, 4.0, 3.5, 5.5, 6.0, 5.0, 5.5]])
# X = np.array([[2.5, 3.0, 3.0, 3.5, 5.5, 6.0, 6.0, 6.5, 25, 25, 25, 25],
#               [3.5, 3.0, 4.0, 3.5, 5.5, 6.0, 5.0, 5.5, 25, 25, 25, 25]])
# X = np.array([[2.5, 3.0, 3.0, 3.5, 5.5, 6.0, 6.0, 6.5, 25, 25, 25, 25],
#               [3.5, 3.0, 4.0, 3.5, 5.5, 6.0, 5.0, 5.5, 25, 25, 25, 25]])

num_rows, N = X.shape
m = 2
is_stop_criterion = 10000
epsilon = 0.00001
c = 2
# c = 3
# c = 4
V = np.zeros((num_rows, c))

U = np.random.rand(c, N)
U /= np.sum(U, axis=0)

t = 0
while is_stop_criterion > epsilon:
    t += 1
    for i in range(c):
        for j in range(num_rows):
            V[j, i] = np.sum(X[j, :] * U[i, :]**m) / np.sum(U[i, :]**m)
    V[np.isnan(V)] = 0

    d = np.zeros((c, N))
    for i in range(c):
        for j in range(N):
            d[i, j] = np.sum((X[:, j] - V[:, i])**2)

    J = 0
    for i in range(c):
        for j in range(N):
            J += (U[i, j]**m) * d[i, j]

    U_save = U.copy()
    for i in range(c):
        for j in range(N):
            U[i, j] = (d[i, j]**(2 / (1 - m))) / np.sum(d[:, j]**(2 / (1 - m)))
    is_stop_criterion = np.linalg.norm(U - U_save)

print("Partition matrix:")
print(U)
print("Cluster centers:")
print(V)
print("Minimum:")
print(J)
print("Number of iterations:")
print(t)

plt.scatter(X[0, :], X[1, :])
plt.scatter(V[0, :], V[1, :])
plt.show()
