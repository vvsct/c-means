import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2.5, 3.0, 3.0, 3.5, 5.5, 6.0, 6.0, 6.5],
              [3.5, 3.0, 4.0, 3.5, 5.5, 6.0, 5.0, 5.5]])

num_rows, N = X.shape
c = 2
# c = 3
# c = 4
V = np.zeros((num_rows, c))
U = np.zeros((c, N))
row_iteration = 0
for i in range(N):
    U[row_iteration, i] = 1
    row_iteration = (row_iteration + 1) % c

print(U)
U = U[:, np.random.permutation(N)]
is_stop_criterion = 10000
epsilon = 0.00001

t = 0
while is_stop_criterion > epsilon:
    t += 1
    for i in range(c):
        for j in range(num_rows):
            V[j, i] = np.sum(X[j, :] * U[i, :]) / np.sum(U[i, :])
    V[np.isnan(V)] = 0

    d = np.zeros((c, N))
    for i in range(c):
        for j in range(N):
            d[i, j] = np.sum((X[:, j] - V[:, i]) ** 2)

    J = np.sum(U * d)

    U_save = U.copy()
    U = np.zeros((c, N))
    for j in range(N):
        min_cluster = np.argmin(d[:, j])
        U[min_cluster, j] = 1

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
