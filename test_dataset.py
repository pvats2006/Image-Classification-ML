import numpy as np

X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("Feature Matrix:", X.shape)
print("Labels:", y.shape)

print("Classes:", set(y))

print("First Label:", y[0])

print("First Feature Vector Length:", len(X[0]))