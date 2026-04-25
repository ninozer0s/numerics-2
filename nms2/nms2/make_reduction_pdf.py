import numpy as np
import matplotlib.pyplot as plt

files = [
    ("Original", "original.dat"),
    ("K = 5", "reduction_K5.dat"),
    ("K = 10", "reduction_K10.dat"),
    ("K = 20", "reduction_K20.dat"),
]

fig, axes = plt.subplots(1, 4, figsize=(12, 4), constrained_layout=True)

for ax, (title, filename) in zip(axes, files):
    A = np.loadtxt(filename)
    ax.imshow(A, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


plt.savefig("reduction.pdf")
