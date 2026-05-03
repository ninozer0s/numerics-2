
import numpy as np
import matplotlib.pyplot as plt

psi0 = np.genfromtxt("psi0.dat", names=True)
psi1 = np.genfromtxt("psi1.dat", names=True)
err = np.genfromtxt("errors.dat", names=True)

fig, ax = plt.subplots(3, 1, figsize=(8, 10))

names = psi0.dtype.names[2:]
num_plots = len(names)
for i, name in enumerate(names):
    color_val = 0.2 + 0.8 * (i / (num_plots - 1) if num_plots > 1 else 1)
    ax[0].plot(psi0["x"], psi0[name], linewidth=0.8, label=name, color=(0,0,color_val))

ax[0].plot(psi0["x"], psi0["exact"], color="red", linestyle="--", linewidth=1.4, label="exact")
ax[0].set_title("ground state psi0")
ax[0].set_xlabel("x")
ax[0].set_ylabel("psi0")
ax[0].legend(fontsize=7, ncol=4)
ax[0].set_xlim(-5,5)



names1 = psi1.dtype.names[2:]
num_plots1 = len(names1)
for i, name in enumerate(names1):
    color_val = 0.2 + 0.8 * (i / (num_plots1 - 1) if num_plots1 > 1 else 1)
    ax[1].plot(psi1["x"], psi1[name], linewidth=0.8, label=name, color=(0,0,color_val))

ax[1].plot(psi1["x"], psi1["exact"], color="red", linestyle="--", linewidth=1.4, label="exact")
ax[1].set_title("first excited state psi1")
ax[1].set_xlabel("x")
ax[1].set_ylabel("psi1")
ax[1].set_xlim(-5,5)
ax[1].legend(fontsize=7, ncol=4)

ax[2].semilogy(err["r"], err["E0_error"], marker="o", label="|E0 - exact|")
ax[2].semilogy(err["r"], err["E1_error"], marker="o", label="|E1 - exact|")
ax[2].set_title("eigenvalue errors")
ax[2].set_xlabel("Davidson iteration r")
ax[2].set_ylabel("absolute error")
ax[2].legend()

fig.tight_layout()
fig.savefig("davidson.pdf")
