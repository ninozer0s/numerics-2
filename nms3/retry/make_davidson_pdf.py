
import numpy as np
import matplotlib.pyplot as plt

psi0 = np.genfromtxt("psi0.dat", names=True)
psi1 = np.genfromtxt("psi1.dat", names=True)
err = np.genfromtxt("errors.dat", names=True)

fig, ax = plt.subplots(3, 1, figsize=(8, 10))

for name in psi0.dtype.names[2:]:
    ax[0].plot(psi0["x"], psi0[name], linewidth=0.8, label=name)

ax[0].plot(psi0["x"], psi0["exact"], linestyle="--", linewidth=1.4, label="exact")
ax[0].set_title("ground state psi0")
ax[0].set_xlabel("x")
ax[0].set_ylabel("psi0")
ax[0].legend(fontsize=7, ncol=4)

for name in psi1.dtype.names[2:]:
    ax[1].plot(psi1["x"], psi1[name], linewidth=0.8, label=name)

ax[1].plot(psi1["x"], psi1["exact"], linestyle="--", linewidth=1.4, label="exact")
ax[1].set_title("first excited state psi1")
ax[1].set_xlabel("x")
ax[1].set_ylabel("psi1")
ax[1].legend(fontsize=7, ncol=4)

ax[2].semilogy(err["r"], err["E0_error"], marker="o", label="|E0 - exact|")
ax[2].semilogy(err["r"], err["E1_error"], marker="o", label="|E1 - exact|")
ax[2].set_title("eigenvalue errors")
ax[2].set_xlabel("Davidson iteration r")
ax[2].set_ylabel("absolute error")
ax[2].legend()

fig.tight_layout()
fig.savefig("davidson.pdf")
