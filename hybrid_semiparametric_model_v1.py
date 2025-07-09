# Hybrid Shallow-FFNN + ODE (very small, single cell demo)
#
# • Single hidden layer tanh network predicts a reaction-rate  v(c;θ)
# • ODE:  dc/dt = v(c) - D*c
# • Training: Levenberg-Marquardt (SciPy `least_squares`, method='lm')
#
# Everything fits in ~100 lines so you can paste it straight into a
# Jupyter/VS Code “canvas” cell.
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# ---------- 1. Network -------------------------------------------------
class ShallowNN:
    def __init__(self, hidden_dim, rng=None):
        self.H = hidden_dim
        self.rng = np.random.default_rng(rng)
        # Xavier initialisation
        self.W1 = self.rng.normal(0, 1 / np.sqrt(1), size=(hidden_dim, 1))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0, 1 / np.sqrt(hidden_dim), size=(1, hidden_dim))
        self.b2 = np.zeros(1)

    # -- packing helpers (for optimiser) --------------------------------
    def pack(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def unpack(self, theta):
        p = 0
        self.W1 = theta[p : p + self.H].reshape(self.H, 1)
        p += self.H
        self.b1 = theta[p : p + self.H]
        p += self.H
        self.W2 = theta[p : p + self.H].reshape(1, self.H)
        p += self.H
        self.b2 = theta[p : p + 1]

    # -- forward --------------------------------------------------------
    def __call__(self, c):
        # c: array-like shape (T,) or scalar
        z = self.W1 @ c.reshape(1, -1) + self.b1[:, None]
        h = np.tanh(z)
        v = (self.W2 @ h + self.b2).ravel()
        return v


# ---------- 2. ODE simulator -------------------------------------------
def simulate(nn, c0, t_eval, D=0.1):
    """Integrate  dc/dt = v(c) - D*c  using solve_ivp."""

    def rhs(t, c):
        return nn(np.array([c]))[0] - D * c

    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        [c0],
        t_eval=t_eval,
        vectorized=False,
        rtol=1e-6,
        atol=1e-8,
    )
    return sol.y[0]


# ---------- 3. Synthetic data ------------------------------------------
rng = np.random.default_rng(0)

true_nn = ShallowNN(hidden_dim=5, rng=1)
c0_true = 1.0
t_meas = np.linspace(0, 10, 41)
c_true = simulate(true_nn, c0_true, t_meas)
sigma = 0.05 * np.ones_like(c_true)
c_obs = c_true + rng.normal(0, sigma)

# ---------- 4. Training -----------------------------------------------
model = ShallowNN(hidden_dim=5, rng=42)
theta0 = model.pack()


def residuals(theta):
    model.unpack(theta)
    c_sim = simulate(model, c0_true, t_meas)
    return (c_sim - c_obs) / sigma


res = least_squares(residuals, theta0, method="lm", xtol=1e-8, ftol=1e-8, gtol=1e-8)
model.unpack(res.x)
print(f"Optimisation finished, WMSE = {np.mean(res.fun**2):.4e}")

# ---------- 5. Plot ----------------------------------------------------
c_fit = simulate(model, c0_true, t_meas)

plt.figure()
plt.errorbar(t_meas, c_obs, yerr=sigma, fmt="o", label="observations")
plt.plot(t_meas, c_fit, label="hybrid model")
plt.xlabel("time")
plt.ylabel("concentration c")
plt.legend()
plt.title("Hybrid shallow NN + ODE fit")
plt.show()
