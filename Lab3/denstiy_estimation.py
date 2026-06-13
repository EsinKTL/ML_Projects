import numpy as np
import matplotlib.pyplot as plt

'''
We will generate 200 data points: 90% from a population with mean 5, variance 1,
and the remaining 10% from another population with mean 10, variance 1.
When plotted, the histogram should be bimodal (two peaks), not unimodal.
'''

'''
What KDE does: It places a smooth peak (called a kernel) on each data point.
Then it sums all these 200 little peaks. The resulting smooth curve is our density estimate.

n: Number of data points (200 in our scenario).
X_i: The actual data points we have.
x: Any location where we want to measure the density (height).
h: Smoothing parameter (Bandwidth). Determines how wide or narrow the peaks are.
K(dot): The chosen Kernel function.
'''

def gaussian_kernel(u):
    # The most popular kernel. Places a standard bell curve on each point.
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def epanechnikov_kernel(u):
    '''
    Mathematically optimal kernel that minimizes variance.
    Returns a parabolic value if |u| <= 1, otherwise zero.
    '''
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

def uniform_kernel(u):
    # The simplest kernel. Places a flat box around each point.
    # Returns constant 1/2 if |u| <= 1, otherwise zero.
    return np.where(np.abs(u) <= 1, 0.5, 0)

def kde_predict(x, X_data, h, kernel_func):
    n = len(X_data)
    # 1. Compute u values vectorially for each X_data point
    u = (x - X_data) / h
    # 2. Apply the chosen kernel function to these u values
    kernel_val = kernel_func(u)
    f_hat = np.sum(kernel_val) / (n * h)
    return f_hat

def true_density(x):
    # N(5,1) component
    pdf_1 = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * (x - 5)**2)
    # N(10,1) component
    pdf_2 = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * (x - 10)**2)
    return 0.9 * pdf_1 + 0.1 * pdf_2

def kde_evaluate(x_grid, X_data, h, kernel_func):
    # Call kde_predict for each point in x_grid and collect results
    return np.array([kde_predict(x, X_data, h, kernel_func) for x in x_grid])

def compute_kde_mse(X_data, h, kernel_func, K=1000):
    # Generate K=1000 iid points uniformly from [2,12]
    x_grid = np.random.uniform(2, 12, size=K)
    # True density values at these points
    y_true = true_density(x_grid)
    # KDE predictions at these points
    y_pred = kde_evaluate(x_grid, X_data, h, kernel_func)
    # Mean squared error
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def silverman_bandwidth(X_data):
    n = len(X_data)
    sigma_hat = np.std(X_data, ddof=1)
    return 1.06 * sigma_hat * (n ** (-1/5))

n_values = [50, 100, 200, 500, 1000]
mse_results = []

# 1. GAUSSIAN MIXTURE DATA SIMULATION
def generate_mixture_data(n=200):
    choice = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    X = np.zeros(n)
    for i in range(n):
        if choice[i] == 0:
            X[i] = np.random.normal(5, 1)
        else:
            X[i] = np.random.normal(10, 1)
    return X

for n in n_values:
    X_sample = generate_mixture_data(n=n)
    h_optimal = silverman_bandwidth(X_sample)
    error = compute_kde_mse(X_sample, h_optimal, gaussian_kernel)
    mse_results.append(error)

# Plot the effect of sample size on MSE
plt.figure(figsize=(8, 5))
plt.plot(n_values, mse_results, marker='o', color='purple', linewidth=2)
plt.xlabel("Sample Size (n)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Effect of Sample Size on MSE")
plt.grid(True)
plt.savefig("KDE_n_influence.pdf", format="pdf")
plt.close()

X_data = generate_mixture_data(n=200)
x_plot = np.linspace(2, 12, 500)  # 500 points for a smooth plot
y_true = true_density(x_plot)

kernels = [gaussian_kernel, epanechnikov_kernel, uniform_kernel]
kernel_names = ["Gaussian", "Epanechnikov", "Uniform"]
h_values = [0.2, 1.0, 3.0]  # Small, optimal, large bandwidths

fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)

for k_idx, kernel in enumerate(kernels):
    for h_idx, h in enumerate(h_values):
        ax = axes[k_idx, h_idx]
        # True density
        ax.plot(x_plot, y_true, label="True Density", color="black", linestyle="--")
        # KDE estimate
        y_pred = kde_evaluate(x_plot, X_data, h, kernel)
        ax.plot(x_plot, y_pred, label="KDE Estimate", color="red")
        ax.set_title(f"{kernel_names[k_idx]} (h={h})")
        if k_idx == 0 and h_idx == 0:
            ax.legend()

plt.tight_layout()
plt.savefig("KDE_kernel_h_influence.pdf", format="pdf")
plt.close()

def generate_artificial_sample(X_data, h, k=5000):
    n = len(X_data)
    X_prime = np.zeros(k)
    for j in range(k):
        i = np.random.randint(0, n)          # uniform index
        epsilon = np.random.normal(0, 1)     # standard normal
        X_prime[j] = X_data[i] + epsilon * h
    return X_prime

# 1. Generate original 200-size data and find optimal h
X_orig = generate_mixture_data(n=200)
h_optimal = silverman_bandwidth(X_orig)

# 2. Method 1: MSE using original data
mse_method_1 = compute_kde_mse(X_orig, h_optimal, gaussian_kernel)

# 3. Method 2: Generate artificial sample of size k=5000 from the original data
X_artif = generate_artificial_sample(X_orig, h_optimal, k=5000)
mse_method_2 = compute_kde_mse(X_artif, h_optimal, gaussian_kernel)

print("#1 Method MSE (Original):", mse_method_1)
print("#2 Method MSE (Artificial):", mse_method_2)

class KDENaiveBayes:
    def __init__(self, h=1.0, kernel_func=gaussian_kernel):
        self.h = h
        self.kernel_func = kernel_func

    def fit(self, X, y):
        # Separate data by class
        self.X_0 = X[y == 0]
        self.X_1 = X[y == 1]
        total_samples = len(y)
        self.prior_0 = self.X_0.shape[0] / total_samples
        self.prior_1 = self.X_1.shape[0] / total_samples

    def predict_proba(self, Xtest):
        num_samples = Xtest.shape[0]
        num_features = Xtest.shape[1]
        probs = np.zeros(num_samples)

        for idx in range(num_samples):
            x_sample = Xtest[idx]
            # Log‑likelihood for class 0
            score_0 = np.log(self.prior_0)
            for f in range(num_features):
                kde_val = kde_predict(x_sample[f], self.X_0[:, f], self.h, self.kernel_func)
                score_0 += np.log(max(kde_val, 1e-10))   # lower bound for numerical stability

            # Log‑likelihood for class 1
            score_1 = np.log(self.prior_1)
            for f in range(num_features):
                kde_val = kde_predict(x_sample[f], self.X_1[:, f], self.h, self.kernel_func)
                score_1 += np.log(max(kde_val, 1e-10))

            # Convert to probability using sigmoid
            probs[idx] = 1 / (1 + np.exp(-(score_1 - score_0)))
        return probs

    def predict(self, Xtest):
        probs = self.predict_proba(Xtest)
        return np.where(probs > 0.5, 1, 0)


# ============================================================================
# ANSWERS TO THE QUESTIONS (added as comments at the end)
# ============================================================================
"""
Question 1 (Effect of too small / too large bandwidth):
- Too small bandwidth (h): The kernel peaks become very narrow. The estimate
  becomes highly variable, showing many spurious bumps (overfitting). It will
  have low bias but high variance.
- Too large bandwidth (h): The kernel peaks become very wide. The estimate
  becomes oversmoothed, losing important features such as the bimodality
  (underfitting). It will have high bias but low variance.

Question 2 (Compare Method 1 and Method 2):
- Method 1 directly uses the original sample X1..Xn to compute the KDE.
- Method 2 generates a much larger artificial sample X' by adding Gaussian
  noise to each original point (X'_j = Xi + ε·h). The KDE computed from this
  larger sample approximates the same density because the added noise exactly
  matches the kernel smoothing operation. In the limit of infinite k, the
  empirical distribution of the artificial sample converges to the KDE itself.
- The two methods should give very similar MSE because the artificial sample
  is essentially a Monte Carlo approximation of the KDE integral. Increasing k
  reduces the Monte Carlo error, making the two methods even closer, but the
  essential density estimate does not change; only the variance due to finite
  sampling from the KDE diminishes.

Question 3 (Naive Bayes comparison on a binary classification dataset):
This part requires selecting a real dataset with quantitative features and
implementing four variants:
  1. Naive Bayes with KDE (as coded above)
  2. Naive Bayes with Gaussian approximation (assume each feature follows a
     normal distribution per class, estimate mean and variance).
  3. Naive Bayes with discretization of quantitative features (e.g., binning
     into intervals and using multinomial frequencies).
  4. Linear Discriminant Analysis (LDA) – assumes Gaussian features with a
     common covariance matrix.
A proper answer would involve running these methods on a chosen dataset,
reporting accuracy (e.g., via cross-validation), and discussing:
- KDE can capture arbitrary shapes but requires bandwidth tuning and more data.
- Gaussian approximation is parametric and works well if features are roughly
  normal; it may fail if the true densities are multimodal or skewed.
- Discretization can capture non‑linear effects but loses ordering information
  and the number of bins must be chosen.
- LDA is also parametric, assumes normality with equal covariances; it can be
  very stable when the assumptions are met, but less flexible.
The actual comparison would show which method performs best on the specific
dataset, depending on the underlying distributions.
"""
