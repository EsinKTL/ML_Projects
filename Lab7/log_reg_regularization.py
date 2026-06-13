import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

y_df = pd.read_csv('prostate_y.txt', sep=r'\s+', index_col=0)
X_df = pd.read_csv('prostate_x.txt', sep=r'\s+', index_col=0)

# .values: converts pandas DataFrame to numpy array
# .ravel(): converts 2D array like [[0],[1],[0],...] to 1D [0,1,0,...]
# y result: (102,) — labels of 102 patients: 0=healthy, 1=cancer
y = y_df.values.ravel()

# X result: (102, 6033) — 102 patients, 6033 genes
X = X_df.values

print("X shape:", X.shape)   # should be (102, 6033)
print("y shape:", y.shape)   # should be (102,)

# Regularization wants to apply equal penalty to each coefficient.
# But genes may be on different scales:
#   gene_1: values between 0.001 and 0.002
#   gene_2: values between 100 and 200
# Applying the same penalty would be unfair.
# StandardScaler transforms each gene as:
#   new_value = (old_value - mean) / standard_deviation
# Result: each gene has mean 0 and standard deviation 1

scaler = StandardScaler()

# fit_transform: first computes mean and std (fit), then transforms (transform)
X_scaled = scaler.fit_transform(X)

# LAMBDA VALUES — WHY LOG SCALE?

# In sklearn, regularization strength is controlled by parameter C
# Relation: C = 1/λ
#   Large C → small λ → weak penalty → coefficients can be large
#   Small C → large λ → strong penalty → coefficients shrink toward zero
#
# np.logspace(-4, 2, 50):
#   generates 50 equally spaced values between 10^(-4)=0.0001 and 10^2=100
#   BUT on a log scale
#   i.e., 0.0001, 0.0002, ..., 1, 2, ..., 100
#   Why log scale? Because the effect of λ appears more uniform on log scale

C_values = np.logspace(-4, 2, 50)   # 50 different C values
lambdas = 1 / C_values              # C → λ conversion

# TASK 1: RIDGE, LASSO, ELASTIC NET MODELS

# We will store coefficients for each regularization type at every λ
# Start with empty lists, fill them in the loop
# coefs_ridge[i] → the 6033 coefficients at i-th λ value

coefs_ridge = []   # ridge coefficients: one row per λ
coefs_lasso = []   # lasso coefficients
coefs_enet  = []   # elastic net coefficients

for C in C_values:
	
	# -------------------------------------------------------
	# RIDGE (L2 Regularization)
	# -------------------------------------------------------
	# What it does: penalizes the SQUARE of each coefficient
	# Formula: Loss + λ * Σ(βj²)
	# Result: all coefficients shrink but NONE becomes exactly zero
	# When to use: when all variables are somewhat important
	#
	# l1_ratio=0: pure L2 penalty (ridge)
	# C=C: current λ value (C = 1/λ)
	# solver='saga': fast algorithm for large and sparse datasets
	# max_iter=10000: maximum iterations for convergence
	# tol=0.01: tolerance — stop if coefficients change less than this
	
	model_ridge = LogisticRegression(
		l1_ratio=0, C=C, solver='saga', max_iter=10000, tol=0.01
	)
	model_ridge.fit(X_scaled, y)
	
	# coef_: fitted coefficients, shape: (1, 6033) — 1 class, 6033 genes
	# coef_[0]: take the first (and only) row → (6033,) vector
	coefs_ridge.append(model_ridge.coef_[0])
	
	# -------------------------------------------------------
	# LASSO (L1 Regularization)
	# -------------------------------------------------------
	# What it does: penalizes the ABSOLUTE VALUE of each coefficient
	# Formula: Loss + λ * Σ|βj|
	# Result: some coefficients become EXACTLY zero → variable selection
	# When to use: when only a few variables are truly important
	#   If only a few genes are relevant for cancer, lasso finds them
	
	model_lasso = LogisticRegression(
		l1_ratio=1, C=C, solver='saga', max_iter=10000, tol=0.01
	)
	model_lasso.fit(X_scaled, y)
	coefs_lasso.append(model_lasso.coef_[0])
	
	# -------------------------------------------------------
	# ELASTIC NET (L1 + L2 Mixture)
	# -------------------------------------------------------
	# What it does: applies both L1 and L2 penalties
	# Formula: Loss + λ * [l1_ratio * Σ|βj| + (1-l1_ratio) * Σβj²]
	# l1_ratio=0.5: 50% lasso + 50% ridge
	# Result: creates sparsity like lasso BUT keeps correlated
	#   variables together (lasso would discard one of them)
	# When to use: when there are correlated genes
	
	model_enet = LogisticRegression(
		l1_ratio=0.5, C=C, solver='saga', max_iter=10000, tol=0.01
	)
	model_enet.fit(X_scaled, y)
	coefs_enet.append(model_enet.coef_[0])

# np.array: convert list to numpy matrix
# Result: (50, 6033) — 50 λ values, each with 6033 coefficients
coefs_ridge = np.array(coefs_ridge)
coefs_lasso = np.array(coefs_lasso)
coefs_enet  = np.array(coefs_enet)

# At the largest λ (most penalty), how many coefficients are non-zero?
# np.sum(... != 0): count non-zero elements
print("\n=== Number of non-zero coefficients at largest λ ===")
print(f"Ridge      : {np.sum(coefs_ridge[-1] != 0)}")  # -1: last element = largest λ
print(f"Lasso      : {np.sum(coefs_lasso[-1] != 0)}")
print(f"Elastic Net: {np.sum(coefs_enet[-1]  != 0)}")

# TASK 1: PROFILE PLOTS

# Profile plot: how does the coefficient of each gene change with λ?
# x-axis: log10(λ) — λ values
# y-axis: coefficient value
# Each line: coefficient path of one gene across λ
#
# 6033 genes are too many for a plot; we only plot the first 20 genes
# plt.subplots(1, 3): three plots side by side
# figsize=(18, 6): figure size (width, height) in inches

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, coefs, title in zip(
		axes,
		[coefs_ridge, coefs_lasso, coefs_enet],
		['Ridge (L2)', 'Lasso (L1)', 'Elastic Net']
):
	for j in range(20):
		# coefs[:, j]: coefficients of j-th gene across all λ
		# np.log10(lambdas): show λ on log scale
		ax.plot(np.log10(lambdas), coefs[:, j])
	
	# y=0 horizontal line: shows where coefficient is zero
	ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
	ax.set_xlabel('log10(λ) — penalty increases as λ grows')
	ax.set_ylabel('Coefficient value')
	ax.set_title(title)
	ax.grid(True)

plt.suptitle('Profile Plots: How coefficients change with λ')
plt.tight_layout()
plt.show()

# TASK 1: CROSS-VALIDATION TO FIND OPTIMAL λ

# Cross-validation: which λ gives the best predictions?
# cv=5: split data into 5 folds, use each fold as test
# LogisticRegressionCV: tries different C values, selects the best
# Cs=C_values: the C (=1/λ) values to try

print("\n=== Optimal λ via cross-validation ===")

for l1_ratio, name in [
	(0,   'Ridge'),
	(1,   'Lasso'),
	(0.5, 'Elastic Net'),
]:
	model_cv = LogisticRegressionCV(
		l1_ratios=[l1_ratio],
		Cs=C_values,
		cv=5,
		solver='saga',
		max_iter=10000,
		tol=0.01
	)
	model_cv.fit(X_scaled, y)
	best_C = model_cv.C_[0]
	best_lambda = 1 / best_C
	print(f"\n{name}:")
	print(f"  Optimal λ = {best_lambda:.6f}")
	print(f"  Number of non-zero coefficients: {np.sum(model_cv.coef_[0] != 0)}")

# TASK 2: LASSO PERFORMANCE ON SIMULATED DATA

# Function to compute PSR and FDR
def compute_psr_fdr(selected, true_relevant):
	"""
	selected     : set of indices of variables selected by the model, e.g., {0, 2, 5}
	true_relevant: set of indices of truly relevant variables, e.g., {0,1,...,9}

	PSR (Positive Selection Rate):
		How many of the truly relevant variables did we find?
		Example: 7 out of 10 relevant variables found → PSR = 7/10 = 0.7
		PSR should be high (close to 1)

	FDR (False Discovery Rate):
		Among the selected variables, how many are actually irrelevant?
		Example: selected 10 variables, 3 are actually irrelevant → FDR = 3/10 = 0.3
		FDR should be low (close to 0)

	& operator: intersection of two sets (elements in both)
	- operator: set difference (elements in first but not in second)
	"""
	selected = set(selected)
	if len(selected) == 0:
		return 0.0, 0.0   # if no variable selected, PSR=0, FDR=0
	
	# |selected ∩ true_relevant| / |true_relevant|
	psr = len(selected & true_relevant) / len(true_relevant)
	
	# |selected - true_relevant| / |selected|
	fdr = len(selected - true_relevant) / len(selected)
	
	return psr, fdr

def run_lasso_simulation(n, n_irrelevant=10, L=20, probit=False):
	"""
	n           : number of observations (patients) to generate
				  larger n → more data → model learns better
	n_irrelevant: number of irrelevant (noise) variables
				  larger n_irrelevant → harder to find the relevant ones
	L           : number of repetitions
				  L=20 → generate 20 different random datasets, average results
	probit      : data generation model
				  False → logistic model: p = 1/(1+exp(-β^T x))
				  True  → probit model: p = Φ(β^T x)
				  Φ: CDF of standard normal distribution
				  To test what happens when we fit logistic regression to probit data
				  — model is mis‑specified, but how well does it perform?
	"""
	
	# Total number of variables: 10 relevant + n_irrelevant irrelevant
	p = 10 + n_irrelevant
	
	# True coefficients:
	# β = [1,1,1,1,1,1,1,1,1,1, 0,0,...,0]
	# First 10 variables are relevant (coefficient=1), the rest are irrelevant (coefficient=0)
	beta = np.array([1]*10 + [0]*n_irrelevant)
	
	# Indices of truly relevant variables: 0, 1, 2, ..., 9
	true_relevant = set(range(10))
	
	psr_list = []   # store PSR for each repetition
	fdr_list = []   # store FDR for each repetition
	
	for _ in range(L):   # repeat L times (_ : loop variable not used)
		
		# Generate data
		# np.random.randn(n, p): n×p matrix of standard normal
		# Each row is an observation (patient), each column a variable (gene)
		X_sim = np.random.randn(n, p)
		
		# Linear combination: β^T * x
		# X_sim @ beta: matrix-vector product → (n,) vector
		# For each observation: β1*x1 + β2*x2 + ... + βp*xp
		linear = X_sim @ beta
		
		if probit:
			# Probit model: p_i = Φ(β^T x)
			# norm.cdf: cumulative distribution function of standard normal
			# Φ(0) = 0.5, Φ(1) = 0.84, Φ(-1) = 0.16
			# Using probit instead of logistic makes the model "mis‑specified"
			# We test whether lasso can still find the variables
			probs = norm.cdf(linear)
		else:
			# Logistic model: p_i = 1/(1+exp(-β^T x))
			# sigmoid function
			probs = 1 / (1 + np.exp(-linear))
		
		# Bernoulli sampling: for each patient, y=1 with probability p_i, y=0 with probability 1-p_i
		# np.random.binomial(1, probs, n):
		#   1: single trial (Bernoulli)
		#   probs: success probability for each observation
		#   n: number of observations to generate
		y_sim = np.random.binomial(1, probs, n)
		
		# Fit lasso with cross-validation to select optimal λ
		# LogisticRegressionCV: tries different C values, selects best via cross-validation
		# l1_ratios=[1]: pure lasso (l1_ratio=1)
		# np.logspace(-3, 2, 30): 30 C values between 0.001 and 100
		model = LogisticRegressionCV(
			l1_ratios=[1],                      # lasso: pure L1 penalty
			solver='saga',
			Cs=np.logspace(-3, 2, 30),
			cv=5,
			max_iter=10000,
			tol=0.01
		)
		model.fit(X_sim, y_sim)
		
		# Indices of non-zero coefficients = variables selected by the model
		# model.coef_[0]: coefficient vector (p,)
		# np.where(...)[0]: indices where condition is true
		selected = set(np.where(model.coef_[0] != 0)[0])
		
		psr, fdr = compute_psr_fdr(selected, true_relevant)
		psr_list.append(psr)
		fdr_list.append(fdr)
	
	# np.mean: average over L repetitions
	return np.mean(psr_list), np.mean(fdr_list)

# PSR and FDR as a function of n (sample size)

# What do we expect as n increases?
# PSR should increase: more data → model finds true variables better
# FDR should decrease: more data → model selects fewer false variables

print("\n=== PSR and FDR vs n ===")
n_values = [50, 100, 300, 500, 1000, 2000]
psr_n = []
fdr_n = []

for n in n_values:
	psr, fdr = run_lasso_simulation(n=n, n_irrelevant=10, L=20)
	psr_n.append(psr)
	fdr_n.append(fdr)
	print(f"n={n:5d}: PSR={psr:.3f}, FDR={fdr:.3f}")

plt.figure(figsize=(10, 5))
plt.plot(n_values, psr_n, 'b-o', label='PSR (should be high)')
plt.plot(n_values, fdr_n, 'r-o', label='FDR (should be low)')
plt.xlabel('n — sample size')
plt.ylabel('Value (0 to 1)')
plt.title('PSR and FDR vs Sample Size\nExpected: PSR↑ and FDR↓ as n increases')
plt.legend()
plt.grid(True)
plt.show()

# PSR and FDR as a function of number of irrelevant variables

# Fix n=300, vary number of irrelevant variables (k)
# What do we expect as k increases?
# PSR may decrease: more noise drowns the signal
# FDR may increase: model selects more false variables

print("\n=== PSR and FDR vs number of irrelevant variables (n=300) ===")
irrelevant_values = [10, 50, 100, 200, 500]
psr_k = []
fdr_k = []

for k in irrelevant_values:
	psr, fdr = run_lasso_simulation(n=300, n_irrelevant=k, L=20)
	psr_k.append(psr)
	fdr_k.append(fdr)
	print(f"k={k:5d}: PSR={psr:.3f}, FDR={fdr:.3f}")

plt.figure(figsize=(10, 5))
plt.plot(irrelevant_values, psr_k, 'b-o', label='PSR')
plt.plot(irrelevant_values, fdr_k, 'r-o', label='FDR')
plt.xlabel('Number of irrelevant variables (k)')
plt.ylabel('Value')
plt.title('PSR and FDR vs Number of Irrelevant Variables\nModel struggles as k increases')
plt.legend()
plt.grid(True)
plt.show()

# LOGISTIC vs PROBIT COMPARISON

# Probit model: we generate data with p = Φ(β^T x)
# Then fit logistic regression with lasso (mis‑specified model)
# Question: can a wrong model still find the relevant variables?

print("\n=== Logistic vs Probit comparison (n=300, k=10) ===")

psr_log, fdr_log = run_lasso_simulation(
	n=300, n_irrelevant=10, L=20, probit=False   # correct model
)
psr_pro, fdr_pro = run_lasso_simulation(
	n=300, n_irrelevant=10, L=20, probit=True    # mis‑specified model (probit data, logistic fit)
)

print(f"Logistic (correct model) : PSR={psr_log:.3f}, FDR={fdr_log:.3f}")
print(f"Probit   (wrong model)   : PSR={psr_pro:.3f}, FDR={fdr_pro:.3f}")
print("\nInterpretation: If PSR and FDR differ substantially, lasso is sensitive to probit data.")
print("If they are similar, lasso is robust to model mis‑specification.")
