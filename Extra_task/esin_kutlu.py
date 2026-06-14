import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode, norm

def generate_dataset(n=5000, p=10, seed=None):
	rng = np.random.default_rng(seed)
	
	# 1. Generate X: (n, p) shape, U(0, pi)
	X = rng.uniform(low=0, high=np.pi, size=(n, p))
	
	# 2. Sum of sines of the first 4 columns
	'''
	1. Select first 4 columns with X[:, :4]
	2. Take np.sin(...)
	3. Sum along axis=1 (row-wise)
	'''
	S = np.sin(X[:, :4]).sum(axis=1)
	
	# 3. Create Y with threshold 2.5
	Y = (S >= 2.5).astype(int)
	
	return X, Y

X, Y = generate_dataset(n=5000, p=10, seed=42)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Y distribution:", np.bincount(Y))  # how many 0's and 1's

X_train, X_test, y_train, y_test = train_test_split(
	X, Y, train_size=1000, test_size=4000, random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Task 2 - RSM CLASS
class RSM:
	def __init__(self, B=100, m=5, max_depth=3):
		self.B = B            # number of classifiers (100)
		self.m = m            # number of features each classifier sees (5)
		self.max_depth = max_depth  # depth of each tree (3)
		self.classifiers = []       # store fitted trees
		self.feature_subsets = []   # store which features each tree sees
	
	def fit(self, X, y):
		p = X.shape[1]   # number of columns in X = number of features = 10
		for b in range(self.B):
			'''
			Iteration 1:
				From p=10 features, randomly select m=5, e.g., selected = [3, 7, 1, 9, 0]
				Train a tree using only these 5 columns: X[:, [3,7,1,9,0]] shape: (1000, 5)
				Store the tree and the selected features.
			'''
			selected = np.random.choice(p, size=self.m, replace=False)  # replace=False: do not select same feature twice
			'''
			There are p=10 numbers (0,...,9 — feature indices). Choose m=5 of them with replace=False.
			'''
			tree = DecisionTreeClassifier(max_depth=self.max_depth)
			tree.fit(X[:, selected], y)
			self.classifiers.append(tree)
			self.feature_subsets.append(selected)
		return self
	
	def predict(self, X):
		'''
		zip(self.classifiers, self.feature_subsets)
		[(tree_0, [3,7,1,9,0]),
		 (tree_1, [2,5,8,0,6]), ...
		 (tree_99, [4,1,6,3,8])]
		
		For each tree:
			clf.predict(X[:, subset])
			tree_0 with [3,7,1,9,0] predicts 4000 observations
			shape: (4000,)  i.e., 4000 values of 0 or 1
		
		Combining with np.array(...):
			predictions shape: (100, 4000)
			100 rows (each tree is a row)
			4000 columns (each test observation is a column)
		
				  obs_0   obs_1   obs_2   ...   obs_3999
		tree_0  [   1,      0,      1,    ...      1      ]
		tree_1  [   1,      1,      0,    ...      1      ]
		tree_2  [   0,      0,      1,    ...      0      ]
		...
		tree_99 [   1,      0,      1,    ...      1      ]
		
		For each observation (column-wise) we want the most frequent value.
		axis=0 : column-wise operation (find mode of the 100 values in each column)
		axis=1 : row-wise operation (wrong)
		'''
		predictions = np.array([
			clf.predict(X[:, subset])
			for clf, subset in zip(self.classifiers, self.feature_subsets)
		])
		final = mode(predictions, axis=0)[0]
		return final

# RSM
rsm = RSM(B=100, m=5, max_depth=3)
rsm.fit(X_train, y_train)
rsm_preds = rsm.predict(X_test)
rsm_acc = np.mean(rsm_preds == y_test)

# Single Tree
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
tree_acc = np.mean(tree_preds == y_test)

print(f"RSM Accuracy        : {rsm_acc:.4f}")
print(f"Single Tree Accuracy: {tree_acc:.4f}")

# TASK 3

n = 1000
np.random.seed(42)

# 1. x_i ~ U(0, 10)
x = np.random.uniform(0, 10, size=n)
# 2. True function f(x)
def f_true(x):
	# Weighted sum of two Gaussians — called a "mixture of Gaussians".
	# f(x) = 0.7 * N(mean=3, sd=1) + 0.3 * N(mean=6, sd=1)
	return 0.7 * norm.pdf(x, loc=3, scale=1) + 0.3 * norm.pdf(x, loc=6, scale=1)  # SMALL peak near x=6 (weight 30%)
	# LARGE peak near x=3 (weight 70%)

# 3. Add noise
epsilon = np.random.normal(loc=0, scale=0.05, size=n)
y = f_true(x) + epsilon  # Each observation scatters around f(x) with std ±0.05.

print("x shape:", x.shape)
print("y shape:", y.shape)
print("y sample values:", y[:5])

# Scatter Plot
x_grid = np.linspace(0, 10, 500)  # smooth grid for visualisation
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.3, s=10, color='steelblue', label='Data (xi, yi)')
plt.plot(x_grid, f_true(x_grid), 'k-', linewidth=2, label='Real f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real Function')
plt.legend()
plt.grid(True)
plt.show()

# Kernel Linear Regression
'''
What does Normal Linear Regression do?
    Fits a single straight line to all data: min sum[ (yi - theta0 - theta1*xi)^2 ] (global)

What does Kernel Linear Regression do?
    For each x0, it fits a SEPARATE linear regression
    But nearby points have larger influence, distant points have smaller influence

    min sum[ K((xi-x0)/h) * (yi - theta0 - theta1*xi)^2 ]
    >> points near x0 get large K weight; points far from x0 get small K weight
    >> prediction: f_hat(x0) = theta0 + theta1 * x0
    
Mathematical Solution
    This is a Weighted Least Squares (WLS) problem. Closed‑form solution exists:
    W = diag(w1, w2, ..., wn)   wi = K((xi - x0) / h)

    [theta0, theta1] = (A^T W A)^(-1) A^T W y

    where A = [[1, x1],
              [1, x2],
              ...
              [1, xn]]
'''

def gaussian_kernel(u):
	'''
	u = (xi - x0) / h
	u=0 (xi exactly at x0): kernel=1 (max weight)
	u large (xi far away): kernel → 0 (zero weight)
	h large: u small, even distant points influence
	h small: u large, only very close points influence
	'''
	return np.exp(-0.5 * u**2)

def kernel_linear_regression(x0, x_train, y_train, h):
	"""
	x0      : single point at which to predict
	x_train : training points (n,)
	y_train : training targets (n,)
	h       : bandwidth
	"""
	# 1. Compute kernel weights
	u = (x_train - x0) / h        # normalised distance of each xi to x0
	w = gaussian_kernel(u)        # shape: (n,)
	
	# 2. Build design matrix A
	# Each row: [1, xi]
	A = np.column_stack([np.ones(len(x_train)), x_train])  # A shape: (n, 2)
	
	# 3. W = diag(w) — weight matrix
	W = np.diag(w)                # shape: (n, n)
	
	# 4. WLS solution: (A^T W A)^(-1) A^T W y
	AtW  = A.T @ W                # shape: (2, n)
	AtWA = AtW @ A                # shape: (2, 2)
	AtWy = AtW @ y_train          # shape: (2,)
	
	theta = np.linalg.solve(AtWA, AtWy)
	# theta[0] = theta0 (intercept)
	# theta[1] = theta1 (slope)
	
	# 5. Prediction: f_hat(x0) = theta0 + theta1 * x0
	return theta[0] + theta[1] * x0

# Test on a single point
x0_test = 3.0
pred = kernel_linear_regression(x0_test, x, y, h=0.5)
true = f_true(x0_test)
print(f"x0={x0_test}: prediction={pred:.4f}, true={true:.4f}")

# Predict for all grid points
x_grid = np.arange(0, 10, 0.01)  # from 0 to 10 step 0.01

# Predictions for h=1, h=0.5, h=0.05
h_values = [1, 0.5, 0.05]

plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.3, s=10, color='steelblue', label='Data')
plt.plot(x_grid, f_true(x_grid), 'k-', linewidth=2, label='Real f(x)')

# Add prediction curve for each h
for h in h_values:
	preds = np.array([
		kernel_linear_regression(x0, x, y, h=h)
		for x0 in x_grid
	])
	plt.plot(x_grid, preds, linewidth=2, label=f'h={h}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Kernel Linear Regression — Different h values')
plt.legend()
plt.grid(True)
plt.show()

'''
Answers:
1.
h=1    : Underfitting
       : peak at x=3 is too low, peak at x=6 disappears
       : distant points have influence, high bias

h=0.5  : Balanced
       : closest to the true f(x)
       : bias and variance in balance

h=0.05 : Overfitting
       : follows every noise
       : large oscillations on the right (x>7)
       : high variance
       
2.
Nadaraya-Watson:
   f_hat(x0) = weighted AVERAGE of y_i
   : only predicts a constant value

Kernel Linear Regression:
   f_hat(x0) = theta0 + theta1 * x0
   : fits a separate LINEAR function at each x0
   : also estimates the slope
   : works better at the boundaries (x=0 and x=10)
     because it can extrapolate linearly

3.
Because each observation has a different weight:

Observations near x0 get large weight and strongly influence the prediction.
Observations far from x0 get small weight and have almost no influence.

Ordinary Least Squares (OLS) gives equal weight to all observations and fits a global straight line. With WLS we obtain a local straight line.
'''