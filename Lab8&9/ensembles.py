import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

try:
	import xgboost as xgb
	XGBOOST_AVAILABLE = True
except ImportError:
	XGBOOST_AVAILABLE = False
	print("WARNING: xgboost not installed. Run: pip install xgboost --break-system-packages")

# PART 1: ADABOOST — IMPLEMENTATION FROM SCRATCH

# AdaBoost (Adaptive Boosting) builds a sequence of "weak" classifiers,
# where each new classifier focuses more on the observations that
# previous classifiers got WRONG. The final prediction is a WEIGHTED
# VOTE of all classifiers, where more accurate classifiers get more vote.
#
# This implementation follows the classic AdaBoost formulation using
# the BETA parameterization (rather than the more common ALPHA = log((1-eps)/eps)
# parameterization). They are mathematically equivalent:
#   beta_k = eps_k / (1 - eps_k)
#   alpha_k = log(1/beta_k) = log((1-eps_k)/eps_k)
#
# Algorithm summary:
#
# TRAINING:
#   1. Initialize weights: w_i = 1/n for all i
#   2. For k = 1 to B:
#      a. Train classifier f_k using sample weights w
#      b. Compute weighted error: eps_k = sum_i [w_i * I(f_k(x_i) != y_i)]
#      c. Handle edge cases (eps_k = 0 or eps_k >= 0.5)
#      d. Compute beta_k = eps_k / (1 - eps_k)
#      e. UPDATE WEIGHTS: for CORRECTLY classified points, multiply weight by beta_k
#         (this DECREASES the weight of correctly classified points,
#          since beta_k < 1 when eps_k < 0.5)
#         INCORRECTLY classified points keep their weight unchanged
#         (relatively, their weight INCREASES after normalization)
#      f. Normalize weights to sum to 1
#
# PREDICTION:
#   y_hat(x) = argmax_y { sum_k [ I(f_k(x) = y) * log(1/beta_k) ] }
#   This is a WEIGHTED VOTE where each classifier's vote is weighted
#   by log(1/beta_k) — classifiers with SMALL eps_k (= small beta_k)
#   get LARGE weight log(1/beta_k).


class AdaBoostFromScratch:
	"""
	AdaBoost implementation using decision trees as base learners.

	Parameters:
		n_estimators : int, number of boosting rounds (B)
		max_depth    : int, max depth of each decision tree (controls
					   complexity of "weak" learners; depth=1 = decision stump)
	"""
	
	def __init__(self, n_estimators=50, max_depth=1):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.classifiers = []   # list of fitted DecisionTreeClassifier objects
		self.betas       = []   # corresponding beta_k values
		self.alphas      = []   # corresponding alpha_k = log(1/beta_k) — voting weights
		self.train_errors = []  # weighted training error at each step (diagnostic)
	
	def fit(self, X, y, verbose=False):
		"""
		Fit AdaBoost ensemble.

		Parameters:
			X : feature matrix, shape (n, p)
			y : labels in {-1, +1}  (IMPORTANT: AdaBoost classically uses
				{-1, +1} labels, not {0, 1}, because the prediction
				formula and many derivations assume +/-1 encoding.
				We will convert {0,1} -> {-1,+1} before calling fit.)
			verbose : if True, print diagnostic info at each step

		Returns:
			self
		"""
		n = len(y)
		
		# Step 1: initialize uniform weights
		# w_i = 1/n for all i — every observation starts with equal importance
		w = np.ones(n) / n
		
		self.classifiers = []
		self.betas = []
		self.alphas = []
		self.train_errors = []
		
		for k in range(self.n_estimators):
			
			# --- Step 2a: train classifier f_k using sample weights w ---
			# sample_weight parameter tells the tree to "pay more attention"
			# to observations with higher weight when choosing splits
			tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=k)
			tree.fit(X, y, sample_weight=w)
			
			y_pred = tree.predict(X)
			
			# --- Step 2b: compute weighted classification error ---
			# eps_k = sum_i [ w_i * I(f_k(x_i) != y_i) ]
			# This is the TOTAL WEIGHT of misclassified points.
			# If eps_k is small, the classifier is doing well on the
			# current (re-weighted) problem.
			incorrect = (y_pred != y)
			eps_k = np.sum(w[incorrect])
			
			# --- Step 2c: handle edge cases ---
			#
			# CASE 1: eps_k == 0 (perfect classifier on weighted data)
			#   -> beta_k = 0 / (1-0) = 0
			#   -> alpha_k = log(1/beta_k) = log(1/0) = +infinity
			#   -> This single classifier would get INFINITE voting weight,
			#      completely dominating (or breaking) the ensemble.
			#   -> HANDLING: we clip eps_k to a small positive value
			#      (e.g., eps_k = 1e-10) to avoid division by zero / log(inf).
			#      This gives the classifier a very large but FINITE weight.
			#      We also STOP adding more classifiers after this point,
			#      since a perfect classifier means the (re-weighted)
			#      training data is fully explained — further boosting
			#      rounds would be based on degenerate weights.
			#
			# CASE 2: eps_k >= 0.5 (classifier no better than random guessing,
			#         or worse)
			#   -> beta_k = eps_k/(1-eps_k) >= 1
			#   -> alpha_k = log(1/beta_k) <= 0  (zero or NEGATIVE voting weight)
			#   -> A negative weight means this classifier's votes would be
			#      SUBTRACTED, effectively flipping its predictions — this
			#      is not the intended behavior of AdaBoost and signals
			#      that boosting has failed to find a useful weak learner
			#      on this re-weighted distribution.
			#   -> HANDLING: we STOP the boosting process (do not add this
			#      classifier to the ensemble). This is the standard approach:
			#      once a weak learner cannot beat random guessing, further
			#      iterations are unlikely to help and may destabilize weights.
			
			eps_min = 1e-10  # numerical floor to avoid division by zero
			
			if eps_k <= 0:
				# Perfect classifier: clip and add it, then stop
				eps_k = eps_min
				beta_k = eps_k / (1 - eps_k)
				alpha_k = np.log(1 / beta_k)
				
				self.classifiers.append(tree)
				self.betas.append(beta_k)
				self.alphas.append(alpha_k)
				self.train_errors.append(eps_k)
				
				if verbose:
					print(f"  Iter {k+1}: eps_k=0 (perfect classifier). "
					      f"Clipping to {eps_min}, alpha_k={alpha_k:.3f}. STOPPING.")
				break
			
			if eps_k >= 0.5:
				# Classifier no better than random: discard and stop
				if verbose:
					print(f"  Iter {k+1}: eps_k={eps_k:.4f} >= 0.5 "
					      f"(no better than random). STOPPING boosting.")
				break
			
			# --- Step 2d: compute beta_k (and alpha_k for prediction) ---
			# beta_k = eps_k / (1 - eps_k)
			# Since 0 < eps_k < 0.5, we have 0 < beta_k < 1
			beta_k = eps_k / (1 - eps_k)
			
			# alpha_k = log(1/beta_k) = log((1-eps_k)/eps_k)
			# Since 0 < beta_k < 1, alpha_k > 0 (positive voting weight, good)
			# Smaller eps_k -> smaller beta_k -> LARGER alpha_k
			alpha_k = np.log(1 / beta_k)
			
			self.classifiers.append(tree)
			self.betas.append(beta_k)
			self.alphas.append(alpha_k)
			self.train_errors.append(eps_k)
			
			if verbose:
				print(f"  Iter {k+1}: eps_k={eps_k:.4f}, beta_k={beta_k:.4f}, "
				      f"alpha_k={alpha_k:.4f}")
			
			# --- Step 2e/2f: update weights ---
			# For CORRECTLY classified points: w_i = w_i * beta_k
			#   Since beta_k < 1, this DECREASES their weight —
			#   "easy" points become less important in the next round.
			# For INCORRECTLY classified points: weight unchanged
			#   (but after normalization, their RELATIVE weight increases,
			#    forcing the next classifier to focus on them)
			correct = ~incorrect
			w[correct] = w[correct] * beta_k
			
			# --- Step 2g: normalize weights to sum to 1 ---
			w = w / np.sum(w)
		
		return self
	
	def predict(self, X):
		"""
		Weighted majority vote prediction.

		y_hat(x) = argmax_y { sum_k [ I(f_k(x) = y) * alpha_k ] }

		For binary {-1, +1} classification, this simplifies to:
		y_hat(x) = sign( sum_k [ alpha_k * f_k(x) ] )

		Returns:
			predictions in {-1, +1}
		"""
		# Weighted sum of predictions (each f_k(x) is -1 or +1)
		n = X.shape[0]
		weighted_sum = np.zeros(n)
		
		for tree, alpha_k in zip(self.classifiers, self.alphas):
			weighted_sum += alpha_k * tree.predict(X)
		
		# sign() gives -1 or +1; treat exactly-0 as +1 (arbitrary tie-break)
		predictions = np.where(weighted_sum >= 0, 1, -1)
		return predictions
	
	def staged_predict(self, X):
		"""
		Generator yielding predictions using only the first k classifiers,
		for k = 1, 2, ..., n_estimators. Useful for plotting how
		train/test error evolves with the number of boosting rounds.
		"""
		n = X.shape[0]
		weighted_sum = np.zeros(n)
		
		for tree, alpha_k in zip(self.classifiers, self.alphas):
			weighted_sum += alpha_k * tree.predict(X)
			yield np.where(weighted_sum >= 0, 1, -1)

# PART 2: TEST ADABOOST ON A SIMPLE DATASET FIRST

print("="*70)
print("PART 1-2: ADABOOST FROM SCRATCH — BASIC TEST")
print("="*70)

# Quick sanity check on the breast cancer dataset
data = load_breast_cancer()
X_bc, y_bc01 = data.data, data.target   # y in {0, 1}

# Convert labels {0,1} -> {-1,+1} as required by our implementation
y_bc = np.where(y_bc01 == 0, -1, 1)

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
	X_bc, y_bc, test_size=0.3, random_state=42
)

print("\nFitting AdaBoost (depth=1 stumps) with verbose output (first 10 rounds):")
ada_test = AdaBoostFromScratch(n_estimators=10, max_depth=1)
ada_test.fit(X_train_bc, y_train_bc, verbose=True)

train_acc = np.mean(ada_test.predict(X_train_bc) == y_train_bc)
test_acc  = np.mean(ada_test.predict(X_test_bc)  == y_test_bc)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy : {test_acc:.4f}")


# PART 3: EFFECT OF BASE LEARNER DEPTH

print("\n" + "="*70)
print("PART 3: EFFECT OF BASE LEARNER DEPTH")
print("="*70)

# We compare AdaBoost with:
#   - depth=1 (decision stumps): the "classic" AdaBoost weak learner
#   - deeper trees (depth=3, depth=5): stronger individual learners
#
# Expected behavior:
#   - depth=1: each tree is a very weak learner (slightly better than random).
#     Boosting needs MANY iterations to build a strong ensemble.
#     Training is FAST per tree (stumps are trivial to fit).
#     Less prone to overfitting because each weak learner has high bias,
#     low variance — boosting primarily reduces BIAS.
#
#   - deeper trees (depth=5): each tree is already a reasonably strong
#     learner with low bias but higher variance.
#     Boosting on top of strong learners can lead to OVERFITTING
#     more quickly (the ensemble can memorize training data).
#     Training is SLOWER per tree (deeper trees take longer to fit).

depths = [1, 3, 5]
n_estimators_compare = 100

results_depth = {}

for depth in depths:
	start_time = time.time()
	ada = AdaBoostFromScratch(n_estimators=n_estimators_compare, max_depth=depth)
	ada.fit(X_train_bc, y_train_bc)
	elapsed = time.time() - start_time
	
	# Compute staged train/test errors
	train_errors_staged = []
	test_errors_staged  = []
	for pred_train, pred_test in zip(ada.staged_predict(X_train_bc),
	                                 ada.staged_predict(X_test_bc)):
		train_errors_staged.append(np.mean(pred_train != y_train_bc))
		test_errors_staged.append(np.mean(pred_test != y_test_bc))
	
	results_depth[depth] = {
		'train_errors': train_errors_staged,
		'test_errors':  test_errors_staged,
		'time': elapsed,
		'n_actual_estimators': len(ada.classifiers)
	}
	
	print(f"\nDepth={depth}:")
	print(f"  Training time         : {elapsed:.3f} seconds")
	print(f"  Estimators actually used: {len(ada.classifiers)} "
	      f"(requested {n_estimators_compare})")
	print(f"  Final train error     : {train_errors_staged[-1]:.4f}")
	print(f"  Final test error      : {test_errors_staged[-1]:.4f}")

# Plot
fig, axes = plt.subplots(1, len(depths), figsize=(5*len(depths), 4.5))
for ax, depth in zip(axes, depths):
	res = results_depth[depth]
	n_iter = len(res['train_errors'])
	ax.plot(range(1, n_iter+1), res['train_errors'], 'b-', label='Train error')
	ax.plot(range(1, n_iter+1), res['test_errors'],  'r-', label='Test error')
	ax.set_title(f'Depth={depth}\nTime={res["time"]:.2f}s, '
	             f'Final test err={res["test_errors"][-1]:.4f}')
	ax.set_xlabel('Boosting iteration')
	ax.set_ylabel('Error rate')
	ax.legend()
	ax.grid(True)

plt.suptitle('AdaBoost: Effect of Base Learner Depth', fontsize=14)
plt.tight_layout()
plt.show()


# PART 4: ARTIFICIAL DATASET (CHI-SQUARED)

print("\n" + "="*70)
print("PART 4: ARTIFICIAL CHI-SQUARED DATASET")
print("="*70)

# Generate the artificial dataset described in the task:
#   X1, ..., X10 ~ N(0,1) independently
#   chi2_10(0.5) = median of chi-squared distribution with 10 df
#   Y = +1 if sum(X_j^2) > chi2_10(0.5), else Y = -1
#
# This creates a BAYES-OPTIMAL decision boundary that is a HYPERSPHERE
# (in 10 dimensions) centered at the origin:
#   sum(x_j^2) = chi2_10(0.5)  <=>  ||x|| = sqrt(chi2_10(0.5))
# Points INSIDE the sphere -> Y=-1, points OUTSIDE -> Y=+1

def generate_chi2_dataset(n, p=10, seed=None):
	"""
	Generate the chi-squared artificial dataset.

	Parameters:
		n    : number of observations
		p    : number of features (default 10, matching chi2 df=10)
		seed : random seed

	Returns:
		X : feature matrix, shape (n, p), each column ~ N(0,1)
		y : labels in {-1, +1}
	"""
	rng = np.random.default_rng(seed)
	X = rng.standard_normal(size=(n, p))
	
	# chi-squared median with df=10
	threshold = chi2.ppf(0.5, df=10)
	
	# sum of squares for each row
	sum_sq = np.sum(X**2, axis=1)
	
	y = np.where(sum_sq > threshold, 1, -1)
	return X, y

# Generate training (n=2000) and test (n=10000) sets
X_train_art, y_train_art = generate_chi2_dataset(2000,  seed=1)
X_test_art,  y_test_art  = generate_chi2_dataset(10000, seed=2)

print(f"\nArtificial dataset:")
print(f"  Training size: {X_train_art.shape}")
print(f"  Test size    : {X_test_art.shape}")
print(f"  chi2_10(0.5) threshold = {chi2.ppf(0.5, df=10):.4f}")
print(f"  Train class balance: Y=+1: {np.mean(y_train_art==1):.3f}, "
      f"Y=-1: {np.mean(y_train_art==-1):.3f}")
print(f"  Test class balance : Y=+1: {np.mean(y_test_art==1):.3f}, "
      f"Y=-1: {np.mean(y_test_art==-1):.3f}")


# PART 5: COMPARE ALL ENSEMBLE METHODS — REAL DATASET (breast cancer)

print("\n" + "="*70)
print("PART 5: METHOD COMPARISON — REAL DATASET (Breast Cancer)")
print("="*70)

# We compare 6 methods:
#   1. Single Tree
#   2. Bagging
#   3. AdaBoost (our implementation)
#   4. Gradient Boosting
#   5. XGBoost (if available)
#   6. Random Forest
#
# All methods are compared on the SAME train/test split for fairness.

n_estimators_final = 150

def run_comparison(X_train, y_train_pm1, X_test, y_test_pm1, dataset_name):
	"""
	Run all 6 ensemble methods on the given dataset and return
	staged train/test errors for plotting, plus final accuracies.

	y_train_pm1, y_test_pm1: labels in {-1, +1} (for AdaBoost compatibility)
	For sklearn methods, we also create {0,1} versions.
	"""
	# sklearn's GradientBoostingClassifier, RandomForest etc. work fine
	# with {-1,+1} labels too (they treat them as two distinct classes),
	# but to be safe and conventional, create {0,1} versions as well
	y_train_01 = np.where(y_train_pm1 == -1, 0, 1)
	y_test_01  = np.where(y_test_pm1  == -1, 0, 1)
	
	results = {}
	
	# --- 1. Single Tree ---
	tree = DecisionTreeClassifier(max_depth=5, random_state=42)
	tree.fit(X_train, y_train_01)
	results['Single Tree'] = {
		'train_acc': tree.score(X_train, y_train_01),
		'test_acc':  tree.score(X_test,  y_test_01),
	}
	
	# --- 2. Bagging ---
	# n_estimators trees, each on a bootstrap sample
	bagging = BaggingClassifier(
		estimator=DecisionTreeClassifier(max_depth=5),
		n_estimators=n_estimators_final,
		random_state=42
	)
	bagging.fit(X_train, y_train_01)
	results['Bagging'] = {
		'train_acc': bagging.score(X_train, y_train_01),
		'test_acc':  bagging.score(X_test,  y_test_01),
	}
	# Staged errors for bagging: refit incrementally with increasing n_estimators
	bagging_train_err, bagging_test_err = [], []
	for n_est in range(1, n_estimators_final+1, 5):
		bg = BaggingClassifier(
			estimator=DecisionTreeClassifier(max_depth=5),
			n_estimators=n_est, random_state=42
		)
		bg.fit(X_train, y_train_01)
		bagging_train_err.append(1 - bg.score(X_train, y_train_01))
		bagging_test_err.append(1 - bg.score(X_test, y_test_01))
	results['Bagging']['staged_train'] = bagging_train_err
	results['Bagging']['staged_test']  = bagging_test_err
	
	# --- 3. AdaBoost (our implementation, depth=1 stumps — classic choice) ---
	ada = AdaBoostFromScratch(n_estimators=n_estimators_final, max_depth=1)
	ada.fit(X_train, y_train_pm1)
	
	ada_train_err, ada_test_err = [], []
	for pred_train, pred_test in zip(ada.staged_predict(X_train),
	                                 ada.staged_predict(X_test)):
		ada_train_err.append(np.mean(pred_train != y_train_pm1))
		ada_test_err.append(np.mean(pred_test != y_test_pm1))
	
	results['AdaBoost (scratch)'] = {
		'train_acc': 1 - ada_train_err[-1],
		'test_acc':  1 - ada_test_err[-1],
		'staged_train': ada_train_err,
		'staged_test':  ada_test_err,
	}
	
	# --- 4. Gradient Boosting ---
	gb = GradientBoostingClassifier(n_estimators=n_estimators_final,
	                                max_depth=3, learning_rate=0.1,
	                                random_state=42)
	gb.fit(X_train, y_train_01)
	results['Gradient Boosting'] = {
		'train_acc': gb.score(X_train, y_train_01),
		'test_acc':  gb.score(X_test,  y_test_01),
	}
	# Staged predictions available natively via staged_predict
	gb_train_err = [1 - np.mean(p == y_train_01)
	                for p in gb.staged_predict(X_train)]
	gb_test_err  = [1 - np.mean(p == y_test_01)
	                for p in gb.staged_predict(X_test)]
	results['Gradient Boosting']['staged_train'] = gb_train_err
	results['Gradient Boosting']['staged_test']  = gb_test_err
	
	# --- 5. XGBoost ---
	if XGBOOST_AVAILABLE:
		xgb_model = xgb.XGBClassifier(
			n_estimators=n_estimators_final, max_depth=3,
			learning_rate=0.1, eval_metric='logloss',
			random_state=42
		)
		xgb_model.fit(X_train, y_train_01)
		results['XGBoost'] = {
			'train_acc': xgb_model.score(X_train, y_train_01),
			'test_acc':  xgb_model.score(X_test,  y_test_01),
		}
		# XGBoost staged predictions via iteration_range
		xgb_train_err, xgb_test_err = [], []
		for n_est in range(1, n_estimators_final+1, 5):
			preds_tr = xgb_model.predict(X_train, iteration_range=(0, n_est))
			preds_te = xgb_model.predict(X_test,  iteration_range=(0, n_est))
			xgb_train_err.append(1 - np.mean(preds_tr == y_train_01))
			xgb_test_err.append(1 - np.mean(preds_te == y_test_01))
		results['XGBoost']['staged_train'] = xgb_train_err
		results['XGBoost']['staged_test']  = xgb_test_err
	else:
		results['XGBoost'] = None
	
	# --- 6. Random Forest ---
	rf = RandomForestClassifier(n_estimators=n_estimators_final,
	                            max_depth=5, random_state=42)
	rf.fit(X_train, y_train_01)
	results['Random Forest'] = {
		'train_acc': rf.score(X_train, y_train_01),
		'test_acc':  rf.score(X_test,  y_test_01),
	}
	# Staged errors for RF: refit with increasing n_estimators
	rf_train_err, rf_test_err = [], []
	for n_est in range(1, n_estimators_final+1, 5):
		rf_temp = RandomForestClassifier(n_estimators=n_est, max_depth=5,
		                                 random_state=42)
		rf_temp.fit(X_train, y_train_01)
		rf_train_err.append(1 - rf_temp.score(X_train, y_train_01))
		rf_test_err.append(1 - rf_temp.score(X_test, y_test_01))
	results['Random Forest']['staged_train'] = rf_train_err
	results['Random Forest']['staged_test']  = rf_test_err
	
	# --- Print summary table ---
	print(f"\n--- {dataset_name}: Final Accuracies ---")
	print(f"{'Method':<22} | {'Train Acc':>10} | {'Test Acc':>10} | "
	      f"{'Train Err':>10} | {'Test Err':>10}")
	print("-" * 70)
	for name, res in results.items():
		if res is None:
			print(f"{name:<22} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
			continue
		print(f"{name:<22} | {res['train_acc']:>10.4f} | {res['test_acc']:>10.4f} | "
		      f"{1-res['train_acc']:>10.4f} | {1-res['test_acc']:>10.4f}")
	
	return results

results_real = run_comparison(X_train_bc, y_train_bc, X_test_bc, y_test_bc,
                              "Breast Cancer (Real Dataset)")


# PART 6: COMPARE ALL ENSEMBLE METHODS — ARTIFICIAL CHI-SQUARED DATASET

print("\n" + "="*70)
print("PART 6: METHOD COMPARISON — ARTIFICIAL DATASET (Chi-squared)")
print("="*70)

results_art = run_comparison(X_train_art, y_train_art, X_test_art, y_test_art,
                             "Chi-squared (Artificial Dataset)")

# PART 7: PLOTS — TRAIN/TEST ERROR vs NUMBER OF ITERATIONS

def plot_staged_errors(results, dataset_name):
	methods_with_staged = {
		name: res for name, res in results.items()
		if res is not None and 'staged_train' in res
	}
	
	fig, axes = plt.subplots(1, len(methods_with_staged),
	                         figsize=(5*len(methods_with_staged), 4.5))
	if len(methods_with_staged) == 1:
		axes = [axes]
	
	for ax, (name, res) in zip(axes, methods_with_staged.items()):
		n_points = len(res['staged_train'])
		
		# x-axis: always just use 1..n_points, with appropriate labeling
		if 'AdaBoost' in name:
			x_vals = range(1, n_points + 1)
			x_label = 'Number of trees/iterations'
		elif name in ('Gradient Boosting', 'XGBoost'):
			# staged_predict yields one value per estimator (1..n_estimators_final)
			x_vals = range(1, n_points + 1)
			x_label = 'Number of trees/iterations'
		else:
			# Bagging, Random Forest: stepped by 5
			x_vals = range(1, n_estimators_final + 1, 5)[:n_points]
			x_label = 'Number of trees/iterations'
		
		ax.plot(x_vals, res['staged_train'], 'b-', label='Train error')
		ax.plot(x_vals, res['staged_test'],  'r-', label='Test error')
		ax.set_title(name)
		ax.set_xlabel(x_label)
		ax.set_ylabel('Error rate')
		ax.legend()
		ax.grid(True)
	
	plt.suptitle(f'Train/Test Error vs Iterations — {dataset_name}', fontsize=14)
	plt.tight_layout()
	plt.show()

# PART 8: VOTING WEIGHT ANALYSIS (Theoretical Question B)

print("\n" + "="*70)
print("PART 8: VOTING WEIGHT ANALYSIS")
print("="*70)

# Question: 999 classifiers have eps_k = 0.4, one classifier has
# eps_1000 = 1e-8. Compute voting weights and discuss dominance.

eps_typical = 0.4
eps_excellent = 1e-8

beta_typical   = eps_typical / (1 - eps_typical)
alpha_typical  = np.log(1 / beta_typical)

beta_excellent  = eps_excellent / (1 - eps_excellent)
alpha_excellent = np.log(1 / beta_excellent)

print(f"\nFor eps_k = {eps_typical} (999 classifiers):")
print(f"  beta_k  = {beta_typical:.6f}")
print(f"  alpha_k = log(1/beta_k) = {alpha_typical:.6f}")

print(f"\nFor eps_k = {eps_excellent} (1 classifier):")
print(f"  beta_k  = {beta_excellent:.2e}")
print(f"  alpha_k = log(1/beta_k) = {alpha_excellent:.6f}")

total_typical_weight = 999 * alpha_typical
print(f"\nTotal weight of 999 typical classifiers: {total_typical_weight:.4f}")
print(f"Weight of 1 excellent classifier        : {alpha_excellent:.4f}")
print(f"Ratio (excellent / total typical)       : "
      f"{alpha_excellent/total_typical_weight:.6f}")
print(f"\n=> The single excellent classifier's weight ({alpha_excellent:.2f}) is "
      f"much SMALLER\n   than the combined weight of the 999 typical "
      f"classifiers ({total_typical_weight:.2f}).\n"
      f"=> It does NOT dominate the final vote by itself.")

# QUESTIONS AND ANSWERS

print("""
=================================================================
QUESTIONS AND ANSWERS — ENSEMBLE METHODS (Labs 8-9)
=================================================================

-----------------------------------------------------------------
Q1. What should happen when eps_k = 0 or eps_k >= 0.5 during
    AdaBoost training? Why are these cases problematic?
-----------------------------------------------------------------
A: CASE eps_k = 0 (PERFECT classifier on the current weighted data):

   Mathematically:
     beta_k = eps_k / (1-eps_k) = 0 / 1 = 0
     alpha_k = log(1/beta_k) = log(1/0) = +infinity

   Why problematic?
   -> An infinite voting weight means this ONE classifier would
      completely override all others in the final weighted vote,
      regardless of what they predict.
   -> Numerically, log(1/0) is undefined (division by zero, then
      log of infinity) -- this causes a crash or NaN in code.
   -> Conceptually, a "perfect" classifier on a re-weighted training
      set might just mean the weighted distribution became trivial
      (e.g., one class has near-zero total weight), not that the
      classifier generalizes perfectly.

   OUR HANDLING:
   -> Clip eps_k to a tiny positive value (e.g., 1e-10) so beta_k
      and alpha_k remain finite (but very large).
   -> STOP the boosting process after adding this classifier --
      since the (re-weighted) training data is already perfectly
      fit, continuing would operate on a degenerate weight
      distribution (all remaining weight is on already-correct points).


   CASE eps_k >= 0.5 (classifier no better than, or worse than,
   random guessing):

   Mathematically:
     beta_k = eps_k/(1-eps_k) >= 1   (since eps_k >= 0.5)
     alpha_k = log(1/beta_k) <= 0    (zero or NEGATIVE)

   Why problematic?
   -> A weight of zero means this classifier contributes nothing
      (acceptable, though wasteful).
   -> A NEGATIVE weight means the classifier's predictions get
      SUBTRACTED from the vote -- effectively, AdaBoost would be
      using the OPPOSITE of what this classifier predicts.
      While mathematically this can sometimes still work
      (a classifier worse than random IS informative -- its
      complement is better than random), the classic AdaBoost
      derivation and convergence guarantees assume eps_k < 0.5
      strictly. Allowing eps_k >= 0.5 breaks these guarantees and
      the weight update rule (multiplying correct-point weights by
      beta_k >= 1 would INCREASE rather than decrease their
      relative importance -- the opposite of AdaBoost's intended
      behavior).

   OUR HANDLING:
   -> STOP the boosting process (do not add this classifier).
      This is the standard, conservative choice: once a weak
      learner cannot beat 50% weighted error, the algorithm
      has reached its limit on this dataset/weight distribution.

-----------------------------------------------------------------
Q2. How does the depth of the base learner affect AdaBoost's
    results and training time?
-----------------------------------------------------------------
A: DEPTH = 1 (decision stumps):
   -> Each tree makes a decision based on a SINGLE feature and
      a SINGLE threshold -- extremely simple, "weak" learner.
   -> Individually, a stump has HIGH BIAS (cannot capture complex
      patterns) but LOW VARIANCE (very stable, simple model).
   -> AdaBoost combines MANY stumps, each correcting errors of
      previous ones. This is primarily a BIAS REDUCTION technique
      when using stumps -- the ensemble can represent much more
      complex functions than any single stump.
   -> Training is VERY FAST per tree (finding the best single split
      is cheap), but MANY iterations (e.g., 100-500) may be needed
      for good performance.
   -> Generally more resistant to overfitting because each addition
      is a small, controlled update.

   DEEPER TREES (e.g., depth=5):
   -> Each tree is already a fairly strong learner -- lower bias,
      higher variance individually.
   -> Boosting on top of strong learners can lead to FASTER
      convergence (fewer iterations needed to fit training data well)
      but also FASTER OVERFITTING -- the ensemble can start fitting
      noise in the training data after just a few rounds.
   -> Training is SLOWER per tree (finding optimal splits at
      multiple levels is more expensive).
   -> In our experiments: depth=1 typically shows a slow, steady
      decrease in both train and test error over many iterations.
      depth=5 typically shows training error dropping to ~0 very
      quickly, while test error may start INCREASING after some
      point (overfitting).

   PRACTICAL RECOMMENDATION:
   -> Classic AdaBoost theory and practice favor shallow trees
      (stumps or depth 2-3) as base learners.
   -> If using deeper trees, fewer boosting iterations and/or
      additional regularization (e.g., shrinkage/learning rate,
      as in Gradient Boosting) become necessary.

-----------------------------------------------------------------
Q3. Why do we want a classifier with smaller eps_k to receive
    a LARGER weight in the final prediction?
-----------------------------------------------------------------
A: The voting weight is alpha_k = log((1-eps_k)/eps_k).

   This is a DECREASING function of eps_k:
   -> eps_k -> 0   :  alpha_k -> +infinity (huge weight)
   -> eps_k = 0.5  :  alpha_k = log(1) = 0 (no weight -- useless,
                       since 50% error = random guessing for
                       binary classification)
   -> eps_k -> 1   :  alpha_k -> -infinity (would need negative
                       weight, i.e., "always believe the opposite")

   INTUITION:
   -> A classifier with eps_k = 0.05 (5% error) is making CORRECT
      predictions 95% of the time -- it carries a lot of reliable
      information and should heavily influence the final decision.
   -> A classifier with eps_k = 0.45 (45% error) is barely better
      than a coin flip -- it carries very little information and
      should have minimal influence.
   -> If we gave EQUAL weight to both, the noisy/weak classifier
      would "drown out" useful signal from the accurate one,
      especially when many weak classifiers vote together.

   This weighting scheme is what allows AdaBoost to combine many
   "weak" learners (each only slightly better than random) into
   a single "strong" learner: the WEIGHTED combination amplifies
   the signal from better classifiers while still incorporating
   the (smaller) useful signal from weaker ones.

-----------------------------------------------------------------
Q4. 999 classifiers have eps_k = 0.4, one classifier has
    eps_1000 = 1e-8. Compute the voting weights. Does the
    almost-perfect classifier dominate the final decision?
-----------------------------------------------------------------
A: For eps_k = 0.4:
     beta_k  = 0.4 / 0.6 = 0.6667
     alpha_k = log(1/0.6667) = log(1.5) ~= 0.4055

   For eps_k = 1e-8:
     beta_k  = 1e-8 / (1 - 1e-8) ~= 1e-8
     alpha_k = log(1/1e-8) = log(1e8) ~= 18.42

   Total weight from the 999 "typical" classifiers:
     999 * 0.4055 ~= 405.1

   Weight from the single near-perfect classifier:
     ~= 18.42

   COMPARISON:
     18.42  vs  405.1
   The near-perfect classifier's weight (~18.4) is roughly
   22 TIMES SMALLER than the combined weight of the 999 typical
   classifiers (~405).

   CONCLUSION:
   -> NO, the single near-perfect classifier does NOT dominate
      the final decision by itself.
   -> Its weight, while large in ABSOLUTE terms compared to any
      INDIVIDUAL typical classifier (18.4 vs 0.4, i.e., ~45x
      larger per-classifier), is small relative to the SUM of
      999 typical classifiers.
   -> If the 999 typical classifiers happen to AGREE with each
      other (and disagree with the near-perfect one) on a
      particular point, their combined weight (405) would
      OVERRULE the near-perfect classifier's weight (18.4).
   -> However, if the 999 typical classifiers' votes are roughly
      evenly SPLIT (since eps_k=0.4 means each is correct 60% of
      the time but their errors may be somewhat independent),
      the near-perfect classifier could still play a meaningful
      "tie-breaking" role on points where the majority is uncertain.

   GENERAL LESSON:
   -> AdaBoost's logarithmic weighting means even a perfect
      classifier gets a FINITE, bounded weight -- it cannot
      single-handedly overrule a large ensemble of weak-but-many
      classifiers. The voting weight grows only LOGARITHMICALLY
      as eps_k -> 0, not without bound in a way that would let
      one classifier dictate everything regardless of ensemble size.

-----------------------------------------------------------------
Q5. Why are very simple base classifiers, such as decision
    stumps, often sufficient in AdaBoost?
-----------------------------------------------------------------
A: Several complementary reasons:

   1. BOOSTING IS AN ADDITIVE MODEL:
      The final AdaBoost classifier is a weighted SUM of many
      simple functions:
        F(x) = sum_k alpha_k * f_k(x)
      Even if each f_k (a stump) is simple -- a step function in
      ONE dimension -- the SUM of many such step functions across
      different dimensions and thresholds can approximate VERY
      COMPLEX, smooth decision boundaries. This is similar to how
      a Fourier series builds complex functions from simple
      sine/cosine terms.

   2. BIAS-VARIANCE PERSPECTIVE:
      A single stump has high bias (oversimplified) but very low
      variance (extremely stable -- there's not much that CAN vary).
      Boosting primarily reduces BIAS by sequentially correcting
      errors. Combining low-variance, high-bias learners through
      boosting is a particularly EFFECTIVE bias-reduction strategy,
      because the ensemble's variance stays controlled even after
      many additions.

   3. AVOIDS OVERFITTING:
      Each stump can only make a tiny, localized correction.
      It takes MANY iterations to overfit, giving more "room" to
      monitor validation/test performance and stop early if needed.
      Deeper trees as base learners can overfit in just a handful
      of iterations.

   4. COMPUTATIONAL EFFICIENCY:
      Finding the best stump (single feature, single threshold) is
      computationally CHEAP -- O(n*p) roughly, versus exponentially
      more for deep trees with many possible split combinations.
      This allows running MANY boosting iterations quickly.

   5. THEORETICAL GUARANTEES:
      The original AdaBoost convergence proofs (related to the
      "weak learning" assumption from PAC learning theory) only
      require each base learner to be SLIGHTLY better than random
      guessing (eps_k < 0.5). A decision stump on most real datasets
      easily satisfies this -- it doesn't need to be a strong
      learner on its own.

-----------------------------------------------------------------
Q6. Why do we expect Y to be balanced in the artificial
    chi-squared dataset?
-----------------------------------------------------------------
A: By construction:
     X_1, ..., X_10 ~ N(0,1) independently
     S = sum_{j=1}^{10} X_j^2  ~  chi-squared distribution with 10 df
     Y = +1 if S > chi2_10(0.5), else Y = -1

   chi2_10(0.5) is defined as the MEDIAN of the chi-squared(10)
   distribution -- i.e., the value m such that:
     P(S <= m) = 0.5  and  P(S > m) = 0.5

   Therefore, by the very definition of the median:
     P(Y = +1) = P(S > chi2_10(0.5)) = 0.5
     P(Y = -1) = P(S <= chi2_10(0.5)) = 0.5

   The dataset is balanced EXACTLY because we chose the threshold
   to be the median of the distribution of S -- this guarantees a
   50/50 split in expectation (and approximately 50/50 in any
   finite sample, with sampling variability).

   Note: this would NOT hold if we used, e.g., the MEAN of the
   chi-squared(10) distribution (which equals 10, but the
   chi-squared distribution is right-skewed, so the mean is
   ABOVE the median -- using the mean as threshold would give
   P(S > mean) < 0.5, an IMBALANCED dataset).

-----------------------------------------------------------------
Q7. Describe the shape of the Bayes decision boundary for the
    artificial dataset. Which methods are better suited to
    approximate it, and why?
-----------------------------------------------------------------
A: SHAPE OF THE BAYES BOUNDARY:

   The decision rule is:
     Y = +1  if  sum_{j=1}^{10} X_j^2 > chi2_10(0.5)
     Y = -1  otherwise

   The boundary sum_{j=1}^{10} x_j^2 = chi2_10(0.5) is a
   HYPERSPHERE (10-dimensional sphere) centered at the ORIGIN,
   with radius r = sqrt(chi2_10(0.5)).
   -> Points INSIDE the sphere (close to origin) -> Y = -1
   -> Points OUTSIDE the sphere (far from origin) -> Y = +1

   This boundary is:
   -> RADIALLY SYMMETRIC (depends only on distance from origin,
      not direction)
   -> SMOOTH and CURVED (a sphere, not flat)
   -> Cannot be represented by ANY single linear function of the
      original features X_1, ..., X_10 (it's fundamentally a
      QUADRATIC function: sum of squares).

   WHICH METHODS ARE BETTER SUITED?

   POORLY suited:
   -> LINEAR methods (logistic regression, linear SVM, LDA):
      cannot represent a spherical boundary at all -- they can only
      produce hyperplane (flat) boundaries. Expected accuracy
      close to 50% (random) unless features are transformed.
   -> Single shallow decision trees: trees create AXIS-ALIGNED
      RECTANGULAR partitions. Approximating a smooth sphere with
      axis-aligned boxes requires MANY splits and is inefficient,
      though with enough depth/many trees it can approximate it.

   WELL suited:
   -> RANDOM FOREST and BAGGING: ensembles of trees can approximate
      smooth curved boundaries by averaging many axis-aligned
      partitions, effectively "rounding off" the corners. With
      enough trees, the spherical boundary can be approximated
      reasonably well.
   -> BOOSTING methods (AdaBoost, Gradient Boosting, XGBoost):
      similarly build up complex boundaries additively. Since the
      true function depends on sum_j X_j^2 (a sum over all 10
      features), boosting with trees of depth >= 2 (which can
      capture some interaction/curvature per tree) tends to perform
      well, building up the spherical shape through many
      "stacked" partial corrections.
   -> KERNEL METHODS (SVM with RBF kernel) or QDA: can directly
      represent a quadratic/spherical boundary very naturally
      (QDA explicitly fits a quadratic decision boundary; RBF
      kernels measure distance from points, naturally suited to
      radially symmetric problems).

   SUMMARY: Tree-ensemble methods (Random Forest, Gradient Boosting,
   XGBoost, AdaBoost with depth>=2) should outperform single trees
   and linear methods, because they can flexibly approximate the
   smooth, curved (spherical) Bayes boundary through aggregation
   of many simple pieces.

-----------------------------------------------------------------
Q8. For each method, does the improvement over a single tree
    come mainly from variance reduction, bias reduction, or both?
-----------------------------------------------------------------
A: SINGLE TREE (baseline):
   -> High variance (small changes in data -> very different tree).
   -> Bias depends on depth: shallow = high bias, deep = low bias
      but then high variance (classic tradeoff).

   BAGGING:
   -> Mainly VARIANCE REDUCTION.
   -> Averaging many trees, each trained on a bootstrap sample,
      reduces the variance of the ensemble (Var(mean) < Var(individual)
      when trees are not perfectly correlated).
   -> Bias is roughly UNCHANGED -- each individual tree still has
      the same bias as before; averaging doesn't fix systematic
      errors shared by all trees.

   RANDOM FOREST:
   -> Mainly VARIANCE REDUCTION, but MORE EFFECTIVE than Bagging.
   -> In addition to bootstrap sampling, RF randomly selects a
      subset of features at each split. This DECORRELATES the
      trees further (two trees are less likely to make the same
      "mistake" from the same dominant feature).
   -> Lower correlation between trees -> averaging reduces variance
      MORE than in Bagging, for the same number of trees.
   -> Bias is again roughly unchanged or slightly increased
      (restricting feature choices at each split can slightly
      hurt individual tree quality, but the variance reduction
      more than compensates).

   ADABOOST:
   -> Mainly BIAS REDUCTION (especially with stumps).
   -> Each new weak learner specifically targets the RESIDUAL
      ERRORS (misclassified/high-weight points) of the current
      ensemble -- this directly attacks systematic bias.
   -> Variance can actually INCREASE somewhat with very many
      iterations (risk of overfitting), but the early/moderate
      iterations primarily reduce bias dramatically (single stump
      ~50% error -> ensemble can reach very low error).

   GRADIENT BOOSTING / XGBOOST:
   -> Primarily BIAS REDUCTION, similar mechanism to AdaBoost but
      using gradient descent on a loss function instead of
      reweighting.
   -> Each tree fits the "gradient" (residuals) of the current
      ensemble's predictions -- directly reduces systematic error.
   -> Includes explicit regularization (learning rate, tree depth,
      L1/L2 penalties in XGBoost) to control variance/overfitting,
      so the method achieves LOW BIAS while keeping variance
      manageable through these controls -- arguably BOTH bias and
      variance are addressed, but the core boosting mechanism is
      bias-driven.

   SUMMARY TABLE:
   +--------------------+-----------------+-----------------------+
   | Method             | Primary Effect  | Mechanism             |
   +--------------------+-----------------+-----------------------+
   | Bagging            | Variance down   | Averaging bootstraps  |
   | Random Forest      | Variance down++ | Averaging + decorrel. |
   | AdaBoost           | Bias down       | Reweighting errors    |
   | Gradient Boosting  | Bias down (+    | Fitting residuals,    |
   | / XGBoost          | variance ctrl)  | + regularization      |
   +--------------------+-----------------+-----------------------+

-----------------------------------------------------------------
Q9. Why may Random Forest outperform Bagging, although both
    methods average many trees?
-----------------------------------------------------------------
A: Both Bagging and Random Forest build B trees on bootstrap
   samples of the training data and average their predictions
   (majority vote for classification).

   The KEY DIFFERENCE: Random Forest adds an EXTRA layer of
   randomness -- at EACH SPLIT in each tree, only a RANDOM SUBSET
   of features (e.g., sqrt(p) out of p features) is considered as
   candidates for splitting. Bagging considers ALL features at
   every split.

   WHY THIS MATTERS -- VARIANCE OF AN AVERAGE:

   For B identically distributed (but not independent) random
   variables (here, tree predictions) with pairwise correlation
   rho and individual variance sigma^2, the variance of their
   average is:

     Var(average) = rho * sigma^2 + (1-rho)/B * sigma^2

   As B -> infinity, this approaches: rho * sigma^2

   -> If rho (correlation between trees) is LARGE, the variance
      of the average remains LARGE regardless of how many trees
      we add -- averaging has DIMINISHING RETURNS.
   -> If rho is SMALL (trees are more INDEPENDENT/decorrelated),
      the variance of the average can be made much SMALLER.

   In BAGGING:
   -> If one feature is very strong (e.g., dominates the splits),
      almost EVERY bootstrap tree will choose that feature for its
      top split(s), making the trees highly CORRELATED (similar
      structure, similar errors).
   -> High correlation rho -> limited variance reduction from
      averaging, even with many trees.

   In RANDOM FOREST:
   -> By forcing each split to consider only a random subset of
      features, sometimes the dominant feature is NOT even available
      as a candidate, forcing the tree to use a DIFFERENT
      (secondary) feature for that split.
   -> This produces trees with more DIVERSE structures --
      LOWER correlation rho between trees.
   -> Lower rho -> averaging reduces variance MORE EFFECTIVELY,
      for the SAME number of trees B.

   TRADE-OFF:
   -> Restricting feature choice at each split can make INDIVIDUAL
      trees slightly WORSE (higher individual variance sigma^2,
      since each tree is built with less information per split).
   -> However, empirically, the REDUCTION IN CORRELATION rho
      typically outweighs the slight increase in individual tree
      variance sigma^2, resulting in a LOWER overall variance
      Var(average) for the Random Forest compared to Bagging.

   CONCLUSION:
   Random Forest's feature subsampling trades a small increase in
   individual tree variance for a larger decrease in inter-tree
   correlation, yielding a net DECREASE in the variance of the
   averaged ensemble -- hence often better generalization than
   plain Bagging.
""")