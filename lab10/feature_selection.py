import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

try:
	from boruta import BorutaPy
	BORUTA_AVAILABLE = True
except ImportError:
	BORUTA_AVAILABLE = False
	print("WARNING: boruta not installed. Run: pip install Boruta --break-system-packages")

# We generate two types of synthetic datasets where ONLY the first k features
# (X1, ..., Xk) are RELEVANT (i.e., determine Y), while the remaining (p-k)
# features (X_{k+1}, ..., X_p) are PURE NOISE — independent of Y.
#
# The goal of feature selection methods is to correctly identify
# X1, ..., Xk as the important features, and X_{k+1}, ..., X_p as irrelevant.

def generate_dataset1(n, p, k, seed=None):
	"""
	Dataset 1: based on a CHI-SQUARED boundary.

	Generate:
		X1, ..., Xp ~ N(0,1) independently
		S = sum_{j=1}^{k} X_j^2   (depends ONLY on the first k features)
		Y = 1 if S > chi2_k(0.5), else Y = 0

	chi2_k(0.5) is the MEDIAN of the chi-squared distribution with k
	degrees of freedom — using the median guarantees P(Y=1) = P(Y=0) = 0.5
	(the dataset is balanced by construction).

	Parameters:
		n    : number of observations
		p    : total number of features
		k    : number of RELEVANT features (X1,...,Xk)
		seed : random seed

	Returns:
		X : feature matrix, shape (n, p)
		y : binary labels, shape (n,), in {0, 1}
	"""
	rng = np.random.default_rng(seed)
	X = rng.standard_normal(size=(n, p))
	
	# Only the first k columns determine Y
	S = np.sum(X[:, :k]**2, axis=1)
	threshold = chi2.ppf(0.5, df=k)
	
	y = (S > threshold).astype(int)
	return X, y


def generate_dataset2(n, p, k, seed=None):
	"""
	Dataset 2: based on an L1 (absolute value sum) boundary.

	Generate:
		X1, ..., Xp ~ N(0,1) independently
		T = sum_{j=1}^{k} |X_j|   (depends ONLY on the first k features)
		Y = 1 if T > k, else Y = 0

	Note: E[|X_j|] = sqrt(2/pi) ~= 0.7979 for X_j ~ N(0,1).
	So E[T] = k * 0.7979 ~= 0.7979*k, which is LESS than k.
	Therefore P(T > k) < 0.5 — this dataset is SOMEWHAT IMBALANCED
	(more 0s than 1s), unlike Dataset 1 which is exactly balanced.

	Parameters:
		n    : number of observations
		p    : total number of features
		k    : number of RELEVANT features (X1,...,Xk)
		seed : random seed

	Returns:
		X : feature matrix, shape (n, p)
		y : binary labels, shape (n,), in {0, 1}
	"""
	rng = np.random.default_rng(seed)
	X = rng.standard_normal(size=(n, p))
	
	T = np.sum(np.abs(X[:, :k]), axis=1)
	y = (T > k).astype(int)
	return X, y


# Quick demonstration with default parameters
n_demo, p_demo, k_demo = 500, 50, 10

X1, y1 = generate_dataset1(n_demo, p_demo, k_demo, seed=42)
X2, y2 = generate_dataset2(n_demo, p_demo, k_demo, seed=42)

print("="*70)
print("PART 1: DATA GENERATION")
print("="*70)
print(f"\nDataset 1 (chi-squared boundary): n={n_demo}, p={p_demo}, k={k_demo}")
print(f"  P(Y=1) = {y1.mean():.3f}  (expected ~0.5, since threshold = median)")
print(f"  chi2_{k_demo}(0.5) threshold = {chi2.ppf(0.5, df=k_demo):.4f}")

print(f"\nDataset 2 (L1 boundary): n={n_demo}, p={p_demo}, k={k_demo}")
print(f"  P(Y=1) = {y2.mean():.3f}  (expected < 0.5, "
      f"since E[sum|Xj|]={k_demo*np.sqrt(2/np.pi):.3f} < {k_demo})")

# PART 2: MARGINAL CORRELATION ANALYSIS

# Question: Are the relevant variables X1,...,Xk expected to be detectable
# by simple MARGINAL CORRELATION with Y?
#
# Marginal correlation: corr(X_j, Y) computed INDEPENDENTLY for each j,
# ignoring all other variables.
#
# Let's COMPUTE this empirically to verify our theoretical reasoning.

print("\n" + "="*70)
print("PART 2: MARGINAL CORRELATION ANALYSIS")
print("="*70)

def marginal_correlations(X, y, k, n_show=15):
	"""
	Compute the Pearson correlation between each feature X_j and Y.

	Parameters:
		X      : feature matrix
		y      : binary labels
		k      : number of TRUE relevant features (for comparison)
		n_show : how many features to display

	Returns:
		correlations : array of correlation coefficients, one per feature
	"""
	p = X.shape[1]
	correlations = np.zeros(p)
	for j in range(p):
		correlations[j], _ = pearsonr(X[:, j], y)
	return correlations

corr1 = marginal_correlations(X1, y1, k_demo)
corr2 = marginal_correlations(X2, y2, k_demo)

print("\nDataset 1 (chi-squared boundary):")
print(f"  Mean |correlation| for RELEVANT features (X1..X{k_demo}):    "
      f"{np.mean(np.abs(corr1[:k_demo])):.4f}")
print(f"  Mean |correlation| for IRRELEVANT features (X{k_demo+1}..Xp): "
      f"{np.mean(np.abs(corr1[k_demo:])):.4f}")
print(f"  -> Relevant features are essentially INDISTINGUISHABLE from "
      f"noise by marginal correlation!")

print("\nDataset 2 (L1 boundary):")
print(f"  Mean |correlation| for RELEVANT features (X1..X{k_demo}):    "
      f"{np.mean(np.abs(corr2[:k_demo])):.4f}")
print(f"  Mean |correlation| for IRRELEVANT features (X{k_demo+1}..Xp): "
      f"{np.mean(np.abs(corr2[k_demo:])):.4f}")
print(f"  -> Same conclusion: relevant features show essentially ZERO "
      f"marginal correlation.")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, corr, title in zip(axes, [corr1, corr2],
                           ['Dataset 1 (chi-squared)', 'Dataset 2 (L1)']):
	colors = ['red' if j < k_demo else 'gray' for j in range(p_demo)]
	ax.bar(range(p_demo), corr, color=colors)
	ax.set_xlabel('Feature index')
	ax.set_ylabel('Pearson correlation with Y')
	ax.set_title(title)
	ax.axvline(x=k_demo-0.5, color='black', linestyle='--',
	           label=f'Relevant | Irrelevant boundary (k={k_demo})')
	ax.legend(fontsize=8)
	ax.grid(True, alpha=0.3)

plt.suptitle('Marginal Correlation: Relevant (red) vs Irrelevant (gray) Features',
             fontsize=13)
plt.tight_layout()
plt.show()

# PART 3: RANDOM FOREST VARIABLE IMPORTANCE (n=500, p=50, k=10)

print("\n" + "="*70)
print("PART 3: RANDOM FOREST VARIABLE IMPORTANCE")
print("="*70)

n_rf, p_rf, k_rf = 500, 50, 10
X_rf, y_rf = generate_dataset1(n_rf, p_rf, k_rf, seed=1)

# Split for permutation importance (permutation importance should be
# computed on held-out data to avoid overfitting-induced bias)
X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_rf, test_size=0.3, random_state=1)

# Fit Random Forest
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_tr, y_tr)

print(f"\nDataset: n={n_rf}, p={p_rf}, k={k_rf}")
print(f"RF Test Accuracy: {rf.score(X_te, y_te):.4f}")

# --- Method 1: Mean Decrease in Impurity (MDI) ---
# This is the DEFAULT feature_importances_ in sklearn.
# For each tree, every time a feature is used to split a node, the
# DECREASE IN IMPURITY (e.g., Gini index) weighted by the number of
# samples reaching that node is recorded. MDI = average of this
# decrease across all trees in the forest, for each feature.
#
# Bias: MDI is biased toward features with MANY possible split points
# (e.g., continuous variables, high-cardinality categoricals) — they
# get more "chances" to be selected for splits, inflating their importance
# even if not truly predictive.

mdi_importance = rf.feature_importances_

# --- Method 2: Permutation Importance ---
# For each feature j:
#   1. Compute baseline model accuracy on TEST data.
#   2. RANDOMLY SHUFFLE (permute) the values of feature j only,
#      breaking its relationship with Y while preserving its
#      marginal distribution.
#   3. Recompute accuracy with the shuffled feature.
#   4. Importance_j = baseline_accuracy - shuffled_accuracy
#      (how much does accuracy DROP when this feature is "destroyed"?)
#
# This directly measures how much the MODEL RELIES on feature j for
# its predictions, computed on UNSEEN data — less biased than MDI,
# but more computationally expensive (requires n_repeats x p model evaluations).

perm_result = permutation_importance(
	rf, X_te, y_te, n_repeats=20, random_state=42, n_jobs=-1
)
perm_importance = perm_result.importances_mean

# Compare: do both methods rank the first k=10 features highest?
print("\n--- Mean Decrease in Impurity (MDI) ---")
mdi_rank = np.argsort(mdi_importance)[::-1]  # descending order
print(f"Top {k_rf} features by MDI: {sorted(mdi_rank[:k_rf])}")
print(f"True relevant features   : {list(range(k_rf))}")
n_correct_mdi = len(set(mdi_rank[:k_rf]) & set(range(k_rf)))
print(f"Correctly identified: {n_correct_mdi}/{k_rf}")

print("\n--- Permutation Importance ---")
perm_rank = np.argsort(perm_importance)[::-1]
print(f"Top {k_rf} features by Permutation Importance: {sorted(perm_rank[:k_rf])}")
print(f"True relevant features                       : {list(range(k_rf))}")
n_correct_perm = len(set(perm_rank[:k_rf]) & set(range(k_rf)))
print(f"Correctly identified: {n_correct_perm}/{k_rf}")

# Visualize both importance measures
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, importance, title in zip(
		axes, [mdi_importance, perm_importance],
		['MDI (Mean Decrease in Impurity)', 'Permutation Importance']):
	colors = ['red' if j < k_rf else 'gray' for j in range(p_rf)]
	ax.bar(range(p_rf), importance, color=colors)
	ax.axvline(x=k_rf-0.5, color='black', linestyle='--',
	           label=f'Relevant | Irrelevant boundary (k={k_rf})')
	ax.set_xlabel('Feature index')
	ax.set_ylabel('Importance')
	ax.set_title(title)
	ax.legend(fontsize=8)
	ax.grid(True, alpha=0.3)

plt.suptitle('Random Forest Variable Importance: Relevant (red) vs Irrelevant (gray)',
             fontsize=13)
plt.tight_layout()
plt.show()

# PART 4: BORUTA ALGORITHM

print("\n" + "="*70)
print("PART 4: BORUTA ALGORITHM")
print("="*70)

# Boruta is a feature selection ALGORITHM (not just a ranking) — it
# produces a final decision: each feature is labeled "Confirmed" (important),
# "Tentative" (undecided), or "Rejected" (unimportant).
#
# How Boruta works:
#   1. For each REAL feature X_j, create a "SHADOW" copy: a random
#      permutation of X_j's values (same distribution, but shuffled —
#      destroys any relationship with Y).
#   2. Fit a Random Forest on the EXTENDED dataset (real features + shadow
#      features).
#   3. Compute feature importance (e.g., MDI) for ALL features
#      (real + shadow).
#   4. For each real feature X_j, compare its importance to the
#      MAXIMUM importance among ALL shadow features (MSI = max shadow
#      importance).
#      - If importance(X_j) > MSI: this is "evidence" that X_j is
#        more important than random noise -> "hit"
#      - If importance(X_j) <= MSI: "miss"
#   5. Repeat steps 1-4 over many iterations. Track the number of
#      "hits" for each feature. Use a binomial test to decide:
#      - If hits are SIGNIFICANTLY MORE than expected by chance (50%):
#        CONFIRM the feature as important.
#      - If hits are SIGNIFICANTLY FEWER than chance: REJECT the feature.
#      - Otherwise: TENTATIVE (inconclusive).

if BORUTA_AVAILABLE:
	rf_boruta = RandomForestClassifier(n_estimators=200, max_depth=5,
	                                   random_state=42, n_jobs=-1)
	boruta_selector = BorutaPy(rf_boruta, n_estimators='auto',
	                           random_state=42, verbose=0, max_iter=50)
	boruta_selector.fit(X_rf, y_rf)
	
	confirmed = np.where(boruta_selector.support_)[0]
	tentative = np.where(boruta_selector.support_weak_)[0]
	rejected  = np.where(~boruta_selector.support_ & ~boruta_selector.support_weak_)[0]
	
	print(f"\nConfirmed features ({len(confirmed)}): {sorted(confirmed)}")
	print(f"Tentative features ({len(tentative)}): {sorted(tentative)}")
	print(f"Rejected  features ({len(rejected)}): {sorted(rejected)[:10]}... "
	      f"(showing first 10)")
	print(f"\nTrue relevant features: {list(range(k_rf))}")
	
	n_correct_boruta = len(set(confirmed) & set(range(k_rf)))
	print(f"Correctly CONFIRMED as relevant: {n_correct_boruta}/{k_rf}")
	
	# Check if any irrelevant features were incorrectly confirmed
	false_positives = set(confirmed) - set(range(k_rf))
	print(f"False positives (irrelevant features confirmed): {len(false_positives)}")
else:
	print("\nBoruta not available — skipping this section.")
	print("Install with: pip install Boruta --break-system-packages")

# PART 5: PROBABILITY OF SUCCESSFUL FEATURE RECOVERY (L repetitions)

print("\n" + "="*70)
print("PART 5: PROBABILITY OF SUCCESSFUL FEATURE RECOVERY (L=50)")
print("="*70)

# "Successful feature recovery" definition:
#   For a given method, we check whether ALL k truly relevant features
#   (X1,...,Xk) are placed among the TOP k features in the importance
#   ranking (for MDI/Permutation), OR are "Confirmed" by Boruta.
#
# We repeat data generation L times (different random seeds) and compute
# the FRACTION of repetitions where recovery is successful.
#
# This gives us a sense of the RELIABILITY of each method — not just
# whether it CAN work, but how OFTEN it works across random samples.

L = 50  # number of repetitions (reduce if too slow)
n_l, p_l, k_l = 500, 50, 10

success_mdi  = 0
success_perm = 0
success_boruta = 0

print(f"\nRunning L={L} simulations with n={n_l}, p={p_l}, k={k_l}...")
print("(This may take a minute...)\n")

for l in range(L):
	X_l, y_l = generate_dataset1(n_l, p_l, k_l, seed=1000+l)
	X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
		X_l, y_l, test_size=0.3, random_state=1000+l
	)
	
	rf_l = RandomForestClassifier(n_estimators=200, random_state=1000+l, n_jobs=-1)
	rf_l.fit(X_tr_l, y_tr_l)
	
	# --- MDI check ---
	mdi_l = rf_l.feature_importances_
	top_k_mdi = set(np.argsort(mdi_l)[::-1][:k_l])
	if top_k_mdi == set(range(k_l)):
		success_mdi += 1
	
	# --- Permutation importance check ---
	perm_l = permutation_importance(rf_l, X_te_l, y_te_l,
	                                n_repeats=5, random_state=1000+l, n_jobs=-1)
	top_k_perm = set(np.argsort(perm_l.importances_mean)[::-1][:k_l])
	if top_k_perm == set(range(k_l)):
		success_perm += 1
	
	# --- Boruta check ---
	if BORUTA_AVAILABLE:
		rf_b = RandomForestClassifier(n_estimators=100, max_depth=5,
		                              random_state=1000+l, n_jobs=-1)
		boruta_l = BorutaPy(rf_b, n_estimators='auto',
		                    random_state=1000+l, verbose=0, max_iter=30)
		boruta_l.fit(X_l, y_l)
		confirmed_l = set(np.where(boruta_l.support_)[0])
		if set(range(k_l)).issubset(confirmed_l):
			success_boruta += 1
	
	if (l+1) % 10 == 0:
		print(f"  Completed {l+1}/{L} simulations...")

print(f"\n--- Results (out of {L} simulations) ---")
print(f"MDI:                   {success_mdi}/{L}  "
      f"(P = {success_mdi/L:.3f})")
print(f"Permutation Importance: {success_perm}/{L}  "
      f"(P = {success_perm/L:.3f})")
if BORUTA_AVAILABLE:
	print(f"Boruta:                {success_boruta}/{L}  "
	      f"(P = {success_boruta/L:.3f})")


# PART 6: ACCURACY vs NUMBER OF TOP-RANKED FEATURES (n=200, p=500, k=20)

print("\n" + "="*70)
print("PART 6: TEST ACCURACY vs NUMBER OF TOP-RANKED FEATURES t")
print("="*70)

# Setup: a HARDER problem — n=200 (small!), p=500 (large!), k=20.
# With n < p, this is a HIGH-DIMENSIONAL problem where feature
# selection becomes crucial.
#
# Procedure:
#   1. Generate training data (n=200, p=500, k=20) and an INDEPENDENT
#      test set (also p=500).
#   2. Fit a Random Forest on the FULL training set to get importance
#      rankings (MDI).
#   3. For each t in {5, 10, 15, 20, 50, 100, 200, 300, 400, 500}:
#      a. Select the top-t features according to the importance ranking.
#      b. Train a NEW classifier using ONLY these t features.
#      c. Evaluate test accuracy.
#   4. Plot test accuracy vs t.
#
# Expected pattern:
#   - t too small (t < k=20): missing some relevant features -> lower accuracy.
#   - t = k = 20 (or slightly more): close to optimal -- all relevant
#     features included, minimal noise.
#   - t >> k: including many noise features -> accuracy may DEGRADE
#     slightly (curse of dimensionality, noise dilutes signal) or
#     plateau (Random Forest is somewhat robust to irrelevant features).

n_t, p_t, k_t = 200, 500, 20

X_train_t, y_train_t = generate_dataset1(n_t, p_t, k_t, seed=7)
X_test_t,  y_test_t  = generate_dataset1(2000, p_t, k_t, seed=8)  # large indep. test set

print(f"\nSetup: n_train={n_t}, n_test=2000, p={p_t}, k={k_t}")

# Step 1: Get feature ranking from a RF trained on ALL features
rf_full = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_full.fit(X_train_t, y_train_t)
importance_full = rf_full.feature_importances_
ranking = np.argsort(importance_full)[::-1]  # descending order of importance

print(f"\nTop 20 features by importance: {sorted(ranking[:20])}")
print(f"True relevant features (0-19): {list(range(20))}")
n_correct_top20 = len(set(ranking[:20]) & set(range(20)))
print(f"Overlap with true relevant set: {n_correct_top20}/20")

# Step 2: For each t, train classifier on top-t features
t_values = [5, 10, 15, 20, 50, 100, 200, 300, 400, 500]
accuracies = []

print(f"\n{'t':>6} | {'Test Accuracy':>14}")
print("-" * 25)

for t in t_values:
	selected_features = ranking[:t]
	
	rf_t = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
	rf_t.fit(X_train_t[:, selected_features], y_train_t)
	
	acc = rf_t.score(X_test_t[:, selected_features], y_test_t)
	accuracies.append(acc)
	print(f"{t:>6} | {acc:>14.4f}")

# Plot
plt.figure(figsize=(9, 5))
plt.plot(t_values, accuracies, 'o-', linewidth=2, markersize=8, color='steelblue')
plt.axvline(x=k_t, color='red', linestyle='--', label=f'k={k_t} (true number of relevant features)')
plt.xlabel('t (number of top-ranked features used)')
plt.ylabel('Test Accuracy')
plt.title(f'Test Accuracy vs Number of Top-Ranked Features\n'
          f'(n={n_t}, p={p_t}, k={k_t})')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# QUESTIONS AND ANSWERS

print("""
=================================================================
QUESTIONS AND ANSWERS — FEATURE SELECTION (Lab 10)
=================================================================

-----------------------------------------------------------------
Q1. Are the relevant variables X1,...,Xk expected to be detectable
    by simple marginal correlation with Y? Why or why not?
    (Applies to BOTH Dataset 1 and Dataset 2)
-----------------------------------------------------------------
A: NO -- marginal correlation FAILS to detect these relevant
   variables, for a SYMMETRY reason that applies to BOTH datasets.

   DATASET 1 (chi-squared boundary):
     Y = 1 if sum_{j=1}^k X_j^2 > chi2_k(0.5)

   Consider corr(X_1, Y). Note that Y depends on X_1 only through
   X_1^2 -- it does NOT depend on the SIGN of X_1.
   By symmetry of the standard normal distribution:
     X_1 ~ N(0,1)  =>  -X_1 ~ N(0,1)  (same distribution)
   And:
     Y(X_1, X_2,...,X_k) = Y(-X_1, X_2,...,X_k)
     (since X_1^2 = (-X_1)^2)

   Therefore, for any fixed value of Y:
     E[X_1 | Y=y] = E[-X_1 | Y=y] = -E[X_1 | Y=y]
   This implies E[X_1 | Y=y] = 0 for both y=0 and y=1.

   So: Cov(X_1, Y) = E[X_1 * Y] - E[X_1]*E[Y]
                    = E[X_1 * Y] - 0
                    = E[Y * E[X_1|Y]]  (by law of total expectation)
                    = E[Y * 0] = 0

   => corr(X_1, Y) = 0  THEORETICALLY, for ALL relevant features
      X_1,...,X_k in Dataset 1.

   DATASET 2 (L1 boundary):
     Y = 1 if sum_{j=1}^k |X_j| > k

   Identical argument: Y depends on X_1 only through |X_1|, which
   is symmetric in the sign of X_1. The same reasoning gives:
     corr(X_1, Y) = 0  for all relevant features in Dataset 2.

   CONCLUSION:
   -> Marginal correlation is ZERO (in expectation) for ALL relevant
      features in BOTH datasets, DESPITE these features being
      CRUCIAL for predicting Y.
   -> This is because the relationship is through a NONLINEAR,
      SIGN-SYMMETRIC function (squares or absolute values) -- a
      purely LINEAR association measure (correlation) cannot
      "see" this kind of dependence.
   -> This demonstrates why marginal/univariate screening methods
      (e.g., simply keeping features with |corr(X_j,Y)| above some
      threshold) can COMPLETELY MISS important variables that
      interact NONLINEARLY with the response.
   -> Methods that capture NONLINEAR and INTERACTION effects
      (Random Forest importance, Boruta) are needed to detect
      such variables.

-----------------------------------------------------------------
Q2. What is Mean Decrease in Impurity (MDI), and what is its
    main weakness?
-----------------------------------------------------------------
A: MDI (also called "Gini importance" for classification) is computed
   as follows:
   -> For every split in every tree of the Random Forest, when
      feature X_j is used to split a node, the IMPURITY DECREASE
      (e.g., decrease in Gini index or entropy) caused by that
      split is recorded, WEIGHTED by the fraction of samples
      reaching that node.
   -> MDI_j = sum over all splits using X_j, across all trees,
      of (weighted impurity decrease), averaged over trees.

   MAIN WEAKNESSES:
   1. BIAS TOWARD HIGH-CARDINALITY FEATURES:
      Features with MANY unique values (continuous variables, or
      categorical variables with many categories) have MORE
      POSSIBLE SPLIT POINTS to choose from. Purely by having more
      "opportunities" to produce a good-looking split (even on
      pure noise, by chance one of the many possible thresholds
      may look informative), such features tend to get artificially
      INFLATED importance scores -- even if they have NO real
      relationship with Y.

   2. COMPUTED ON TRAINING DATA:
      MDI is computed using the SAME data used to grow the trees.
      It reflects how much a feature was used to FIT the training
      data, which can be inflated by OVERFITTING -- a feature might
      get high MDI because it helps memorize training-specific noise,
      not because it generalizes.

   3. CORRELATED FEATURES SPLIT IMPORTANCE:
      If two features are highly correlated and both informative,
      the importance "credit" gets split between them, potentially
      making each look less important than it really is.

   In our experiments: irrelevant features (X_{k+1},...,X_p) can
   still receive SMALL BUT NON-ZERO MDI scores purely due to random
   chance in tree splits -- this "background noise level" must be
   accounted for, which is exactly what Boruta's shadow features do.

-----------------------------------------------------------------
Q3. What is Permutation Importance, and how does it differ from MDI?
-----------------------------------------------------------------
A: Permutation Importance is computed as:
   1. Compute the model's accuracy (or another metric) on a
      (preferably held-out / OUT-OF-BAG) dataset: baseline_score.
   2. For feature j: randomly SHUFFLE (permute) the values of
      column j across all observations. This destroys any
      relationship between X_j and Y while preserving X_j's
      marginal distribution (mean, variance, etc. unchanged).
   3. Recompute the model's accuracy with this shuffled feature:
      permuted_score.
   4. Importance_j = baseline_score - permuted_score
      (how much WORSE does the model perform when X_j's information
       is destroyed?)
   5. Repeat steps 2-4 multiple times (n_repeats) and average, to
      reduce the randomness from a single permutation.

   KEY DIFFERENCES FROM MDI:
   -> MDI is computed DURING tree construction (training time),
      reflecting how features were USED to build splits.
      Permutation importance is computed AFTER training, by
      measuring the EFFECT of removing a feature's information
      on PREDICTIVE PERFORMANCE.

   -> Permutation importance can (and SHOULD) be computed on
      HELD-OUT data, giving a measure of how much the model RELIES
      on a feature for GENERALIZATION -- not just how much it was
      used to fit the training data.

   -> Permutation importance is NOT biased toward high-cardinality
      features in the same way MDI is, because it doesn't depend
      on how splits were chosen -- only on the final model's
      sensitivity to each feature's information content.

   -> Permutation importance is MORE COMPUTATIONALLY EXPENSIVE:
      requires n_repeats * p model evaluations on the test set,
      versus MDI which is essentially "free" (already computed
      during training).

   -> CAVEAT: if features are highly CORRELATED, permuting one
      feature may have little effect (the model can "substitute"
      information from the correlated feature), leading to
      UNDERESTIMATED importance for correlated groups of features.

-----------------------------------------------------------------
Q4. Explain the role of shadow variables in Boruta. Why is
    comparison with the best shadow variable more meaningful
    than simply checking whether an importance score is positive?
-----------------------------------------------------------------
A: SHADOW VARIABLES:
   For each real feature X_j, Boruta creates a "shadow" copy
   X_j_shadow by taking a RANDOM PERMUTATION of the values of X_j.
   -> X_j_shadow has the EXACT SAME marginal distribution as X_j
      (same values, just shuffled order).
   -> By construction, X_j_shadow has ZERO TRUE RELATIONSHIP with Y
      (the shuffling destroys any dependence).
   -> Shadow variables therefore represent "PURE NOISE" features
      that have the SAME STATISTICAL PROPERTIES (scale, distribution)
      as the real features.

   THE COMPARISON:
   Boruta fits a Random Forest on [real features + shadow features]
   and computes importance for ALL of them. Then, for each real
   feature X_j, it compares importance(X_j) to MSI = the MAXIMUM
   importance among ALL shadow features.

   WHY NOT JUST CHECK "importance > 0"?

   1. RANDOM FOREST IMPORTANCE IS ALMOST NEVER EXACTLY ZERO:
      Even for completely irrelevant features, by random chance
      some splits in some trees will happen to use that feature
      and produce SOME (small) impurity decrease. So PURE NOISE
      features will have SMALL BUT POSITIVE importance scores,
      simply due to the randomness of tree construction
      (bootstrap sampling, random feature subsets at each split).
      A threshold of "importance > 0" would essentially select
      ALMOST ALL features, including pure noise -- useless.

   2. THE "BACKGROUND NOISE LEVEL" DEPENDS ON THE DATASET:
      How much importance a PURE NOISE feature receives "by chance"
      depends on factors like n, p, the number of trees, tree depth,
      etc. There's no UNIVERSAL threshold value that works across
      different datasets/settings.

   3. SHADOW FEATURES PROVIDE A DATASET-SPECIFIC, EMPIRICAL
      "NULL DISTRIBUTION":
      Since shadow features are GUARANTEED to have zero true
      relationship with Y, but are subjected to the SAME random
      forest fitting process (same n, same number of trees, same
      depth, etc.) as the real features, their importance scores
      represent EXACTLY the "by chance" importance level for THIS
      SPECIFIC dataset and model configuration.
      MSI = max(shadow importances) represents the HIGHEST
      importance achievable by PURE NOISE under these exact conditions.

   4. THE DECISION RULE BECOMES MEANINGFUL:
      A real feature X_j is considered "evidence of relevance" in
      one iteration (a "hit") only if:
        importance(X_j) > MSI
      i.e., X_j beat the BEST of all the noise impostors -- a much
      stronger and more meaningful claim than "importance > 0".
      Over many iterations, features that CONSISTENTLY beat MSI
      (statistically significantly more often than 50% of the time,
      via a binomial test) are CONFIRMED as truly important.

   ANALOGY: It's like a competitive exam where instead of asking
   "did you score above zero?" (everyone does), you ask "did you
   score higher than the BEST RANDOM GUESSER in the room?" --
   a much more informative bar.

-----------------------------------------------------------------
Q5. In Part 6 (n=200, p=500, k=20), how does test accuracy depend
    on t (number of top-ranked features used)? Interpret the shape
    of the curve.
-----------------------------------------------------------------
A: Expected pattern (based on bias-variance / signal-to-noise
   reasoning):

   t < k (e.g., t=5, 10, 15):
   -> Some of the truly relevant features (out of the 20) are
      EXCLUDED from the model.
   -> The model is missing PART of the signal needed to predict Y
      correctly -> UNDERFITTING with respect to the true structure.
   -> Test accuracy is LOWER than optimal.
   -> As t increases toward k, accuracy generally IMPROVES, since
      more of the true signal becomes available.

   t ~= k (t=20):
   -> (Approximately) all truly relevant features are included,
      with minimal noise features.
   -> This is close to the BEST achievable configuration --
      maximum signal, minimum noise.
   -> Test accuracy should be at or near its PEAK here.

   t >> k (t=50, 100, 200, ..., 500):
   -> All 20 relevant features are still included (assuming the
      ranking correctly placed them near the top), PLUS many
      (t-20) irrelevant/noise features.
   -> Random Forest is SOMEWHAT ROBUST to irrelevant features
      (it can mostly ignore them via the feature subsampling at
      each split), so accuracy may remain RELATIVELY STABLE.
   -> However, with n=200 being quite small relative to growing
      t, including hundreds of noise features can:
        - dilute the "attention" given to relevant features at
          each split (lower probability that a relevant feature
          is in the random subset considered at any given split),
        - slightly increase variance of the fitted model.
   -> Accuracy may show a SLIGHT DECLINE or PLATEAU at large t,
      but typically not a dramatic collapse, because RF handles
      high-dimensional noise reasonably well.

   OVERALL SHAPE: an increasing curve from small t, peaking around
   t ~ k (or slightly above), followed by a roughly flat or
   mildly decreasing curve for t >> k. This illustrates the
   PRACTICAL VALUE of feature selection: using t ~ k features
   (a small, informative subset) achieves NEAR-OPTIMAL accuracy
   with a MUCH SIMPLER, more interpretable, and computationally
   cheaper model than using all p=500 features.

-----------------------------------------------------------------
Q6. Why is it important to use an INDEPENDENT test set when
    evaluating accuracy vs t (rather than reusing the training data
    or the data used for ranking)?
-----------------------------------------------------------------
A: The feature ranking itself was derived from the TRAINING data
   (rf_full was fit on X_train_t, y_train_t). If we then trained
   AND evaluated the t-feature models on the SAME training data,
   we would be DOUBLE-DIPPING:
   -> The ranking might have picked up on TRAINING-SET-SPECIFIC
      noise patterns (some "irrelevant" feature might have appeared
      useful purely by chance in this particular training sample).
   -> Evaluating on the same data would not reveal this -- the
      noise pattern that helped the ranking would ALSO help the
      evaluation, giving an OVERLY OPTIMISTIC accuracy estimate
      that does not reflect true generalization.

   By using a genuinely INDEPENDENT test set (generated separately,
   even though from the same data-generating process), we ensure
   that:
   -> Any "lucky" noise patterns exploited during training/ranking
      will NOT be present in the test set (different random draws).
   -> The reported accuracy reflects how well the SELECTED FEATURES
      and TRAINED MODEL generalize to NEW data -- the true quantity
      of interest.

   This is the same principle as the train/validation/test split
   discipline: information used for model/feature SELECTION must
   not be the same information used for FINAL EVALUATION.

-----------------------------------------------------------------
Q7. Summarize: MDI vs Permutation Importance vs Boruta -- when
    would you use each?
-----------------------------------------------------------------
A:
   +--------------------+---------------------------------------+
   | Method             | Best Used When...                      |
   +--------------------+---------------------------------------+
   | MDI                | Quick, cheap exploratory ranking.      |
   | (feature_          | Features have similar cardinality/     |
   |  importances_)     | scale. Computational budget is tight.  |
   +--------------------+---------------------------------------+
   | Permutation        | More reliable ranking is needed.       |
   | Importance         | Features vary in type/cardinality.     |
   |                    | Held-out data is available.            |
   |                    | Willing to pay extra computation cost. |
   +--------------------+---------------------------------------+
   | Boruta             | Need a DEFINITIVE Confirmed/Rejected   |
   |                    | DECISION (not just a ranking).         |
   |                    | Statistical significance / control of  |
   |                    | false positives matters.               |
   |                    | Willing to pay HIGH computation cost   |
   |                    | (many RF fits over many iterations).   |
   +--------------------+---------------------------------------+

   In practice: MDI for a first quick look, Permutation Importance
   for a more trustworthy ranking, and Boruta when a formal
   selected/rejected decision with statistical backing is required
   (e.g., for downstream model simplification or reporting).
""")