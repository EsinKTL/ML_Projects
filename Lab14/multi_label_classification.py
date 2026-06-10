import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score

# =============================================================
# DATA LOADING
# =============================================================

# fetch_openml: downloads the "emotions" dataset from OpenML
# This is a music emotion dataset:
#   X: audio features (rhythmic, spectral, etc.)
#   Y: 6 binary labels (amazed-suprised, happy-pleased, relaxing-calm,
#                        quiet-still, sad-lonely, angry-aggresive)
# version=4: specific version of the dataset
# return_X_y=True: return X and Y separately instead of a Bunch object
X, Y = fetch_openml("emotions", version=4, return_X_y=True)

# Y comes as string "TRUE"/"FALSE" — convert to integer 0/1
# (Y == "TRUE") creates a boolean array, .astype(int) converts True->1, False->0
Y = (Y == "TRUE").astype(int)

# Split into 70% training, 30% test
# random_state=42: reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(
	X, Y, test_size=0.3, random_state=42
)

print("Dataset loaded successfully.")
print(f"X shape     : {X.shape}   (observations x features)")
print(f"Y shape     : {Y.shape}   (observations x labels)")
print(f"X_train     : {X_train.shape}")
print(f"X_test      : {X_test.shape}")

# =============================================================
# TASK 1: PRELIMINARY ANALYSIS OF LABELS
# =============================================================

print("\n" + "="*60)
print("TASK 1: PRELIMINARY LABEL ANALYSIS")
print("="*60)

# --- 1. How many labels are there? ---
K = Y.shape[1]   # number of columns = number of labels
print(f"\n1. Number of labels: {K}")
print(f"   Label names: {list(Y.columns)}")

# --- 2. Empirical frequency of each label ---
# Mean of each column = proportion of observations where that label is 1
print("\n2. Empirical frequency of each label:")
for col in Y.columns:
	freq = Y[col].mean()
	print(f"   {col}: {freq:.3f}  ({int(Y[col].sum())} out of {len(Y)} observations)")

# --- 3. Average number of active labels per observation ---
# For each row, sum across columns to get number of active labels
# Then take the mean across all rows
avg_active = Y.sum(axis=1).mean()
print(f"\n3. Average number of active labels per observation: {avg_active:.3f}")

# Also show distribution
print("   Distribution of active label counts:")
for n in range(K + 1):
	count = (Y.sum(axis=1) == n).sum()
	print(f"   {n} labels: {count} observations ({100*count/len(Y):.1f}%)")

# --- 4. Which pairs of labels are most strongly correlated? ---
# Pearson correlation between label columns
print("\n4. Label correlation matrix:")
label_corr = Y.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(label_corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True)
plt.title("Label Correlation Matrix")
plt.tight_layout()
plt.show()

# Find top 5 most correlated pairs
corr_pairs = []
cols = list(Y.columns)
for i in range(len(cols)):
	for j in range(i+1, len(cols)):
		corr_pairs.append((cols[i], cols[j], label_corr.iloc[i, j]))

# Sort by absolute correlation descending
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("   Top 5 most correlated label pairs:")
for c1, c2, r in corr_pairs[:5]:
	print(f"   {c1} <-> {c2}: r = {r:.3f}")

# --- 5. Are labels approximately independent? ---
print("\n5. Label independence check:")
print("   Non-zero off-diagonal correlations suggest label dependence.")
print("   This matters because Binary Relevance assumes independence,")
print("   while Classifier Chains and CCC try to model dependencies.")

# =============================================================
# EVALUATION FUNCTION
# =============================================================

def evaluate(name, Y_true, Y_pred):
	"""
	Computes and prints three evaluation metrics for multi-label classification.

	Parameters:
		name   : string label for the method being evaluated
		Y_true : true label matrix  (n_samples x K)
		Y_pred : predicted label matrix (n_samples x K)

	Metrics:
		Subset Accuracy : fraction of observations where ALL labels are correct
						  very strict — even one wrong label counts as a full mistake
		Hamming Score   : fraction of individual label predictions that are correct
						  = 1 - Hamming Loss
						  averages over all labels and all observations
		Jaccard Score   : for each observation, |predicted ∩ true| / |predicted ∪ true|
						  focuses on active (=1) labels, ignores correct 0 predictions
						  more informative when active labels are rare
	"""
	subset_acc = accuracy_score(Y_true, Y_pred)
	hamming    = 1 - hamming_loss(Y_true, Y_pred)
	jacc       = jaccard_score(Y_true, Y_pred, average="samples")
	print(
		f"{name:<30} -> "
		f"Subset accuracy: {subset_acc:.3f}, "
		f"Hamming score: {hamming:.3f}, "
		f"Jaccard score: {jacc:.3f}"
	)
	return subset_acc, hamming, jacc

# =============================================================
# TASK 2: STANDARD MULTI-LABEL METHODS
# =============================================================

print("\n" + "="*60)
print("TASK 2: STANDARD MULTI-LABEL METHODS")
print("="*60)

# -----------------------------------------------------------
# METHOD 1: BINARY RELEVANCE (BR)
# -----------------------------------------------------------
# Binary Relevance: fit one independent binary classifier per label
# Yk <- X  for k = 1, ..., K
# Assumption: labels are INDEPENDENT given X (ignores label correlations)
#
# OneVsRestClassifier wraps a base classifier and trains K separate models
# solver="liblinear": fast solver for small/medium datasets
# Each model sees only X and tries to predict one label at a time

br = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=1000))
br.fit(X_train, Y_train)

Y_pred_br_train = br.predict(X_train)
Y_pred_br_test  = br.predict(X_test)

print("\n--- Binary Relevance (BR) ---")
evaluate("BR (train)", Y_train, Y_pred_br_train)
evaluate("BR (test) ", Y_test,  Y_pred_br_test)

# -----------------------------------------------------------
# METHOD 2: CLASSIFIER CHAINS (CC)
# -----------------------------------------------------------
# Classifier Chains: fit K classifiers sequentially
# Y1 <- X
# Y2 <- X, Y1
# Y3 <- X, Y1, Y2
# ...
# YK <- X, Y1, ..., Y(K-1)
#
# Each classifier gets X PLUS all previously predicted labels as features
# This allows the model to capture label dependencies
# IMPORTANT: the order of labels matters — earlier labels influence later ones
# We test 10 different random orders to see variability

print("\n--- Classifier Chains (CC) with 10 random orders ---")

cc_results = []
for seed in range(10):
	# order="random": randomly shuffle the label order
	# random_state=seed: different seed = different order
	# base_estimator vs estimator depends on sklearn version
	try:
		# sklearn >= 1.7
		cc = ClassifierChain(
			LogisticRegression(solver="liblinear", max_iter=1000),
			order="random", random_state=seed
		)
	except TypeError:
		# sklearn <= 1.6.1
		cc = ClassifierChain(
			base_estimator=LogisticRegression(solver="liblinear", max_iter=1000),
			order="random", random_state=seed
		)
	
	cc.fit(X_train, Y_train)
	Y_pred_cc = cc.predict(X_test)
	
	subset_acc = accuracy_score(Y_test, Y_pred_cc)
	hamming    = 1 - hamming_loss(Y_test, Y_pred_cc)
	jacc       = jaccard_score(Y_test, Y_pred_cc, average="samples")
	
	cc_results.append((subset_acc, hamming, jacc))
	print(f"  Seed {seed:2d}: SubsetAcc={subset_acc:.3f}, "
	      f"Hamming={hamming:.3f}, Jaccard={jacc:.3f}")

# Average and std across 10 random orders
cc_results = np.array(cc_results)
print(f"\n  Mean over 10 random orders:")
print(f"  SubsetAcc = {cc_results[:,0].mean():.3f} ± {cc_results[:,0].std():.3f}")
print(f"  Hamming   = {cc_results[:,1].mean():.3f} ± {cc_results[:,1].std():.3f}")
print(f"  Jaccard   = {cc_results[:,2].mean():.3f} ± {cc_results[:,2].std():.3f}")

# -----------------------------------------------------------
# METHOD 3: ENSEMBLE OF CLASSIFIER CHAINS (ECC)
# -----------------------------------------------------------
# ECC: train M chains with different random orders
# Final prediction: majority vote across all chains
# If >= 50% of chains predict label k = 1, then predict 1, else 0
#
# This reduces variance compared to a single chain
# Different chains make different errors, averaging reduces noise

M = 20   # number of chains in the ensemble
print(f"\n--- Ensemble of Classifier Chains (ECC, M={M}) ---")

chains = []
for m in range(M):
	try:
		cc_m = ClassifierChain(
			LogisticRegression(solver="liblinear", max_iter=1000),
			order="random", random_state=m
		)
	except TypeError:
		cc_m = ClassifierChain(
			base_estimator=LogisticRegression(solver="liblinear", max_iter=1000),
			order="random", random_state=m
		)
	cc_m.fit(X_train, Y_train)
	chains.append(cc_m)

# Collect predictions from all M chains
# predictions: list of M arrays, each of shape (n_test, K)
all_preds = np.array([c.predict(X_test) for c in chains])  # shape: (M, n_test, K)

# Majority vote: for each observation and label,
# if more than half the chains predict 1, predict 1
# all_preds.mean(axis=0): average prediction per observation per label
# >= 0.5: threshold for majority vote
Y_pred_ecc = (all_preds.mean(axis=0) >= 0.5).astype(int)

evaluate("ECC (test)", Y_test, Y_pred_ecc)

# -----------------------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------------------
print("\n--- Summary: All methods on test set ---")
evaluate("BR          ", Y_test, Y_pred_br_test)
# Use best CC (seed with highest Jaccard)
best_seed = cc_results[:, 2].argmax()
try:
	cc_best = ClassifierChain(
		LogisticRegression(solver="liblinear", max_iter=1000),
		order="random", random_state=int(best_seed)
	)
except TypeError:
	cc_best = ClassifierChain(
		base_estimator=LogisticRegression(solver="liblinear", max_iter=1000),
		order="random", random_state=int(best_seed)
	)
cc_best.fit(X_train, Y_train)
evaluate(f"CC (best seed={best_seed})", Y_test, cc_best.predict(X_test))
evaluate("ECC         ", Y_test, Y_pred_ecc)

# =============================================================
# TASK 3: CIRCULAR CHAIN CLASSIFIER (CCC)
# =============================================================

print("\n" + "="*60)
print("TASK 3: CIRCULAR CHAIN CLASSIFIER (CCC)")
print("="*60)

def fit_ccc(X_train, Y_train):
	"""
	Train the Circular Chain Classifier.

	For each label k (k = 0, ..., K-1):
		- Remove column k from the label matrix
		- Append all remaining labels Y_{-k} as additional features to X
		- Train a binary logistic regression to predict Y_k from (X, Y_{-k})

	This means each classifier uses the TRUE values of all other labels
	during training (oracle training) — which is valid because at training
	time we know all labels.

	Parameters:
		X_train : feature matrix (n_train, n_features)
		Y_train : label matrix   (n_train, K)

	Returns:
		models : list of K fitted LogisticRegression models
		K      : number of labels
	"""
	# Convert to numpy array if pandas DataFrame
	X_np = np.array(X_train)
	Y_np = np.array(Y_train)
	
	n, K = Y_np.shape
	models = []
	
	for k in range(K):
		# Remove column k from the label matrix
		# np.delete(..., k, axis=1) removes the k-th column
		Y_minus_k = np.delete(Y_np, k, axis=1)   # shape: (n, K-1)
		
		# Append remaining labels to features
		X_augmented = np.hstack([X_np, Y_minus_k])   # shape: (n, n_features + K-1)
		
		# Fit binary logistic regression for label k
		model = LogisticRegression(solver="liblinear", max_iter=1000)
		model.fit(X_augmented, Y_np[:, k])
		models.append(model)
	
	return models, K


def predict_ccc(models, X_test, K, init="zeros", max_iter=20,
                random_state=None, br_preds=None):
	"""
	Predict labels using the Circular Chain Classifier.

	Since true labels are unknown at test time, we use iterative updates:
	1. Initialize label estimates y_hat
	2. Repeatedly update each label using current estimates of all other labels
	3. Stop when no label changes after a full pass, or max_iter is reached

	Parameters:
		models       : list of K fitted models from fit_ccc
		X_test       : test feature matrix (n_test, n_features)
		K            : number of labels
		init         : initialization strategy
					   "zeros"  -> start with all zeros
					   "random" -> random 0/1 initialization
					   "br"     -> use Binary Relevance predictions as start
		max_iter     : maximum number of full passes (to avoid infinite loops)
		random_state : random seed for reproducibility (used when init="random")
		br_preds     : BR predictions to use when init="br" (n_test, K array)

	Returns:
		Y_pred       : predicted label matrix (n_test, K)
		avg_iters    : average number of iterations until convergence
	"""
	X_np = np.array(X_test)
	n = X_np.shape[0]
	
	# --- Initialize label estimates ---
	if init == "zeros":
		Y_hat = np.zeros((n, K), dtype=int)
	elif init == "random":
		rng = np.random.default_rng(random_state)
		Y_hat = rng.integers(0, 2, size=(n, K))
	elif init == "br":
		if br_preds is None:
			raise ValueError("br_preds must be provided for BR initialization")
		Y_hat = np.array(br_preds, dtype=int)
	else:
		raise ValueError(f"Unknown init='{init}'. Use 'zeros', 'random', or 'br'.")
	
	# --- Random permutation of label update order ---
	# sigma: which order to update labels in each pass
	rng_sigma = np.random.default_rng(random_state)
	sigma = rng_sigma.permutation(K)
	
	iterations = np.zeros(n, dtype=int)
	converged  = np.zeros(n, dtype=bool)
	
	for iteration in range(max_iter):
		Y_hat_old = Y_hat.copy()
		
		# Full pass through all labels in order sigma
		for j in sigma:
			# Remove current estimate of label j from Y_hat
			# Use all other K-1 label estimates as additional features
			Y_minus_j = np.delete(Y_hat, j, axis=1)   # (n, K-1)
			X_aug     = np.hstack([X_np, Y_minus_j])  # (n, n_features + K-1)
			
			# Predict label j using the j-th trained model
			# proba[:,1]: probability of class 1
			proba = models[j].predict_proba(X_aug)[:, 1]
			
			# Apply threshold 0.5
			Y_hat[:, j] = (proba >= 0.5).astype(int)
		
		# Check convergence for each observation individually
		# An observation converged if its label vector did not change
		just_converged = (~converged) & (np.all(Y_hat == Y_hat_old, axis=1))
		iterations[just_converged] = iteration + 1
		converged[just_converged]  = True
		
		# Stop if all observations have converged
		if converged.all():
			break
	
	# Observations that never converged: record max_iter
	iterations[~converged] = max_iter
	if not converged.all():
		n_not = (~converged).sum()
		print(f"    WARNING: {n_not} observation(s) did not converge within {max_iter} iterations.")
	
	avg_iters = iterations.mean()
	return Y_hat, avg_iters


# --- Train CCC ---
print("\nTraining CCC...")
ccc_models, K_ccc = fit_ccc(X_train, Y_train)
print("CCC training complete.")

# --- Experiment 1: Compare three initializations ---
print("\n--- CCC with different initializations ---")

# zeros initialization
Y_pred_ccc_zeros, iters_zeros = predict_ccc(
	ccc_models, X_test, K_ccc, init="zeros", random_state=0
)
print(f"\nZeros init (avg iters = {iters_zeros:.2f}):")
evaluate("CCC zeros    ", Y_test, Y_pred_ccc_zeros)

# random initialization (try 5 different seeds to check stability)
print("\nRandom init (5 seeds):")
random_results = []
for seed in range(5):
	Y_pred_ccc_rand, iters_rand = predict_ccc(
		ccc_models, X_test, K_ccc, init="random", random_state=seed
	)
	sa = accuracy_score(Y_test, Y_pred_ccc_rand)
	hm = 1 - hamming_loss(Y_test, Y_pred_ccc_rand)
	jc = jaccard_score(Y_test, Y_pred_ccc_rand, average="samples")
	random_results.append((sa, hm, jc))
	print(f"  Seed {seed}: SubsetAcc={sa:.3f}, Hamming={hm:.3f}, "
	      f"Jaccard={jc:.3f}, avg_iters={iters_rand:.2f}")

random_results = np.array(random_results)
print(f"  Std across seeds: SubsetAcc={random_results[:,0].std():.4f}, "
      f"Hamming={random_results[:,1].std():.4f}, "
      f"Jaccard={random_results[:,2].std():.4f}")

# BR initialization (use BR predictions as starting point)
Y_pred_ccc_br, iters_br = predict_ccc(
	ccc_models, X_test, K_ccc, init="br",
	br_preds=Y_pred_br_test, random_state=0
)
print(f"\nBR init (avg iters = {iters_br:.2f}):")
evaluate("CCC BR init  ", Y_test, Y_pred_ccc_br)

# --- Experiment 2: Check dependence on permutation sigma ---
print("\n--- CCC with zeros init: 5 different label permutations ---")
perm_results = []
for seed in range(5):
	# init="zeros" but different random_state changes sigma permutation
	Y_pred_perm, iters_perm = predict_ccc(
		ccc_models, X_test, K_ccc, init="zeros", random_state=seed
	)
	sa = accuracy_score(Y_test, Y_pred_perm)
	hm = 1 - hamming_loss(Y_test, Y_pred_perm)
	jc = jaccard_score(Y_test, Y_pred_perm, average="samples")
	perm_results.append((sa, hm, jc))
	print(f"  Sigma seed {seed}: SubsetAcc={sa:.3f}, Hamming={hm:.3f}, "
	      f"Jaccard={jc:.3f}")

perm_results = np.array(perm_results)
print(f"  Std across permutations: SubsetAcc={perm_results[:,0].std():.4f}, "
      f"Hamming={perm_results[:,1].std():.4f}, "
      f"Jaccard={perm_results[:,2].std():.4f}")

# --- Final comparison: BR vs CC vs ECC vs CCC ---
print("\n" + "="*60)
print("FINAL COMPARISON: ALL METHODS ON TEST SET")
print("="*60)
evaluate("BR             ", Y_test, Y_pred_br_test)
evaluate(f"CC (best seed) ", Y_test, cc_best.predict(X_test))
evaluate("ECC            ", Y_test, Y_pred_ecc)
evaluate("CCC (zeros)    ", Y_test, Y_pred_ccc_zeros)
evaluate("CCC (random)   ", Y_test, Y_pred_ccc_rand)   # last seed
evaluate("CCC (BR init)  ", Y_test, Y_pred_ccc_br)

print(f"\nAverage iterations to convergence:")
print(f"  CCC zeros  init: {iters_zeros:.2f}")
print(f"  CCC random init: {iters_rand:.2f}")
print(f"  CCC BR     init: {iters_br:.2f}")

print("""

1. What label-dependence assumption is made by Binary Relevance?
   -> Binary Relevance assumes that all labels are CONDITIONALLY
      INDEPENDENT given the features X.
      That is: P(Y1, ..., YK | X) = P(Y1|X) * P(Y2|X) * ... * P(YK|X).
      Each label is predicted by a separate model that only sees X,
      ignoring all information about other labels.
      This assumption is often violated in practice — labels tend to
      co-occur in patterns (e.g., "happy" and "relaxing" emotions
      may correlate), so BR can miss these dependencies.

2. Why can the order of labels influence Classifier Chains?
   -> In a Classifier Chain, each classifier for label Yk uses the
      PREDICTED values of all earlier labels Y1, ..., Y(k-1) as
      additional features. If an early label is predicted incorrectly,
      this error propagates to all later classifiers.
      A label that appears early in the chain can only use X as input,
      while a label appearing late benefits from many other predicted
      labels as context. Therefore different orderings lead to
      different amounts of label-context available to each classifier,
      and to different patterns of error propagation.

3. Why should ECC be more stable than a single Classifier Chain?
   -> A single chain depends heavily on its specific label order.
      One bad ordering (where an important label appears early and
      is predicted poorly) can degrade all downstream predictions.
      ECC trains M chains with M different random orders. Each chain
      makes different ordering-specific errors. By combining via
      majority vote, random errors in individual chains cancel out,
      and the ensemble converges toward more reliable predictions.
      This is the same principle as Random Forest vs a single tree.

4. Why is subset accuracy usually much smaller than Hamming score?
   -> Subset accuracy requires ALL K labels to be predicted correctly
      for an observation to count as correct — it is a very strict metric.
      Even if 5 out of 6 labels are correct, subset accuracy counts it as 0.
      Hamming score averages over individual label predictions, so
      getting 5 out of 6 labels right gives a Hamming score of 5/6 ≈ 0.83.
      With K=6 labels, even a model with 90% per-label accuracy would
      have subset accuracy of only 0.9^6 ≈ 0.53.

5. Why is Jaccard score often more informative than Hamming score
   when active labels are rare?
   -> If most labels are 0 (inactive), a model that always predicts 0
      achieves a very high Hamming score (most labels are correctly
      predicted as 0) but completely fails to identify any active labels.
      Jaccard score computes |predicted ∩ true| / |predicted ∪ true|
      and only considers active (=1) labels. A model that predicts all
      zeros gets a Jaccard score of 0 for observations with active labels.
      Therefore Jaccard score punishes missing active labels, making it
      much more sensitive to the quality of positive predictions.

6. Why can CCC be viewed as searching for a self-consistent label vector?
   -> In CCC, each label is predicted conditioned on all other labels.
      During iterative prediction, we keep updating each label using
      the current estimates of all others. The algorithm converges when
      no label changes after a full pass — meaning each label prediction
      is CONSISTENT with the predictions of all other labels.
      A self-consistent vector y_hat satisfies:
        y_hat_k = argmax P(Yk | X, y_hat_{-k})  for all k simultaneously.
      This is analogous to a fixed point of the iterative update operator.

7. Does CCC define one coherent joint model P(Y1,...,YK | X=x),
   or only a collection of conditional models?
   Why is this distinction important?
   -> CCC defines only a COLLECTION of K conditional models, one per label:
        p_k(x, y_{-k}) ≈ P(Yk=1 | X=x, Y_{-k}=y_{-k}), k=1,...,K.
      These K conditional distributions are not guaranteed to be
      consistent with any single joint distribution P(Y1,...,YK|X).
      In general, there may be no joint distribution whose conditionals
      are exactly the K fitted models (this is the issue of
      "compatible conditionals" in probabilistic graphical models).
      This distinction matters because:
      - Without a coherent joint model
""")