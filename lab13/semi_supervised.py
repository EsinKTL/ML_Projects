import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

# =============================================================
# STEP 1: GENERATE DATASET
# =============================================================

# make_moons: two interleaved half-circles — a classic non-linearly separable dataset
# n_samples=1000: total observations
# noise=0.1: small amount of Gaussian noise added to the coordinates
# random_state=42: reproducibility — same data every run

X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Visualize the full dataset to understand the structure
plt.figure(figsize=(8, 5))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.4, s=15, label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red',  alpha=0.4, s=15, label='Class 1')
plt.title('Full Dataset (make_moons)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================
# STEP 2: SPLIT INTO TRAINING AND TEST SETS
# =============================================================

# test_size=0.3: 30% for testing, 70% for training
# random_state=42: reproducibility
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=42
)

print(f"Total samples      : {len(X)}")
print(f"Training samples   : {len(X_train)}")
print(f"Test samples       : {len(X_test)}")

# =============================================================
# STEP 3: STANDARDIZE THE FEATURES
# =============================================================

# SVM is sensitive to feature scale — standardization is mandatory
# StandardScaler: transforms each feature to mean=0, std=1
# IMPORTANT: fit only on training data, then transform both sets
# Fitting on test data would be "data leakage"

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit + transform on train
X_test_s  = scaler.transform(X_test)        # only transform on test

# =============================================================
# STEP 4: CREATE PARTIALLY LABELED TRAINING SET
# =============================================================

# g = number of LABELED observations in training set
# The rest are UNLABELED — their labels are set to -1
# -1 is the sklearn convention for "unlabeled" in semi-supervised methods
# This simulates the real-world scenario where labeling is expensive

g = 20  # only 20 labeled examples out of 700 training observations

# Create label array where most entries are -1 (unlabeled)
y_train_ssl = np.full(len(y_train), -1, dtype=int)

# Randomly select g indices to be labeled
# random_state=0 for reproducibility
rng = np.random.default_rng(seed=0)
labeled_idx = rng.choice(len(y_train), size=g, replace=False)

# Assign true labels only to those g selected observations
y_train_ssl[labeled_idx] = y_train[labeled_idx]

print(f"\nLabeled observations   : {g}")
print(f"Unlabeled observations : {(y_train_ssl == -1).sum()}")
print(f"Fraction labeled       : {g/len(y_train):.3f}")

# =============================================================
# EVALUATION HELPER FUNCTION
# =============================================================

def evaluate(name, model, X_test, y_test):
	"""
	Computes and prints four metrics for a fitted classifier.

	Parameters:
		name   : string name of the method
		model  : fitted sklearn estimator
		X_test : standardized test features
		y_test : true test labels

	Metrics:
		Accuracy         : fraction of correct predictions
						   = (TP + TN) / (TP + TN + FP + FN)
		Balanced Accuracy: average recall per class
						   = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
						   better than accuracy for imbalanced datasets
		F1 Score         : harmonic mean of precision and recall
						   = 2 * precision * recall / (precision + recall)
						   focuses on positive class performance
		ROC AUC          : area under the ROC curve
						   = probability that model ranks a random positive
							 higher than a random negative
						   1.0 = perfect, 0.5 = random
	"""
	y_pred = model.predict(X_test)
	
	acc  = accuracy_score(y_test, y_pred)
	bacc = balanced_accuracy_score(y_test, y_pred)
	f1   = f1_score(y_test, y_pred)
	
	# AUC requires probability scores, not all models support this
	if hasattr(model, 'predict_proba'):
		y_score = model.predict_proba(X_test)[:, 1]
		auc = roc_auc_score(y_test, y_score)
	elif hasattr(model, 'decision_function'):
		y_score = model.decision_function(X_test)
		auc = roc_auc_score(y_test, y_score)
	else:
		auc = float('nan')
	
	print(f"  {name:<30}: Acc={acc:.3f}, BAcc={bacc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
	return acc, bacc, f1, auc

# =============================================================
# METHOD 1: NAIVE METHOD (only labeled data)
# =============================================================

# The naive approach: ignore all unlabeled data entirely
# Train only on the g=20 labeled observations
# This is the baseline — SSL methods should improve on this

# Extract only the labeled observations
labeled_mask = (y_train_ssl != -1)  # boolean mask: True where labeled

# SVC with RBF kernel and probability=True for AUC computation
# C=1.0: default regularization
# probability=True: enables predict_proba() using Platt scaling
naive_svc = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
naive_svc.fit(X_train_s[labeled_mask], y_train_ssl[labeled_mask])

print("\n=== Results with g=20 labeled observations ===\n")
print("Method comparison:")
r_naive = evaluate("Naive (g labeled only)", naive_svc, X_test_s, y_test)

# =============================================================
# METHOD 2: SELF-TRAINING
# =============================================================

# Self-Training algorithm:
# 1. Start with a base classifier trained on labeled data
# 2. Predict labels for unlabeled data
# 3. Add high-confidence predictions to the labeled set
# 4. Repeat until no more unlabeled data or convergence
#
# threshold: minimum prediction confidence to add as pseudo-label
# criterion='threshold': add if max probability >= threshold
# max_iter: maximum rounds of pseudo-labeling
#
# KEY IDEA: the classifier iteratively "teaches itself" using
# its own confident predictions on unlabeled data

base_svc_st = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
self_train = SelfTrainingClassifier(
	base_estimator=base_svc_st,
	threshold=0.75,    # only use predictions with confidence >= 75%
	criterion='threshold',
	max_iter=10,       # maximum 10 rounds of self-labeling
	verbose=False
)
self_train.fit(X_train_s, y_train_ssl)

r_st = evaluate("Self-Training           ", self_train, X_test_s, y_test)

# =============================================================
# METHOD 3: LABEL PROPAGATION
# =============================================================

# Label Propagation builds a graph where nodes are all observations
# (labeled + unlabeled). Edges are weighted by similarity (kernel).
# Labels "flow" from labeled nodes to unlabeled nodes through the graph.
#
# Algorithm:
# 1. Build affinity matrix W where W_ij = similarity(x_i, x_j)
# 2. Propagate labels: F = D^(-1) * W * F (D = degree matrix)
# 3. Labeled nodes are "clamped" — their labels cannot change
# 4. Repeat until convergence
#
# kernel='rbf': similarity = exp(-gamma * ||xi - xj||^2)
# gamma: controls how quickly similarity decays with distance
#   gamma large  = only very close points are similar (local)
#   gamma small  = all points somewhat similar (global)
#
# IMPORTANT: labeled nodes are FIXED (clamped) — their labels never change
# This can be sensitive to label noise

lp = LabelPropagation(
	kernel='rbf',
	gamma=20,          # RBF bandwidth parameter
	max_iter=1000,     # maximum propagation iterations
	tol=1e-3           # convergence tolerance
)
lp.fit(X_train_s, y_train_ssl)

r_lp = evaluate("Label Propagation       ", lp, X_test_s, y_test)

# =============================================================
# METHOD 4: LABEL SPREADING
# =============================================================

# Label Spreading is a variant of Label Propagation with two key differences:
# 1. Labeled nodes are NOT fully clamped — they can change slightly
#    controlled by alpha parameter
# 2. Uses normalized Laplacian instead of raw affinity matrix
#    this makes it more robust to noise and outliers
#
# alpha: controls balance between original labels and propagated labels
#   alpha = 0: fully trust original labels (= Label Propagation)
#   alpha = 1: ignore original labels, rely entirely on graph structure
#   alpha = 0.2: 20% weight on graph structure, 80% on original labels
#
# WHY is this better? If some labeled points are outliers or mislabeled,
# Label Propagation locks in those wrong labels. Label Spreading can
# "correct" them slightly using the surrounding graph structure.

ls = LabelSpreading(
	kernel='rbf',
	gamma=20,          # same as LabelPropagation for fair comparison
	alpha=0.2,         # 20% propagation, 80% original labels
	max_iter=1000,
	tol=1e-3
)
ls.fit(X_train_s, y_train_ssl)

r_ls = evaluate("Label Spreading         ", ls, X_test_s, y_test)

# =============================================================
# DECISION REGION VISUALIZATION
# =============================================================

def plot_ssl_decision_regions(models_dict, X_train_s, y_train_ssl,
                              X_test_s, y_test, g):
	"""
	Plots decision regions for all SSL methods side by side.

	Parameters:
		models_dict  : dict of {name: fitted_model}
		X_train_s    : standardized training features
		y_train_ssl  : training labels with -1 for unlabeled
		X_test_s     : standardized test features
		y_test       : true test labels
		g            : number of labeled observations
	"""
	n_methods = len(models_dict)
	fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
	
	# Create grid over the feature space
	x1_min = min(X_train_s[:,0].min(), X_test_s[:,0].min()) - 0.5
	x1_max = max(X_train_s[:,0].max(), X_test_s[:,0].max()) + 0.5
	x2_min = min(X_train_s[:,1].min(), X_test_s[:,1].min()) - 0.5
	x2_max = max(X_train_s[:,1].max(), X_test_s[:,1].max()) + 0.5
	
	xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
	                     np.linspace(x2_min, x2_max, 300))
	grid = np.column_stack([xx.ravel(), yy.ravel()])
	
	for ax, (name, model) in zip(axes, models_dict.items()):
		# Decision regions
		Z = model.predict(grid).reshape(xx.shape)
		ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
		ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
		
		# Unlabeled training points (gray, small)
		unlabeled_mask = (y_train_ssl == -1)
		ax.scatter(X_train_s[unlabeled_mask, 0], X_train_s[unlabeled_mask, 1],
		           c='gray', alpha=0.2, s=8, label='Unlabeled')
		
		# Labeled training points (colored, larger with black edge)
		labeled_mask = (y_train_ssl != -1)
		ax.scatter(X_train_s[labeled_mask & (y_train_ssl==0), 0],
		           X_train_s[labeled_mask & (y_train_ssl==0), 1],
		           c='blue', s=80, edgecolors='black', linewidths=1.5,
		           zorder=5, label='Labeled 0')
		ax.scatter(X_train_s[labeled_mask & (y_train_ssl==1), 0],
		           X_train_s[labeled_mask & (y_train_ssl==1), 1],
		           c='red', s=80, edgecolors='black', linewidths=1.5,
		           zorder=5, label='Labeled 1')
		
		acc = model.score(X_test_s, y_test)
		ax.set_title(f'{name}\nTest Acc={acc:.3f}')
		ax.set_xlabel('X1 (standardized)')
		ax.set_ylabel('X2 (standardized)')
		ax.legend(fontsize=7, markerscale=0.8)
	
	plt.suptitle(f'Decision Regions — g={g} labeled observations', fontsize=13)
	plt.tight_layout()
	plt.show()

models_dict = {
	'Naive': naive_svc,
	'Self-Training': self_train,
	'Label Prop.': lp,
	'Label Spreading': ls
}
plot_ssl_decision_regions(models_dict, X_train_s, y_train_ssl,
                          X_test_s, y_test, g)

# =============================================================
# STEP 5: ANALYZE HOW g AFFECTS RESULTS
# =============================================================

# g = number of labeled observations
# As g increases: all methods should improve
# SSL methods should be much better than Naive for small g
# As g grows large, SSL advantage shrinks (enough labels for supervised)

g_values = [5, 10, 20, 30, 50, 100, 200, 350]

# Store results for plotting
results = {
	'Naive':            {'acc': [], 'bacc': [], 'f1': [], 'auc': []},
	'Self-Training':    {'acc': [], 'bacc': [], 'f1': [], 'auc': []},
	'Label Prop.':      {'acc': [], 'bacc': [], 'f1': [], 'auc': []},
	'Label Spreading':  {'acc': [], 'bacc': [], 'f1': [], 'auc': []},
}

print("\n=== Effect of g on all methods ===\n")
print(f"{'g':>5} | {'Method':<22} | {'Acc':>6} | {'BAcc':>6} | {'F1':>6} | {'AUC':>6}")
print("-" * 70)

for g_val in g_values:
	# Create partially labeled training set for this g
	y_ssl = np.full(len(y_train), -1, dtype=int)
	idx   = np.random.default_rng(seed=g_val).choice(
		len(y_train), size=g_val, replace=False)
	y_ssl[idx] = y_train[idx]
	lmask = (y_ssl != -1)
	
	# --- Naive ---
	m_naive = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
	m_naive.fit(X_train_s[lmask], y_ssl[lmask])
	a,b,f,u = (accuracy_score(y_test, m_naive.predict(X_test_s)),
	           balanced_accuracy_score(y_test, m_naive.predict(X_test_s)),
	           f1_score(y_test, m_naive.predict(X_test_s)),
	           roc_auc_score(y_test, m_naive.predict_proba(X_test_s)[:,1]))
	results['Naive']['acc'].append(a); results['Naive']['bacc'].append(b)
	results['Naive']['f1'].append(f);  results['Naive']['auc'].append(u)
	print(f"{g_val:>5} | {'Naive':<22} | {a:>6.3f} | {b:>6.3f} | {f:>6.3f} | {u:>6.3f}")
	
	# --- Self-Training ---
	m_st = SelfTrainingClassifier(
		SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
		threshold=0.75, max_iter=10)
	m_st.fit(X_train_s, y_ssl)
	a,b,f,u = (accuracy_score(y_test, m_st.predict(X_test_s)),
	           balanced_accuracy_score(y_test, m_st.predict(X_test_s)),
	           f1_score(y_test, m_st.predict(X_test_s)),
	           roc_auc_score(y_test, m_st.predict_proba(X_test_s)[:,1]))
	results['Self-Training']['acc'].append(a); results['Self-Training']['bacc'].append(b)
	results['Self-Training']['f1'].append(f);  results['Self-Training']['auc'].append(u)
	print(f"{g_val:>5} | {'Self-Training':<22} | {a:>6.3f} | {b:>6.3f} | {f:>6.3f} | {u:>6.3f}")
	
	# --- Label Propagation ---
	m_lp = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
	m_lp.fit(X_train_s, y_ssl)
	a,b,f,u = (accuracy_score(y_test, m_lp.predict(X_test_s)),
	           balanced_accuracy_score(y_test, m_lp.predict(X_test_s)),
	           f1_score(y_test, m_lp.predict(X_test_s)),
	           roc_auc_score(y_test, m_lp.predict_proba(X_test_s)[:,1]))
	results['Label Prop.']['acc'].append(a); results['Label Prop.']['bacc'].append(b)
	results['Label Prop.']['f1'].append(f);  results['Label Prop.']['auc'].append(u)
	print(f"{g_val:>5} | {'Label Prop.':<22} | {a:>6.3f} | {b:>6.3f} | {f:>6.3f} | {u:>6.3f}")
	
	# --- Label Spreading ---
	m_ls = LabelSpreading(kernel='rbf', gamma=20, alpha=0.2, max_iter=1000)
	m_ls.fit(X_train_s, y_ssl)
	a,b,f,u = (accuracy_score(y_test, m_ls.predict(X_test_s)),
	           balanced_accuracy_score(y_test, m_ls.predict(X_test_s)),
	           f1_score(y_test, m_ls.predict(X_test_s)),
	           roc_auc_score(y_test, m_ls.predict_proba(X_test_s)[:,1]))
	results['Label Spreading']['acc'].append(a); results['Label Spreading']['bacc'].append(b)
	results['Label Spreading']['f1'].append(f);  results['Label Spreading']['auc'].append(u)
	print(f"{g_val:>5} | {'Label Spreading':<22} | {a:>6.3f} | {b:>6.3f} | {f:>6.3f} | {u:>6.3f}")
	print("-" * 70)

# =============================================================
# STEP 6: PLOT HOW g AFFECTS ACCURACY
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {'Naive': 'black', 'Self-Training': 'blue',
          'Label Prop.': 'green', 'Label Spreading': 'red'}
linestyles = {'Naive': '--', 'Self-Training': '-',
              'Label Prop.': '-', 'Label Spreading': '-'}

for ax, metric, title in zip(axes, ['acc', 'auc'],
                             ['Accuracy vs g', 'ROC AUC vs g']):
	for method in results:
		ax.plot(g_values, results[method][metric],
		        color=colors[method],
		        linestyle=linestyles[method],
		        marker='o', linewidth=2,
		        label=method)
	ax.set_xlabel('g (number of labeled observations)', fontsize=12)
	ax.set_ylabel(metric.upper(), fontsize=12)
	ax.set_title(title, fontsize=13)
	ax.legend()
	ax.grid(True)

plt.suptitle('Effect of Labeled Sample Size on SSL Methods', fontsize=14)
plt.tight_layout()
plt.show()

# =============================================================
# QUESTIONS AND ANSWERS
# =============================================================

print("""
=================================================================
QUESTIONS AND ANSWERS — SEMI-SUPERVISED LEARNING (Lab 13)
=================================================================

General understanding questions based on the lab tasks:

─────────────────────────────────────────────────────────────────
Q1. What is semi-supervised learning and why is it useful?
─────────────────────────────────────────────────────────────────
A: Semi-supervised learning (SSL) is a learning paradigm that uses
   BOTH labeled and unlabeled data during training.
   In supervised learning, every training observation has a label.
   In SSL, only a small subset g of training observations are labeled,
   and the rest (often much larger) are unlabeled.

   Why useful?
   -> In many real-world problems, labeling data is expensive and
      time-consuming (e.g., medical images, legal documents, speech).
      Collecting raw unlabeled data is cheap (just gather it),
      but having an expert label each example costs money and time.
      SSL allows us to leverage the large pool of unlabeled data
      to improve performance beyond what g labeled examples alone
      can achieve.

   Example: You have 10,000 medical scans but only 50 have been
   diagnosed by a doctor. SSL uses all 10,000 to train, while
   supervised learning uses only 50.

─────────────────────────────────────────────────────────────────
Q2. What is the Naive Method and why is it the baseline?
─────────────────────────────────────────────────────────────────
A: The Naive Method simply discards all unlabeled data and trains
   a standard supervised classifier using ONLY the g labeled examples.

   It is the baseline because:
   -> It represents what a purely supervised approach achieves
      with the same small labeled set.
   -> Any SSL method that is worth using should outperform this
      baseline, especially for small g.
   -> If an SSL method performs WORSE than naive, it means the
      unlabeled data is hurting rather than helping (e.g., wrong
      manifold assumption, cluster mismatch).

   Expected behavior:
   -> For small g (e.g., g=5,10), naive performs poorly because
      there is too little data to learn the decision boundary.
   -> As g grows, naive improves but SSL methods converge to
      similar performance — the labeled data becomes sufficient.

─────────────────────────────────────────────────────────────────
Q3. How does Self-Training work? What is its key assumption?
─────────────────────────────────────────────────────────────────
A: Self-Training is an iterative SSL method:
   1. Train a base classifier on labeled data only.
   2. Predict labels for all unlabeled observations.
   3. Take the MOST CONFIDENT predictions (above threshold).
   4. Add those pseudo-labeled observations to the training set.
   5. Retrain the classifier on the expanded labeled set.
   6. Repeat until convergence or max_iter.

   Key assumption: HIGH-CONFIDENCE predictions are likely correct.
   -> If the model is very sure (e.g., probability 0.95 for class 1),
      we trust that prediction and use it as a label.
   -> Low-confidence predictions are ignored to avoid error propagation.

   Risk: If the initial classifier makes confident but wrong predictions
   (e.g., at the start with g=5), these errors get added to training
   data and the model "reinforces" its own mistakes.
   This is called "confirmation bias" in Self-Training.

   Parameter choices:
   -> threshold=0.75: only pseudo-label if confidence >= 75%
      (lower = more data but noisier, higher = less data but cleaner)
   -> max_iter=10: prevents infinite loops

─────────────────────────────────────────────────────────────────
Q4. How does Label Propagation work?
─────────────────────────────────────────────────────────────────
A: Label Propagation treats ALL observations (labeled + unlabeled)
   as nodes in a GRAPH. Edges between nodes are weighted by
   similarity (using a kernel function).

   Algorithm:
   1. Build affinity matrix W: W_ij = exp(-gamma * ||xi - xj||^2)
      (high weight if xi and xj are close together)
   2. Normalize: compute transition matrix T = D^(-1) W
      where D_ii = sum_j W_ij (degree of node i)
   3. Initialize: labeled nodes get their true labels,
      unlabeled nodes start with uniform distribution [0.5, 0.5]
   4. Update: F_t+1 = T * F_t  (propagate labels through edges)
   5. Clamp labeled nodes back to their true labels after each step
   6. Repeat until convergence

   Key idea: Labels "flow" through the graph from labeled to unlabeled
   nodes. If unlabeled point A is very similar to labeled point B
   (class 1), then A will likely also get label 1.

   Key assumption: CLUSTER ASSUMPTION — points in the same cluster
   (connected dense region) tend to have the same label.

   Limitation: Labeled nodes are fully clamped (frozen). A noisy or
   mislabeled point permanently propagates the wrong label.

─────────────────────────────────────────────────────────────────
Q5. How does Label Spreading differ from Label Propagation?
─────────────────────────────────────────────────────────────────
A: Label Spreading has two key differences:

   1. SOFT CLAMPING (controlled by alpha):
      -> In Label Propagation, labeled nodes are HARD CLAMPED —
         their labels never change during propagation.
      -> In Label Spreading, labeled nodes can be influenced by
         their neighbors to a degree controlled by alpha.
      -> Update rule: F_t+1 = alpha * T * F_t + (1 - alpha) * Y_0
         where Y_0 is the initial label matrix.
      -> alpha=0: fully trust original labels (= Label Propagation)
      -> alpha=1: ignore original labels entirely (pure propagation)
      -> alpha=0.2: 20% from graph, 80% from original labels

   2. NORMALIZED LAPLACIAN (instead of row-normalized transition matrix):
      -> Label Spreading uses the symmetric normalized Laplacian.
      -> This is more theoretically grounded and robust.

   When to prefer Label Spreading over Label Propagation?
   -> When labeled data may contain noise or mislabeled points.
   -> When you want smoother, more robust label propagation.
   -> When the graph structure should partially override label information.

─────────────────────────────────────────────────────────────────
Q6. What is the role of gamma in Label Propagation/Spreading?
─────────────────────────────────────────────────────────────────
A: gamma controls the RBF kernel: k(xi, xj) = exp(-gamma * ||xi-xj||^2)

   gamma SMALL (e.g., 0.1):
   -> The Gaussian is WIDE — even distant points have high similarity.
   -> The graph is densely connected.
   -> Labels propagate far, creating smooth boundaries.
   -> Risk: labels spread too far across class boundaries.

   gamma LARGE (e.g., 100):
   -> The Gaussian is NARROW — only very close points are similar.
   -> The graph is sparsely connected (mostly local connections).
   -> Labels propagate only to immediate neighbors.
   -> Risk: isolated labeled points may not reach far unlabeled regions.

   Choice: gamma should reflect the intrinsic scale of the data.
   Typically chosen by cross-validation or based on domain knowledge.
   Rule of thumb: gamma ≈ 1 / (2 * sigma^2) where sigma is the
   typical distance between nearby points.

─────────────────────────────────────────────────────────────────
Q7. Why use SVC (Support Vector Classifier) as the base classifier?
─────────────────────────────────────────────────────────────────
A: SVC with RBF kernel is a good base classifier for SSL because:

   1. Effective with few labeled examples:
      SVM maximizes the margin, which provides good generalization
      even when the training set is very small (g=20).
      Margin maximization acts as implicit regularization.

   2. Non-linear boundaries:
      The RBF kernel allows SVC to learn curved decision boundaries
      like the make_moons crescent shapes without explicit feature
      engineering.

   3. Well-calibrated uncertainty (with probability=True):
      SVC + Platt scaling provides probability estimates needed for
      Self-Training thresholding.

   4. Works well in low-to-medium dimensional spaces:
      make_moons is 2D — SVC excels here.

   Alternative base classifiers: Random Forest, Logistic Regression,
   kNN. The choice can affect Self-Training behavior significantly.

─────────────────────────────────────────────────────────────────
Q8. How does g affect the results? What do you expect to observe?
─────────────────────────────────────────────────────────────────
A: Expected behavior as g increases:

   Small g (5-20):
   -> Naive performs poorly — too few examples to learn the boundary.
   -> SSL methods (Label Prop, Label Spreading) leverage unlabeled
      data and should outperform naive significantly.
   -> Self-Training may struggle if initial classifier is very poor
      (confirmation bias risk).

   Medium g (50-100):
   -> All methods improve. The gap between SSL and naive shrinks.
   -> SSL methods still have an advantage from unlabeled structure.

   Large g (200-350):
   -> All methods reach similar performance.
   -> The labeled set is large enough that unlabeled data adds
      little additional information.
   -> Naive may even match or beat SSL methods (no noise from
      unlabeled data).

   General trend:
   -> SSL advantage is largest for VERY SMALL g.
   -> Label Propagation / Label Spreading typically outperform
      Self-Training for very small g because they use ALL
      unlabeled data simultaneously rather than iteratively.

─────────────────────────────────────────────────────────────────
Q9. What are the key assumptions underlying SSL methods?
─────────────────────────────────────────────────────────────────
A: SSL methods rely on at least one of these assumptions:

   1. SMOOTHNESS ASSUMPTION:
      If two points x1 and x2 are close in the input space,
      their labels should be the same.
      -> Basis for kernel-based methods (Label Propagation).

   2. CLUSTER ASSUMPTION:
      Points in the same cluster tend to share the same label.
      -> If you can identify clusters in unlabeled data, all
         points in a cluster likely have the same class.

   3. MANIFOLD ASSUMPTION:
      High-dimensional data lies on a low-dimensional manifold.
      Points on the same manifold component share the same label.
      -> The crescent shapes in make_moons are 1D manifolds
         embedded in 2D space.

   If these assumptions are VIOLATED:
   -> SSL can HURT performance compared to naive supervised learning.
   -> Example: if unlabeled data comes from a different distribution
      than labeled data, propagated labels will be wrong.

─────────────────────────────────────────────────────────────────
Q10. What is the difference between accuracy and balanced accuracy?
     When does balanced accuracy matter more?
─────────────────────────────────────────────────────────────────
A: Accuracy:
   = (TP + TN) / (TP + TN + FP + FN)
   = fraction of ALL predictions that are correct
   -> Problem: if 90% of data is class 0, a model that always
      predicts class 0 gets 90% accuracy without learning anything.

   Balanced Accuracy:
   = 0.5 * (TPR + TNR)
   = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
   = average recall per class
   -> A model that always predicts class 0 gets 0.5 * (0 + 1) = 0.5
      (no better than random).

   When does balanced accuracy matter more?
   -> When the dataset is IMBALANCED (one class much more frequent).
   -> When both classes are equally important (e.g., disease detection).
   -> In SSL with small g: with few labeled examples, the model may
      be biased toward predicting the majority class — balanced accuracy
      reveals this.

   For make_moons (balanced 50/50), both metrics should be similar.

─────────────────────────────────────────────────────────────────
Q11. Why might Label Propagation sometimes fail?
─────────────────────────────────────────────────────────────────
A: Label Propagation can fail in several scenarios:

   1. WRONG GRAPH STRUCTURE:
      If gamma is too large or too small, the graph does not
      capture the true data geometry. Labels propagate to wrong
      regions.

   2. LABEL NOISE:
      Since labeled nodes are hard-clamped, a single mislabeled
      point permanently corrupts the propagation in its neighborhood.
      Label Spreading handles this better with soft clamping.

   3. DISCONNECTED COMPONENTS:
      If the graph has isolated components with no labeled nodes,
      labels cannot propagate to those components.
      The unlabeled points remain with their initial distribution.

   4. ASSUMPTION VIOLATION:
      If the true decision boundary cuts through a dense cluster
      (not the gap between clusters), the cluster assumption fails
      and Label Propagation propagates wrong labels.

   5. SCALABILITY:
      Label Propagation requires storing and computing the full
      n×n affinity matrix. For large n (millions of points),
      this is computationally infeasible.

─────────────────────────────────────────────────────────────────
Q12. Compare the four methods: Naive, Self-Training,
     Label Propagation, Label Spreading
─────────────────────────────────────────────────────────────────
A:
   ┌─────────────────┬────────────────────────────────────────────┐
   │ Method          │ Key Characteristics                        │
   ├─────────────────┼────────────────────────────────────────────┤
   │ Naive           │ Supervised only. Ignores unlabeled data.   │
   │                 │ Baseline. Poor for small g.                │
   ├─────────────────┼────────────────────────────────────────────┤
   │ Self-Training   │ Iterative pseudo-labeling. Requires        │
   │                 │ confident predictions. Risk of             │
   │                 │ confirmation bias.                         │
   ├─────────────────┼────────────────────────────────────────────┤
   │ Label           │ Graph-based. Hard-clamped labels.          │
   │ Propagation     │ Sensitive to label noise. Cluster          │
   │                 │ assumption. Good when data well-clustered. │
   ├─────────────────┼────────────────────────────────────────────┤
   │ Label           │ Graph-based. Soft-clamped labels (alpha).  │
   │ Spreading       │ More robust to noise. Generally preferred  │
   │                 │ over Label Propagation in practice.        │
   └─────────────────┴────────────────────────────────────────────┘

   Expected ranking for small g on make_moons:
   Label Spreading ≈ Label Propagation > Self-Training > Naive

   Expected ranking for large g:
   All methods converge to similar performance.
""")