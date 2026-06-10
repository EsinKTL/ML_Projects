import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Veriyi üret — hocanın verdiği kod
np.random.seed(42)
n_strip = 16      # sınır yakınındaki küçük grup
n_blob  = 6000    # uzaktaki büyük Gaussian kümesi

# Sınıf 0: büyük küme [0,10] etrafında + sınır yakınında küçük grup [0, 0.3-0.8]
x0_1 = np.hstack([
	np.random.normal(scale=0.6, size=(n_strip, 1)),
	np.random.uniform(0.3, 0.8, size=(n_strip, 1))
])
x0_2 = np.random.multivariate_normal(
	mean=[0, 10],
	cov=np.array([[4, 3.5], [3.5, 4]]),
	size=n_blob
)
X0 = np.vstack([x0_1, x0_2])
y0 = np.zeros(n_strip + n_blob)

# Sınıf 1: büyük küme [0,-10] etrafında + sınır yakınında küçük grup [0, -0.8 ile -0.3]
x1_1 = np.hstack([
	np.random.normal(scale=0.6, size=(n_strip, 1)),
	np.random.uniform(-0.8, -0.3, size=(n_strip, 1))
])
x1_2 = np.random.multivariate_normal(
	mean=[0, -10],
	cov=np.array([[4, 3.5], [3.5, 4]]),
	size=n_blob
)
X1 = np.vstack([x1_1, x1_2])
y1 = np.ones(n_strip + n_blob)

X = np.vstack([X0, X1])
y = np.concatenate([y0, y1])

# %80 eğitim, %20 test
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# StandardScaler: her değişkeni ortalama=0, std=1 yapar
# SVM ve logistic regression ölçeğe duyarlıdır — standartlaştırma şart
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # eğitim verisiyle fit et ve dönüştür
X_test_s  = scaler.transform(X_test)         # test verisini sadece dönüştür (fit etme!)

# solver='lbfgs': büyük veri için etkili optimizasyon algoritması
# max_iter=1000: yakınsama için yeterli iterasyon
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train_s, y_train)

train_acc_lr = logreg.score(X_train_s, y_train)
test_acc_lr  = logreg.score(X_test_s,  y_test)

print("=== Logistic Regression ===")
print(f"Training Accuracy : {train_acc_lr:.4f}")
print(f"Testing  Accuracy : {test_acc_lr:.4f}")

# LİNEER SVM

# kernel='linear': düz çizgi karar sınırı
# C=1.0: ceza parametresi
#   C büyük → margin küçük, az hata tolere edilir → overfitting riski
#   C küçük → margin büyük, daha fazla hata tolere edilir → underfitting riski
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_s, y_train)

train_acc_svm = svm_linear.score(X_train_s, y_train)
test_acc_svm  = svm_linear.score(X_test_s,  y_test)

print("\n=== Linear SVM ===")
print(f"Training Accuracy : {train_acc_svm:.4f}")
print(f"Testing  Accuracy : {test_acc_svm:.4f}")
print(f"Number of support vectors: {svm_linear.support_vectors_.shape[0]}")

def plot_decision_regions(ax, model, X, y, title,
                          show_margin=False, show_sv=False):
	"""
	ax          : matplotlib ekseni
	model       : eğitilmiş sınıflandırıcı
	X           : standartlaştırılmış veri (görselleştirme için)
	y           : etiketler
	title       : grafik başlığı
	show_margin : SVM margin çizgilerini göster
	show_sv     : support vector'ları vurgula
	"""
	
	# Grid oluştur — tüm alanı kaplayan ızgara
	x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
	x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
	xx, yy = np.meshgrid(
		np.linspace(x1_min, x1_max, 300),
		np.linspace(x2_min, x2_max, 300)
	)
	grid = np.column_stack([xx.ravel(), yy.ravel()])
	
	# Her grid noktası için tahmin yap
	Z = model.predict(grid).reshape(xx.shape)
	
	# Karar bölgelerini renklendir
	# alpha=0.3: yarı saydam — veri noktaları görünsün
	ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
	
	# Karar sınırı: Z=0.5 eşiğinde siyah çizgi
	ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
	
	# SVM için separating hyperplane ve margin çizgileri
	if show_margin and hasattr(model, 'decision_function'):
		# decision_function: her noktanın karar sınırına olan uzaklığı
		# =0  → karar sınırı (separating hyperplane)
		# =+1 → pozitif margin sınırı
		# =-1 → negatif margin sınırı
		Z_df = model.decision_function(grid).reshape(xx.shape)
		ax.contour(xx, yy, Z_df, levels=[-1, 0, 1],
		           colors=['blue', 'black', 'red'],
		           linestyles=['--', '-', '--'],
		           linewidths=[1.5, 2, 1.5])
	
	# Veri noktalarını çiz
	ax.scatter(X[y==0, 0], X[y==0, 1],
	           c='blue', alpha=0.3, s=5, label='Class 0')
	ax.scatter(X[y==1, 0], X[y==1, 1],
	           c='red', alpha=0.3, s=5, label='Class 1')
	
	# Support vector'ları vurgula
	if show_sv and hasattr(model, 'support_vectors_'):
		sv = model.support_vectors_
		ax.scatter(sv[:, 0], sv[:, 1],
		           s=100, facecolors='none',
		           edgecolors='green', linewidths=2,
		           label='Support Vectors', zorder=5)
	
	ax.set_title(title)
	ax.set_xlabel('X1 (standardized)')
	ax.set_ylabel('X2 (standardized)')
	ax.legend(markerscale=3)

# Görselleştirmek için veri boyutunu küçültelim (12016 nokta çok fazla)
# Rastgele 1000 nokta al
idx = np.random.choice(len(X_train_s), 1000, replace=False)
X_vis = X_train_s[idx]
y_vis = y_train[idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plot_decision_regions(axes[0], logreg, X_vis, y_vis,
                      title='Logistic Regression\nDecision Boundary')

plot_decision_regions(axes[1], svm_linear, X_vis, y_vis,
                      title='Linear SVM\nDecision Boundary + Margin + Support Vectors',
                      show_margin=True, show_sv=True)

plt.suptitle('Task 1: Linear Classifiers Comparison', fontsize=14)
plt.tight_layout()
plt.show()

# TASK 1f-g: C PARAMETRESİNİN ETKİSİ

C_values = [0.01, 0.1, 1, 10, 100]

print("\n=== C parametresi ve support vector sayısı ===")
print(f"{'C':>8} | {'Train Acc':>10} | {'Test Acc':>10} | {'# Support Vectors':>18}")
print("-" * 55)

fig, axes = plt.subplots(1, len(C_values), figsize=(20, 4))

for i, C in enumerate(C_values):
	# Her C için yeni SVM modeli
	svm_c = SVC(kernel='linear', C=C)
	svm_c.fit(X_train_s, y_train)
	
	train_acc = svm_c.score(X_train_s, y_train)
	test_acc  = svm_c.score(X_test_s,  y_test)
	n_sv      = svm_c.support_vectors_.shape[0]
	
	print(f"{C:>8.2f} | {train_acc:>10.4f} | {test_acc:>10.4f} | {n_sv:>18}")
	
	plot_decision_regions(axes[i], svm_c, X_vis, y_vis,
	                      title=f'SVM C={C}\nSV={n_sv}',
	                      show_margin=True, show_sv=True)

plt.suptitle('Task 1f-g: Effect of C parameter on SVM', fontsize=14)
plt.tight_layout()
plt.show()

# TASK 2: KERNEL SVM

# make_moons: iki hilal şeklinde sınıf — lineer ayrılamaz
# n_samples=300: 300 gözlem
# noise=0.25: gürültü miktarı
# random_state=123: tekrarlanabilirlik
X_m, y_m = make_moons(n_samples=300, noise=0.25, random_state=123)

# Train/test split
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
	X_m, y_m, test_size=0.2, random_state=42
)

# Standartlaştır
scaler_m = StandardScaler()
X_m_train_s = scaler_m.fit_transform(X_m_train)
X_m_test_s  = scaler_m.transform(X_m_test)

# 1. Logistic Regression (düz çizgi — zayıf performans beklenir)
lr_plain = LogisticRegression(max_iter=1000)
lr_plain.fit(X_m_train_s, y_m_train)

# 2. Logistic Regression + Polynomial Features degree=2
# PolynomialFeatures: x1, x2 → x1, x2, x1², x1*x2, x2²
# Pipeline: önce polynomial özellikler ekle, sonra logistic regression
lr_poly2 = Pipeline([
	('poly', PolynomialFeatures(degree=2, include_bias=False)),
	('lr',   LogisticRegression(max_iter=1000))
])
lr_poly2.fit(X_m_train_s, y_m_train)

# 3. Logistic Regression + Polynomial Features degree=3
lr_poly3 = Pipeline([
	('poly', PolynomialFeatures(degree=3, include_bias=False)),
	('lr',   LogisticRegression(max_iter=1000))
])
lr_poly3.fit(X_m_train_s, y_m_train)

# 4. Linear SVM (düz çizgi — zayıf performans beklenir)
svm_lin_m = SVC(kernel='linear', C=1.0)
svm_lin_m.fit(X_m_train_s, y_m_train)

# 5. SVM with Polynomial Kernel degree=2
# kernel='poly': k(x,z) = (coef0 + x^T z)^d
# C=10: ceza parametresi
# coef0=10: sabit terim — polinom kernel için önemli
# degree=2,3,4: polinom derecesi
svm_poly2 = SVC(kernel='poly', C=10, coef0=10, degree=2)
svm_poly3 = SVC(kernel='poly', C=10, coef0=10, degree=3)
svm_poly4 = SVC(kernel='poly', C=10, coef0=10, degree=4)

svm_poly2.fit(X_m_train_s, y_m_train)
svm_poly3.fit(X_m_train_s, y_m_train)
svm_poly4.fit(X_m_train_s, y_m_train)

# 6. SVM with RBF Kernel (C=1, varsayılan gamma)
# kernel='rbf': Radial Basis Function — Gaussian kernel
# k(x,z) = exp(-gamma * ||x-z||²)
# Her noktanın etki alanı gamma ile kontrol edilir
# gamma büyük → dar etki alanı → karmaşık sınır → overfitting
# gamma küçük → geniş etki alanı → düzgün sınır → underfitting
svm_rbf = SVC(kernel='rbf', C=1.0)
svm_rbf.fit(X_m_train_s, y_m_train)

# Tüm modellerin accuracy'sini yazdır
models = [
	('Logistic Regression',       lr_plain),
	('LR + Poly degree=2',        lr_poly2),
	('LR + Poly degree=3',        lr_poly3),
	('Linear SVM',                svm_lin_m),
	('SVM Poly kernel d=2',       svm_poly2),
	('SVM Poly kernel d=3',       svm_poly3),
	('SVM Poly kernel d=4',       svm_poly4),
	('SVM RBF kernel (C=1)',      svm_rbf),
]

print("\n=== Task 2: Model Comparison (Moons Dataset) ===")
print(f"{'Model':<30} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 55)
for name, model in models:
	tr = model.score(X_m_train_s, y_m_train)
	te = model.score(X_m_test_s,  y_m_test)
	print(f"{name:<30} | {tr:>10.4f} | {te:>10.4f}")

# TASK 2: GRAFİKLER — TÜM MODELLER

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(models):
	tr = model.score(X_m_train_s, y_m_train)
	te = model.score(X_m_test_s,  y_m_test)
	plot_decision_regions(
		axes[i], model,
		X_m_train_s, y_m_train,
		title=f'{name}\nTrain={tr:.3f} Test={te:.3f}'
	)

plt.suptitle('Task 2: Decision Boundaries — All Models', fontsize=14)
plt.tight_layout()
plt.show()

# TASK 2: RBF KERNEL — GAMMA ETKİSİ

gamma_values = [0.01, 0.1, 1, 10, 100]

print("\n=== RBF Kernel: gamma etkisi ===")
print(f"{'gamma':>8} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 35)

fig, axes = plt.subplots(1, len(gamma_values), figsize=(20, 4))

for i, gamma in enumerate(gamma_values):
	# C=1 sabit, sadece gamma değişiyor
	svm_g = SVC(kernel='rbf', C=1.0, gamma=gamma)
	svm_g.fit(X_m_train_s, y_m_train)
	
	tr = svm_g.score(X_m_train_s, y_m_train)
	te = svm_g.score(X_m_test_s,  y_m_test)
	
	print(f"{gamma:>8.2f} | {tr:>10.4f} | {te:>10.4f}")
	
	plot_decision_regions(
		axes[i], svm_g,
		X_m_train_s, y_m_train,
		title=f'RBF gamma={gamma}\nTrain={tr:.3f} Test={te:.3f}'
	)

plt.suptitle('Task 2: RBF Kernel — Effect of Gamma', fontsize=14)
plt.tight_layout()
plt.show()

print("""

(a) Both methods use linear decision boundaries. Why are the boundaries different?
    → Logistic regression maximizes log-likelihood using ALL observations.
      The large Gaussian clusters far from the boundary strongly pull
      the decision boundary toward them.
      SVM maximizes the margin and only depends on support vectors —
      observations close to the boundary. The large distant clusters
      have no influence on the SVM boundary at all.
      Therefore the two boundaries end up in different positions.

(b) Which observations become support vectors?
    → Observations that lie within the margin or on the wrong side
      of the boundary. In this dataset, the small group of points
      close to the separating boundary (n_strip = 16 points per class)
      become the support vectors.

(c) Are support vectors necessarily misclassified observations?
    → No. Support vectors fall into three categories:
      1. Correctly classified points that lie inside the margin
      2. Points exactly on the margin boundary
      3. Misclassified points (on the wrong side of the boundary)
      Most support vectors are correctly classified but simply
      close to the decision boundary.

(d) Does linear SVM estimate posterior probabilities P(Y = 1 | X = x)?
    → No. Linear SVM only outputs a class label (+1 or -1).
      It does not naturally produce probability estimates.
      In sklearn, setting probability=True enables approximate
      probabilities via Platt scaling, but this is not a native
      output of SVM — it requires an additional calibration step.

(e) Why does the SVM boundary depend mainly on observations close to
    the separating boundary, while logistic regression is influenced
    by all observations?
    → The SVM optimization problem is defined entirely by the support
      vectors. Points far from the boundary that are correctly
      classified contribute nothing to the solution — moving them
      does not change the boundary at all.
      Logistic regression sums the log-likelihood contribution of
      every single observation, so every point influences the
      estimated coefficients and thus the boundary.

(f) How does the SVM boundary depend on the C parameter?
    Consider C ∈ {0.01, 0.1, 1, 10, 100}.
    → C = 0.01 (very small): large margin is enforced, many
      violations are tolerated, boundary is smooth and stable.
    → C = 0.1:  moderate margin, fewer violations allowed.
    → C = 1:    default balance between margin size and violations.
    → C = 10:   small margin, very few violations tolerated,
      boundary starts fitting closely to boundary points.
    → C = 100 (very large): almost no violations allowed,
      the boundary hugs the support vectors tightly,
      risk of overfitting increases.

(g) For each value of C, report the number of support vectors.
    → As C increases, the margin shrinks and fewer points
      fall inside or violate the margin, so the number of
      support vectors decreases.
      As C decreases, the margin widens and more points
      fall inside it, so the number of support vectors increases.
      (Exact numbers are printed in the table above.)

=================================================================
TASK 2 — QUESTIONS AND ANSWERS
=================================================================

(a) Why do ordinary logistic regression and linear SVM perform
    poorly on this dataset?
    → The make_moons dataset is not linearly separable — the two
      classes form interleaved crescent shapes that cannot be
      separated by a straight line. Both ordinary logistic regression
      and linear SVM produce only linear decision boundaries,
      so they fundamentally cannot capture the curved structure
      of this data.

(b) Does the logistic regression with polynomial features of degree 3
    produce a proper decision boundary for this task?
    → Yes, generally it does. Adding degree-3 polynomial features
      (x1², x1·x2, x2², x1³, ...) transforms the input space so
      that a linear boundary in the expanded space corresponds to
      a curved boundary in the original space. This gives the model
      enough flexibility to approximate the crescent shapes.
      However, the degree must be chosen carefully — too low underfits,
      too high overfits.

(c) What does the RBF kernel allow the SVM to do?
    → The RBF (Radial Basis Function) kernel implicitly maps the data
      into an infinite-dimensional feature space using the formula:
      k(x, z) = exp(-gamma * ||x - z||²)
      This allows the SVM to find a linear boundary in that
      high-dimensional space, which corresponds to a highly
      complex nonlinear boundary in the original 2D space.
      The kernel trick means this is done efficiently without
      explicitly computing the high-dimensional coordinates.

(d) What happens when gamma is very small?
    → When gamma is very small, each point's Gaussian has a very
      wide radius of influence. All points look similar to each
      other, and the decision boundary becomes very smooth and
      nearly linear. The model underfits — both training and
      test accuracy are low.

(e) What happens when gamma is very large?
    → When gamma is very large, each point's Gaussian has a very
      narrow radius of influence. The model effectively memorizes
      the training data, producing a highly irregular boundary
      that wraps tightly around individual training points.
      Training accuracy approaches 1.0 but test accuracy drops
      significantly — classic overfitting.

(f) Compare logistic regression with polynomial features and SVM
    with RBF kernel. In what sense are these approaches similar,
    and in what sense are they different?
    → Similarity:
      Both extend linear models to capture nonlinear boundaries.
      Both can approximate complex decision surfaces in 2D.
    → Differences:
      LR + Poly: explicitly constructs new features (x², x·y, ...),
        so the dimensionality grows rapidly with degree.
        The degree must be chosen manually.
      SVM + RBF: uses the kernel trick to implicitly work in an
        infinite-dimensional space without ever constructing
        the features explicitly. The complexity is controlled
        by gamma and C rather than a polynomial degree.
        SVM + RBF is generally more flexible and computationally
        efficient for nonlinear problems.
""")

