# Applied Machine Learning — Complete Final Study Notes

---

## Table of Contents

1. [Bayesian Classification — LDA, QDA, Naive Bayes (Lab 1-2)](#1-bayesian-classification)
2. [Density Estimation (Lab 3)](#2-density-estimation)
3. [Logistic Regression (Lab 4)](#3-logistic-regression)
4. [Evaluation Methods (Lab 5)](#4-evaluation-methods)
5. [Classification Trees, Bagging, Random Forest (Lab 6)](#5-classification-trees)
6. [Logistic Regression with Regularization (Lab 7)](#6-regularization)
7. [Ensemble Methods — AdaBoost, Gradient Boosting (Lab 8-9)](#7-ensemble-methods)
8. [Feature Selection (Lab 10)](#8-feature-selection)
9. [SVM — Support Vector Machines (Lab 11)](#9-svm)
10. [Regression I — Nadaraya-Watson, Smoothing Splines (Lab 12)](#10-regression)
11. [Semi-Supervised Learning (Lab 13)](#11-semi-supervised-learning)
12. [Multi-label Classification (Lab 14)](#12-multi-label-classification)
13. [Key Formulas Quick Reference](#13-key-formulas)
14. [Code Patterns Quick Reference](#14-code-patterns)

---

## 1. Bayesian Classification

### Temel Fikir
Bayes teoremini kullanarak her sınıfın olasılığını hesapla, en yüksek olasılıklı sınıfa ata.

$$P(Y=k \mid X=x) = \frac{P(X=x \mid Y=k) \cdot P(Y=k)}{P(X=x)}$$

- **Prior:** $P(Y=k)$ — eğitim verisindeki sınıf oranı
- **Likelihood:** $P(X=x \mid Y=k)$ — sınıfa göre özellik dağılımı
- **Posterior:** $P(Y=k \mid X=x)$ — tahmin etmek istediğimiz

---

### LDA — Linear Discriminant Analysis

**Varsayım:** Her sınıf aynı kovaryans matrisine sahip çok değişkenli normal dağılım.

$$P(X=x \mid Y=k) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_k)^T \Sigma^{-1}(x-\mu_k)\right)$$

**Ortak kovaryans:** $\Sigma_0 = \Sigma_1 = \Sigma$ (her iki sınıf aynı)

**Parametreler:**
- $\hat{\mu}_k = \frac{1}{n_k}\sum_{i:y_i=k} x_i$ — sınıf ortalamaları
- $\hat{\Sigma} = \frac{1}{n-K}\sum_k\sum_{i:y_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T$ — pooled kovaryans
- $\hat{\pi}_k = n_k/n$ — prior

**Karar sınırı:** Düz çizgi (lineer)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
```

**Scratch implementasyonu:**
```python
import numpy as np
from scipy.stats import multivariate_normal

class LDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        n, p = X.shape
        Sigma_pooled = np.zeros((p, p))

        for c in self.classes:
            Xc = X[y == c]
            self.priors[c] = len(Xc) / n
            self.means[c] = Xc.mean(axis=0)
            diff = Xc - self.means[c]
            Sigma_pooled += diff.T @ diff

        self.cov = Sigma_pooled / (n - len(self.classes))

    def predict_proba(self, X):
        # Use log-likelihood for numerical stability
        log_probs = []
        for c in self.classes:
            log_like = multivariate_normal.logpdf(X, self.means[c], self.cov)
            log_prior = np.log(self.priors[c])
            log_probs.append(log_like + log_prior)
        log_probs = np.array(log_probs).T
        # Normalize
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]  # class 1 probability

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
```

---

### QDA — Quadratic Discriminant Analysis

**Varsayım:** Her sınıf kendi kovaryans matrisine sahip.

**Fark:** $\Sigma_0 \neq \Sigma_1$ → her sınıf için ayrı kovaryans tahmini

**Karar sınırı:** Eğri (quadratic/ikinci dereceden)

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
```

**Scratch:**
```python
class QDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        self.covs = {}
        n = len(y)
        for c in self.classes:
            Xc = X[y == c]
            self.priors[c] = len(Xc) / n
            self.means[c] = Xc.mean(axis=0)
            diff = Xc - self.means[c]
            self.covs[c] = (diff.T @ diff) / (len(Xc) - 1)

    def predict_proba(self, X):
        log_probs = []
        for c in self.classes:
            log_like = multivariate_normal.logpdf(X, self.means[c], self.covs[c])
            log_prior = np.log(self.priors[c])
            log_probs.append(log_like + log_prior)
        log_probs = np.array(log_probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
```

---

### Naive Bayes (NB)

**Varsayım:** Özellikler birbirinden bağımsız (koşullu bağımsızlık).

$$P(X=x \mid Y=k) = \prod_{j=1}^p P(X_j=x_j \mid Y=k)$$

**Karar sınırı:** Eğri olabilir

**Ne zaman kullanılır:** Çok sayıda özellik varsa veya kovaryans matrisini tahmin etmek güçse.

**Neden korelasyon olsa bile iyi çalışabilir?**
→ Sınıflandırma kararı için tam olasılık değil, sadece hangi sınıfın daha olası olduğu önemli. Korelasyon hatası her iki sınıf için de benzer şekilde olursa karar sınırı değişmeyebilir.

```python
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
```

---

### LDA vs QDA vs NB Karşılaştırması

| | LDA | QDA | NB |
|---|---|---|---|
| Kovaryans | Ortak Σ | Ayrı Σ_k | Diyagonal (bağımsız) |
| Karar sınırı | Lineer | Kuadratik | Kuadratik |
| Parametre sayısı | Az | Çok | Az |
| Ne zaman iyi | n küçük, LDA doğru | n büyük, farklı Σ | Yüksek boyut |

**Önemli:** QDA'da n_k < p ise kovaryans matrisi tekil (singular) olur → tahmin güvenilmez!

---

### Veri Üretme (Scheme 1 ve 2)

```python
import numpy as np

# Scheme 1: LDA varsayımları geçerli
def generate_scheme1(n=1000, a=2, rho=0.5):
    y = np.random.binomial(1, 0.5, n)
    X = np.zeros((n, 2))
    # Sınıf 0: N(0, I)
    X[y==0] = np.random.multivariate_normal([0, 0], np.eye(2), size=(y==0).sum())
    # Sınıf 1: N(a, I)
    X[y==1] = np.random.multivariate_normal([a, a], np.eye(2), size=(y==1).sum())
    return X, y

# Scheme 2: LDA varsayımları geçersiz (farklı kovaryans)
def generate_scheme2(n=1000, a=2, rho=0.5):
    y = np.random.binomial(1, 0.5, n)
    X = np.zeros((n, 2))
    cov0 = np.array([[1, rho],  [rho,  1]])   # korelasyon +rho
    cov1 = np.array([[1, -rho], [-rho, 1]])   # korelasyon -rho
    X[y==0] = np.random.multivariate_normal([0, 0], cov0, size=(y==0).sum())
    X[y==1] = np.random.multivariate_normal([a, a], cov1, size=(y==1).sum())
    return X, y
```

---

### Teorik Sorular ve Cevapları

**Neden NB korelasyon olsa bile iyi çalışabilir?**
→ Sınıflandırma için posterior'ın sıralaması önemli, tam değeri değil. Korelasyon hatası iki sınıf için simetrik ise karar sınırı değişmez.

**QDA'da n_k ≈ p olursa ne olur?**
→ Kovaryans matrisi singular (tekil) hale gelir, inverse alınamaz. Sonuç güvenilmez.

**LDA ve NB'yi birleştiren model nasıl olur?**
→ Her sınıf için diyagonal kovaryans matrisi kullan ama ortak Σ = Σ_0 = Σ_1 varsay. Bu "diagonal LDA" dır.

**Accuracy yerine ne kullanılmalı?**
→ Log-loss (cross-entropy), Brier score — olasılıkların kalitesini ölçer.

---

## 2. Density Estimation

### Kernel Density Estimation (KDE)

$$\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^n K\left(\frac{x-x_i}{h}\right)$$

- **K:** kernel fonksiyonu (Gaussian, Epanechnikov, Uniform...)
- **h:** bandwidth (smoothing parameter)
  - h küçük → çok pürüzlü, overfitting
  - h büyük → çok düzgün, underfitting

```python
from sklearn.neighbors import KernelDensity
import numpy as np

# Gaussian mixture'dan veri üret
def generate_mixture(n=200):
    # 0.9 * N(5,1) + 0.1 * N(10,1)
    n1 = int(0.9 * n)
    n2 = n - n1
    x1 = np.random.normal(5, 1, n1)
    x2 = np.random.normal(10, 1, n2)
    return np.concatenate([x1, x2])

# Gerçek yoğunluk
from scipy.stats import norm
def true_density(x):
    return 0.9 * norm.pdf(x, 5, 1) + 0.1 * norm.pdf(x, 10, 1)

# KDE fit
X_sample = generate_mixture(200)
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(X_sample.reshape(-1, 1))

# Tahmin
x_grid = np.linspace(2, 12, 500).reshape(-1, 1)
log_density = kde.score_samples(x_grid)
density_est = np.exp(log_density)

# MSE hesapla
x_eval = np.random.uniform(2, 12, 1000)
f_true = true_density(x_eval)
log_d  = kde.score_samples(x_eval.reshape(-1, 1))
f_est  = np.exp(log_d)
mse    = np.mean((f_true - f_est)**2)
print(f"MSE: {mse:.6f}")
```

### Farklı Kernel ve Bandwidth

```python
kernels = ['gaussian', 'epanechnikov', 'tophat']
bandwidths = [0.1, 0.5, 1.0, 2.0]

for kernel in kernels:
    for h in bandwidths:
        kde = KernelDensity(kernel=kernel, bandwidth=h)
        kde.fit(X_sample.reshape(-1, 1))
        # ...
```

### Method 2: Artificial Sample Generation

```python
def generate_artificial_sample(X_orig, h, k=10000):
    """
    KDE'den yapay örnek üret:
    1. i'yi {1,...,n}'den uniform çek
    2. epsilon ~ N(0,1) üret
    3. X'_j = X_i + epsilon * h
    """
    n = len(X_orig)
    indices = np.random.randint(0, n, size=k)
    epsilon = np.random.randn(k)
    X_artificial = X_orig[indices] + epsilon * h
    return X_artificial
```

**Neden ikisi benzer?** Yapay örneklem KDE'nin implicit örneklemesidir. k büyütmek yoğunluk tahminini değiştirmez çünkü KDE deterministik — sadece hesaplama hassasiyetini artırır.

---

## 3. Logistic Regression

### Model

$$p_i = P(Y=1 \mid X=x_i) = \frac{1}{1+e^{-(\beta_0 + \beta^T x_i)}}$$

### Log-Likelihood

$$\ell(\beta) = \sum_{i=1}^n \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right]$$

### Veri Üretme

```python
n = 1000
beta = np.array([0.5, 1, 1, 1, 1, 1])  # beta_0=0.5, beta_1..5=1
X = np.random.randn(n, 5)
linear = beta[0] + X @ beta[1:]
p = 1 / (1 + np.exp(-linear))
y = np.random.binomial(1, p, n)
```

### Log-Likelihood Hesaplama

```python
def log_likelihood(model, X, y):
    probs = model.predict_proba(X)
    # Her gözlem için doğru sınıfın log olasılığını topla
    ll = np.sum(np.log(probs[np.arange(len(y)), y] + 1e-15))
    return ll
```

### Karar Sınırı

```python
def plot_decision_boundary(model, X, y, body, surface):
    body_range = np.linspace(body.min()-0.5, body.max()+0.5, 100)
    # surface = (-β0 - β1*body) / β2
    b0 = model.intercept_[0]
    b1, b2 = model.coef_[0]
    boundary = (-b0 - b1 * body_range) / b2
    plt.plot(body_range, boundary, 'b-', label='Decision boundary')
```

---

## 4. Evaluation Methods

### Yöntemler Karşılaştırması

| Yöntem | Açıklama | Yanlılık | Güvenilirlik |
|---|---|---|---|
| Refitting | Train=Test | Çok iyimser | Düşük |
| Cross-validation | k-fold | Az | Yüksek |
| Bootstrap | Yerine koyarak örneklem | Biraz kötümser | Yüksek |
| Bootstrap 0.632 | 0.632*boot + 0.368*refit | Dengeli | En yüksek |

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
error_cv = 1 - scores.mean()
```

### Bootstrap

```python
B = 100
errors = []
for i in range(B):
    # Yerine koyarak örneklem
    train_idx = np.random.choice(n, n, replace=True)
    test_idx  = np.setdiff1d(np.arange(n), train_idx)  # seçilmeyenler

    model.fit(X[train_idx], y[train_idx])
    error = np.mean(model.predict(X[test_idx]) != y[test_idx])
    errors.append(error)

bootstrap_error = np.mean(errors)
```

### Bootstrap 0.632

```python
# Formül: 0.632 * bootstrap_error + 0.368 * refitting_error
error_0632 = 0.632 * bootstrap_error + 0.368 * refitting_error
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision)
```

**ROC vs PR:** Dengesiz veri (α=2 gibi) durumunda PR daha bilgilendirici.

### Threshold Analizi

```python
thresholds = np.linspace(0, 1, 100)
accuracies = []
balanced_accs = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    accuracies.append(np.mean(y_pred == y_test))
    balanced_accs.append(balanced_accuracy_score(y_test, y_pred))

# Accuracy: maksimum genellikle t=0.5 civarında
# Balanced accuracy: maksimum t = eğitim setindeki 1'lerin oranı civarında
t_balanced = y_train.mean()
```

---

## 5. Classification Trees

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,           # maksimum derinlik
    min_samples_split=10,  # bölünmek için min gözlem
    criterion='gini'       # 'gini' veya 'entropy'
)
model.fit(X_train, y_train)
print("Depth:", model.get_depth())
print("Leaves:", model.get_n_leaves())
```

### Cost-Complexity Pruning

```python
path = model.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

for alpha in alphas:
    pruned_model = DecisionTreeClassifier(ccp_alpha=alpha)
    pruned_model.fit(X_train, y_train)
    acc = pruned_model.score(X_test, y_test)
```

### Bagging (Kendi İmplementasyonu)

```python
B = 100
trees = []
for i in range(B):
    train_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    tree = DecisionTreeClassifier()
    tree.fit(X_train[train_idx], y_train[train_idx])
    trees.append(tree)

# Çoğunluk oyu
from scipy.stats import mode
predictions = np.array([tree.predict(X_test) for tree in trees])
y_pred_bagging = mode(predictions, axis=0).mode
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test)
```

### Karşılaştırma

| Yöntem | Avantaj | Dezavantaj |
|---|---|---|
| Single Tree | Yorumlanabilir | Yüksek varyans, overfitting |
| Bagging | Düşük varyans | Ağaçlar korelasyonlu |
| Random Forest | En iyi performans | Yorumlanması zor |

**Neden RF > Bagging?** RF her ağaçta rastgele değişken alt kümesi kullanır → ağaçlar daha az korelasyonlu → averaging daha etkili.

---

## 6. Regularization

### Formüller

$$\text{Ridge: } \ell(\beta) - \lambda\sum\beta_j^2$$
$$\text{Lasso: } \ell(\beta) - \lambda\sum|\beta_j|$$
$$\text{Elastic Net: } \ell(\beta) - \lambda\left[\alpha\sum|\beta_j| + (1-\alpha)\sum\beta_j^2\right]$$

### C ve λ ilişkisi

$$C = \frac{1}{\lambda} \Rightarrow C \text{ büyük} = \lambda \text{ küçük} = \text{az ceza}$$

```python
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np

# Ridge
model_ridge = LogisticRegression(l1_ratio=0, C=1.0, solver='saga', max_iter=5000)

# Lasso
model_lasso = LogisticRegression(l1_ratio=1, C=1.0, solver='saga', max_iter=5000)

# Elastic Net
model_enet  = LogisticRegression(l1_ratio=0.5, C=1.0, solver='saga', max_iter=5000)

# Optimal C seçimi
C_values = np.logspace(-4, 2, 50)
model_cv = LogisticRegressionCV(
    l1_ratios=[1],       # lasso
    Cs=C_values,
    cv=5,
    solver='saga',
    max_iter=5000
)
model_cv.fit(X_scaled, y)
best_lambda = 1 / model_cv.C_[0]
```

### Profile Plot

```python
coefs_lasso = []
for C in C_values:
    m = LogisticRegression(l1_ratio=1, C=C, solver='saga', max_iter=5000)
    m.fit(X_scaled, y)
    coefs_lasso.append(m.coef_[0])

coefs_lasso = np.array(coefs_lasso)  # (n_lambda, n_features)

# İlk 20 gen için çiz
for j in range(20):
    plt.plot(np.log10(1/C_values), coefs_lasso[:, j])
plt.xlabel('log10(λ)')
plt.ylabel('Coefficient')
plt.title('Lasso Profile Plot')
```

### PSR ve FDR

$$PSR = \frac{|\hat{t} \cap t|}{|t|}, \quad FDR = \frac{|\hat{t} \setminus t|}{|\hat{t}|}$$

```python
def compute_psr_fdr(selected, true_relevant):
    selected = set(selected)
    true_relevant = set(true_relevant)
    if len(selected) == 0:
        return 0.0, 0.0
    psr = len(selected & true_relevant) / len(true_relevant)
    fdr = len(selected - true_relevant) / len(selected)
    return psr, fdr

# Seçilen değişkenler = sıfır olmayan katsayılar
selected = set(np.where(model.coef_[0] != 0)[0])
```

---

## 7. Ensemble Methods

### AdaBoost Algoritması

**Eğitim:**

1. Başlangıç ağırlıkları: $w_i = 1/n$
2. Her $k=1,...,B$ için:
   - Ağırlıklı tree fit et: $f_k$
   - Ağırlıklı hata: $\varepsilon_k = \sum_{i=1}^n w_i \cdot \mathbf{1}[f_k(x_i) \neq y_i]$
   - $\beta_k = \varepsilon_k / (1-\varepsilon_k)$
   - Doğru sınıflandırmalar için: $w_i = w_i \cdot \beta_k$
   - Ağırlıkları normalize et: $w_i = w_i / \sum_j w_j$

**Tahmin:**

$$\hat{y}(x) = \arg\max_y \sum_{k=1}^B \mathbf{1}[f_k(x)=y] \cdot \log\left(\frac{1}{\beta_k}\right)$$

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoost:
    def __init__(self, B=50, max_depth=1):
        self.B = B
        self.max_depth = max_depth
        self.classifiers = []
        self.betas = []

    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n  # uniform başlangıç ağırlıkları

        for k in range(self.B):
            # Ağırlıklı tree fit et
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=w)

            y_pred = tree.predict(X)
            # Ağırlıklı hata
            incorrect = (y_pred != y).astype(float)
            eps = np.dot(w, incorrect)

            # Edge cases
            if eps == 0:
                # Mükemmel sınıflandırıcı — tüm ağırlığı al
                self.classifiers.append(tree)
                self.betas.append(1e-10)
                break
            if eps >= 0.5:
                # Rastgele tahmin kadar kötü — durdur
                break

            beta = eps / (1 - eps)

            # Doğru sınıflandırmalar için ağırlığı azalt
            w[y_pred == y] *= beta

            # Normalize
            w /= w.sum()

            self.classifiers.append(tree)
            self.betas.append(beta)

    def predict(self, X):
        # Ağırlıklı oy
        classes = np.unique([0, 1])
        votes = np.zeros((len(X), 2))

        for tree, beta in zip(self.classifiers, self.betas):
            weight = np.log(1 / (beta + 1e-10))
            preds = tree.predict(X)
            for i, p in enumerate(preds):
                votes[i, int(p)] += weight

        return np.argmax(votes, axis=1)
```

### Edge Cases (ε_k)

- **ε_k = 0:** Mükemmel sınıflandırıcı → β_k → 0 → log(1/β_k) → ∞ → tek classifier tüm kararı alır. Erken durdur.
- **ε_k ≥ 0.5:** Rastgele tahminden bile kötü → negatif katkı. Durdur veya atla.

### Yapay Veri Seti (Chi-squared)

```python
from scipy.stats import chi2

def generate_chi2_data(n=2000, p=10):
    X = np.random.randn(n, p)
    # χ²_10 medyanı
    chi2_median = chi2.ppf(0.5, df=10)
    # Y=1 eğer sum(X_j²) > medyan
    y = (np.sum(X**2, axis=1) > chi2_median).astype(int)
    return X, y

# Neden dengeli? X_j ~ N(0,1) → sum(X_j²) ~ χ²_10
# Medyan tanımı gereği %50 üstünde, %50 altında → Y dengeli
```

### Gradient Boosting ve XGBoost

```python
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

gb = GradientBoostingClassifier(n_estimators=100, max_depth=3)
gb.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
```

### Teorik Sorular

**Küçük ε_k → büyük ağırlık neden?**
→ Daha az hata yapan classifier daha güvenilir → ona daha çok oy hakkı verilmeli.

**999 classifier ε=0.4, 1 classifier ε=10^-8 ise?**
→ log(1/β) = log((1-ε)/ε). ε=0.4 → weight≈0.405. ε=10^-8 → weight≈18.4. Tek mükemmel classifier yaklaşık 45 kötü classifier'a eşdeğer. Dominant olmaz ama çok etkili.

**Neden decision stumps yeterli?**
→ Boosting biası azaltır. Birçok zayıf classifier toplandığında strong classifier oluşur. Stumps az varyansa sahip, boosting da varyansı artırmaz.

**RF neden Bagging'den iyi?**
→ Bagging'de tüm ağaçlar benzer (correlated), averaging etkisi sınırlı. RF'de rastgele feature subset → ağaçlar farklı → averaging daha etkili varyans azaltır.

---

## 8. Feature Selection

### Veri Üretme

```python
from scipy.stats import chi2

# Dataset 1: Chi-squared boundary
def generate_dataset1(n=500, p=50, k=10):
    X = np.random.randn(n, p)
    chi2_median = chi2.ppf(0.5, df=k)
    y = (np.sum(X[:, :k]**2, axis=1) > chi2_median).astype(int)
    return X, y

# Dataset 2: L1 boundary
def generate_dataset2(n=500, p=50, k=10):
    X = np.random.randn(n, p)
    y = (np.sum(np.abs(X[:, :k]), axis=1) > k).astype(int)
    return X, y
```

**Marginal korelasyon ile tespit edilebilir mi?**
→ Hayır. Y = f(X_1²+...+X_k²) — X_j'nin işaretinden bağımsız (simetrik). E[X_j·Y] = 0 olduğundan marginal korelasyon sıfır. Önemli değişkenler marjinal olarak Y ile ilişkili değil.

### Random Forest Variable Importance

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Mean Decrease in Impurity (MDI)
importances_mdi = rf.feature_importances_

# Permutation-based importance
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10)
importances_perm = result.importances_mean

# Görselleştir
plt.bar(range(p), importances_mdi)
plt.axvline(x=k-0.5, color='red', linestyle='--', label='True boundary')
plt.xlabel('Feature index')
plt.ylabel('Importance')
```

### Boruta Algoritması

```python
from boruta import BorutaPy

rf_boruta = RandomForestClassifier(n_estimators=100, max_depth=5)
boruta = BorutaPy(rf_boruta, n_estimators='auto', random_state=42)
boruta.fit(X_train, y_train)

selected = np.where(boruta.support_)[0]
print("Boruta selected features:", selected)
```

**Shadow variables neden önemli?**
→ Boruta her gerçek değişken için rastgele permütasyon (shadow) yaratır. Gerçek değişkenin önemi en iyi shadow değişkenden büyük olmalı. Sadece "pozitif önem" kontrolü yeterli değil çünkü gürültülü değişkenler de pozitif önem alabilir.

---

## 9. SVM

### Temel Kavramlar

- **Support vectors:** Karar sınırına en yakın noktalar — tüm modeli bunlar belirler
- **Margin:** İki sınıf arasındaki boşluk — SVM bunu maksimize eder
- **C parametresi:** Hata toleransı
  - C küçük → geniş margin, daha fazla hata tolere edilir
  - C büyük → dar margin, hata tolere edilmez

```python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_s, y_train)

# Support vectors
sv = svm_linear.support_vectors_
n_sv = sv.shape[0]

# RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma=0.1)
svm_rbf.fit(X_train_s, y_train)
```

### C Parametresi Etkisi

```python
for C in [0.01, 0.1, 1, 10, 100]:
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train_s, y_train)
    print(f"C={C}: #SV={svm.support_vectors_.shape[0]}, "
          f"Test acc={svm.score(X_test_s, y_test):.3f}")
# C küçük → çok SV, C büyük → az SV
```

### Gamma Parametresi (RBF)

$$k(x,z) = \exp(-\gamma \|x-z\|^2)$$

```python
for gamma in [0.01, 0.1, 1, 10, 100]:
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma)
    svm.fit(X_train_s, y_train)
    tr = svm.score(X_train_s, y_train)
    te = svm.score(X_test_s, y_test)
    print(f"gamma={gamma}: Train={tr:.3f}, Test={te:.3f}")
# gamma küçük → underfitting, gamma büyük → overfitting
```

### Karar Sınırı + Margin Görselleştirme

```python
def plot_svm_boundary(model, X, y):
    x1_min, x1_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    x2_min, x2_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                         np.linspace(x2_min, x2_max, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Decision function: 0=boundary, ±1=margin
    Z = model.decision_function(grid).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1],
                colors=['blue', 'black', 'red'],
                linestyles=['--', '-', '--'])

    # Support vectors
    sv = model.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], s=100, facecolors='none',
                edgecolors='green', linewidths=2, label='Support Vectors')
```

### SVM vs Logistic Regression

| | SVM | Logistic Regression |
|---|---|---|
| Karar sınırı | Margin maximization | Log-likelihood max |
| Hangi noktalar etkili? | Sadece support vectors | Tüm gözlemler |
| Olasılık tahmini | Hayır (default) | Evet |
| Doğrusal olmayan sınır | Kernel trick | Polynomial features |

---

## 10. Regression

### Nadaraya-Watson Kernel Regression

$$\hat{g}_h(x) = \frac{\sum_{i=1}^n K\left(\frac{x-x_i}{h}\right) y_i}{\sum_{i=1}^n K\left(\frac{x-x_i}{h}\right)}$$

```python
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

def nadaraya_watson(x_new, X_train, y_train, h):
    """
    x_new   : tek bir test noktası
    X_train : eğitim noktaları
    y_train : eğitim hedefleri
    h       : bandwidth
    """
    u = (x_new - X_train) / h
    weights = gaussian_kernel(u)
    if weights.sum() == 0:
        return 0.0
    return np.dot(weights, y_train) / weights.sum()

def nw_predict(X_new, X_train, y_train, h):
    return np.array([nadaraya_watson(x, X_train, y_train, h) for x in X_new])
```

### Gerçek Fonksiyon

```python
def g_true(x):
    return 4.26 * (np.exp(-x) - 4*np.exp(-2*x) + 3*np.exp(-3*x))

# Veri üret
n_train = 200
X_train = np.random.uniform(0, 4, n_train)
eps     = np.random.normal(0, 0.1, n_train)
y_train = g_true(X_train) + eps
```

### MSE Türleri

```python
# Function estimation error (sadece simülasyonda mevcut — gerçek g bilinir)
MSE_g = np.mean((g_true(X_test) - y_pred)**2)

# Prediction error (gerçek veri setlerinde de hesaplanabilir)
MSE_y = np.mean((y_test - y_pred)**2)

# MSE_y = MSE_g + noise_variance (yaklaşık olarak)
# MSE_g: modelin gerçek fonksiyonu ne kadar iyi öğrendiği
# MSE_y: modelin gözlemleri ne kadar iyi tahmin ettiği
```

### Smoothing Splines

```python
from scipy.interpolate import UnivariateSpline

spline = UnivariateSpline(X_train_sorted, y_train_sorted, s=smoothing_param)
y_pred_spline = spline(X_test)
```

### Bandwidth Seçimi

```python
# h küçük → çok pürüzlü (overfitting), h büyük → çok düzgün (underfitting)
# n arttıkça optimal h azalmalı: h ~ n^(-1/5) teorik optimaldir

for h in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
    y_pred = nw_predict(X_test, X_train, y_train, h)
    mse = np.mean((y_test - y_pred)**2)
    print(f"h={h:.2f}: Test MSE={mse:.4f}")
```

**Nadaraya-Watson (local) vs Smoothing Splines (global):**
- NW: her tahmin yakın komşulara bağlı — yeni nokta uzak bölgeleri etkilemez
- Splines: global fitting — tek bir yeni nokta tüm eğriyi etkileyebilir

---

## 11. Semi-Supervised Learning

### Temel Fikir
Çok az etiketli veri + çok fazla etiketsiz veri. SSL hem etiketlileri hem de etiketsizleri kullanır.

```python
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

# Veri
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Sadece g gözlem etiketli, geri kalan -1 (unlabeled)
g = 20  # etiketli gözlem sayısı
y_train_ssl = y_train.copy()
unlabeled_idx = np.random.choice(len(y_train), len(y_train)-g, replace=False)
y_train_ssl[unlabeled_idx] = -1  # -1 = etiketsiz

# 1. Naive Method: sadece g etiketliye eğit
labeled_idx = y_train_ssl != -1
base_svc = SVC(probability=True, kernel='rbf')
base_svc.fit(X_train[labeled_idx], y_train[labeled_idx])

# 2. Self-Training: base classifier'ı iteratif olarak etiketlenmiş veyle genişlet
self_train = SelfTrainingClassifier(SVC(probability=True, kernel='rbf'))
self_train.fit(X_train, y_train_ssl)

# 3. Label Propagation
lp = LabelPropagation(kernel='rbf', gamma=20)
lp.fit(X_train, y_train_ssl)

# 4. Label Spreading
ls = LabelSpreading(kernel='rbf', gamma=20, alpha=0.2)
ls.fit(X_train, y_train_ssl)

# Değerlendirme
def evaluate_ssl(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    if hasattr(model, 'predict_proba'):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    else:
        auc = float('nan')
    print(f"{name:<25}: Acc={acc:.3f}, BAcc={bacc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

evaluate_ssl("Naive (g labeled)",  base_svc,    X_test, y_test)
evaluate_ssl("Self-Training",      self_train,  X_test, y_test)
evaluate_ssl("Label Propagation",  lp,          X_test, y_test)
evaluate_ssl("Label Spreading",    ls,          X_test, y_test)
```

### g'nin Etkisi

```python
for g in [5, 10, 20, 50, 100, 200]:
    # her g için yukarıdaki kodun aynısı
    # g arttıkça tüm metodlar iyileşir
    # SSL metodlar az g'de naive'dan çok daha iyi
    pass
```

### Label Propagation vs Spreading

| | Label Propagation | Label Spreading |
|---|---|---|
| Etiketli noktalar | Sabit (değişmez) | Değişebilir (alpha ile) |
| Gürültüye dayanıklılık | Düşük | Yüksek |
| Parametre | gamma | gamma, alpha |

---

## 12. Multi-label Classification

### Temel Kavramlar

- **Binary Relevance (BR):** Her etiket bağımsız — K ayrı ikili sınıflandırıcı
- **Classifier Chain (CC):** Etiketler sıralı — önceki etiketler sonraki için feature
- **ECC:** M farklı sırayla CC, majority vote
- **CCC:** Her etiket diğerlerini kullanır, iteratif güncelleme

### Evaluation Metrikleri

$$\text{Subset Accuracy} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\{Y^{(i)} = \hat{Y}^{(i)}\}$$

$$\text{Hamming Score} = \frac{1}{NK}\sum_{i=1}^N\sum_{k=1}^K \mathbf{1}\{Y^{(i)}_k = \hat{Y}^{(i)}_k\}$$

$$\text{Jaccard Score} = \frac{1}{N}\sum_{i=1}^N \frac{|\{k: Y^{(i)}_k=1\} \cap \{k: \hat{Y}^{(i)}_k=1\}|}{|\{k: Y^{(i)}_k=1\} \cup \{k: \hat{Y}^{(i)}_k=1\}|}$$

```python
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score

def evaluate(name, Y_true, Y_pred):
    subset_acc = accuracy_score(Y_true, Y_pred)
    hamming    = 1 - hamming_loss(Y_true, Y_pred)
    jacc       = jaccard_score(Y_true, Y_pred, average="samples")
    print(f"{name}: Subset={subset_acc:.3f}, Hamming={hamming:.3f}, Jaccard={jacc:.3f}")
```

### Binary Relevance

```python
from sklearn.multiclass import OneVsRestClassifier
br = OneVsRestClassifier(LogisticRegression(solver="liblinear"))
br.fit(X_train, Y_train)
Y_pred_br = br.predict(X_test)
```

### Classifier Chain

```python
from sklearn.multioutput import ClassifierChain
cc = ClassifierChain(LogisticRegression(solver="liblinear"), order="random", random_state=42)
cc.fit(X_train, Y_train)
Y_pred_cc = cc.predict(X_test)
```

### ECC (Ensemble of Classifier Chains)

```python
M = 20
chains = [ClassifierChain(LogisticRegression(solver="liblinear"),
                          order="random", random_state=m) for m in range(M)]
for c in chains:
    c.fit(X_train, Y_train)

all_preds = np.array([c.predict(X_test) for c in chains])  # (M, n_test, K)
Y_pred_ecc = (all_preds.mean(axis=0) >= 0.5).astype(int)   # majority vote
```

### Teorik Sorular

**BR'nin varsayımı nedir?**
→ Etiketler X verildiğinde koşullu bağımsız: P(Y1,...,YK|X) = ΠP(Yk|X)

**CC neden sıraya bağımlı?**
→ Erken etiketlerin tahmini yanlışsa bu hata sonraki sınıflandırıcılara propagate olur.

**ECC neden daha stabil?**
→ Farklı sıralamalar farklı hatalar yapar. Averaging ile hatalar cancel out.

**Subset acc neden Hamming'den çok küçük?**
→ Subset acc TÜM K etiketin doğru olmasını gerektirir. K=6'da %90 per-label acc → 0.9^6 ≈ 0.53 subset acc.

**Jaccard neden daha bilgilendirici?**
→ Aktif (=1) etiketlere odaklanır. Hamming her zaman 0 tahmin eden modele yüksek skor verir.

---

## 13. Key Formulas

### Olasılık

| Formül | Ne zaman kullanılır |
|---|---|
| $p = \frac{1}{1+e^{-(\beta_0+\beta^T x)}}$ | Logistic regression |
| $p = \Phi(\beta^T x)$ | Probit model |
| $y \sim Bern(p)$ | Binary veri üretme |
| $\sum X_j^2 \sim \chi^2_k$ | Chi-squared veri üretme |

### Regularization

| | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|---|---|---|---|
| Ceza | $\lambda\sum\|\beta_j\|$ | $\lambda\sum\beta_j^2$ | Her ikisi |
| Sıfır katsayı | Evet | Hayır | Evet |
| Değişken seçimi | Evet | Hayır | Kısmen |
| sklearn parametresi | l1_ratio=1 | l1_ratio=0 | 0<l1_ratio<1 |

### Hata Metrikleri

| Metrik | Formül | Ne ölçer |
|---|---|---|
| MSE | $\frac{1}{n}\sum(y_i-\hat{y}_i)^2$ | Regresyon hatası |
| Accuracy | $\frac{1}{n}\sum\mathbf{1}\{y_i=\hat{y}_i\}$ | Sınıflandırma doğruluğu |
| PSR | $\frac{|\hat{t}\cap t|}{|t|}$ | Önemli değişken bulma oranı |
| FDR | $\frac{|\hat{t}\setminus t|}{|\hat{t}|}$ | Yanlış seçim oranı |
| AUC | Alan altı ROC | Genel sınıflandırma kalitesi |

---

## 14. Code Patterns

### Veri Üretme

```python
# Normal dağılım
X = np.random.randn(n, p)                          # N(0,1)
X = np.random.normal(mu, sigma, n)                  # N(mu, sigma^2)
X = np.random.multivariate_normal(mean, cov, n)     # Çok değişkenli

# Binary hedef
y = np.random.binomial(1, 0.5, n)                   # Bernoulli(0.5)
y = np.random.binomial(1, p_array, n)               # Farklı olasılıklar

# Logistic model
linear = beta[0] + X @ beta[1:]
p = 1 / (1 + np.exp(-linear))
y = np.random.binomial(1, p, n)

# Chi-squared boundary
chi2_median = chi2.ppf(0.5, df=k)
y = (np.sum(X[:,:k]**2, axis=1) > chi2_median).astype(int)
```

### Model Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Standartlaştırma + Model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
```

### Cross-Validation

```python
# Basit
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
error  = 1 - scores.mean()

# Manuel
kf = KFold(n_splits=10, shuffle=True)
for train_idx, test_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    errors.append(np.mean(model.predict(X[test_idx]) != y[test_idx]))
```

### Karar Sınırı (2D)

```python
def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300),
                         np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X[y==0,0], X[y==0,1], c='blue', alpha=0.5, label='Class 0')
    plt.scatter(X[y==1,0], X[y==1,1], c='red',  alpha=0.5, label='Class 1')
    plt.legend()
```

### Overfitting Kontrolü

```python
train_acc = model.score(X_train, y_train)
test_acc  = model.score(X_test,  y_test)
gap = train_acc - test_acc

print(f"Train: {train_acc:.3f}, Test: {test_acc:.3f}, Gap: {gap:.3f}")
# gap < 0.02  → OK
# gap > 0.10  → Overfitting var → Regularization artır veya model basitleştir
```

### Hızlı Cheatsheet

```python
# Fit
model.fit(X_train, y_train)

# Tahmin
y_pred  = model.predict(X_test)           # etiket
y_proba = model.predict_proba(X_test)[:,1] # olasılık

# Değerlendirme
model.score(X_test, y_test)               # accuracy
np.mean(y_pred != y_test)                 # error
np.mean((y_pred - y_test)**2)             # MSE

# Katsayılar
model.coef_[0]       # logistic regression katsayıları
model.feature_importances_  # random forest önem
model.support_vectors_       # SVM support vectors

# Optimal lambda
model_cv = LogisticRegressionCV(cv=5, Cs=np.logspace(-4,4,50))
model_cv.fit(X, y)
best_C = model_cv.C_[0]

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

---

*Bu notlar AML dersinin tüm lablarını kapsamaktadır. Her lab için temel formüller, Python kodu ve teorik sorular derlenmiştir.*
