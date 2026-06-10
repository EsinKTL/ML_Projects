import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# =============================================================
# VERİ YÜKLEME
# =============================================================

# pd.read_csv: metin dosyasını okuyup tablo (DataFrame) oluşturur
# sep=r'\s+': sütunlar arasındaki ayraç boşluk veya tab karakteri
# index_col=0: ilk sütun satır numarası, veri değil — onu indeks yap

y_df = pd.read_csv('prostate_y.txt', sep=r'\s+', index_col=0)
X_df = pd.read_csv('prostate_x.txt', sep=r'\s+', index_col=0)

# .values: pandas DataFrame'i numpy array'e çevirir
# .ravel(): 2 boyutlu [[0],[1],[0],...] diziyi 1 boyutlu [0,1,0,...] yapar
# y sonuç: (102,) — 102 hastanın etiketi: 0=sağlıklı, 1=kanserli
y = y_df.values.ravel()

# X sonuç: (102, 6033) — 102 hasta, 6033 gen
X = X_df.values

print("X boyutu:", X.shape)   # (102, 6033) olmalı
print("y boyutu:", y.shape)   # (102,) olmalı

# =============================================================
# VERİYİ STANDARTLAŞTIRMA — NEDEN GEREKLİ?
# =============================================================

# Regularization her katsayıya eşit ceza uygulamak ister.
# Ama genler farklı ölçeklerde olabilir:
#   gen_1: değerler 0.001 ile 0.002 arasında
#   gen_2: değerler 100 ile 200 arasında
# Bu durumda aynı katsayı cezası haksız olur.
# StandardScaler her geni şöyle dönüştürür:
#   yeni_değer = (eski_değer - ortalama) / standart_sapma
# Sonuç: her genin ortalaması 0, standart sapması 1 olur

scaler = StandardScaler()

# fit_transform: önce ortalama ve std hesaplar (fit), sonra dönüştürür (transform)
X_scaled = scaler.fit_transform(X)

# =============================================================
# LAMBDA DEĞERLERİ — NEDEN LOG ÖLÇEĞINDE?
# =============================================================

# sklearn'de regularization gücü C parametresiyle kontrol edilir
# C = 1/λ ilişkisi var:
#   C büyük → λ küçük → az ceza → katsayılar büyük olabilir
#   C küçük → λ büyük → çok ceza → katsayılar sıfıra yaklaşır
#
# np.logspace(-4, 2, 50):
#   10^(-4) = 0.0001 ile 10^2 = 100 arasında
#   50 eşit aralıklı değer üretir AMA log ölçeğinde
#   yani: 0.0001, 0.0002, ..., 1, 2, ..., 100
#   neden log ölçeği? çünkü lambda'nın etkisi log ölçeğinde düzgün görünür

C_values = np.logspace(-4, 2, 50)   # 50 farklı C değeri
lambdas = 1 / C_values              # C → lambda dönüşümü

# =============================================================
# TASK 1: RIDGE, LASSO, ELASTIC NET MODELLERİ
# =============================================================

# Her regularization için her lambda değerindeki katsayıları saklayacağız
# Başlangıçta boş liste, döngüde dolduracağız
# coefs_ridge[i] → i. lambda değerindeki 6033 katsayı

coefs_ridge = []   # ridge katsayıları: her lambda için bir satır
coefs_lasso = []   # lasso katsayıları
coefs_enet  = []   # elastic net katsayıları

for C in C_values:
	
	# -------------------------------------------------------
	# RIDGE (L2 Regularization)
	# -------------------------------------------------------
	# Ne yapar: her katsayının KARESİNİ cezalandırır
	# Formül: Loss + λ * Σ(βj²)
	# Sonuç: tüm katsayılar küçülür ama HİÇBİRİ tam sıfır olmaz
	# Ne zaman kullanılır: tüm değişkenler biraz önemliyse
	#
	# penalty='l2': ridge regularization
	# C=C: şu anki lambda değeri (C = 1/lambda)
	# solver='saga': büyük ve seyrek veri setleri için hızlı algoritma
	# max_iter=5000: modelin yakınsaması için maksimum iterasyon sayısı
	#   (yeterli iterasyon olmazsa model yakınsamaz ve uyarı verir)
	
	# RIDGE: l1_ratio=0 → tamamen L2 cezası
	model_ridge = LogisticRegression(
		l1_ratio=0, C=C, solver='saga', max_iter=5000
	)
	model_ridge.fit(X_scaled, y)
	coefs_ridge.append(model_ridge.coef_[0])
	
	# coef_: fit edilen katsayılar, shape: (1, 6033) — 1 sınıf, 6033 gen
	# coef_[0]: ilk (ve tek) satırı al → (6033,) vektör
	
	# -------------------------------------------------------
	# LASSO (L1 Regularization)
	# -------------------------------------------------------
	# Ne yapar: her katsayının MUTLAK DEĞERİNİ cezalandırır
	# Formül: Loss + λ * Σ|βj|
	# Sonuç: bazı katsayılar TAM SIFIR olur → değişken seçimi yapar
	# Ne zaman kullanılır: az sayıda değişken gerçekten önemliyse
	#   6033 genden sadece birkaçı kanserle ilgiliyse lasso onları bulur
	
	model_lasso = LogisticRegression(
		l1_ratio=1, C=C, solver='saga', max_iter=5000
	)
	model_lasso.fit(X_scaled, y)
	coefs_lasso.append(model_lasso.coef_[0])
	
	# -------------------------------------------------------
	# ELASTIC NET (L1 + L2 Karışımı)
	# -------------------------------------------------------
	# Ne yapar: hem L1 hem L2 cezası uygular
	# Formül: Loss + λ * [l1_ratio * Σ|βj| + (1-l1_ratio) * Σβj²]
	# l1_ratio=0.5: %50 lasso + %50 ridge
	# Sonuç: lasso gibi sıfır yapar AMA correlated (birbirine bağlı)
	#   değişkenleri birlikte tutar (lasso bunlardan birini atar)
	# Ne zaman kullanılır: birbiriyle ilişkili genler varsa
	
	model_enet = LogisticRegression(
		l1_ratio=0.5, C=C, solver='saga', max_iter=5000
	)
	model_enet.fit(X_scaled, y)
	coefs_enet.append(model_enet.coef_[0])

# np.array: listeyi numpy matrise çevirir
# Sonuç: (50, 6033) — 50 lambda değeri, her biri için 6033 katsayı
coefs_ridge = np.array(coefs_ridge)
coefs_lasso = np.array(coefs_lasso)
coefs_enet  = np.array(coefs_enet)

# En büyük lambda'da (en çok ceza) kaç katsayı sıfır değil?
# np.sum(... != 0): sıfır olmayan elemanları say
print("\n=== En büyük lambda'da sıfır olmayan katsayı sayısı ===")
print(f"Ridge      : {np.sum(coefs_ridge[-1] != 0)}")  # -1: son eleman = en büyük lambda
print(f"Lasso      : {np.sum(coefs_lasso[-1] != 0)}")
print(f"Elastic Net: {np.sum(coefs_enet[-1]  != 0)}")

# =============================================================
# TASK 1: PROFILE PLOTS
# =============================================================

# Profile plot: her genin katsayısı lambda'ya göre nasıl değişiyor?
# x ekseni: log10(lambda) — lambda değerleri
# y ekseni: katsayı değeri
# Her çizgi: bir genin katsayısının lambda'ya göre değişimi
#
# 6033 gen çok fazla görsel için, sadece ilk 20 geni çiziyoruz
# plt.subplots(1, 3): yan yana 3 grafik
# figsize=(18, 6): grafik boyutu (genişlik, yükseklik) inç cinsinden

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, coefs, title in zip(
		axes,
		[coefs_ridge, coefs_lasso, coefs_enet],
		['Ridge (L2)', 'Lasso (L1)', 'Elastic Net']
):
	for j in range(20):
		# coefs[:, j]: j. genin tüm lambda değerlerindeki katsayıları
		# np.log10(lambdas): lambda'yı log ölçeğinde göster
		ax.plot(np.log10(lambdas), coefs[:, j])
	
	# y=0 yatay çizgisi: katsayının sıfır olduğu noktayı gösterir
	ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
	ax.set_xlabel('log10(λ) — lambda büyüdükçe ceza artar')
	ax.set_ylabel('Katsayı değeri')
	ax.set_title(title)
	ax.grid(True)

plt.suptitle('Profile Plots: Katsayılar λ ile nasıl değişiyor?')
plt.tight_layout()
plt.show()

# =============================================================
# TASK 1: CROSS-VALIDATION İLE OPTİMAL LAMBDA
# =============================================================

# Cross-validation: hangi lambda en iyi tahmin yapar?
# cv=5: veriyi 5 parçaya böl, her parçayı test olarak kullan
# LogisticRegressionCV: farklı C değerlerini dener, en iyisini seçer
# Cs=C_values: denenecek C (=1/lambda) değerleri

print("\n=== Cross-validation ile optimal lambda ===")

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
		max_iter=5000
	)
	model_cv.fit(X_scaled, y)
	best_C = model_cv.C_[0]
	best_lambda = 1 / best_C
	print(f"\n{name}:")
	print(f"  Optimal λ = {best_lambda:.6f}")
	print(f"  Sıfır olmayan katsayı sayısı: {np.sum(model_cv.coef_[0] != 0)}")

# =============================================================
# TASK 2: SİMÜLE VERİ İLE LASSO PERFORMANSI
# =============================================================

# PSR ve FDR hesaplama fonksiyonu
def compute_psr_fdr(selected, true_relevant):
	"""
	selected     : modelin seçtiği değişkenlerin indeks seti, örn: {0, 2, 5}
	true_relevant: gerçekten önemli değişkenlerin indeks seti, örn: {0,1,...,9}

	PSR (Positive Selection Rate):
		Gerçek önemli değişkenlerin kaçını bulduk?
		Örnek: 10 önemli değişkenden 7'sini bulduk → PSR = 7/10 = 0.7
		PSR yüksek olmalı (1'e yakın)

	FDR (False Discovery Rate):
		Seçtiğimiz değişkenlerin kaçı aslında önemsiz?
		Örnek: 10 değişken seçtik, 3'ü aslında önemsiz → FDR = 3/10 = 0.3
		FDR düşük olmalı (0'a yakın)

	& operatörü: iki setin kesişimi (her ikisinde de olanlar)
	- operatörü: fark (birincide olup ikincide olmayanlar)
	"""
	selected = set(selected)
	if len(selected) == 0:
		return 0.0, 0.0   # hiç değişken seçilmediyse PSR=0, FDR=0
	
	# |seçilen ∩ gerçek| / |gerçek|
	psr = len(selected & true_relevant) / len(true_relevant)
	
	# |seçilen - gerçek| / |seçilen|
	fdr = len(selected - true_relevant) / len(selected)
	
	return psr, fdr

def run_lasso_simulation(n, n_irrelevant=10, L=100, probit=False):
	"""
	n           : kaç gözlem (hasta) üretileceği
				  n büyük → daha fazla veri → model daha iyi öğrenir
	n_irrelevant: kaç tane önemsiz (gürültü) değişken olacağı
				  n_irrelevant büyük → model daha zor önemliyi bulur
	L           : kaç kez tekrar edeceğiz
				  L=100 → 100 farklı rastgele veri seti üret, ortalamasını al
	probit      : veri üretim modeli
				  False → logistic model: p = 1/(1+exp(-β^T x))
				  True  → probit model: p = Φ(β^T x)
				  Φ: standart normal dağılımın CDF'i (kümülatif dağılım fonksiyonu)
				  Probit model gerçekte logistic regression uyguladığımızda
				  ne olur diye test etmek için — model yanlış ama ne kadar başarılı?
	"""
	
	# Toplam değişken sayısı: 10 önemli + n_irrelevant önemsiz
	p = 10 + n_irrelevant
	
	# Gerçek katsayılar:
	# β = [1,1,1,1,1,1,1,1,1,1, 0,0,...,0]
	# İlk 10 değişken önemli (katsayı=1), geri kalanlar önemsiz (katsayı=0)
	beta = np.array([1]*10 + [0]*n_irrelevant)
	
	# Gerçek önemli değişkenlerin indeksleri: 0, 1, 2, ..., 9
	true_relevant = set(range(10))
	
	psr_list = []   # her tekrardaki PSR değerlerini sakla
	fdr_list = []   # her tekrardaki FDR değerlerini sakla
	
	for _ in range(L):   # L kez tekrarla (_ : kullanmadığımız döngü değişkeni)
		
		# Veri üret
		# np.random.randn(n, p): n×p boyutunda standart normal matris
		# Her satır bir gözlem (hasta), her sütun bir değişken (gen)
		X_sim = np.random.randn(n, p)
		
		# Lineer kombinasyon: β^T * x
		# X_sim @ beta: matris-vektör çarpımı → (n,) vektör
		# Her gözlem için: β1*x1 + β2*x2 + ... + βp*xp
		linear = X_sim @ beta
		
		if probit:
			# Probit model: p_i = Φ(β^T x)
			# norm.cdf: standart normal dağılımın kümülatif dağılım fonksiyonu
			# Φ(0) = 0.5, Φ(1) = 0.84, Φ(-1) = 0.16
			# Logistic yerine probit kullanmak modeli "yanlış" yapar
			# ama lasso hâlâ değişkenleri bulabilir mi diye test ediyoruz
			probs = norm.cdf(linear)
		else:
			# Logistic model: p_i = 1/(1+exp(-β^T x))
			# sigmoid fonksiyonu
			probs = 1 / (1 + np.exp(-linear))
		
		# Bernoulli örnekleme: her hasta için p_i olasılıkla y=1, 1-p_i olasılıkla y=0
		# np.random.binomial(1, probs, n):
		#   1: tek deneme (Bernoulli)
		#   probs: her gözlem için başarı olasılığı
		#   n: kaç gözlem üret
		y_sim = np.random.binomial(1, probs, n)
		
		# Lasso ile fit et, cross-validation ile optimal lambda seç
		# LogisticRegressionCV: farklı C değerlerini dener, cross-validation ile en iyisini seçer
		# np.logspace(-3, 2, 30): 0.001 ile 100 arasında 30 C değeri
		model = LogisticRegressionCV(
			penalty='l1',
			solver='saga',
			Cs=np.logspace(-3, 2, 30),
			cv=5,
			max_iter=5000
		)
		model.fit(X_sim, y_sim)
		
		# Sıfır olmayan katsayıların indeksleri = modelin seçtiği değişkenler
		# model.coef_[0]: katsayı vektörü (p,)
		# np.where(...)[0]: koşulu sağlayan indeksler
		selected = set(np.where(model.coef_[0] != 0)[0])
		
		psr, fdr = compute_psr_fdr(selected, true_relevant)
		psr_list.append(psr)
		fdr_list.append(fdr)
	
	# np.mean: listenin ortalaması — L tekrarın ortalama PSR ve FDR'si
	return np.mean(psr_list), np.mean(fdr_list)

# =============================================================
# PSR ve FDR'nin n'ye (örneklem büyüklüğü) bağımlılığı
# =============================================================

# n büyüdükçe ne bekliyoruz?
# PSR artmalı: daha fazla veri → model gerçek değişkenleri daha iyi bulur
# FDR azalmalı: daha fazla veri → model yanlış değişken seçmez

print("\n=== PSR ve FDR vs n ===")
n_values = [50, 100, 300, 500, 1000, 2000]
psr_n = []
fdr_n = []

for n in n_values:
	psr, fdr = run_lasso_simulation(n=n, n_irrelevant=10, L=100)
	psr_n.append(psr)
	fdr_n.append(fdr)
	print(f"n={n:5d}: PSR={psr:.3f}, FDR={fdr:.3f}")

plt.figure(figsize=(10, 5))
plt.plot(n_values, psr_n, 'b-o', label='PSR (yüksek olmalı)')
plt.plot(n_values, fdr_n, 'r-o', label='FDR (düşük olmalı)')
plt.xlabel('n — örneklem büyüklüğü')
plt.ylabel('Değer (0 ile 1 arası)')
plt.title('PSR ve FDR vs Örneklem Büyüklüğü\nn arttıkça PSR↑ FDR↓ bekliyoruz')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================
# PSR ve FDR'nin önemsiz değişken sayısına bağımlılığı
# =============================================================

# n=300 sabit, önemsiz değişken sayısı (k) değişiyor
# k büyüdükçe ne bekliyoruz?
# PSR azalabilir: gürültü artar, sinyal gömülür
# FDR artabilir: model daha fazla yanlış değişken seçer

print("\n=== PSR ve FDR vs önemsiz değişken sayısı (n=300) ===")
irrelevant_values = [10, 50, 100, 200, 500]
psr_k = []
fdr_k = []

for k in irrelevant_values:
	psr, fdr = run_lasso_simulation(n=300, n_irrelevant=k, L=100)
	psr_k.append(psr)
	fdr_k.append(fdr)
	print(f"k={k:5d}: PSR={psr:.3f}, FDR={fdr:.3f}")

plt.figure(figsize=(10, 5))
plt.plot(irrelevant_values, psr_k, 'b-o', label='PSR')
plt.plot(irrelevant_values, fdr_k, 'r-o', label='FDR')
plt.xlabel('Önemsiz değişken sayısı (k)')
plt.ylabel('Değer')
plt.title('PSR ve FDR vs Önemsiz Değişken Sayısı\nk arttıkça model zorlanır')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================
# LOGİSTİK vs PROBİT KARŞILAŞTIRMASI
# =============================================================

# Probit model: gerçekte p = Φ(β^T x) ile veri üretiyoruz
# Ama lasso ile logistic regression fit ediyoruz (model yanlış)
# Soru: yanlış model bile değişkenleri bulabilir mi?

print("\n=== Logistic vs Probit karşılaştırması (n=300, k=10) ===")

psr_log, fdr_log = run_lasso_simulation(
	n=300, n_irrelevant=10, L=100, probit=False   # doğru model
)
psr_pro, fdr_pro = run_lasso_simulation(
	n=300, n_irrelevant=10, L=100, probit=True    # yanlış model (probit veri, logistic fit)
)

print(f"Logistic (doğru model) : PSR={psr_log:.3f}, FDR={fdr_log:.3f}")
print(f"Probit   (yanlış model): PSR={psr_pro:.3f}, FDR={fdr_pro:.3f}")
print("\nYorum: PSR ve FDR çok farklıysa lasso probit veriye karşı hassas demektir.")
print("Çok farklı değilse lasso model yanlışlığına karşı robust (sağlam) demektir.")
