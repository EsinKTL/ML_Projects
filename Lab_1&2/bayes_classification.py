import numpy as np
import matplotlib.pyplot as plt

class LDA:
	'''
	LDA, sınıfları birbirinden en iyi şekilde ayıracak doğrusal bir sınır
	(decision boundary) bulmaya çalışan generative sınıflandırma algoritmasıdır.
	Sınıfların (features) Çok Değişkenli Normal Dağılım'a
	(Multivariate Normal Distribution) uyduğunu varsayar.
	
	Normal Dağılım: Her sınıfın verisi uzayda hiper-eliptik bir bulut oluşturur.
		Bu bulutun merkez üssü ortalama vektörüdür (u_k).
	Eşit Kovaryans (Kritik Nokta): LDA, tüm sınıfların aynı varyans ve
		korelasyon yapısına (yani aynı kovaryans matrisine =sum=) sahip
		olduğunu varsayar.
		Geometrik olarak bunun anlamı şudur: Sınıf bulutlarının uzaydaki genişlikleri,
		eğimleri ve şekilleri birebir aynıdır; sadece merkezleri (mu_0 ve mu_1)
		farklı yerlerdedir.
	Doğrusal Sınır: Madem sınıfların şekilleri/eğimleri aynı, o zaman onları
		ayıran sınır mükemmel bir doğru (veya hiper-düzlem) olur.
		QDA'de bu matrisler farklı olacağı için sınır eğrileşecektir
		(quadratic).
	'''
	def fit(self,X,y):
		# 1. Sınıfları ayır (X_0 ve X_1)
		# y'nin 0'a eşit olduğu indekslerdeki X satırlarını alıyoruz
		X_0 = X[y == 0]
		
		# y'nin 1'e eşit olduğu indekslerdeki X satırlarını alıyoruz
		X_1 = X[y == 1]
	
		# 2. Prior (Önsel Olasılık) Hesaplama
		# Toplam gözlem sayısını bulalım
		total_samples = len(y)
		
		self.prior_0 = X_0.shape[0] / total_samples
		self.prior_1 = X_1.shape[0] / total_samples
	
		# 3. Mean (Ortalama) Vektörlerini Hesaplama
		self.mu_0 = np.mean(X_0, axis=0)
		self.mu_1 = np.mean(X_1, axis=0)
	
		# 4. Sınıf kovaryanslarını np.cov(..., rowvar=False) ile bul
		'''
		LDA'de her iki sınıfın da kovaryans matrislerini hesaplayıp,
			ardından bunları ağırlıklı ortalamayla tek bir ortak kovaryans
			matrisine (self.sigma) dönüştürmeliyiz.
		Kovaryans matrisi, özelliklerin kendi varyanslarını ve birbirleriyle
			olan ilişkilerini (korelasyonlarını) gösterir.
		NumPy'da kovaryans np.cov() fonksiyonu ile hesaplanır. fonksiyonun
			satırları değil sütunları özellik kabul etmesi için rowvar=False demeliyiz.
		'''
		
		cov_0 = np.cov(X_0, rowvar=False)
		cov_1 = np.cov(X_1, rowvar=False)
		
		# 5. Ortak kovaryansı (self.sigma) ağırlıklı ortalamayla hesapla
		self.sigma = (X_0.shape[0] * cov_0 + X_1.shape[0] * cov_1) / total_samples
		
		self.sigma_inv = np.linalg.inv(self.sigma) # ...
		
	def predict_proba(self, Xtest):
		# log-likelihood tabanlı karar skoru (discriminant score)
		score_0 = (Xtest @ self.sigma_inv @ self.mu_0) - (0.5 * self.mu_0 @ self.sigma_inv @ self.mu_0) + np.log(self.prior_0)
		score_1 = (Xtest @ self.sigma_inv @ self.mu_1) - (0.5 * self.mu_1 @ self.sigma_inv @ self.mu_1) + np.log(self.prior_1)
		
		# Sayısal kararlılık için farkı sigmoid fonksiyonuna sokuyoruz
		probs = 1 / (1 + np.exp(-(score_1 - score_0)))
		
		return probs
	
	def predict(self, Xtest):
		probs = self.predict_proba(Xtest)
		return np.where(probs > 0.5, 1, 0)

	def get_params(self):
		return {'prior_0': self.prior_0, 'prior_1': self.prior_1, "mu_0": self.mu_0, "mu_1": self.mu_1, "sigma": self.sigma}
		
class QDA:
	'''
    QDA, her sınıfın kendine ait ayrı bir kovaryans matrisine sahip olduğunu varsayar.
    Bu yüzden karar sınırı karesel (eğri) bir yapıya bürünür.
    '''
	
	def fit(self,X,y):
		# 1. Sınıfları ayır (X_0 ve X_1)
		X_0 = X[y == 0]
		X_1 = X[y == 1]
		
		# 2. Prior (Önsel Olasılık) Hesaplama
		total_samples = len(y)
		
		self.prior_0 = X_0.shape[0] / total_samples
		self.prior_1 = X_1.shape[0] / total_samples
		
		# 3. Mean (Ortalama) Vektörlerini Hesaplama
		self.mu_0 = np.mean(X_0, axis=0)
		self.mu_1 = np.mean(X_1, axis=0)
		
		# 4. Sınıf kovaryanslarını np.cov(..., rowvar=False) ile bul
		cov_0 = np.cov(X_0, rowvar=False)
		cov_1 = np.cov(X_1, rowvar=False)
		
		#5. Ayrı cov hesabı
		# Doğrudan cov_0 ve cov_1 matrislerinin tersini (inverse) alıp nesne niteliği yap.
		
		self.sigma_inv_0 = np.linalg.inv(cov_0)
		self.sigma_inv_1 = np.linalg.inv(cov_1)
		
		'''
		NumPy'da bir matrisin determinantını hesaplamak için np.linalg.det() fonksiyonu kullanılır. Ancak sayısal kararlılık (numerical stability) sağlamak ve çok küçük/büyük
		sayılarda taşma yaşamamak için NumPy bize harika bir kolaylık sunar: np.linalg.slogdet(). Bu fonksiyon bir matrisin determinantının logaritmasını doğrudan hesaplar.
		'''
		
		#6. Log_determinantlarını hesapla (QDA fakrı)
		# np.linalg.slogdet fonksiyonu bize iki değer döner: (sign, logdet).
		# Bizim sadece ikinci değer olan logdet'e ihtiyacımız var, o yüzden [1] indeksiyle alıyoruz.
		_, self.logdet_0 = np.linalg.slogdet(cov_0)
		_, self.logdet_1 = np.linalg.slogdet(cov_1)
	
	def predict_proba(self,Xtest):
		# Her test verisinin Sınıf 0 ve sınıf 1'ın merkezinden olan farkını buluyoruz
		diff_0 = Xtest - self.mu_0
		diff_1 = Xtest - self.mu_1
	
		# Mahalanobis mesafesini satır bazlı matris çarpımıyla hesaplıyoruz
		# diff_0 @ self.sigma_inv_0 çarpımının diff_0 ile eleman bazlı çarpılıp
		# (mul), satır bazlı toplanması (sum) gerekir.
		mahalanobis_0 = np.sum((diff_0 @ self.sigma_inv_0) * diff_0, axis=1)
		mahalanobis_1 = np.sum((diff_1 @ self.sigma_inv_1) * diff_1, axis=1)
	
		# Nihai QDA skoru
		score_0 = -0.5 * self.logdet_0 - 0.5 * mahalanobis_0 + np.log(self.prior_0)
		score_1 = -0.5 * self.logdet_1 - 0.5 * mahalanobis_1 + np.log(self.prior_1)
		
		probs = 1 / (1 + np.exp(-(score_1 - score_0)))
		return probs
	
	def predict(self, Xtest):
		probs = self.predict_proba(Xtest)
		return np.where(probs > 0.5, 1, 0)
	
	def get_params(self):
		return {
			"prior_0": self.prior_0, "prior_1": self.prior_1,
			"mu_0": self.mu_0, "mu_1": self.mu_1,
			"sigma_0": np.linalg.inv(self.sigma_inv_0),
			"sigma_1": np.linalg.inv(self.sigma_inv_1)}

class NB:
	
	'''
	Naive Bayes, özelliklerin bağımsız olduğunu varsaydığı için özellikler
	arasındaki korelasyonu sıfır kabul eder. Bu durumda kovaryans matrisinin
	köşegeninin (diagonal) dışında kalan tüm elemanlar 0 olur.
	Matrisimiz sadece her özelliğin kendi varyansını barındıran bir Köşegen
	Matrise (Diagonal Matrix) dönüşür
	'''
	def fit(self,X,y):
	
		'''Gausyen Naive Bayes modelinde her sınıfın her bir özelliği için ortalama
		(mu) ve varyans (o-^2) değerlerini hesaplayıp saklamamız gerekir.
		
		NumPy'da varyans hesaplamak için np.var(..., axis=0) fonksiyonu kullanılır.
		axis=0 yine sütun bazlı (özellik bazlı) varyans almamızı sağlar.
		'''
		# 1. Sınıfları ayır
		X_0 = X[y == 0]
		X_1 = X[y == 1]
		
		# 2. Priorları hesapla
		total_samples = len(y)
		self.prior_0 = X_0.shape[0] / total_samples
		self.prior_1 = X_1.shape[0] / total_samples
		
		# 3. Ortalama (Mean) vektörlerini hesapla
		self.mu_0 = np.mean(X_0, axis=0)
		self.mu_1 = np.mean(X_1, axis=0)
		
		# 4. Varyansları (Variance) sütun bazlı hesapla
		self.var_0 = np.var(X_0, axis=0)
		self.var_1 = np.var(X_1, axis=0)

	def predict_proba(self,Xtest):
		'''
		Gausyen Naive Bayes'te özelliklerin birbirinden bağımsız olduğunu
		varsaydığımız için, her bir özelliğin olasılık yoğunluk fonksiyonunu
		(PDF) ayrı ayrı hesaplayıp birbiriyle çarpabiliriz.
		Olasılıkları çarpmak yerine, sayısal kararlılık için logaritmasını
		alıp toplayacağız (Log-Likelihood).
		'''
		# Her iki terimde de ortak olan 2 * pi ifadesini logaritma içinde kullanıyoruz
		# np.sum(..., axis=1) diyerek her bir özelliğin getirdiği skoru satır bazlı topluyoruz
		score_0 = np.sum(-0.5 * np.log(2 * np.pi * self.var_0) - ((Xtest - self.mu_0) ** 2) / (2 * self.var_0), axis=1) + np.log(self.prior_0)
		score_1 = np.sum(-0.5 * np.log(2* np.pi * self.var_1) - ((Xtest - self.mu_1) ** 2) / (2 * self.var_1), axis=1) + np.log(self.prior_1)
	
		# Sigmoid ile olasılığa dönüştürme
		probs = 1 / (1 + np.exp(-(score_1 - score_0)))
		return probs
	
	def predict(self, Xtest):
		probs = self.predict_proba(Xtest)
		return np.where(probs > 0.5, 1, 0)

	def get_params(self):
		return {
			"prior_0": self.prior_0, "prior_1": self.prior_1,
			"mu_0": self.mu_0, "mu_1": self.mu_1,
			"var_0": self.var_0, "var_1": self.var_1}

def generate_scheme_1(n,a, p=2):
	
	'''
	Toplam n=1000 gözlem üretilecek.
		Hedef değişken y (0 veya 1), başarı olasılığı 0.5 olan bir Bernoulli
	dağılımından üretilecek. (Yani verinin yaklaşık yarısı 0, yarısı 1 olacak).
		Sınıf 0'a ait olan özellikler (p=2), ortalaması 0 ve varyansı 1 olan
	standart normal dağılımdan bağımsız olarak üretilecek.
		Sınıf 1'in özellikleri ise ortalaması a ve varyansı 1 olan normal
	dağılımdan üretilecek.
		Bu testi a = 0.1, 0.5, 1, 2, 3, 5$ gibi farklı değerler için tekrarlayacağız.
	'''
	
	# 1 denemeli binom dağılımı, Bernoulli dağılımına eşittir
	# 1. Bernoulli dağılımından y vektörünü (0 ve 1'ler) üret
	y = np.random.binomial(n=1, p=0.5, size=n)
	
	# Boş bir X matrisi oluşturalım (n satır, p sütun)
	X = np.zeros((n, p))
	
	# y'nin 0 olduğu indekslerin sayısını bulalım
	n_0 = np.sum(y == 0)
	# y'nin 1 olduğu indekslerin sayısını bulalım
	n_1 = np.sum(y == 1)
	
	# np.random.normal fonksiyonunu kullanarak X[y == 0] kısmını doldur:
	X[y == 0] = np.random.normal(loc=0, scale=1, size=(n_0, p))
	X[y == 1] = np.random.normal(loc=a, scale=1, size=(n_1, p))
	
	return X, y

def generate_scheme_2(n,a, rho, p=2):
	
	'''
		Sınıf 0'ın özellikleri, iki boyutlu bir normal dağılımdan üretilecek;
	ortalaması 0, varyansı 1 ve aralarındaki korelasyon (rho) olacak.
		Sınıf 1'in özellikleri ise yine iki boyutlu bir normal dağılımdan üretilecek;
	ortalaması a, varyansı 1 ve aralarındaki korelasyon (eksi rho) olacak.
		İşin içine korelasyon (iki değişkenin birbiriyle ilişkisi) girdiği an,
	özellikleri birbirinden bağımsız üretemeyiz. Bu yüzden np.random.normal
	yerine, Çok Değişkenli Normal Dağılımdan veri üreten
	np.random.multivariate_normal fonksiyonunu kullanmak zorundayız.
	'''
	
	# 1. Bernoulli dağılımından y vektörünü (0 ve 1'ler) üret
	y = np.random.binomial(n=1, p=0.5, size=n)
	
	X = np.zeros((n, p))

	n_0 = np.sum(y == 0)
	n_1 = np.sum(y == 1)

	# Sınıf 0 için parametreler (Ortalama 0, korelasyon rho)
	mean_0 =[0,0]
	cov_0 =[[1,rho],[rho,1]]
	X[y == 0] = np.random.multivariate_normal(mean_0, cov_0, size=n_0)
	
	# Sınıf 1 için parametreler
	mean_1 =[a,a]
	cov_1 =[[1,-rho],[-rho,1]]
	X[y == 1] = np.random.multivariate_normal(mean_1, cov_1, size=n_1)
	
	return X, y

def compute_accuracy(y_true, y_pred):
	return np.mean(y_true == y_pred)

#Deney Parametreleri
a_values = [0.1, 0.5, 1, 2, 3, 5]
n_splits = 30
rho = 0.5

results_scheme_1 = {"LDA": [], "QDA":[], "NB":[] }

# Deney döngüsü
for a in a_values:
	# her bir a değeri için 30 farklı rastgele split
	for split in range(n_splits):
		
		#1. Veriyi üret
		X, y = generate_scheme_1(n =1000, a=a)
		
		#2. Veriyi %70 /%30 bölüyoruz
		num_samples = X.shape[0]
		shuffled_indices = np.random.permutation(num_samples)
		train_size = int(num_samples * 0.7)
		
		train_idx = shuffled_indices[:train_size]
		test_idx = shuffled_indices[train_size:]
		
		X_train = X[train_idx]
		X_test = X[test_idx]
		
		y_train = y[train_idx]
		y_test = y[test_idx]
		
		# 3. Modelleri eğitim tahmin alıyoruz.
		# -----------------------------------
		# LDA
		lda = LDA()
		lda.fit(X_train,y_train)
		preds_lda = lda.predict(X_test)
		acc_lda = compute_accuracy(y_test, preds_lda)
		results_scheme_1["LDA"].append({"a": a, "accuracy": acc_lda})
		
		#QDA
		qda = QDA()
		qda.fit(X_train,y_train)
		preds_qda = qda.predict(X_test)
		acc_qda = compute_accuracy(y_test, preds_qda)
		results_scheme_1["QDA"].append({"a": a, "accuracy": acc_qda})
		
		#Naive Bayes
		nb = NB()
		nb.fit(X_train,y_train)
		preds_nb = nb.predict(X_test)
		acc_nb = compute_accuracy(y_test, preds_nb)
		results_scheme_1["NB"].append({"a": a, "accuracy": acc_nb})

# Scheme 2 için skorları saklayacağımız yeni bir sözlük açıyoruz
results_scheme_2 = {"LDA": [], "QDA": [], "NB": []}

for a in a_values:
	for split in range(n_splits):
		# 1. Scheme 2 fonksiyonu çağrılıyor (rho=0.5 sabit)
		X, y = generate_scheme_2(n=1000, a=a, rho=rho)
		
		# 2. %70 / %30 Bölme işlemi
		num_samples = X.shape[0]
		shuffled_indices = np.random.permutation(num_samples)
		train_size = int(num_samples * 0.7)
		train_idx = shuffled_indices[:train_size]
		test_idx = shuffled_indices[train_size:]
		
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]
		
		# 3. Modeller
		# --- LDA ---
		lda = LDA()
		lda.fit(X_train, y_train)
		preds_lda = lda.predict(X_test)
		acc_lda = compute_accuracy(y_test, preds_lda)
		results_scheme_2["LDA"].append({"a": a, "accuracy": acc_lda})
		
		# --- QDA ---
		qda = QDA()
		qda.fit(X_train, y_train)
		preds_qda = qda.predict(X_test)
		acc_qda = compute_accuracy(y_test, preds_qda)
		results_scheme_2["QDA"].append({"a": a, "accuracy": acc_qda})
		
		# --- Naive Bayes ---
		nb = NB()
		nb.fit(X_train, y_train)
		preds_nb = nb.predict(X_test)
		acc_nb = compute_accuracy(y_test, preds_nb)
		results_scheme_2["NB"].append({"a": a, "accuracy": acc_nb})

lda_data1 = [[d["accuracy"] for d in results_scheme_1["LDA"] if d["a"] == a] for a in a_values]
qda_data1 = [[d["accuracy"] for d in results_scheme_1["QDA"] if d["a"] == a] for a in a_values]
nb_data1 = [[d["accuracy"] for d in results_scheme_1["NB"] if d["a"] == a] for a in a_values]



lda_data2 = [[d["accuracy"] for d in results_scheme_2["LDA"] if d["a"] == a] for a in a_values]
qda_data2 = [[d["accuracy"] for d in results_scheme_2["QDA"] if d["a"] == a] for a in a_values]
nb_data2 = [[d["accuracy"] for d in results_scheme_2["NB"] if d["a"] == a] for a in a_values]

lda

# GRAFİK 1: SCHEME 1 (BayesianSimulatedData1.pdf)
fig1, ax1 = plt.subplots(figsize=(10, 6))
positions = np.arange(len(a_values))

# Kutuları yan yana yerleştiriyoruz
ax1.boxplot(lda_data1, positions=positions - 0.2, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightblue"))
ax1.boxplot(qda_data1, positions=positions, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
ax1.boxplot(nb_data1, positions=positions + 0.2, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightpink"))

# Eksen ve Etiket Ayarları (Kapatmadan hemen önce yapılmalı!)
ax1.set_xticks(positions)
ax1.set_xticklabels(a_values)
ax1.set_xlabel("a Parametresi")
ax1.set_ylabel("Doğruluk (Accuracy)")
ax1.set_title("Scheme 1: Farklı a Değerleri İçin Model Karşılaştırmaları")
ax1.legend(["LDA", "QDA", "NB"])

# Figür 1'i kaydet ve hafızayı temizle
plt.savefig("BayesianSimulatedData1.pdf", format="pdf")
plt.close(fig1)
print("Scheme 1 Grafiği PDF olarak kaydedildi! [BayesianSimulatedData1.pdf]")


# GRAFİK 2: SCHEME 2 (BayesianSimulatedData2.pdf)
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Kutuları yan yana yerleştiriyoruz
ax2.boxplot(lda_data2, positions=positions - 0.2, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightblue"))
ax2.boxplot(qda_data2, positions=positions, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
ax2.boxplot(nb_data2, positions=positions + 0.2, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightpink"))

# Eksen ve Etiket Ayarları (Dökümandaki isteğe göre başlık güncellendi)
ax2.set_xticks(positions)
ax2.set_xticklabels(a_values)
ax2.set_xlabel("a Parametresi (Fixed rho=0.5)")
ax2.set_ylabel("Doğruluk (Accuracy)")
ax2.set_title("Scheme 2: Farklı a Değerleri İçin Model Karşılaştırmaları (Korelasyonlu)")
ax2.legend(["LDA", "QDA", "NB"])

# Figür 2'yi kaydet ve hafızayı temizle
plt.savefig("BayesianSimulatedData2.pdf", format="pdf")
plt.close(fig2)
print("Scheme 2 Grafiği PDF olarak kaydedildi! [BayesianSimulatedData2.pdf]")


# SCHEME 3

# 1. Sabit parametrelerle Scheme 2 verisi üretiyoruz (a=2, rho=0.5)
X_train, y_train = generate_scheme_2(n=1000, a=2, rho=0.5)

# 2. Modellerimizi bu veriyle eğitiyoruz
lda = LDA()
lda.fit(X_train, y_train)

qda = QDA()
qda.fit(X_train, y_train)

# 3. Grafik alanının sınırlarını belirliyoruz
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

# np.meshgrid ile ızgarayı oluşturuyoruz
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Izgara üzerindeki tüm noktaları iki sütunlu bir test matrisi haline getiriyoruz
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 4. Modellere bu ızgara noktaları için tahmin yaptırıyoruz
Z_lda = lda.predict(grid_points).reshape(xx.shape)

Z_qda = qda.predict(grid_points).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 8))

# 1. Eğitim verilerini sınıflarına göre ayırarak çiziyoruz (Döküman istekleri)
# Sınıf 0: Mavi renk ('blue') ve daire ('o') sembolü
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
           color="blue", marker="o", label="Sınıf 0", alpha=0.5)

# Sınıf 1: Kırmızı renk ('red') ve artı ('+') sembolü
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
           color="red", marker="+", label="Sınıf 1", alpha=0.5)

# 2. Karar Sınırlarını (Decision Boundaries) Çiziyoruz
# levels=[0.5] diyerek tam 0'dan 1'e geçiş çizgisini yakalıyoruz
ax.contour(xx, yy, Z_lda, levels=[0.5], colors="black", linestyles="--", linewidths=2)
ax.contour(xx, yy, Z_qda, levels=[0.5], colors="purple", linestyles="-", linewidths=2)

# Grafiğin etiketleri ve kozmetiği
ax.set_xlabel("Özellik 1 (Feature 1)")
ax.set_ylabel("Özellik 2 (Feature 2)")
ax.set_title("Eğitim Kümesi ve Karar Sınırları (a=2, rho=0.5)")

# Sınır çizgilerini lejantta göstermek için boş proxy sanal çizgiler ekleyebilirsin
from matplotlib.lines import Line2D
custom_lines = [
	Line2D([0], [0], color="blue", marker="o", linestyle=""),
	Line2D([0], [0], color="red", marker="+", linestyle=""),
	Line2D([0], [0], color="black", linestyle="--"),
	Line2D([0], [0], color="purple", linestyle="-")
]
ax.legend(custom_lines, ["Sınıf 0", "Sınıf 1", "LDA Sınırı (Doğrusal)", "QDA Sınırı (Karesel)"])

# Üçüncü PDF olarak kaydetme
plt.savefig("BayesianSimulatedData3.pdf", format="pdf")
plt.close()
print("Karar sınırları grafiği başarıyla kaydedildi! [BayesianSimulatedData3.pdf]")