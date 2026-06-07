import numpy as np
import matplotlib.pyplot as plt

'''
Üreteceğimiz 200 adet verinin %90'ı ortalaması 5, varyansı 1 olan bir
popülasyondan; kalan %10'u ise ortalaması 10, varyansı 1 olan başka bir
popülasyondan rastgele seçilecek. Grafik çizdiğinde tek bir tepe noktası
(unimodal) değil, biri büyük biri küçük iki tepeli (bimodal) bir histogram
elde etmeliyiz.
'''

'''
KDE'nin yaptığı sihir şudur: Verideki her bir noktanın üzerine yumuşak bir
tepe noktası (buna Kernel denir) yerleştirir. Sonra bu 200 tane küçük
tepeciği üst üste toplar. Ortaya çıkan nihai pürüzsüz eğri, bizim yoğunluk
tahminimiz olur.

n: Veri sayısı (bizim senaryoda 200).
X_i: Elimizdeki gerçek veri noktaları.
x: Yoğunluğunu (yüksekliğini) ölçmek istediğimiz herhangi bir konum.
h: Smoothing parameter (Bandwidth / Yumuşatma parametresi).
Tepelerin ne kadar geniş veya dar olacağını belirler.
K(dot): Seçtiğimiz Kernel Fonksiyonu.'''

def gaussian_kernel(u):
	# En popüler kerneldir. Noktaların üzerine standart çan eğrisi yerleştirir.
	return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def epanechnikov_kernel(u):
	'''
	Matematiksel olarak varyansı en aza indiren en optimal kerneldir.
	Eğer |u| <= 1 ise parabolik değer üretir, dışındaysa sıfır döner.
	'''
	
	# u'nun mutlak değeri 1'den küçük eşitse formülü uygula, değilse 0 dön
	return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

def uniform_kernel(u):
	#En basit kerneldir. Noktanın etrafına düz bir kutu yerleştirir. |u| <= 1 ise sabit 1/2 değerini üretir.
	return np.where(np.abs(u) <= 1, 0.5, 0)

def kde_predict(x, X_data, h, kernel_func):
	n = len(X_data)
	
	# 1. Her bir X_data noktası için u değerlerini vektörel olarak hesapla
	u = (x - X_data) / h
	
	#2. Bu u değerlerini seçilen kernel fonksiyonuna gönder
	kernel_val = kernel_func(u)
	
	f_hat = np.sum(kernel_val) / (n * h)
	
	return f_hat

def true_density(x):
	# N(5,1) bileşeni
	pdf_1 = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 *(x - 5)**2)
	
	# N(10,1) bileşeni
	pdf_2 = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 *(x - 10)**2)
	
	return 0.9 * pdf_1 + 0.1 * pdf_2

def kde_evaluate(x_grid, X_data, h, kernel_func):
	# x_grid içindeki her bir nokta için tek tek kde_predict çağırıp listeye ekliyoruz
	return np.array([kde_predict(x, X_data, h, kernel_func) for x in x_grid])

def compute_kde_mse(X_data, h, kernel_func, K=1000):
	# [2, 12] aralığında rastgele $K=1000$ tane iid sayı üretmek için np.random.uniform(low=2, high=12, size=1000) fonksiyonunu kullanacaksın.
	
	# 1. [2,12] Aralığında K tane rastgele x_i noktası üret
	x_grid = np.random.uniform(2, 12, size=K)
	
	# 2. Bu noktalar için gerçek yoğunluk değerlerini bul.
	y_true = true_density(x_grid)
	
	# 3. Bu noktalar için bizim KDE modelimizin tahminlerini bul
	y_pred = kde_evaluate(x_grid, X_data, h, kernel_func)
	
	# 4. Formüle göre gerçek değerler ile tahminler arasındaki farkın karesinin ortalamasını (MSE) dönmeliyiz.
	mse = np.mean((y_true - y_pred) ** 2)
	
	return mse

def silverman_bandwidth(X_data):
	
	n = len(X_data)
	sigma_hat = np.std(X_data, ddof=1)
	
	return 1.06 * sigma_hat * (n **(-1/5))

n_values = [50, 100, 200, 500, 1000]
mse_results = []


# 1. GAUSSIAN MIXTURE VERİ SİMÜLASYONU
def generate_mixture_data(n=200):
	secim = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
	X = np.zeros(n)
	for i in range(n):
		if secim[i] == 0:
			X[i] = np.random.normal(5, 1)
		else:
			X[i] = np.random.normal(10, 1)
	return X


for n in n_values:
	# 1. n boyutunda veri üret
	X_sample = generate_mixture_data(n=n)
	
	# 2. Silverman ile her n için h parametresini dinamik hesapla
	h_optimal = silverman_bandwidth(X_sample)
	
	# 3. Gaussian kernel kullanarak MSE hesapla
	error = compute_kde_mse(X_sample, h_optimal, gaussian_kernel)
	mse_results.append(error)

# Grafiğe dökelim
plt.figure(figsize=(8, 5))
plt.plot(n_values, mse_results, marker='o', color='purple', linewidth=2)
plt.xlabel("Örneklem Boyutu (n)")
plt.ylabel("Ortalama Karesel Hata (MSE)")
plt.title("Örneklem Boyutunun MSE Üzerindeki Etkisi")
plt.grid(True)
plt.savefig("KDE_n_influence.pdf", format="pdf")
plt.close()

X_data = generate_mixture_data(n=200)
x_plot = np.linspace(2, 12, 500) # Grafiği pürüzsüz çizdirmek için 500 nokta
y_true = true_density(x_plot)

kernels = [gaussian_kernel, epanechnikov_kernel, uniform_kernel]
kernel_names = ["Gaussian", "Epanechnikov", "Uniform"]
h_values = [0.2, 1.0, 3.0] # Küçük, Optimal, Büyük h değerleri

fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)

for k_idx, kernel in enumerate(kernels):
	for h_idx, h in enumerate(h_values):
		ax = axes[k_idx, h_idx]
		
		# 1. Gerçek yoğunluğu çizelim
		ax.plot(x_plot, y_true, label="Gerçek Yoğunluk", color="black", linestyle="--")
		
		# Soru: Bizim eğittiğimiz KDE modelinin tahminlerini üret:
		# İpucu: x_plot, X_data, h ve kernel değişkenlerini kullanacaksın.
		y_pred = kde_evaluate(x_plot, X_data, h, kernel)
		
		# 2. Tahmin edilen yoğunluğu çizelim
		ax.plot(x_plot, y_pred, label="KDE Tahmini", color="red")
		
		ax.set_title(f"{kernel_names[k_idx]} (h={h})")
		if k_idx == 0 and h_idx == 0:
			ax.legend()

plt.tight_layout()
plt.savefig("KDE_kernel_h_influence.pdf", format="pdf")
plt.close()

def generate_artificial_sample(X_data, h, k=5000):
	n = len(X_data)
	X_prime = np.zeros(k)
	
	for j in range(k):
		# 1. 0 ile n-1 arasında rastgele bir indeks seç (Uniform)
		i = np.random.randint(0, n)
		
		# 2. Standart normal dağılımdan epsilon üret
		epsilon = np.random.normal(0, 1)
		
		X_prime[j] = X_data[i] + epsilon * h
	
	return X_prime

# 1. Orijinal 200 boyutlu veriyi üret ve optimal h'ı bul
X_orig = generate_mixture_data(n=200)
h_optimal = silverman_bandwidth(X_orig)

# 2. Yöntem 1: Orijinal veriyle direkt MSE hesapla
mse_method_1 = compute_kde_mse(X_orig, h_optimal, gaussian_kernel)

# 3. Yöntem 2: Orijinal veriden k=5000 boyutlu yapay örneklem üret
X_artif = generate_artificial_sample(X_orig, h_optimal, k=5000)

# Yapay veri üzerinden tekrar MSE hesapla
mse_method_2 = compute_kde_mse(X_artif, h_optimal, gaussian_kernel)

print("#1 Yöntem MSE (Orijinal):", mse_method_1)
print("#2 Yöntem MSE (Yapay):  ", mse_method_2)


class KDENaiveBayes:
	def __init__(self, h=1.0, kernel_func=gaussian_kernel):
		self.h = h
		self.kernel_func = kernel_func
	
	def fit(self, X, y):
		# Sadece verileri sınıflarına göre ayırıp saklıyoruz
		self.X_0 = X[y == 0]
		self.X_1 = X[y == 1]
		
		total_samples = len(y)
		self.prior_0 = self.X_0.shape[0] / total_samples
		self.prior_1 = self.X_1.shape[0] / total_samples
	
	def predict_proba(self, Xtest):
		num_samples = Xtest.shape[0]
		num_features = Xtest.shape[1]
		
		probs = np.zeros(num_samples)
		
		# Her bir test örneği için tek tek olasılık hesaplıyoruz
		for idx in range(num_samples):
			x_sample = Xtest[idx]
			
			# Sınıf 0 için log-likelihood hesabı
			score_0 = np.log(self.prior_0)
			for f in range(num_features):
				# f. özelliğin Sınıf 0 verilerindeki KDE yoğunluğunu hesapla
				kde_val = kde_predict(x_sample[f], self.X_0[:, f], self.h, self.kernel_func)
				# Sayısal kararlılık için alt sınır (underflow koruması) ekliyoruz
				score_0 += np.log(max(kde_val, 1e-10))
			
			# Sınıf 1 için log-likelihood hesabı
			score_1 = np.log(self.prior_1)
			for f in range(num_features):
				# f. özelliğin Sınıf 1 verilerindeki KDE yoğunluğunu hesapla
				kde_val = kde_predict(x_sample[f], self.X_1[:, f], self.h, self.kernel_func)
				# Sayısal kararlılık ve underflow koruması
				score_1 += np.log(max(kde_val, 1e-10))
			
			# İki skoru sigmoid fonksiyonuna besleyerek olasılığa dönüştür
			probs[idx] = 1 / (1 + np.exp(-(score_1 - score_0)))
		
		return probs
	
	def predict(self, Xtest):
		probs = self.predict_proba(Xtest)
		return np.where(probs > 0.5, 1, 0)