# ML_Projects

## 1. Logistic Regression (Lab 4, 7)
Ne yapar: İki sınıfı ayıran olasılık hesaplar.
Ne zaman kullanırsın: Binary classification (0/1 tahmin).
Anahtar kavramlar:

Katsayılar → hangi değişken ne kadar önemli
Log-likelihood → modelin ne kadar iyi fit ettiği
Regularization (L1/L2/Elastic Net) → katsayıları küçültür, overfitting'i önler
λ büyük → çok ceza → katsayılar sıfıra yaklaşır
Lasso (L1) → katsayıları tam sıfır yapar → değişken seçimi
Ridge (L2) → katsayıları küçültür ama sıfırlamaz


## 2. Evaluation Methods (Lab 5)
Ne yapar: Modelin ne kadar iyi tahmin ettiğini ölçer.
Anahtar kavramlar:

Refitting → en iyimser, aldatıcı
Cross-validation → güvenilir
Bootstrap → güvenilir
Bootstrap 0.632 → en dengeli
ROC curve → dengeli veri için
Precision-Recall → dengesiz veri için daha bilgilendirici
Threshold → 0.5 yerine farklı değer seçilebilir


## 3. Classification Trees + Bagging + Random Forest (Lab 6)
Ne yapar: Veriyi sorularla bölerek sınıflandırır.
Anahtar kavramlar:

Decision Tree → tek ağaç, overfitting'e yatkın
Bagging → 100 ağaç, çoğunluk oyu → daha stabil
Random Forest → bagging + her ağaçta rastgele değişken → en iyi
Pruning (budama) → ağacı küçültür, overfitting azalır
max_depth, min_samples_split → ağacın büyüklüğünü kontrol eder


## 4. Logistic Regression with Regularization (Lab 7)
Ne yapar: 6033 gen gibi çok değişkenli veride önemli değişkenleri bulur.
Anahtar kavramlar:

PSR → önemli değişkenlerin kaçını bulduk (yüksek olmalı)
FDR → seçtiklerimizin kaçı aslında önemsiz (düşük olmalı)
n büyüdükçe → PSR artar, FDR azalır
Önemsiz değişken arttıkça → FDR artar
Probit vs Logistic → lasso her iki modelde de iyi çalışır (robust)


## 5. SVM (Lab 11)
Ne yapar: İki sınıfı ayıran en geniş boşluğu bulur.
Anahtar kavramlar:

Support vectors → sınıra en yakın noktalar, kararı bunlar verir
Margin → iki sınıf arasındaki boşluk, SVM bunu maksimize eder
C parametresi → küçük C = geniş margin, büyük C = dar margin
Kernel trick → lineer olmayan sınırlar çizebilir
RBF kernel → gamma küçük = düzgün sınır, gamma büyük = overfitting
SVM vs Logistic → SVM sadece sınıra yakın noktaları dinler, LR hepsini


## 6. Multi-label Classification (Lab 14)
Ne yapar: Bir gözleme aynı anda birden fazla etiket atar.
Anahtar kavramlar:

Binary Relevance (BR) → her etiket bağımsız, basit ama etiket ilişkilerini kaçırır
Classifier Chain (CC) → etiketler sıralı öğrenir, sıraya bağımlı
ECC → 20 farklı sırayla CC, daha stabil
CCC → döngüsel, her etiket diğerlerini kullanır, iteratif yakınsama
Subset accuracy → çok katı, tüm etiketler doğru olmalı
Hamming score → etiket bazında doğruluk
Jaccard score → aktif etiketlere odaklanır


Hoca Ne Sorarsa Ne Yaparsın

"Veri üret" → np.random.randn, np.random.binomial, make_moons
"Fit et / train et" → .fit(X_train, y_train)
"Accuracy hesapla" → .score(X_test, y_test) veya np.mean(y_pred == y_test)
"Cross-validation" → cross_val_score(..., cv=10)
"Regularization" → LogisticRegression(penalty='l1'/'l2', C=...)
"Karar sınırı çiz" → plt.contour ile meshgrid
"Support vector" → model.support_vectors_
"Feature importance" → model.coef_
"Overfitting var mı?" → train accuracy yüksek, test accuracy düşükse evet
"Optimal lambda/C" → LogisticRegressionCV veya cross-validation loop

Kodu Okurken Ne Sorarsın
Bir kod gördüğünde şu 3 soruyu sor:

Veri nereden geliyor? → X ve y nasıl üretilmiş/yüklenmiş
Model ne yapıyor? → hangi sklearn sınıfı, hangi parametreler
Sonuç nasıl değerlendiriliyor? → hangi metrik kullanılmış
