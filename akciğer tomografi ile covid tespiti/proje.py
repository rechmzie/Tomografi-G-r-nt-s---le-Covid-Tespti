import numpy as np
import PIL.Image as img
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Klasör yolları
covidli = "COVID/"
covid_olmayan = "non-COVID/"
test_klasoru = "test/"

# Klasördeki dosyaları almak için fonksiyon
def dosya(yol):
    return [os.path.join(yol, f) for f in os.listdir(yol) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Görüntüleri dönüştürmek için fonksiyon
def veri_donustur(klasor_adi, sinif_adi):
    goruntuler = dosya(klasor_adi)
    goruntu_sinif = []
    
    for goruntu in goruntuler:
        try:
            goruntu_oku = img.open(goruntu).convert("L")  # Grayscale'e çevir
            goruntu_boyutlandırma = goruntu_oku.resize((28, 28))  # Yeniden boyutlandır
            goruntu_donusturme = np.array(goruntu_boyutlandırma).flatten()  # Tek boyutlu vektöre çevir
            
            # Sınıf etiketini ekle
            if sinif_adi == "covidli":
                veriler = np.append(goruntu_donusturme, [0])  # COVID için etiket: 0
            elif sinif_adi == "covid_olmayan":
                veriler = np.append(goruntu_donusturme, [1])  # non-COVID için etiket: 1
            else:
                continue
            
            goruntu_sinif.append(veriler)
        
        except Exception as e:
            print(f"Hata: {goruntu}, {e}")
            continue
    
    return goruntu_sinif

# Veri setlerini oluştur
covidli_veri = veri_donustur(covidli, "covidli")
covidli_df = pd.DataFrame(covidli_veri)
covid_olmayan_veri = veri_donustur(covid_olmayan, "covid_olmayan")
covid_olmayan_df = pd.DataFrame(covid_olmayan_veri)

# Veri setlerini birleştir
tum_veri = pd.concat([covidli_df, covid_olmayan_df], ignore_index=True)

# Giriş ve çıkış ayrımı
Giris = np.array(tum_veri.iloc[:, :-1])  # Son sütun hariç tüm sütunlar
Cikis = np.array(tum_veri.iloc[:, -1])  # Sadece son sütun

# Veriyi eğitim ve test setine ayır
Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris, Cikis, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model = DecisionTreeClassifier()
clf = model.fit(Giris_train, Cikis_train)
Cikis_pred = clf.predict(Giris_test)

# Doğruluk metriğini yazdır
accuracy = metrics.accuracy_score(Cikis_test, Cikis_pred)
print("Doğruluk:", accuracy)

# Kaç adet COVID'li ve COVID olmayan veri olduğunu yazdır
covidli_sayi = len(covidli_veri)
covid_olmayan_sayi = len(covid_olmayan_veri)
print("COVID'li veri sayısı:", covidli_sayi)
print("COVID olmayan veri sayısı:", covid_olmayan_sayi)

# ROC eğrisi çizimi
fpr, tpr, threshold = metrics.roc_curve(Cikis_test, Cikis_pred)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC Eğrisi')
plt.plot(tpr, fpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Yanlış Pozitif Oranı')
plt.xlabel('Doğru Pozitif Oranı')
plt.show()

# Test klasöründeki görüntüleri değerlendirme
def test_veri_degerlendirme():
    test_goruntuler = dosya(test_klasoru)
    test_sonuclari = []
    
    for goruntu in test_goruntuler:
        try:
            goruntu_oku = img.open(goruntu).convert("L")  # Grayscale'e çevir
            goruntu_boyutlandırma = goruntu_oku.resize((28, 28))  # Yeniden boyutlandır
            goruntu_donusturme = np.array(goruntu_boyutlandırma).flatten()  # Tek boyutlu vektöre çevir
            goruntu_donusturme = goruntu_donusturme.reshape(1, -1)
            
            tahmin = clf.predict(goruntu_donusturme)
            sonuc = "COVID'li" if tahmin[0] == 0 else "COVID olmayan"
            test_sonuclari.append((goruntu, sonuc))
        
        except Exception as e:
            print(f"Hata: {goruntu}, {e}")
            continue
    
    return test_sonuclari

# Test verilerini değerlendir ve sonucu yazdır
test_sonuclari = test_veri_degerlendirme()
for goruntu, sonuc in test_sonuclari:
    print(f"Görüntü: {goruntu} -> Sonuç: {sonuc}")



    
    
