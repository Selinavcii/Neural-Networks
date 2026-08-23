[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Selinavcii/Neural-Networks/blob/main/SimpleNeuralNetwork.ipynb)

# Simple Neural Network (Sıfırdan Basit Sinir Ağı)

Bu proje, herhangi bir hazır derin öğrenme kütüphanesi (TensorFlow, PyTorch, Keras vb.) kullanmadan — sadece temel Python ile — bir yapay sinir ağının forward pass, loss hesabı ve gradient descent adımlarının nasıl çalıştığını sıfırdan anlamak için hazırlanmıştır.

## İçerik


### 1. Tek nöronlu, 3 girişli forward pass

3 girişi, kendi ağırlıklarıyla çarpıp bias ekleyerek tek bir nöronun çıktısını hesaplar.

```python
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)  # 2.3
```

### 2. 3 nöronlu, 3 girişli katman (fully connected layer) forward pass

Aynı 3 giriş, her biri kendi ağırlık setine ve bias'ına sahip 3 nörona bağlanır; katman 3 ayrı çıktı üretir.

```python
inputs = [1, 2, 3]
weights = [[1, 2, 3], [-0.5, 0.3, 0.8], [0.2, 0.5, -0.3]]
biases = [2, 3, 0.5]
output = [
    inputs[0]*weights[0][0] + inputs[1]*weights[0][1] + inputs[2]*weights[0][2] + biases[0],
    inputs[0]*weights[1][0] + inputs[1]*weights[1][1] + inputs[2]*weights[1][2] + biases[1],
    inputs[0]*weights[2][0] + inputs[1]*weights[2][1] + inputs[2]*weights[2][2] + biases[2],
]
print(output)  # [16, 5.5, 0.8]
```

### 3. Loss (hata) fonksiyonu — MSE

Katmanın 3 çıktısı, bir hedef (`y_true`) listesiyle karşılaştırılır; her çıktının hatası kareye alınıp ortalaması hesaplanır (Mean Squared Error).

```python
y_true = [10, 3, 4]
error0 = output[0] - y_true[0]
error1 = output[1] - y_true[1]
error2 = output[2] - y_true[2]
loss = (error0**2 + error1**2 + error2**2) / 3
print(loss)  # 17.4967
```

### 4. Parametreyi manuel değiştirerek loss'u gözlemleme

Tek nöronlu sistemde sadece `weights[0]` (w0) değiştirilip diğer her şey sabit tutularak, farklı w0 değerlerinde loss'un nasıl değiştiği karşılaştırılır.

| Durum | weights[0] | output | loss |
|---|---|---|---|
| 1 | 0.2 | 2.30 | 0.0900 |
| 2 | 0.4 | 2.50 | 0.2500 |
| 3 | 0.8 | 2.90 | 0.8100 |

### 5. Loss eğrisinin çizilmesi

`w0` bir aralıkta adım adım tarandı, her adımda loss hesaplanıp listeye eklendi ve `matplotlib` ile çizildi. Ortaya çıkan grafik, loss fonksiyonunun w0'a göre bir "çukur" (parabol) şeklinde olduğunu ve bir minimuma sahip olduğunu gösteriyor.

### 6. Sayısal türev ile basit gradient descent

Son adımda, w0'ın loss üzerindeki etkisi sayısal türev (`(loss(w0+h) - loss(w0)) / h`) ile yaklaşık olarak hesaplanıp, w0 bu eğime göre küçük adımlarla güncellendi — yani kütüphanesiz bir gradient descent uygulaması:

```
1. Adım  w0: 1.5    Loss: 2.5600
2. Adım  w0: 1.1800  Loss: 1.6384
3. Adım  w0: 0.9240  Loss: 1.0485
Sonuç    w0: 0.7192  Loss: 0.6710
```

Görüldüğü gibi, her adımda w0 güncellendikçe loss düzenli olarak azalıyor — bu, sinir ağı eğitiminin temel mantığı.

## Kullanılan kütüphaneler

- Sadece standart Python (forward pass, loss, manuel parametre denemeleri ve gradient descent adımlarının hepsi saf Python ile, harici kütüphane olmadan yazıldı)
- `matplotlib` — yalnızca loss eğrisini görselleştirmek için



## Öğrenilen kavramlar

- Girdi (input), ağırlık (weight), bias
- Tek nöron ve çok nöronlu katmanda forward pass
- Loss / Mean Squared Error (MSE)
- Parametre değişiminin loss üzerindeki etkisi, loss yüzeyi/eğrisi
- Sayısal türev ile gradient hesaplama ve gradient descent'in temel mantığı

## Notlar

Bu proje, sinir ağlarının temel çalışma mantığını (forward pass → loss → gradient descent) adım adım, kütüphaneler olmadan görmek amacıyla hazırlanan bir öğrenme çalışmasıdır.
