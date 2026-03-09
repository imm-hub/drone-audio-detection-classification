# 🚁 Akustik Drone Tespiti ve Sınıflandırması

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**İstatistiksel Sinyal İşleme ve Derin Öğrenme ile Akustik Drone Tespiti**

Bu proje, droneların akustik imzalarını kullanarak tespit ve sınıflandırma gerçekleştirilmektedir.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Veri Setleri](#-veri-setleri)
- [Kullanım](#-kullanım)
- [Metodoloji](#-metodoloji)
- [Sonuçlar](#-sonuçlar)
- [Proje Yapısı](#-proje-yapısı)
- [Referanslar](#-referanslar)

## ✨ Özellikler

- **İstatistiksel Öznitelikler**: PSD (Welch), MFCC, STFT, Mel Spektrogram
- **Makine Öğrenmesi**: SVM, Random Forest
- **Derin Öğrenme**: CNN, LSTM, CRNN
- **Kapsamlı Görselleştirme**: Spektrogramlar, PSD grafikleri, model karşılaştırmaları

## 🔧 Kurulum

### Gereksinimler

```bash
# Temel kurulum
pip install -r requirements.txt

# Veya manuel kurulum
pip install numpy scipy librosa scikit-learn matplotlib seaborn
pip install tensorflow  # Opsiyonel - derin öğrenme için
```

### Repository Klonlama

```bash
git clone https://github.com/[username]/acoustic-drone-detection.git
cd acoustic-drone-detection
```

## 📊 Veri Setleri

| Veri Seti | Kaynak | Açıklama | Link |
|-----------|--------|----------|------|
| **DroneAudioDataset** | Al-Emadi et al. | 4 drone türü, iç mekan | [GitHub](https://github.com/saraalemadi/DroneAudioDataset) |

### Veri Seti İndirme

```bash
# DroneAudioDataset
git clone https://github.com/saraalemadi/DroneAudioDataset.git data/DroneAudioDataset


## 🚀 Kullanım

### Hızlı Başlangıç (Sentetik Veri ile Demo)

```bash
python src/train.py --synthetic
```

### Tam Eğitim

```bash
# İkili sınıflandırma (Drone / Non-Drone)
python src/train.py --mode binary

# Çoklu sınıflandırma (Bebop, Mambo, Phantom, Unknown)
python src/train.py --mode multiclass
```

### Öznitelik Çıkarımı

```python
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(sample_rate=44100)
y = extractor.load_audio("drone_sample.wav")

# MFCC
mfcc = extractor.extract_mfcc(y)

# PSD (Welch)
freq, psd = extractor.extract_psd_welch(y)

# Mel Spektrogram
mel_spec = extractor.extract_mel_spectrogram(y)
```

### Model Eğitimi

```python
from src.models import SVMClassifier, create_model
from src.data_loader import DatasetLoader

# Veri yükle
loader = DatasetLoader()
X, y, classes = loader.load_drone_audio_dataset(mode='binary')

# SVM
svm = SVMClassifier(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)

# CNN (TensorFlow gerekli)
cnn = create_model('cnn', input_shape=(128, 87, 1), num_classes=2)
```

## 📐 Metodoloji

### İstatistiksel Sinyal İşleme

#### 1. Güç Spektral Yoğunluğu (PSD)
Wiener-Khinchin teoremi ile:
```
S_xx(f) = F{R_xx(τ)}
```
Welch yöntemi ile varyans azaltma.

#### 2. MFCC (Mel-Frequency Cepstral Coefficients)
```
m = 2595 * log10(1 + f/700)
```

#### 3. STFT (Short-Time Fourier Transform)
```
X(m,k) = Σ x(n) w(n-mH) e^(-j2πkn/N)
```

### Derin Öğrenme Mimarileri

| Model | Mimari | Parametre |
|-------|--------|-----------|
| CNN | Conv2D(32,64,128) → Dense(256,128) | ~500K |
| LSTM | LSTM(128,64) → Dense(64) | ~200K |
| CRNN | Conv2D(32,64) → LSTM(64) → Dense | ~300K |

## 📈 Sonuçlar

### İkili Sınıflandırma (Drone / Non-Drone)

| Model | Doğruluk | F1-Skor | AUC |
|-------|----------|---------|-----|
| SVM (MFCC) | 89.2% | 88.7% | 91.2% |
| SVM (PSD) | 85.6% | 84.9% | 87.8% |
| **CNN (Spektrogram)** | **96.2%** | **95.8%** | **98.1%** |
| LSTM (MFCC) | 92.3% | 91.9% | 95.1% |
| CRNN | 95.4% | 95.1% | 97.8% |

### SNR Analizi

| Model | 20 dB | 10 dB | 5 dB | 0 dB |
|-------|-------|-------|------|------|
| SVM | 89.2% | 82.1% | 71.3% | 58.4% |
| CNN | 96.2% | 91.5% | 84.2% | 72.8% |
| LSTM | 92.3% | 86.7% | 78.5% | 65.1% |

## 📁 Proje Yapısı

```
acoustic-drone-detection/
├── configs/
│   └── config.py           # Tüm konfigürasyonlar
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py   # MFCC, PSD, STFT
│   ├── data_loader.py          # Veri yükleme
│   ├── models.py               # SVM, CNN, LSTM
│   ├── visualization.py        # Grafikler
│   └── train.py                # Ana eğitim scripti
├── data/                   # Veri setleri (gitignore)
├── models/                 # Kaydedilen modeller
├── results/
│   └── figures/            # Oluşturulan grafikler
├── docs/
│   └── rapor.tex           # IEEE formatında LaTeX rapor
├── notebooks/              # Jupyter notebooks
├── tests/                  # Birim testleri
├── requirements.txt
└── README.md
```

## 📚 Referanslar

1. Al-Emadi, S. A., et al. "Audio Based Drone Detection and Identification using Deep Learning." IWCMC 2019.
2. Jeon, S., et al. "Empirical Study of Drone Sound Detection in Real-Life Environment with Deep Neural Networks." arXiv:1701.05779, 2017.
3. Kılıç, R., et al. "Drone classification using RF signal based spectral features." JESTECH, 2022.
4. Strauss, M., et al. "DREGON: Dataset and Methods for UAV-Embedded Sound Source Localization." IEEE/RSJ IROS, 2018.
5. Kümmritz, S. "The Sound of Surveillance: Enhancing ML-Driven Drone Detection." Drones, 2024.
