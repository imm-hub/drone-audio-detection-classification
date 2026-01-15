"""
Akustik Drone Tespiti - Öznitelik Çıkarımı Modülü
==================================================
İstatistiksel sinyal işleme tabanlı öznitelik çıkarım fonksiyonları.

İçerik:
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - PSD (Power Spectral Density) - Welch yöntemi
    - STFT (Short-Time Fourier Transform) / Spektrogram
    - Mel Spektrogram
    - Chroma öznitelikleri
    - Spectral Contrast
"""

import numpy as np
import librosa
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Konfigürasyon import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (AUDIO_CONFIG, MFCC_CONFIG, 
                            SPECTROGRAM_CONFIG, PSD_CONFIG)


class FeatureExtractor:
    """
    Akustik öznitelik çıkarımı için ana sınıf.
    
    İstatistiksel Sinyal İşleme Temelleri:
    --------------------------------------
    1. MFCC: İnsan işitme sisteminin logaritmik frekans algısını modeller
       - Mel ölçeği: m = 2595 * log10(1 + f/700)
       
    2. PSD: Wiener-Khinchin teoremi ile frekans enerji dağılımı
       - S_xx(f) = F{R_xx(τ)}
       
    3. STFT: Durağan olmayan sinyallerin zaman-frekans analizi
       - X(m,k) = Σ x(n) w(n-mH) e^(-j2πkn/N)
    """
    
    def __init__(self, 
                 sample_rate: int = AUDIO_CONFIG.sample_rate,
                 n_fft: int = AUDIO_CONFIG.n_fft,
                 hop_length: int = AUDIO_CONFIG.hop_length,
                 win_length: int = AUDIO_CONFIG.win_length):
        """
        Args:
            sample_rate: Örnekleme frekansı (Hz)
            n_fft: FFT pencere boyutu
            hop_length: Pencereler arası kayma
            win_length: Pencere uzunluğu
        """
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
    def load_audio(self, file_path: str, duration: float = None) -> np.ndarray:
        """
        Ses dosyasını yükler ve normalize eder.
        
        Args:
            file_path: Ses dosyası yolu
            duration: İsteğe bağlı süre limiti (saniye)
            
        Returns:
            Normalize edilmiş ses sinyali
        """
        y, sr = librosa.load(file_path, sr=self.sr, duration=duration)
        
        # Normalize et
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
            
        return y
    
    # =========================================================================
    # MFCC ÖZNİTELİKLERİ
    # =========================================================================
    
    def extract_mfcc(self, 
                     y: np.ndarray,
                     n_mfcc: int = MFCC_CONFIG.n_mfcc,
                     n_mels: int = MFCC_CONFIG.n_mels,
                     include_delta: bool = MFCC_CONFIG.include_delta,
                     include_delta2: bool = MFCC_CONFIG.include_delta2) -> np.ndarray:
        """
        MFCC (Mel-Frequency Cepstral Coefficients) çıkarımı.
        
        Teorik Altyapı:
        ---------------
        1. Ön-vurgu filtresi: y(n) = x(n) - 0.97*x(n-1)
        2. Çerçeveleme ve Hamming pencereleme
        3. FFT ile güç spektrumu
        4. Mel filtre bankası: m = 2595 * log10(1 + f/700)
        5. Logaritma
        6. DCT (Discrete Cosine Transform)
        
        Args:
            y: Ses sinyali
            n_mfcc: MFCC katsayı sayısı
            n_mels: Mel filtre sayısı
            include_delta: Birinci türev (delta) dahil mi
            include_delta2: İkinci türev (delta-delta) dahil mi
            
        Returns:
            MFCC öznitelik matrisi (n_features x n_frames)
        """
        # Temel MFCC
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        features = [mfccs]
        
        # Delta (birinci türev) - zamansal değişim
        if include_delta:
            delta = librosa.feature.delta(mfccs)
            features.append(delta)
            
        # Delta-delta (ikinci türev) - ivme
        if include_delta2:
            delta2 = librosa.feature.delta(mfccs, order=2)
            features.append(delta2)
            
        return np.concatenate(features, axis=0)
    
    def extract_mfcc_statistics(self, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        MFCC'nin istatistiksel özetini çıkarır (SVM için uygun).
        
        Returns:
            1D öznitelik vektörü [ortalama, std, min, max, medyan]
        """
        mfcc = self.extract_mfcc(y, **kwargs)
        
        stats = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.min(mfcc, axis=1),
            np.max(mfcc, axis=1),
            np.median(mfcc, axis=1)
        ])
        
        return stats
    
    # =========================================================================
    # GÜÇ SPEKTRAL YOĞUNLUĞU (PSD)
    # =========================================================================
    
    def extract_psd_welch(self, 
                          y: np.ndarray,
                          nperseg: int = PSD_CONFIG.nperseg,
                          noverlap: int = PSD_CONFIG.noverlap) -> Tuple[np.ndarray, np.ndarray]:
        """
        Welch yöntemi ile PSD tahmini.
        
        Teorik Altyapı:
        ---------------
        Welch yöntemi, periodogram varyansını azaltmak için:
        1. Sinyal K örtüşen segmente bölünür
        2. Her segment pencerelenir (Hamming)
        3. Periodogramların ortalaması alınır
        
        S_xx^W(f) = (1/K) Σ |X_k(f)|² / (L*U)
        
        Args:
            y: Ses sinyali
            nperseg: Segment uzunluğu
            noverlap: Örtüşme miktarı
            
        Returns:
            (frequencies, psd): Frekans ve PSD değerleri
        """
        frequencies, psd = signal.welch(
            y, 
            fs=self.sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hamming',
            scaling='density'
        )
        
        return frequencies, psd
    
    def extract_psd_band_features(self, 
                                   y: np.ndarray,
                                   bands: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        Frekans bantlarına göre PSD öznitelikleri.
        
        Drone sesleri tipik olarak:
        - 100-400 Hz: Pervane temel frekansı
        - 4000-8000 Hz: Harmonikler
        
        Returns:
            Her bant için [ortalama, std, max, toplam_enerji]
        """
        if bands is None:
            bands = PSD_CONFIG.frequency_bands
            
        frequencies, psd = self.extract_psd_welch(y)
        psd_db = 10 * np.log10(psd + 1e-10)
        
        features = []
        for low, high in bands:
            mask = (frequencies >= low) & (frequencies <= high)
            if np.any(mask):
                band_psd = psd_db[mask]
                features.extend([
                    np.mean(band_psd),
                    np.std(band_psd),
                    np.max(band_psd),
                    np.sum(10 ** (band_psd / 10))  # Lineer enerji
                ])
            else:
                features.extend([0, 0, 0, 0])
                
        return np.array(features)
    
    # =========================================================================
    # SPEKTROGRAM
    # =========================================================================
    
    def extract_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        STFT tabanlı spektrogram.
        
        Teorik Altyapı:
        ---------------
        STFT: X(m,k) = Σ x(n) w(n-mH) e^(-j2πkn/N)
        Spektrogram: |X(m,k)|²
        
        Returns:
            Spektrogram matrisi (frequency x time)
        """
        D = librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        spectrogram = np.abs(D) ** 2
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        return spectrogram_db
    
    def extract_mel_spectrogram(self, 
                                 y: np.ndarray,
                                 n_mels: int = SPECTROGRAM_CONFIG.n_mels) -> np.ndarray:
        """
        Mel ölçekli spektrogram.
        
        Mel ölçeği insan işitme sistemini modeller.
        
        Returns:
            Mel spektrogram (n_mels x time)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    # =========================================================================
    # EK ÖZNİTELİKLER
    # =========================================================================
    
    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Ek spektral öznitelikler.
        
        Returns:
            Dict içinde: centroid, bandwidth, rolloff, contrast, flatness
        """
        features = {}
        
        # Spektral Centroid (ağırlık merkezi frekansı)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spektral Bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spektral Rolloff (%85 enerjinin altındaki frekans)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spektral Contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spektral Flatness (tonalite ölçüsü)
        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Zero Crossing Rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            y, frame_length=self.win_length, hop_length=self.hop_length
        )
        
        # RMS Energy
        features['rms_energy'] = librosa.feature.rms(
            y=y, frame_length=self.win_length, hop_length=self.hop_length
        )
        
        return features
    
    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """
        Chroma öznitelikleri (pitch sınıfı dağılımı).
        """
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return chroma
    
    # =========================================================================
    # BİRLEŞİK ÖZNİTELİK ÇIKARIMI
    # =========================================================================
    
    def extract_all_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Tüm öznitelikleri çıkarır.
        
        Returns:
            Dict içinde tüm öznitelik türleri
        """
        features = {
            'mfcc': self.extract_mfcc(y),
            'mfcc_stats': self.extract_mfcc_statistics(y),
            'mel_spectrogram': self.extract_mel_spectrogram(y),
            'spectrogram': self.extract_spectrogram(y),
            'psd_bands': self.extract_psd_band_features(y),
            'chroma': self.extract_chroma(y),
        }
        
        # Spektral öznitelikler
        spectral = self.extract_spectral_features(y)
        features.update(spectral)
        
        return features
    
    def extract_flat_features(self, y: np.ndarray) -> np.ndarray:
        """
        SVM için tek boyutlu öznitelik vektörü.
        
        Returns:
            1D numpy array
        """
        mfcc_stats = self.extract_mfcc_statistics(y)
        psd_bands = self.extract_psd_band_features(y)
        
        spectral = self.extract_spectral_features(y)
        spectral_stats = []
        for key, val in spectral.items():
            spectral_stats.extend([np.mean(val), np.std(val)])
            
        return np.concatenate([mfcc_stats, psd_bands, spectral_stats])


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def extract_features_from_file(file_path: str, 
                                feature_type: str = "mfcc_stats",
                                duration: float = 1.0) -> Optional[np.ndarray]:
    """
    Dosyadan öznitelik çıkarımı için kısayol fonksiyon.
    
    Args:
        file_path: Ses dosyası yolu
        feature_type: "mfcc", "mfcc_stats", "mel_spectrogram", "psd_bands", "flat"
        duration: Ses süresi (saniye)
        
    Returns:
        Öznitelik array'i veya None (hata durumunda)
    """
    try:
        extractor = FeatureExtractor()
        y = extractor.load_audio(file_path, duration=duration)
        
        if feature_type == "mfcc":
            return extractor.extract_mfcc(y)
        elif feature_type == "mfcc_stats":
            return extractor.extract_mfcc_statistics(y)
        elif feature_type == "mel_spectrogram":
            return extractor.extract_mel_spectrogram(y)
        elif feature_type == "psd_bands":
            return extractor.extract_psd_band_features(y)
        elif feature_type == "flat":
            return extractor.extract_flat_features(y)
        else:
            raise ValueError(f"Bilinmeyen öznitelik türü: {feature_type}")
            
    except Exception as e:
        print(f"Hata ({file_path}): {e}")
        return None


if __name__ == "__main__":
    # Test
    print("Öznitelik Çıkarımı Modülü - Test")
    print("=" * 50)
    
    # Sentetik sinyal oluştur (test için)
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Drone benzeri sinyal: temel frekans + harmonikler
    f0 = 200  # Temel frekans (pervane)
    y = (0.5 * np.sin(2 * np.pi * f0 * t) +
         0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
         0.2 * np.sin(2 * np.pi * 3 * f0 * t) +
         0.1 * np.random.randn(len(t)))
    
    extractor = FeatureExtractor(sample_rate=sr)
    
    # MFCC test
    mfcc = extractor.extract_mfcc(y)
    print(f"MFCC shape: {mfcc.shape}")
    
    # PSD test
    freq, psd = extractor.extract_psd_welch(y)
    print(f"PSD shape: {psd.shape}")
    
    # Mel spektrogram test
    mel_spec = extractor.extract_mel_spectrogram(y)
    print(f"Mel Spectrogram shape: {mel_spec.shape}")
    
    # Flat öznitelikler
    flat = extractor.extract_flat_features(y)
    print(f"Flat features shape: {flat.shape}")
    
    print("\nTest başarılı!")
