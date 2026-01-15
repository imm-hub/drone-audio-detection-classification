"""
Akustik Drone Tespiti - Konfigürasyon Dosyası
==============================================
Tüm model ve sinyal işleme parametrelerini içerir.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# =============================================================================
# PROJE YOLLARI
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# =============================================================================
# VERİ SETLERİ
# =============================================================================
@dataclass
class DatasetConfig:
    """Veri seti konfigürasyonları"""
    name: str
    url: str
    description: str
    sample_rate: int = 44100
    classes: List[str] = field(default_factory=list)

DATASETS = {
    "DroneAudioDataset": DatasetConfig(
        name="DroneAudioDataset",
        url="https://github.com/saraalemadi/DroneAudioDataset",
        description="Al-Emadi et al. - İç mekan drone ses kayıtları",
        sample_rate=44100,
        classes=["Bebop", "Mambo", "Phantom", "Unknown"]
    ),
    "DREGON": DatasetConfig(
        name="DREGON",
        url="http://dregon.inria.fr/datasets/dregon/",
        description="INRIA - 8 kanallı mikrofon dizisi ile UAV kayıtları",
        sample_rate=44100,
        classes=["drone_noise", "speech", "white_noise"]
    ),
    "SPCup19_Egonoise": DatasetConfig(
        name="SPCup19_Egonoise",
        url="http://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/",
        description="IEEE SPCup 2019 - Çeşitli drone ego-noise kayıtları",
        sample_rate=44100,
        classes=["Phantom4", "YH-19HW", "Phantom3", "Custom"]
    ),
    "DroneDetectionDataset": DatasetConfig(
        name="DroneDetectionDataset",
        url="https://github.com/DroneDetectionThesis/Drone-detection-dataset",
        description="IR, görsel ve ses verileri içeren çoklu sensör veri seti",
        sample_rate=44100,
        classes=["drone", "no_drone"]
    ),
    "AcousticUAV": DatasetConfig(
        name="AcousticUAV",
        url="https://github.com/pcasabianca/Acoustic-UAV-Identification",
        description="Casabianca - Deneysel olarak toplanan UAV ses verileri",
        sample_rate=44100,
        classes=["UAV1", "UAV2", "noise"]
    ),
    "ESC50": DatasetConfig(
        name="ESC-50",
        url="https://github.com/karolpiczak/ESC-50",
        description="Çevresel ses sınıflandırma - Arka plan gürültüsü için",
        sample_rate=44100,
        classes=["animals", "natural_soundscapes", "human_non_speech", 
                 "interior_sounds", "exterior_urban"]
    ),
}

# =============================================================================
# SİNYAL İŞLEME PARAMETRELERİ
# =============================================================================
@dataclass
class AudioConfig:
    """Ses işleme parametreleri"""
    sample_rate: int = 44100
    duration: float = 1.0  # saniye
    n_fft: int = 512
    hop_length: int = 256
    win_length: int = 512
    window: str = "hamming"
    
@dataclass
class MFCCConfig:
    """MFCC öznitelik çıkarım parametreleri"""
    n_mfcc: int = 13
    n_mels: int = 40
    fmin: float = 0.0
    fmax: float = 8000.0
    include_delta: bool = True
    include_delta2: bool = True
    
@dataclass
class SpectrogramConfig:
    """Spektrogram parametreleri"""
    n_mels: int = 128
    power: float = 2.0
    normalize: bool = True
    
@dataclass
class PSDConfig:
    """Güç Spektral Yoğunluğu parametreleri"""
    nperseg: int = 1024
    noverlap: int = 512
    window: str = "hamming"
    scaling: str = "density"
    # Frekans bantları (Hz)
    frequency_bands: List[Tuple[int, int]] = field(default_factory=lambda: [
        (50, 200),    # Düşük frekans
        (200, 500),   # Temel frekans (pervane)
        (500, 1500),  # Orta frekans
        (1500, 4000), # Yüksek-orta
        (4000, 8000)  # Yüksek frekans harmonikleri
    ])

# Varsayılan konfigürasyonlar
AUDIO_CONFIG = AudioConfig()
MFCC_CONFIG = MFCCConfig()
SPECTROGRAM_CONFIG = SpectrogramConfig()
PSD_CONFIG = PSDConfig()

# =============================================================================
# MODEL PARAMETRELERİ
# =============================================================================
@dataclass
class SVMConfig:
    """SVM model parametreleri"""
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    random_state: int = 42

@dataclass
class CNNConfig:
    """CNN model parametreleri"""
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    dense_units: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.5
    
@dataclass
class LSTMConfig:
    """LSTM model parametreleri"""
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    dense_units: int = 64
    dropout_rate: float = 0.3
    
@dataclass 
class CRNNConfig:
    """CRNN (Convolutional Recurrent) model parametreleri"""
    conv_filters: List[int] = field(default_factory=lambda: [32, 64])
    lstm_units: int = 64
    dense_units: int = 64
    dropout_rate: float = 0.3

# Varsayılan model konfigürasyonları
SVM_CONFIG = SVMConfig()
CNN_CONFIG = CNNConfig()
LSTM_CONFIG = LSTMConfig()
CRNN_CONFIG = CRNNConfig()

# =============================================================================
# EĞİTİM PARAMETRELERİ
# =============================================================================
@dataclass
class TrainingConfig:
    """Eğitim parametreleri"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42
    optimizer: str = "adam"
    
TRAINING_CONFIG = TrainingConfig()

# =============================================================================
# GÖRSELLEŞTİRME PARAMETRELERİ
# =============================================================================
@dataclass
class PlotConfig:
    """Görselleştirme parametreleri"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "Set2"
    font_size: int = 12
    title_size: int = 14
    save_format: str = "png"
    
PLOT_CONFIG = PlotConfig()

# =============================================================================
# RENK PALETLERİ
# =============================================================================
COLORS = {
    "drone": "#E74C3C",      # Kırmızı
    "no_drone": "#27AE60",   # Yeşil
    "bebop": "#3498DB",      # Mavi
    "mambo": "#9B59B6",      # Mor
    "phantom": "#F39C12",    # Turuncu
    "unknown": "#95A5A6",    # Gri
    "svm": "#2ECC71",
    "cnn": "#E74C3C",
    "lstm": "#3498DB",
    "crnn": "#9B59B6"
}

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================
def ensure_directories():
    """Gerekli dizinleri oluşturur"""
    for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        
def get_dataset_info(dataset_name: str) -> DatasetConfig:
    """Veri seti bilgilerini döndürür"""
    if dataset_name not in DATASETS:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}. "
                        f"Mevcut: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]

if __name__ == "__main__":
    # Test konfigürasyonları
    ensure_directories()
    print("Proje Konfigürasyonu")
    print("=" * 50)
    print(f"Proje Kökü: {PROJECT_ROOT}")
    print(f"Veri Dizini: {DATA_DIR}")
    print(f"Sonuçlar: {RESULTS_DIR}")
    print(f"\nMevcut Veri Setleri:")
    for name, config in DATASETS.items():
        print(f"  - {name}: {config.description}")
