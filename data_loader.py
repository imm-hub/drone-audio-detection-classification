"""
Akustik Drone Tespiti - Veri Yükleme Modülü
============================================
Çoklu veri seti desteği ile veri yükleme ve ön işleme.

Desteklenen Veri Setleri:
    - DroneAudioDataset (Al-Emadi et al.)
    - DREGON (INRIA)
    - SPCup19 Egonoise
    - ESC-50 (arka plan gürültüsü)
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (DATASETS, DATA_DIR, TRAINING_CONFIG, 
                            get_dataset_info, ensure_directories)
from src.feature_extraction import FeatureExtractor, extract_features_from_file


class DatasetLoader:
    """
    Çoklu veri seti yükleme sınıfı.
    """
    
    def __init__(self, data_dir: str = DATA_DIR):
        """
        Args:
            data_dir: Veri setlerinin bulunduğu ana dizin
        """
        self.data_dir = data_dir
        self.extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_drone_audio_dataset(self, 
                                  mode: str = "binary",
                                  feature_type: str = "mfcc_stats"
                                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        DroneAudioDataset (Al-Emadi et al.) yükler.
        
        Args:
            mode: "binary" (Drone/Non-Drone) veya "multiclass"
            feature_type: Öznitelik türü
            
        Returns:
            (features, labels, class_names)
        """
        dataset_path = os.path.join(self.data_dir, "DroneAudioDataset")
        
        if mode == "binary":
            folder = os.path.join(dataset_path, "Binary_Drone_Audio")
            classes = ["Drone", "Unknown"]
        else:
            folder = os.path.join(dataset_path, "Multiclass_Drone_Audio")
            classes = ["Bebop", "Mambo", "Phantom", "Unknown"]
            
        return self._load_from_folders(folder, classes, feature_type)
    
    def load_dregon_dataset(self, 
                            feature_type: str = "mfcc_stats"
                            ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        DREGON veri setini yükler.
        """
        dataset_path = os.path.join(self.data_dir, "DREGON")
        classes = ["drone_noise", "speech", "white_noise"]
        
        return self._load_from_folders(dataset_path, classes, feature_type)
    
    def load_spcup19_dataset(self,
                              feature_type: str = "mfcc_stats"
                              ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        SPCup19 Egonoise veri setini yükler.
        """
        dataset_path = os.path.join(self.data_dir, "SPCup19_Egonoise")
        # SPCup19 klasör yapısına göre düzenle
        classes = os.listdir(dataset_path) if os.path.exists(dataset_path) else []
        classes = [c for c in classes if os.path.isdir(os.path.join(dataset_path, c))]
        
        return self._load_from_folders(dataset_path, classes, feature_type)
    
    def load_esc50_background(self,
                               categories: List[str] = None,
                               feature_type: str = "mfcc_stats"
                               ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        ESC-50 veri setinden arka plan gürültüsü örnekleri yükler.
        
        Args:
            categories: Yüklenecek kategoriler (None = hepsi)
        """
        dataset_path = os.path.join(self.data_dir, "ESC-50", "audio")
        
        if categories is None:
            categories = ["natural_soundscapes", "exterior_urban", 
                         "interior_sounds"]
            
        return self._load_from_folders(dataset_path, categories, feature_type)
    
    def _load_from_folders(self, 
                           base_path: str,
                           classes: List[str],
                           feature_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Klasör yapısından veri yükler.
        """
        features = []
        labels = []
        
        if not os.path.exists(base_path):
            print(f"Uyarı: {base_path} bulunamadı!")
            return np.array([]), np.array([]), classes
            
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                print(f"Uyarı: {class_path} bulunamadı, atlanıyor...")
                continue
                
            for audio_file in os.listdir(class_path):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(class_path, audio_file)
                    feat = extract_features_from_file(file_path, feature_type)
                    
                    if feat is not None:
                        features.append(feat)
                        labels.append(class_name)
                        
        return np.array(features), np.array(labels), classes
    
    def load_combined_dataset(self,
                               datasets: List[str] = None,
                               feature_type: str = "mfcc_stats",
                               binary_mode: bool = True
                               ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Birden fazla veri setini birleştirir.
        
        Args:
            datasets: Yüklenecek veri seti isimleri
            feature_type: Öznitelik türü
            binary_mode: True ise Drone/Non-Drone olarak etiketle
            
        Returns:
            Birleştirilmiş (features, labels, class_names)
        """
        if datasets is None:
            datasets = ["DroneAudioDataset"]
            
        all_features = []
        all_labels = []
        
        for ds_name in datasets:
            if ds_name == "DroneAudioDataset":
                X, y, _ = self.load_drone_audio_dataset("binary" if binary_mode else "multiclass", 
                                                        feature_type)
            elif ds_name == "DREGON":
                X, y, _ = self.load_dregon_dataset(feature_type)
            elif ds_name == "SPCup19_Egonoise":
                X, y, _ = self.load_spcup19_dataset(feature_type)
            elif ds_name == "ESC-50":
                X, y, _ = self.load_esc50_background(feature_type=feature_type)
            else:
                print(f"Uyarı: Bilinmeyen veri seti: {ds_name}")
                continue
                
            if len(X) > 0:
                if binary_mode:
                    # Drone vs Non-Drone olarak etiketle
                    y = np.array(["Drone" if "drone" in label.lower() or 
                                  label in ["Bebop", "Mambo", "Phantom"]
                                  else "Non-Drone" for label in y])
                    
                all_features.append(X)
                all_labels.append(y)
                print(f"{ds_name}: {len(X)} örnek yüklendi")
                
        if len(all_features) == 0:
            return np.array([]), np.array([]), []
            
        X_combined = np.vstack(all_features)
        y_combined = np.concatenate(all_labels)
        classes = list(np.unique(y_combined))
        
        return X_combined, y_combined, classes


class DataPreprocessor:
    """
    Veri ön işleme sınıfı.
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._fitted = False
        
    def fit_transform(self, 
                      X: np.ndarray, 
                      y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eğitim verisi için fit ve transform.
        """
        # Label encoding
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Standardizasyon
        X_scaled = self.scaler.fit_transform(X)
        
        self._fitted = True
        return X_scaled, y_encoded
    
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Test verisi için transform (fit edilmiş parametrelerle).
        """
        if not self._fitted:
            raise RuntimeError("Önce fit_transform çağrılmalı!")
            
        X_scaled = self.scaler.transform(X)
        
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
            
        return X_scaled, None
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Kodlanmış etiketleri orijinal haline çevirir.
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    @property
    def classes(self) -> np.ndarray:
        """Sınıf isimleri"""
        return self.label_encoder.classes_


def prepare_dataset(X: np.ndarray,
                    y: np.ndarray,
                    test_size: float = TRAINING_CONFIG.test_split,
                    val_size: float = TRAINING_CONFIG.validation_split,
                    random_state: int = TRAINING_CONFIG.random_state,
                    stratify: bool = True
                    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Veri setini eğitim/doğrulama/test olarak böler.
    
    Returns:
        Dict: {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    """
    # Önce test setini ayır
    stratify_y = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    
    # Kalan veriden validation ayır
    val_ratio = val_size / (1 - test_size)
    stratify_y_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, 
        stratify=stratify_y_temp
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def create_synthetic_dataset(n_samples: int = 500,
                              n_classes: int = 2,
                              n_features: int = 78,
                              random_state: int = 42
                              ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Test amaçlı sentetik veri seti oluşturur.
    
    Args:
        n_samples: Toplam örnek sayısı
        n_classes: Sınıf sayısı
        n_features: Öznitelik boyutu
        random_state: Rastgelelik kontrolü
        
    Returns:
        (X, y, class_names)
    """
    np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    
    X = []
    y = []
    
    if n_classes == 2:
        class_names = ["Drone", "Non-Drone"]
    else:
        class_names = ["Bebop", "Mambo", "Phantom", "Unknown"][:n_classes]
    
    for i, class_name in enumerate(class_names):
        # Her sınıf için farklı dağılım
        mean = i * 0.5
        std = 0.3 + i * 0.1
        class_samples = np.random.randn(samples_per_class, n_features) * std + mean
        
        X.append(class_samples)
        y.extend([class_name] * samples_per_class)
        
    return np.vstack(X), np.array(y), class_names


if __name__ == "__main__":
    print("Veri Yükleme Modülü - Test")
    print("=" * 50)
    
    # Sentetik veri ile test
    X, y, classes = create_synthetic_dataset(n_samples=200, n_classes=2)
    print(f"Sentetik veri: X={X.shape}, y={y.shape}")
    print(f"Sınıflar: {classes}")
    
    # Preprocessor test
    preprocessor = DataPreprocessor()
    X_scaled, y_encoded = preprocessor.fit_transform(X, y)
    print(f"Ölçeklenmiş veri: X={X_scaled.shape}")
    print(f"Kodlanmış etiketler: {np.unique(y_encoded)}")
    
    # Veri bölme test
    splits = prepare_dataset(X_scaled, y_encoded)
    print(f"\nVeri Bölümü:")
    for name, (X_split, y_split) in splits.items():
        print(f"  {name}: X={X_split.shape}, y={y_split.shape}")
        
    print("\nTest başarılı!")
