"""
Akustik Drone Tespiti - Model Tanımları
========================================
SVM, CNN, LSTM ve CRNN model mimarileri.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (SVM_CONFIG, CNN_CONFIG, LSTM_CONFIG, 
                            CRNN_CONFIG, TRAINING_CONFIG)


# =============================================================================
# GELENEKSEL MAKİNE ÖĞRENMESİ MODELLERİ
# =============================================================================

class SVMClassifier:
    """
    Destek Vektör Makineleri (SVM) sınıflandırıcı.
    
    Teorik Altyapı:
    ---------------
    SVM, sınıflar arasında maksimum marjinli hiper-düzlem bulur.
    RBF çekirdeği: K(x,x') = exp(-γ||x-x'||²)
    """
    
    def __init__(self,
                 kernel: str = SVM_CONFIG.kernel,
                 C: float = SVM_CONFIG.C,
                 gamma: str = SVM_CONFIG.gamma,
                 random_state: int = SVM_CONFIG.random_state):
        """
        Args:
            kernel: Çekirdek türü ('rbf', 'linear', 'poly')
            C: Düzenlileştirme parametresi
            gamma: RBF çekirdek parametresi
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=random_state
        )
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Model eğitimi"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahmin"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Olasılık tahmini"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Model değerlendirme"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
        }
        
        # Binary classification için AUC
        if len(np.unique(y)) == 2:
            metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            
        return metrics


class RandomForestModel:
    """Random Forest sınıflandırıcı"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def feature_importances(self) -> np.ndarray:
        """Öznitelik önem skorları"""
        return self.model.feature_importances_


# =============================================================================
# DERİN ÖĞRENME MODELLERİ
# =============================================================================

def build_cnn_model(input_shape: Tuple[int, ...],
                    num_classes: int,
                    config: Any = CNN_CONFIG) -> Optional[Any]:
    """
    CNN (Evrişimsel Sinir Ağı) modeli oluşturur.
    
    Mimari:
    -------
    [Input] -> [Conv2D -> BN -> ReLU -> MaxPool] x 3 -> [Flatten] -> 
    [Dense -> Dropout] x 2 -> [Softmax Output]
    
    Teorik Altyapı:
    ---------------
    Evrişim: y[i,j] = Σ_m Σ_n w[m,n] * x[i+m,j+n] + b
    Spektrogram görüntülerinden uzamsal öznitelikleri öğrenir.
    
    Args:
        input_shape: Giriş boyutu (height, width, channels)
        num_classes: Sınıf sayısı
        config: CNN konfigürasyonu
        
    Returns:
        Keras model veya None (TensorFlow yoksa)
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                            Dense, Dropout, BatchNormalization,
                                            Input)
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            Input(shape=input_shape),
            
            # 1. Evrişim bloğu
            Conv2D(config.conv_filters[0], config.kernel_size, 
                   activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(config.pool_size),
            
            # 2. Evrişim bloğu
            Conv2D(config.conv_filters[1], config.kernel_size,
                   activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(config.pool_size),
            
            # 3. Evrişim bloğu
            Conv2D(config.conv_filters[2], config.kernel_size,
                   activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(config.pool_size),
            
            # Tam bağlantılı katmanlar
            Flatten(),
            Dense(config.dense_units[0], activation='relu'),
            Dropout(config.dropout_rate),
            Dense(config.dense_units[1], activation='relu'),
            Dropout(config.dropout_rate / 2),
            
            # Çıkış
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=TRAINING_CONFIG.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except ImportError:
        print("TensorFlow yüklü değil. CNN modeli oluşturulamadı.")
        return None


def build_lstm_model(input_shape: Tuple[int, int],
                     num_classes: int,
                     config: Any = LSTM_CONFIG) -> Optional[Any]:
    """
    LSTM (Long Short-Term Memory) modeli oluşturur.
    
    Mimari:
    -------
    [Input] -> [LSTM] -> [Dropout] -> [LSTM] -> [Dropout] -> 
    [Dense] -> [Softmax Output]
    
    Teorik Altyapı:
    ---------------
    LSTM kapı mekanizmaları:
    - Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    - Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    - Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    - Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t] + b_C)
    
    Args:
        input_shape: (timesteps, features)
        num_classes: Sınıf sayısı
        
    Returns:
        Keras model veya None
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            Input(shape=input_shape),
            
            # İlk LSTM katmanı
            LSTM(config.lstm_units[0], return_sequences=True),
            Dropout(config.dropout_rate),
            
            # İkinci LSTM katmanı
            LSTM(config.lstm_units[1], return_sequences=False),
            Dropout(config.dropout_rate),
            
            # Tam bağlantılı katman
            Dense(config.dense_units, activation='relu'),
            Dropout(config.dropout_rate / 2),
            
            # Çıkış
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=TRAINING_CONFIG.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except ImportError:
        print("TensorFlow yüklü değil. LSTM modeli oluşturulamadı.")
        return None


def build_crnn_model(input_shape: Tuple[int, ...],
                     num_classes: int,
                     config: Any = CRNN_CONFIG) -> Optional[Any]:
    """
    CRNN (Convolutional Recurrent Neural Network) modeli.
    
    Mimari:
    -------
    [Input] -> [Conv2D blocks] -> [Reshape] -> [LSTM] -> [Dense] -> [Output]
    
    CNN'in uzamsal öznitelik çıkarımını LSTM'in zamansal modelleme
    kapasitesiyle birleştirir.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Conv2D, MaxPooling2D, LSTM,
                                            Dense, Dropout, BatchNormalization,
                                            Reshape, TimeDistributed, Input)
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            Input(shape=input_shape),
            
            # Evrişim katmanları
            Conv2D(config.conv_filters[0], (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(config.conv_filters[1], (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Reshape for LSTM (batch, timesteps, features)
            Reshape((-1, input_shape[1] // 4 * config.conv_filters[1])),
            
            # LSTM
            LSTM(config.lstm_units, return_sequences=False),
            Dropout(config.dropout_rate),
            
            # Çıkış
            Dense(config.dense_units, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=TRAINING_CONFIG.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except ImportError:
        print("TensorFlow yüklü değil. CRNN modeli oluşturulamadı.")
        return None


# =============================================================================
# MODEL FABRİKA
# =============================================================================

def create_model(model_type: str,
                 input_shape: Tuple = None,
                 num_classes: int = 2,
                 **kwargs) -> Any:
    """
    Model fabrika fonksiyonu.
    
    Args:
        model_type: 'svm', 'rf', 'cnn', 'lstm', 'crnn'
        input_shape: Derin öğrenme modelleri için giriş boyutu
        num_classes: Sınıf sayısı
        
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'svm':
        return SVMClassifier(**kwargs)
    elif model_type == 'rf':
        return RandomForestModel(**kwargs)
    elif model_type == 'cnn':
        return build_cnn_model(input_shape, num_classes)
    elif model_type == 'lstm':
        return build_lstm_model(input_shape, num_classes)
    elif model_type == 'crnn':
        return build_crnn_model(input_shape, num_classes)
    else:
        raise ValueError(f"Bilinmeyen model türü: {model_type}. "
                        f"Geçerli: svm, rf, cnn, lstm, crnn")


# =============================================================================
# DEĞERLENDİRME FONKSİYONLARI
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = "Model") -> Dict[str, Any]:
    """
    Model performansını değerlendirir.
    
    Returns:
        Dict: metrics, confusion_matrix, classification_report
    """
    # Tahmin
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        # Keras model
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
    # Olasılık (varsa)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, 'predict'):
        try:
            y_proba = model.predict(X_test)
        except:
            pass
    
    # Metrikler
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    # Binary için AUC
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            if y_proba.ndim == 2:
                metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_test, y_proba)
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def compare_models(results: Dict[str, Dict]) -> None:
    """
    Model sonuçlarını karşılaştırır ve yazdırır.
    """
    print("\n" + "=" * 70)
    print("MODEL KARŞILAŞTIRMASI")
    print("=" * 70)
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC':>10}")
    print("-" * 70)
    
    for name, result in results.items():
        m = result['metrics']
        auc = m.get('auc', '-')
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
        print(f"{name:<15} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {auc_str:>10}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("Model Modülü - Test")
    print("=" * 50)
    
    # Sentetik veri
    np.random.seed(42)
    X_train = np.random.randn(200, 78)
    y_train = np.random.randint(0, 2, 200)
    X_test = np.random.randn(50, 78)
    y_test = np.random.randint(0, 2, 50)
    
    # SVM test
    print("\nSVM testi...")
    svm = SVMClassifier()
    svm.fit(X_train, y_train)
    svm_results = evaluate_model(svm, X_test, y_test, "SVM")
    print(f"SVM Accuracy: {svm_results['metrics']['accuracy']:.4f}")
    
    # Random Forest test
    print("\nRandom Forest testi...")
    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(rf, X_test, y_test, "RF")
    print(f"RF Accuracy: {rf_results['metrics']['accuracy']:.4f}")
    
    # Karşılaştırma
    compare_models({'SVM': svm_results, 'Random Forest': rf_results})
    
    print("\nTest başarılı!")
