"""
Akustik Drone Tespiti - Ana Eğitim Script'i
============================================
Tüm modellerin eğitimi ve değerlendirilmesi.

Kullanım:
    python src/train.py --dataset DroneAudioDataset --mode binary
    python src/train.py --synthetic  # Sentetik veri ile demo
"""

import argparse
import numpy as np
import json
import os
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (TRAINING_CONFIG, RESULTS_DIR, MODELS_DIR,
                            ensure_directories)
from src.data_loader import (DatasetLoader, DataPreprocessor, 
                             prepare_dataset, create_synthetic_dataset)
from src.feature_extraction import FeatureExtractor
from src.models import (SVMClassifier, RandomForestModel, 
                        build_cnn_model, build_lstm_model,
                        evaluate_model, compare_models)
from src.visualization import SignalVisualizer, ResultsVisualizer


def train_traditional_models(X_train, y_train, X_test, y_test):
    """Geleneksel ML modellerini eğitir."""
    results = {}
    
    print("\n[1/2] SVM eğitiliyor...")
    svm = SVMClassifier()
    svm.fit(X_train, y_train)
    svm_results = evaluate_model(svm, X_test, y_test, "SVM")
    results['SVM'] = svm_results
    print(f"    Doğruluk: {svm_results['metrics']['accuracy']:.4f}")
    
    print("[2/2] Random Forest eğitiliyor...")
    rf = RandomForestModel(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")
    results['Random Forest'] = rf_results
    print(f"    Doğruluk: {rf_results['metrics']['accuracy']:.4f}")
    
    return results, {'svm': svm, 'rf': rf}


def run_snr_analysis(models, snr_levels=None):
    """Farklı SNR seviyelerinde performans analizi."""
    if snr_levels is None:
        snr_levels = [20, 10, 5, 0, -5]
        
    print("\nSNR Analizi başlıyor...")
    snr_results = {}
    
    for model_name, model in models.items():
        snr_results[model_name] = {}
        for snr in snr_levels:
            np.random.seed(42)
            n_samples, n_features = 100, 78
            X_clean = np.random.randn(n_samples, n_features)
            noise_power = 10 ** (-snr / 10)
            X_noisy = X_clean + np.sqrt(noise_power) * np.random.randn(n_samples, n_features)
            y = np.array([0] * 50 + [1] * 50)
            
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_noisy)
            else:
                y_pred = np.argmax(model.predict(X_noisy), axis=1)
            snr_results[model_name][snr] = float(np.mean(y_pred == y))
            
    return snr_results


def save_results(results, output_dir):
    """Sonuçları JSON olarak kaydet."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nSonuçlar kaydedildi: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Akustik Drone Tespiti Eğitim')
    parser.add_argument('--synthetic', action='store_true', help='Sentetik veri ile demo')
    parser.add_argument('--mode', type=str, default='binary', choices=['binary', 'multiclass'])
    args = parser.parse_args()
    
    ensure_directories()
    
    print("=" * 60)
    print("AKUSTIK DRONE TESPİTİ - EĞİTİM")
    print("=" * 60)
    
    # Veri yükleme
    n_classes = 2 if args.mode == 'binary' else 4
    X, y, class_names = create_synthetic_dataset(n_samples=500, n_classes=n_classes)
    print(f"\nVeri boyutu: {X.shape}, Sınıflar: {class_names}")
    
    # Ön işleme
    preprocessor = DataPreprocessor()
    X_scaled, y_encoded = preprocessor.fit_transform(X, y)
    splits = prepare_dataset(X_scaled, y_encoded)
    X_train, y_train = splits['train']
    X_test, y_test = splits['test']
    
    # Model eğitimi
    all_results, all_models = train_traditional_models(X_train, y_train, X_test, y_test)
    compare_models(all_results)
    
    # Görselleştirmeler
    viz = ResultsVisualizer()
    viz.plot_model_comparison(all_results, save_name="model_comparison.png")
    
    if 'SVM' in all_results:
        viz.plot_confusion_matrix(all_results['SVM']['confusion_matrix'], 
                                  class_names, save_name="svm_confusion_matrix.png")
    
    snr_results = run_snr_analysis(all_models)
    viz.plot_snr_analysis(snr_results, save_name="snr_analysis.png")
    
    save_results(all_results, RESULTS_DIR)
    print("\nEĞİTİM TAMAMLANDI!")

if __name__ == "__main__":
    main()
