"""
Akustik Drone Tespiti - Görselleştirme Modülü
=============================================
Spektrogram, PSD, MFCC ve model performans grafikleri.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import PLOT_CONFIG, COLORS, FIGURES_DIR, ensure_directories

# Matplotlib ayarları
plt.rcParams['figure.figsize'] = PLOT_CONFIG.figure_size
plt.rcParams['figure.dpi'] = PLOT_CONFIG.dpi
plt.rcParams['font.size'] = PLOT_CONFIG.font_size
plt.rcParams['axes.titlesize'] = PLOT_CONFIG.title_size

try:
    plt.style.use(PLOT_CONFIG.style)
except:
    plt.style.use('seaborn-v0_8-whitegrid')


class SignalVisualizer:
    """
    Sinyal görselleştirme sınıfı.
    """
    
    def __init__(self, save_dir: str = FIGURES_DIR):
        self.save_dir = save_dir
        ensure_directories()
        
    def _save_figure(self, fig, filename: str):
        """Figürü kaydet"""
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, dpi=PLOT_CONFIG.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Kaydedildi: {filepath}")
        return filepath
    
    def plot_waveform(self, y: np.ndarray, sr: int, 
                      title: str = "Dalga Formu",
                      save_name: str = None) -> plt.Figure:
        """
        Ses dalga formu grafiği.
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        time = np.linspace(0, len(y) / sr, len(y))
        ax.plot(time, y, linewidth=0.5, color=COLORS['drone'])
        
        ax.set_xlabel('Zaman (s)', fontsize=12)
        ax.set_ylabel('Genlik', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_spectrogram(self, spectrogram: np.ndarray, sr: int,
                         hop_length: int = 256,
                         title: str = "Spektrogram",
                         save_name: str = None) -> plt.Figure:
        """
        Spektrogram görselleştirme.
        """
        import librosa.display
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        img = librosa.display.specshow(
            spectrogram, 
            sr=sr, 
            hop_length=hop_length,
            x_axis='time', 
            y_axis='hz',
            ax=ax,
            cmap='magma'
        )
        
        ax.set_xlabel('Zaman (s)', fontsize=12)
        ax.set_ylabel('Frekans (Hz)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Güç (dB)', fontsize=11)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_mel_spectrogram(self, mel_spec: np.ndarray, sr: int,
                              hop_length: int = 256,
                              title: str = "Mel Spektrogram",
                              save_name: str = None) -> plt.Figure:
        """
        Mel spektrogram görselleştirme.
        """
        import librosa.display
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        img = librosa.display.specshow(
            mel_spec,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        
        ax.set_xlabel('Zaman (s)', fontsize=12)
        ax.set_ylabel('Mel Frekansı', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Güç (dB)', fontsize=11)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_mfcc(self, mfcc: np.ndarray, sr: int,
                  hop_length: int = 256,
                  title: str = "MFCC",
                  save_name: str = None) -> plt.Figure:
        """
        MFCC görselleştirme.
        """
        import librosa.display
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        img = librosa.display.specshow(
            mfcc,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            ax=ax,
            cmap='coolwarm'
        )
        
        ax.set_xlabel('Zaman (s)', fontsize=12)
        ax.set_ylabel('MFCC Katsayıları', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Değer', fontsize=11)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_psd(self, frequencies: np.ndarray, psd: np.ndarray,
                 title: str = "Güç Spektral Yoğunluğu (PSD)",
                 log_scale: bool = True,
                 highlight_bands: bool = True,
                 save_name: str = None) -> plt.Figure:
        """
        PSD görselleştirme.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if log_scale:
            ax.semilogy(frequencies, psd, linewidth=1.5, color=COLORS['cnn'])
        else:
            psd_db = 10 * np.log10(psd + 1e-10)
            ax.plot(frequencies, psd_db, linewidth=1.5, color=COLORS['cnn'])
            
        # Drone frekans bantlarını vurgula
        if highlight_bands:
            bands = [
                (100, 400, 'Pervane Temel', 'yellow'),
                (4000, 8000, 'Harmonikler', 'lightgreen')
            ]
            for low, high, label, color in bands:
                ax.axvspan(low, high, alpha=0.2, color=color, label=label)
                
        ax.set_xlabel('Frekans (Hz)', fontsize=12)
        ax.set_ylabel('PSD (V²/Hz)' if log_scale else 'PSD (dB)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0, 10000])
        ax.grid(True, alpha=0.3)
        
        if highlight_bands:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_signal_analysis(self, y: np.ndarray, sr: int,
                              title: str = "Sinyal Analizi",
                              save_name: str = None) -> plt.Figure:
        """
        Kapsamlı sinyal analizi - 4 panel.
        """
        import librosa
        from scipy import signal as sig
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Dalga formu
        time = np.linspace(0, len(y) / sr, len(y))
        axes[0, 0].plot(time, y, linewidth=0.5, color=COLORS['drone'])
        axes[0, 0].set_xlabel('Zaman (s)')
        axes[0, 0].set_ylabel('Genlik')
        axes[0, 0].set_title('Dalga Formu', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spektrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img1 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',
                                        ax=axes[0, 1], cmap='magma')
        axes[0, 1].set_title('STFT Spektrogram', fontweight='bold')
        fig.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')
        
        # 3. Mel Spektrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img2 = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                                        ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title('Mel Spektrogram', fontweight='bold')
        fig.colorbar(img2, ax=axes[1, 0], format='%+2.0f dB')
        
        # 4. PSD
        frequencies, psd = sig.welch(y, fs=sr, nperseg=1024)
        axes[1, 1].semilogy(frequencies, psd, linewidth=1.5, color=COLORS['cnn'])
        axes[1, 1].axvspan(100, 400, alpha=0.2, color='yellow', label='Pervane')
        axes[1, 1].axvspan(4000, 8000, alpha=0.2, color='lightgreen', label='Harmonik')
        axes[1, 1].set_xlabel('Frekans (Hz)')
        axes[1, 1].set_ylabel('PSD')
        axes[1, 1].set_title('Güç Spektral Yoğunluğu', fontweight='bold')
        axes[1, 1].set_xlim([0, 10000])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig


class ResultsVisualizer:
    """
    Model sonuçları görselleştirme sınıfı.
    """
    
    def __init__(self, save_dir: str = FIGURES_DIR):
        self.save_dir = save_dir
        ensure_directories()
        
    def _save_figure(self, fig, filename: str):
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, dpi=PLOT_CONFIG.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Kaydedildi: {filepath}")
        return filepath
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                               class_names: List[str],
                               title: str = "Karışıklık Matrisi",
                               save_name: str = None) -> plt.Figure:
        """
        Karışıklık matrisi görselleştirme.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, annot_kws={'size': 14})
        
        ax.set_xlabel('Tahmin', fontsize=12)
        ax.set_ylabel('Gerçek', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict],
                               metrics: List[str] = None,
                               title: str = "Model Karşılaştırması",
                               save_name: str = None) -> plt.Figure:
        """
        Birden fazla modelin performans karşılaştırması.
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
        model_names = list(results.keys())
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_metrics)
        width = 0.8 / len(model_names)
        
        colors = [COLORS.get(name.lower(), '#95A5A6') for name in model_names]
        
        for i, (model_name, result) in enumerate(results.items()):
            values = [result['metrics'].get(m, 0) for m in metrics]
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, 
                         color=colors[i], edgecolor='white', linewidth=0.5)
            
            # Değerleri bar üzerine yaz
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Skor', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.15])
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_roc_curves(self, results: Dict[str, Dict],
                        title: str = "ROC Eğrileri",
                        save_name: str = None) -> plt.Figure:
        """
        ROC eğrileri karşılaştırması.
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = list(COLORS.values())
        
        for i, (model_name, result) in enumerate(results.items()):
            if 'probabilities' in result and result['probabilities'] is not None:
                y_true = result.get('y_true', None)
                y_proba = result['probabilities']
                
                if y_true is not None and y_proba is not None:
                    if y_proba.ndim == 2:
                        y_proba = y_proba[:, 1]
                    
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, linewidth=2, 
                           label=f'{model_name} (AUC = {roc_auc:.3f})',
                           color=colors[i % len(colors)])
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Rastgele')
        ax.set_xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
        ax.set_ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_training_history(self, history: Dict,
                               title: str = "Eğitim Geçmişi",
                               save_name: str = None) -> plt.Figure:
        """
        Derin öğrenme eğitim geçmişi.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Eğitim', linewidth=2, color=COLORS['cnn'])
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Doğrulama', 
                        linewidth=2, color=COLORS['lstm'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Kayıp (Loss)')
        axes[0].set_title('Kayıp Değişimi', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['accuracy'], label='Eğitim', linewidth=2, color=COLORS['cnn'])
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Doğrulama',
                        linewidth=2, color=COLORS['lstm'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Doğruluk')
        axes[1].set_title('Doğruluk Değişimi', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_snr_analysis(self, snr_results: Dict[str, Dict[float, float]],
                          title: str = "SNR Analizi",
                          save_name: str = None) -> plt.Figure:
        """
        Farklı SNR seviyelerinde performans analizi.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [COLORS.get(name.lower(), '#95A5A6') for name in snr_results.keys()]
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (model_name, snr_data) in enumerate(snr_results.items()):
            snr_levels = sorted(snr_data.keys())
            accuracies = [snr_data[snr] for snr in snr_levels]
            
            ax.plot(snr_levels, accuracies, marker=markers[i % len(markers)],
                   linewidth=2, markersize=8, label=model_name,
                   color=colors[i])
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Doğruluk', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_feature_importance(self, importances: np.ndarray,
                                 feature_names: List[str] = None,
                                 top_n: int = 20,
                                 title: str = "Öznitelik Önem Skorları",
                                 save_name: str = None) -> plt.Figure:
        """
        Random Forest öznitelik önem skorları.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if feature_names is None:
            feature_names = [f'Öznitelik {i}' for i in range(len(importances))]
        
        # En önemli özellikleri seç
        indices = np.argsort(importances)[-top_n:]
        
        ax.barh(range(len(indices)), importances[indices], color=COLORS['svm'])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Önem Skoru', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig


if __name__ == "__main__":
    print("Görselleştirme Modülü - Test")
    print("=" * 50)
    
    import librosa
    
    # Sentetik sinyal
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    y = (0.5 * np.sin(2 * np.pi * 200 * t) +
         0.3 * np.sin(2 * np.pi * 400 * t) +
         0.1 * np.random.randn(len(t)))
    
    # Sinyal görselleştirme testi
    sig_viz = SignalVisualizer()
    fig = sig_viz.plot_signal_analysis(y, sr, "Test Sinyal Analizi", 
                                        "test_signal_analysis.png")
    
    # Sonuç görselleştirme testi
    results_viz = ResultsVisualizer()
    
    # Örnek sonuçlar
    test_results = {
        'SVM': {'metrics': {'accuracy': 0.892, 'precision': 0.887, 
                           'recall': 0.891, 'f1_score': 0.889}},
        'CNN': {'metrics': {'accuracy': 0.962, 'precision': 0.958,
                           'recall': 0.965, 'f1_score': 0.961}},
        'LSTM': {'metrics': {'accuracy': 0.923, 'precision': 0.919,
                            'recall': 0.927, 'f1_score': 0.923}}
    }
    
    fig = results_viz.plot_model_comparison(test_results, 
                                            save_name="test_model_comparison.png")
    
    print("\nTest başarılı! Grafikler results/figures/ klasörüne kaydedildi.")
    plt.show()
