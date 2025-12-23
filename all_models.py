import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import warnings
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Model açıklamaları
MODEL_DESCRIPTIONS = {
    'LSTM': {
        'full_name': 'Long Short-Term Memory',
        'type': 'Deep Learning (Recurrent Neural Network)',
        'purpose': 'Zaman serilerindeki uzun vadeli bağımlılıkları öğrenmek için tasarlanmış derin öğrenme modeli',
        'strengths': [
            'Karmaşık zaman serisi desenlerini yakalama',
            'Uzun vadeli bağımlılıkları öğrenme',
            'Doğrusal olmayan ilişkileri modelleme'
        ],
        'use_case': 'Su tüketimindeki karmaşık mevsimsel ve trend desenlerini yakalamak için kullanıldı'
    },
    'Prophet': {
        'full_name': 'Facebook Prophet',
        'type': 'Time Series Forecasting',
        'purpose': 'Mevsimsellik ve tatil etkilerini otomatik yakalayan zaman serisi tahmin modeli',
        'strengths': [
            'Güçlü mevsimsellik modelleme',
            'Eksik veri ve aykırı değerlere dayanıklılık',
            'Trend değişim noktalarını otomatik algılama'
        ],
        'use_case': 'Yıllık mevsimsel desenleri ve trend değişimlerini modellemek için kullanıldı'
    },
    'SARIMA': {
        'full_name': 'Seasonal AutoRegressive Integrated Moving Average',
        'type': 'Statistical Time Series Model',
        'purpose': 'Mevsimsel desenleri olan zaman serilerini istatistiksel olarak modelleyen klasik yöntem',
        'strengths': [
            'İstatistiksel olarak sağlam temel',
            'Mevsimsel desenleri açık modelleme',
            'Yorumlanabilir parametreler'
        ],
        'use_case': 'Aylık su tüketimindeki periyodik desenleri istatistiksel olarak modellemek için kullanıldı'
    },
    'XGBoost': {
        'full_name': 'eXtreme Gradient Boosting',
        'type': 'Machine Learning (Ensemble)',
        'purpose': 'Gradient boosting ile güçlü tahminler yapan ensemble makine öğrenmesi modeli',
        'strengths': [
            'Yüksek tahmin doğruluğu',
            'Özellik önemliliğini belirleme',
            'Doğrusal olmayan ilişkileri yakalama'
        ],
        'use_case': 'Farklı özelliklerin su tüketimine etkisini öğrenmek ve yüksek doğruluklu tahminler yapmak için kullanıldı'
    }
}

print("=" * 100)
print("TÜM MODELLER - KARŞILAŞTIRMALI ANALİZ".center(100))
print("=" * 100)

# ========== VERİ YÜKLEME ==========
print("\n📊 VERİ YÜKLEME")
print("-" * 100)

df = pd.read_csv('fabrika_clean.csv')
df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Mont'].astype(str) + '-01')
df = df.sort_values('date').reset_index(drop=True)

print(f"✓ Toplam kayıt: {len(df)} ay")
print(f"✓ Tarih aralığı: {df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}")

# ========== ÖZELLİK SEÇİMİ ==========
print("\n" + "=" * 100)
print("ÖZELLİK SEÇİMİ")
print("=" * 100)

feature_cols = [col for col in df.columns if col not in ['Year', 'Mont', 'date', 'W-Water']]
correlations = df[feature_cols + ['W-Water']].corr()['W-Water'].abs().sort_values(ascending=False)
top_3_features = correlations.drop('W-Water').head(3).index.tolist()

print(f"✓ SEÇİLEN 3 ÖZELLİK: {top_3_features}")

# ========== VERİ HAZIRLAMA ==========
print("\n" + "=" * 100)
print("VERİ HAZIRLAMA")
print("=" * 100)

split_idx = int(len(df) * 0.8)
dates_test = df['date'].values[split_idx:]
y_test_real = df['W-Water'].values[split_idx:]

print(f"✓ Eğitim boyutu: {split_idx} ay")
print(f"✓ Test boyutu: {len(df) - split_idx} ay")

# Sonuçları saklamak için dictionary
results = {}

# ========================================
# 1. LSTM MODEL
# ========================================
print("\n" + "=" * 100)
print("1. LSTM MODEL EĞİTİMİ")
print("=" * 100)

X = df[top_3_features].values
y = df['W-Water'].values.reshape(-1, 1)

X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

scaler_x_lstm = MinMaxScaler()
scaler_y_lstm = MinMaxScaler()

X_train_lstm = scaler_x_lstm.fit_transform(X_train_raw).reshape(len(X_train_raw), 1, 3)
X_test_lstm = scaler_x_lstm.transform(X_test_raw).reshape(len(X_test_raw), 1, 3)
y_train_lstm = scaler_y_lstm.fit_transform(y_train_raw)
y_test_lstm = scaler_y_lstm.transform(y_test_raw)

model_lstm = Sequential([
    LSTM(50, input_shape=(1, 3)),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(0.01), loss='mse')

print("⏳ LSTM modeli eğitiliyor...")
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=16, verbose=0)
print("✓ LSTM eğitimi tamamlandı!")

y_pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)
y_pred_lstm = scaler_y_lstm.inverse_transform(y_pred_lstm_scaled).flatten()

results['LSTM'] = {
    'predictions': y_pred_lstm,
    'r2': r2_score(y_test_real, y_pred_lstm),
    'mae': mean_absolute_error(y_test_real, y_pred_lstm),
    'rmse': np.sqrt(mean_squared_error(y_test_real, y_pred_lstm)),
    'mape': mean_absolute_percentage_error(y_test_real, y_pred_lstm),
    'color': '#E63946'
}
results['LSTM']['accuracy'] = (1 - results['LSTM']['mape']) * 100

print(f"✓ LSTM R² Score: {results['LSTM']['r2']:.4f}")
print(f"✓ LSTM Doğruluk: %{results['LSTM']['accuracy']:.2f}")

# ========================================
# 2. PROPHET MODEL
# ========================================
print("\n" + "=" * 100)
print("2. PROPHET MODEL EĞİTİMİ")
print("=" * 100)

prophet_df = df[['date', 'W-Water'] + top_3_features].copy()
prophet_df.columns = ['ds', 'y'] + [f'feat_{i}' for i in range(len(top_3_features))]
prophet_df = prophet_df.dropna()

scaler_prophet = StandardScaler()
feat_cols = [f'feat_{i}' for i in range(len(top_3_features))]
prophet_df[feat_cols] = scaler_prophet.fit_transform(prophet_df[feat_cols])

train_prophet = prophet_df[:split_idx]
test_prophet = prophet_df[split_idx:]

model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    seasonality_mode='additive'
)

for i in range(len(top_3_features)):
    model_prophet.add_regressor(f'feat_{i}', prior_scale=10)

print("⏳ Prophet modeli eğitiliyor...")
model_prophet.fit(train_prophet, algorithm='LBFGS')
print("✓ Prophet eğitimi tamamlandı!")

forecast_prophet = model_prophet.predict(test_prophet)
y_pred_prophet = forecast_prophet['yhat'].values

results['Prophet'] = {
    'predictions': y_pred_prophet,
    'r2': r2_score(y_test_real, y_pred_prophet),
    'mae': mean_absolute_error(y_test_real, y_pred_prophet),
    'rmse': np.sqrt(mean_squared_error(y_test_real, y_pred_prophet)),
    'mape': mean_absolute_percentage_error(y_test_real, y_pred_prophet),
    'color': '#A23B72'
}
results['Prophet']['accuracy'] = (1 - results['Prophet']['mape']) * 100

print(f"✓ Prophet R² Score: {results['Prophet']['r2']:.4f}")
print(f"✓ Prophet Doğruluk: %{results['Prophet']['accuracy']:.2f}")

# ========================================
# 3. SARIMA MODEL
# ========================================
print("\n" + "=" * 100)
print("3. SARIMA MODEL EĞİTİMİ")
print("=" * 100)

df_sarima = df.copy()
df_sarima = df_sarima.set_index('date')
y_sarima = df_sarima['W-Water']
X_exog_sarima = df_sarima[top_3_features]

scaler_sarima = StandardScaler()
X_exog_sarima_scaled = pd.DataFrame(
    scaler_sarima.fit_transform(X_exog_sarima),
    index=X_exog_sarima.index,
    columns=X_exog_sarima.columns
)

y_train_sarima = y_sarima[:split_idx]
y_test_sarima = y_sarima[split_idx:]
X_train_sarima = X_exog_sarima_scaled[:split_idx]
X_test_sarima = X_exog_sarima_scaled[split_idx:]

# Basit parametre seti (hız için)
print("⏳ SARIMA parametreleri optimize ediliyor...")
best_aic = float('inf')
best_params_sarima = (1, 1, 1)
best_seasonal_sarima = (1, 1, 1, 12)

# Sadece birkaç kombinasyonu dene
param_combinations = [
    ((1, 1, 1), (1, 1, 1, 12)),
    ((1, 0, 1), (1, 0, 1, 12)),
    ((2, 1, 1), (1, 1, 1, 12)),
]

for params, seasonal in param_combinations:
    try:
        model_test = SARIMAX(y_train_sarima,
                             exog=X_train_sarima,
                             order=params,
                             seasonal_order=seasonal,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        results_test = model_test.fit(disp=False, maxiter=100)
        if results_test.aic < best_aic:
            best_aic = results_test.aic
            best_params_sarima = params
            best_seasonal_sarima = seasonal
    except:
        continue

print(f"✓ En iyi parametreler: {best_params_sarima}x{best_seasonal_sarima}")

model_sarima = SARIMAX(y_train_sarima,
                       exog=X_train_sarima,
                       order=best_params_sarima,
                       seasonal_order=best_seasonal_sarima,
                       enforce_stationarity=False,
                       enforce_invertibility=False)

print("⏳ SARIMA modeli eğitiliyor...")
results_sarima = model_sarima.fit(disp=False, maxiter=200)
print("✓ SARIMA eğitimi tamamlandı!")

forecast_sarima = results_sarima.forecast(steps=len(y_test_sarima), exog=X_test_sarima)
y_pred_sarima = forecast_sarima.values

results['SARIMA'] = {
    'predictions': y_pred_sarima,
    'r2': r2_score(y_test_real, y_pred_sarima),
    'mae': mean_absolute_error(y_test_real, y_pred_sarima),
    'rmse': np.sqrt(mean_squared_error(y_test_real, y_pred_sarima)),
    'mape': mean_absolute_percentage_error(y_test_real, y_pred_sarima),
    'color': '#06A77D'
}
results['SARIMA']['accuracy'] = (1 - results['SARIMA']['mape']) * 100

print(f"✓ SARIMA R² Score: {results['SARIMA']['r2']:.4f}")
print(f"✓ SARIMA Doğruluk: %{results['SARIMA']['accuracy']:.2f}")

# ========================================
# 4. XGBoost MODEL
# ========================================
print("\n" + "=" * 100)
print("4. XGBoost MODEL EĞİTİMİ")
print("=" * 100)

X_xgb = df[top_3_features].values
y_xgb = df['W-Water'].values

scaler_xgb = StandardScaler()
X_xgb_scaled = scaler_xgb.fit_transform(X_xgb)

X_train_xgb = X_xgb_scaled[:split_idx]
X_test_xgb = X_xgb_scaled[split_idx:]
y_train_xgb = y_xgb[:split_idx]

params_xgb = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.05,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

print("⏳ XGBoost modeli eğitiliyor...")
model_xgb = xgb.XGBRegressor(**params_xgb)
model_xgb.fit(X_train_xgb, y_train_xgb, verbose=False)
print("✓ XGBoost eğitimi tamamlandı!")

y_pred_xgb = model_xgb.predict(X_test_xgb)

results['XGBoost'] = {
    'predictions': y_pred_xgb,
    'r2': r2_score(y_test_real, y_pred_xgb),
    'mae': mean_absolute_error(y_test_real, y_pred_xgb),
    'rmse': np.sqrt(mean_squared_error(y_test_real, y_pred_xgb)),
    'mape': mean_absolute_percentage_error(y_test_real, y_pred_xgb),
    'color': '#F4A460'
}
results['XGBoost']['accuracy'] = (1 - results['XGBoost']['mape']) * 100

print(f"✓ XGBoost R² Score: {results['XGBoost']['r2']:.4f}")
print(f"✓ XGBoost Doğruluk: %{results['XGBoost']['accuracy']:.2f}")

# ========================================
# GÖRSELLEŞTİRME - PDF OLUŞTURMA
# ========================================
print("\n" + "=" * 100)
print("PDF RAPORU OLUŞTURULUYOR")
print("=" * 100)

pdf_filename = 'model_comparison_report.pdf'
with PdfPages(pdf_filename) as pdf:
    # ====================================
    # SAYFA 0: KAPAK VE METODOLOJİ
    # ====================================
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle('ELEKTRİK TÜKETİMİ TAHMİN MODELLERİ\nKARŞILAŞTIRMALI ANALİZ RAPORU',
                 fontsize=22, fontweight='bold', y=0.98)

    # Proje Özeti
    ax1 = plt.subplot(6, 1, 1)
    ax1.axis('off')
    summary_text = f"""
    PROJE ÖZETİ

    Bu çalışmada, su tüketimi tahminlemesi için 4 farklı makine öğrenmesi ve derin öğrenme modeli 
    karşılaştırılmıştır. Modeller, fabrika su tüketim verisi üzerinde eğitilmiş ve test edilmiştir.

    • Veri Seti: {len(df)} aylık su tüketim verisi ({df['date'].min().strftime('%Y-%m')} - {df['date'].max().strftime('%Y-%m')})
    • Eğitim Seti: {split_idx} ay (%80)
    • Test Seti: {len(df) - split_idx} ay (%20)
    • Kullanılan Özellikler: {', '.join(top_3_features)}
    • Hedef Değişken: W-Water (Su Tüketimi)
    """
    ax1.text(0.05, 0.5, summary_text, fontsize=11, va='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Model Açıklamaları
    for idx, (model_name, desc) in enumerate(MODEL_DESCRIPTIONS.items(), 2):
        ax = plt.subplot(6, 1, idx)
        ax.axis('off')

        model_text = f"""
        {idx - 1}. {model_name} - {desc['full_name']}
        Tür: {desc['type']}

        Amaç: {desc['purpose']}

        Güçlü Yönler:
        """ + '\n        '.join([f"• {s}" for s in desc['strengths']]) + f"""

        Kullanım: {desc['use_case']}
        """

        ax.text(0.05, 0.5, model_text, fontsize=9.5, va='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================
    # SAYFA 0.5: TÜM VERİ GRAFİĞİ (Training Data)
    # ====================================
    fig = plt.figure(figsize=(16, 10))
    ax = plt.subplot(111)

    # Tüm veriyi göster
    ax.plot(df['date'].values, df['W-Water'].values, '-',
            linewidth=2, color='#2E86AB', alpha=0.8)

    # Training ve test bölgelerini vurgula
    ax.axvspan(df['date'].values[0], df['date'].values[split_idx - 1],
               alpha=0.2, color='green', label='Training Data (80%)')
    ax.axvspan(df['date'].values[split_idx], df['date'].values[-1],
               alpha=0.2, color='red', label='Test Data (20%)')

    ax.set_title('Water Consumption Data Used in Training Phase\nMonthly Data (2015-2021)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Time (month)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Water Consumption (m³)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45, labelsize=11)

    # İstatistik kutusu
    stats_text = f"""Training Data Statistics:
    • Period: {len(df[:split_idx])} months
    • Mean: {df['W-Water'][:split_idx].mean():.2f}
    • Std Dev: {df['W-Water'][:split_idx].std():.2f}
    • Min: {df['W-Water'][:split_idx].min():.2f}
    • Max: {df['W-Water'][:split_idx].max():.2f}

    Test Data:
    • Period: {len(df[split_idx:])} months"""

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================
    # SAYFA 0.75: TRAINING/TESTING GRAPHS - HER MODEL İÇİN
    # ====================================
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle('Training/Testing Graphs of All Models', fontsize=18, fontweight='bold', y=0.995)

    for idx, (model_name, model_data) in enumerate(results.items(), 1):
        ax = plt.subplot(4, 1, idx)

        # Training verisi (gerçek)
        ax.plot(df['date'].values[:split_idx], df['W-Water'].values[:split_idx],
                '-', linewidth=2, color='blue', alpha=0.6, label='Training (Real)')

        # Test verisi (gerçek ve tahmin)
        ax.plot(dates_test, y_test_real,
                '-', linewidth=2.5, color='blue', label='Testing (Real)')
        ax.plot(dates_test, model_data['predictions'],
                '--', linewidth=2.5, color=model_data['color'],
                label=f'{model_name} (Predicted)', alpha=0.9)

        # Training/Test bölge ayırıcı
        ax.axvline(x=df['date'].values[split_idx], color='red',
                   linestyle=':', linewidth=2, alpha=0.5, label='Train/Test Split')

        ax.set_title(f'{model_name} Model - Training/Testing Graph',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Time (month)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Water Consumption (m³)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=9)

        # Performans bilgisi
        perf_text = f'R²={model_data["r2"]:.4f}, MAE={model_data["mae"]:.2f}'
        ax.text(0.98, 0.02, perf_text, transform=ax.transAxes,
                fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================
    # SAYFA 1: Tüm Modellerin Karşılaştırmalı Grafiği
    # ====================================
    fig = plt.figure(figsize=(16, 10))
    ax = plt.subplot(111)

    # Gerçek değerler
    ax.plot(dates_test, y_test_real, 'o-', label='Gerçek Değer',
            linewidth=3, markersize=8, color='#2E86AB', zorder=5)

    # Her model için tahminler
    for model_name, model_data in results.items():
        ax.plot(dates_test, model_data['predictions'], 's--',
                label=f"{model_name} (R²={model_data['r2']:.3f})",
                linewidth=2.5, markersize=7, color=model_data['color'], alpha=0.8)

    ax.set_title('TÜM MODELLERİN KARŞILAŞTIRMALI TEST SONUÇLARI\nSu Tüketimi Tahmini',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Tarih (Ay)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Su Tüketimi (m³)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

    # Sayfa 2: Her Model İçin Ayrı Grafik
    fig = plt.figure(figsize=(16, 20))

    for idx, (model_name, model_data) in enumerate(results.items(), 1):
        ax = plt.subplot(4, 1, idx)

        ax.plot(dates_test, y_test_real, 'o-', label='Gerçek Değer',
                linewidth=3, markersize=10, color='#2E86AB')
        ax.plot(dates_test, model_data['predictions'], 's--',
                label=f'{model_name} Tahmini',
                linewidth=3, markersize=10, color=model_data['color'])

        ax.set_title(f"{model_name} MODEL - TEST SONUÇLARI\n"
                     f"R² = {model_data['r2']:.4f} | Doğruluk = %{model_data['accuracy']:.2f} | "
                     f"MAE = {model_data['mae']:.2f}",
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Tarih', fontsize=12, fontweight='bold')
        ax.set_ylabel('Su Tüketimi', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=9)

        # Performans kutusu
        textstr = (f'Performans Metrikleri:\n'
                   f'R² Score: {model_data["r2"]:.4f}\n'
                   f'Doğruluk: %{model_data["accuracy"]:.2f}\n'
                   f'MAE: {model_data["mae"]:.2f}\n'
                   f'RMSE: {model_data["rmse"]:.2f}\n'
                   f'MAPE: %{model_data["mape"] * 100:.2f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================
    # SAYFA: HER MODEL İÇİN DETAYLI ANALİZ SAYFALARı
    # ====================================
    for model_name, model_data in results.items():
        fig = plt.figure(figsize=(16, 20))
        fig.suptitle(f'{model_name} Model - Detaylı Analiz',
                     fontsize=20, fontweight='bold', y=0.98)

        # Model açıklaması
        ax1 = plt.subplot(5, 1, 1)
        ax1.axis('off')

        desc = MODEL_DESCRIPTIONS[model_name]
        analysis_text = f"""
        MODEL BİLGİLERİ

        Model Adı: {desc['full_name']}
        Model Tipi: {desc['type']}

        Amaç ve Kullanım:
        {desc['purpose']}

        Bu Çalışmadaki Rolü:
        {desc['use_case']}

        Güçlü Yönleri:
        """ + '\n        '.join([f"• {s}" for s in desc['strengths']])

        ax1.text(0.05, 0.5, analysis_text, fontsize=10, va='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

        # Test tahmin grafiği
        ax2 = plt.subplot(5, 1, 2)
        ax2.plot(dates_test, y_test_real, 'o-', label='Gerçek Değer',
                 linewidth=3, markersize=8, color='#2E86AB')
        ax2.plot(dates_test, model_data['predictions'], 's--',
                 label=f'{model_name} Tahmini',
                 linewidth=3, markersize=8, color=model_data['color'])

        ax2.set_title(f'Test Seti Tahmin Sonuçları', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Tarih', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Su Tüketimi (m³)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45, labelsize=10)

        # Hata analizi grafiği
        ax3 = plt.subplot(5, 1, 3)
        errors = model_data['predictions'] - y_test_real
        ax3.plot(dates_test, errors, 'o-', color=model_data['color'],
                 linewidth=2, markersize=6, alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax3.fill_between(dates_test, errors, 0, alpha=0.3, color=model_data['color'])

        ax3.set_title('Tahmin Hataları (Predicted - Actual)', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Tarih', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Hata (m³)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(axis='x', rotation=45, labelsize=10)

        # Hata dağılımı histogramı
        ax4 = plt.subplot(5, 2, 7)
        ax4.hist(errors, bins=15, color=model_data['color'], alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Hata Dağılımı', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Hata (m³)', fontsize=10)
        ax4.set_ylabel('Frekans', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        # Scatter plot (Gerçek vs Tahmin)
        ax5 = plt.subplot(5, 2, 8)
        ax5.scatter(y_test_real, model_data['predictions'],
                    alpha=0.6, s=80, color=model_data['color'], edgecolors='black')

        # İdeal çizgi (45 derece)
        min_val = min(y_test_real.min(), model_data['predictions'].min())
        max_val = max(y_test_real.max(), model_data['predictions'].max())
        ax5.plot([min_val, max_val], [min_val, max_val],
                 'r--', linewidth=2, label='İdeal Tahmin')

        ax5.set_title('Gerçek vs Tahmin', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Gerçek Değer (m³)', fontsize=10)
        ax5.set_ylabel('Tahmin Değeri (m³)', fontsize=10)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # Performans metrikleri tablosu
        ax6 = plt.subplot(5, 1, 5)
        ax6.axis('off')

        metrics_text = f"""
        PERFORMANS METRİKLERİ ve DEĞERLENDİRME

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  Metrik          │  Değer           │  Açıklama                             ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  R² Score        │  {model_data['r2']:6.4f}        │  Model varyansın %{model_data['r2'] * 100:.1f}'ini açıklıyor       ║
        ║  Doğruluk        │  %{model_data['accuracy']:5.2f}        │  Tahminlerin ortalama doğruluğu              ║
        ║  MAE             │  {model_data['mae']:6.2f} m³     │  Ortalama mutlak hata                        ║
        ║  RMSE            │  {model_data['rmse']:6.2f} m³     │  Kök ortalama kare hata                      ║
        ║  MAPE            │  %{model_data['mape'] * 100:5.2f}        │  Ortalama mutlak yüzde hata                  ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        Hata İstatistikleri:
        • Ortalama Hata: {errors.mean():.2f} m³
        • Std Sapma: {errors.std():.2f} m³
        • Min Hata: {errors.min():.2f} m³ (Gerçeğin altında)
        • Max Hata: {errors.max():.2f} m³ (Gerçeğin üstünde)
        """

        # R² Score'a göre performans yorumu
        if model_data['r2'] >= 0.9:
            performance = "Mükemmel"
        elif model_data['r2'] >= 0.8:
            performance = "Çok İyi"
        elif model_data['r2'] >= 0.7:
            performance = "İyi"
        elif model_data['r2'] >= 0.6:
            performance = "Orta"
        else:
            performance = "Geliştirilmeli"

        metrics_text += f"\n        Genel Performans Değerlendirmesi: {performance}"

        ax6.text(0.05, 0.5, metrics_text, fontsize=9, va='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()

    # ====================================
    # SAYFA: Performans Karşılaştırma Tablosu ve Grafikler
    # ====================================
    fig = plt.figure(figsize=(16, 20))

    # Tablo
    ax1 = plt.subplot(4, 1, 1)
    ax1.axis('tight')
    ax1.axis('off')

    table_data = []
    table_data.append(['Model', 'R² Score', 'Doğruluk (%)', 'MAE', 'RMSE', 'MAPE (%)'])

    for model_name, model_data in results.items():
        table_data.append([
            model_name,
            f"{model_data['r2']:.4f}",
            f"{model_data['accuracy']:.2f}",
            f"{model_data['mae']:.2f}",
            f"{model_data['rmse']:.2f}",
            f"{model_data['mape'] * 100:.2f}"
        ])

    table = ax1.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Başlık satırını renklendir
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Satırları renklendir
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    ax1.set_title('PERFORMANS METRİKLERİ KARŞILAŞTIRMA TABLOSU',
                  fontsize=16, fontweight='bold', pad=20)

    # R² Score Karşılaştırması
    ax2 = plt.subplot(4, 1, 2)
    models = list(results.keys())
    r2_scores = [results[m]['r2'] for m in models]
    colors = [results[m]['color'] for m in models]

    bars = ax2.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('R² Score Karşılaştırması', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Bar üzerine değerleri yaz
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Doğruluk Karşılaştırması
    ax3 = plt.subplot(4, 1, 3)
    accuracies = [results[m]['accuracy'] for m in models]

    bars = ax3.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_title('Doğruluk (%) Karşılaştırması', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel('Doğruluk (%)', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # MAE ve RMSE Karşılaştırması
    ax4 = plt.subplot(4, 1, 4)
    mae_values = [results[m]['mae'] for m in models]
    rmse_values = [results[m]['rmse'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, mae_values, width, label='MAE',
                    color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width / 2, rmse_values, width, label='RMSE',
                    color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax4.set_title('MAE ve RMSE Karşılaştırması', fontsize=14, fontweight='bold', pad=15)
    ax4.set_ylabel('Hata Değeri', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Bar üzerine değerleri yaz
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================
    # SON SAYFA: GENEL DEĞERLENDİRME ve ÖNERİLER
    # ====================================
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle('GENEL DEĞERLENDİRME ve ÖNERİLER',
                 fontsize=20, fontweight='bold', y=0.98)

    # Model karşılaştırma özeti
    ax1 = plt.subplot(4, 1, 1)
    ax1.axis('off')

    sorted_by_r2 = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)

    summary_text = f"""
    MODEL PERFORMANS SIRALAMAS (R² Score'a Göre)

    🥇 1. {sorted_by_r2[0][0]:12s} - R² = {sorted_by_r2[0][1]['r2']:.4f} | Doğruluk = %{sorted_by_r2[0][1]['accuracy']:.2f} | MAE = {sorted_by_r2[0][1]['mae']:.2f}
    🥈 2. {sorted_by_r2[1][0]:12s} - R² = {sorted_by_r2[1][1]['r2']:.4f} | Doğruluk = %{sorted_by_r2[1][1]['accuracy']:.2f} | MAE = {sorted_by_r2[1][1]['mae']:.2f}
    🥉 3. {sorted_by_r2[2][0]:12s} - R² = {sorted_by_r2[2][1]['r2']:.4f} | Doğruluk = %{sorted_by_r2[2][1]['accuracy']:.2f} | MAE = {sorted_by_r2[2][1]['mae']:.2f}
       4. {sorted_by_r2[3][0]:12s} - R² = {sorted_by_r2[3][1]['r2']:.4f} | Doğruluk = %{sorted_by_r2[3][1]['accuracy']:.2f} | MAE = {sorted_by_r2[3][1]['mae']:.2f}

    EN İYİ MODEL: {sorted_by_r2[0][0]}
    Bu model, test verisindeki varyansın %{sorted_by_r2[0][1]['r2'] * 100:.1f}'ini açıklayarak en yüksek tahmin 
    performansını göstermiştir.
    """

    ax1.text(0.05, 0.5, summary_text, fontsize=11, va='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Her modelin güçlü ve zayıf yönleri
    ax2 = plt.subplot(4, 1, 2)
    ax2.axis('off')

    analysis_text = """
    MODEL ANALİZİ ve KARŞILAŞTIRMA

    LSTM (Long Short-Term Memory):
    ✓ Güçlü Yönler: Karmaşık zaman serisi desenlerini yakalama, uzun vadeli bağımlılıklar
    ✗ Zayıf Yönler: Eğitim süresi uzun, daha fazla veri gerektirir, hiperparametre ayarı kritik
    📊 Kullanım Önerisi: Uzun geçmişi olan, karmaşık desenli zaman serileri için idealdir

    Prophet (Facebook Prophet):
    ✓ Güçlü Yönler: Mevsimsellik modelleme, eksik veri toleransı, trend değişim tespiti
    ✗ Zayıf Yönler: Karmaşık doğrusal olmayan ilişkilerde sınırlı, aşırı basitleştirme riski
    📊 Kullanım Önerisi: Güçlü mevsimsel desenleri olan iş verileri için mükemmel

    SARIMA (Seasonal ARIMA):
    ✓ Güçlü Yönler: İstatistiksel temel, yorumlanabilir parametreler, mevsimsellik modelleme
    ✗ Zayıf Yönler: Doğrusal olmayan ilişkilerde zayıf, parametre seçimi karmaşık
    📊 Kullanım Önerisi: Klasik zaman serisi analizi ve istatistiksel güvenlik gerektiğinde

    XGBoost (Gradient Boosting):
    ✓ Güçlü Yönler: Yüksek doğruluk, özellik önemi, doğrusal olmayan ilişkiler
    ✗ Zayıf Yönler: Zaman serisi yapısını doğrudan modellemez, özellik mühendisliği gerektirir
    📊 Kullanım Önerisi: Çok değişkenli tahminlerde ve özellik etkilerinin analizinde güçlü
    """

    ax2.text(0.05, 0.5, analysis_text, fontsize=9.5, va='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Öneriler ve sonuç
    ax3 = plt.subplot(4, 1, 3)
    ax3.axis('off')

    recommendations_text = f"""
    ÖNERİLER ve SONUÇ

    1. MODEL SEÇİMİ:
       • Üretim ortamı için: {sorted_by_r2[0][0]} (En yüksek R² Score)
       • Hızlı tahmin için: XGBoost (Eğitim ve tahmin hızı dengesi)
       • Yorumlanabilirlik için: SARIMA veya Prophet (İstatistiksel parametreler)
       • Karmaşık desenler için: LSTM (Derin öğrenme gücü)

    2. ENSEMBLE YAKLAŞIMI:
       • En iyi 2-3 modelin tahminlerinin ortalaması alınarak daha robust sonuçlar elde edilebilir
       • Önerilen ensemble: {sorted_by_r2[0][0]} + {sorted_by_r2[1][0]} + {sorted_by_r2[2][0]}

    3. İYİLEŞTİRME ÖNERİLERİ:
       • Daha fazla özellik mühendisliği (gecikme özellikleri, hareketli ortalamalar)
       • Hiperparametre optimizasyonu (Grid Search, Bayesian Optimization)
       • Daha uzun eğitim periyodu (özellikle LSTM için)
       • Cross-validation ile model stabilitesini test etme

    4. UYGULAMA ÖNERİLERİ:
       • Model performansını düzenli olarak izleyin
       • Yeni verilerle modeli periyodik olarak yeniden eğitin
       • Tahmin aralıklarını (confidence intervals) hesaplayın
       • Aykırı değer tespiti mekanizması ekleyin

    SONUÇ:
    Bu çalışmada 4 farklı model karşılaştırıldı. {sorted_by_r2[0][0]} modeli {sorted_by_r2[0][1]['r2']:.4f} R² 
    Score ile en iyi performansı gösterdi. Ancak, her modelin kendine özgü güçlü yönleri vardır ve 
    kullanım senaryosuna göre farklı modeller tercih edilebilir.
    """

    ax3.text(0.05, 0.5, recommendations_text, fontsize=9.5, va='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

    # Teknik detaylar
    ax4 = plt.subplot(4, 1, 4)
    ax4.axis('off')

    technical_text = f"""
    TEKNİK DETAYLAR

    Veri Seti:
    • Toplam Veri: {len(df)} ay
    • Eğitim Verisi: {split_idx} ay (%80)
    • Test Verisi: {len(df) - split_idx} ay (%20)
    • Tarih Aralığı: {df['date'].min().strftime('%Y-%m')} - {df['date'].max().strftime('%Y-%m')}

    Kullanılan Özellikler (Top 3):
    • {top_3_features[0]}
    • {top_3_features[1]}
    • {top_3_features[2]}

    Değerlendirme Metrikleri:
    • R² Score: Modelin veri varyansını açıklama gücü (0-1 arası, 1 en iyi)
    • MAE (Mean Absolute Error): Ortalama mutlak hata
    • RMSE (Root Mean Square Error): Kök ortalama kare hata (büyük hatalara daha duyarlı)
    • MAPE (Mean Absolute Percentage Error): Yüzdesel ortalama hata
    • Doğruluk: 100 - MAPE (yüzde olarak)

    Yazılım ve Kütüphaneler:
    • Python 3.x
    • TensorFlow/Keras (LSTM)
    • Facebook Prophet
    • Statsmodels (SARIMA)
    • XGBoost
    • Scikit-learn (Metrikler ve ön işleme)
    """

    ax4.text(0.05, 0.5, technical_text, fontsize=9, va='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

print(f"✓ PDF raporu oluşturuldu: {pdf_filename}")

# Karşılaştırmalı grafik PNG olarak da kaydet
fig = plt.figure(figsize=(16, 10))
ax = plt.subplot(111)

ax.plot(dates_test, y_test_real, 'o-', label='Gerçek Değer (Real)',
        linewidth=3, markersize=8, color='#2E86AB', zorder=5)

for model_name, model_data in results.items():
    ax.plot(dates_test, model_data['predictions'], 's--',
            label=f"{model_name}",
            linewidth=2.5, markersize=7, color=model_data['color'], alpha=0.8)

ax.set_title('TÜM MODELLERİN KARŞILAŞTIRMALI TEST SONUÇLARI\nWater Consumption Prediction',
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Time (month)', fontsize=14, fontweight='bold')
ax.set_ylabel('Water consumption (m³)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=45, labelsize=11)
ax.tick_params(axis='y', labelsize=11)

plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Karşılaştırmalı grafik kaydedildi: all_models_comparison.png")
plt.close()

# ========================================
# SONUÇ ÖZETİ
# ========================================
print("\n" + "=" * 100)
print("GENEL DEĞERLENDİRME VE SONUÇLAR".center(100))
print("=" * 100)

print("\n📊 MODEL PERFORMANS SIRALARMASI:")
print("-" * 100)

# R² Score'a göre sırala
sorted_by_r2 = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)

print("\n🏆 R² Score Sıralaması (En İyi → En Kötü):")
for rank, (model_name, model_data) in enumerate(sorted_by_r2, 1):
    print(f"   {rank}. {model_name:12s} → R² = {model_data['r2']:.4f} | "
          f"Doğruluk = %{model_data['accuracy']:.2f} | "
          f"MAE = {model_data['mae']:.2f}")

# Doğruluk'a göre sırala
sorted_by_acc = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("\n🎯 Doğruluk Sıralaması (En İyi → En Kötü):")
for rank, (model_name, model_data) in enumerate(sorted_by_acc, 1):
    print(f"   {rank}. {model_name:12s} → Doğruluk = %{model_data['accuracy']:.2f}")

# MAE'ye göre sırala (düşük olan iyi)
sorted_by_mae = sorted(results.items(), key=lambda x: x[1]['mae'])

print("\n📉 MAE Sıralaması (En İyi → En Kötü):")
for rank, (model_name, model_data) in enumerate(sorted_by_mae, 1):
    print(f"   {rank}. {model_name:12s} → MAE = {model_data['mae']:.2f}")

print("\n" + "=" * 100)
print("OLUŞTURULAN DOSYALAR".center(100))
print("=" * 100)
print(f"\n📄 Dosyalar:")
print(f"   • {pdf_filename:40s} - Tüm modellerin detaylı PDF raporu")
print(f"   • all_models_comparison.png                - Karşılaştırmalı grafik (PNG)")

print("\n" + "=" * 100)
print(f"🎉 EN İYİ MODEL: {sorted_by_r2[0][0]}")
print(f"   R² Score: {sorted_by_r2[0][1]['r2']:.4f}")
print(f"   Doğruluk: %{sorted_by_r2[0][1]['accuracy']:.2f}")
print(f"   MAE: {sorted_by_r2[0][1]['mae']:.2f}")
print(f"   RMSE: {sorted_by_r2[0][1]['rmse']:.2f}")
print("=" * 100)

print("\n✅ ANALİZ TAMAMLANDI!")
print("=" * 100)