import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
from itertools import product

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("SARIMA MODEL - SU TÜKETİMİ TAHMİNİ (TOP 3 ÖZELLİK)".center(100))
print("=" * 100)

# ========== VERİ YÜKLEME ==========
print("\n📊 VERİ YÜKLEME")
print("-" * 100)

df = pd.read_csv('fabrika_clean.csv')
df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Mont'].astype(str) + '-01')
df = df.sort_values('date').reset_index(drop=True)
df = df.set_index('date')

print(f"✓ Toplam kayıt: {len(df)} ay")
print(f"✓ Tarih aralığı: {df.index.min().strftime('%Y-%m')} → {df.index.max().strftime('%Y-%m')}")

# ========== ÖZELLİK SEÇİMİ ==========
print("\n" + "=" * 100)
print("ÖZELLİK SEÇİMİ")
print("=" * 100)

feature_cols = [col for col in df.columns if col not in ['Year', 'Mont', 'W-Water']]
correlations = df[feature_cols + ['W-Water']].corr()['W-Water'].abs().sort_values(ascending=False)
top_3_features = correlations.drop('W-Water').head(3).index.tolist()

print(f"✓ SEÇİLEN 3 ÖZELLİK: {top_3_features}")

# ========== VERİ HAZIRLAMA ==========
print("\n" + "=" * 100)
print("SARIMA İÇİN VERİ HAZIRLAMA")
print("=" * 100)

# Hedef değişken ve özellikler
y = df['W-Water']
X_exog = df[top_3_features]

# Normalizasyon
scaler = StandardScaler()
X_exog_scaled = pd.DataFrame(
    scaler.fit_transform(X_exog),
    index=X_exog.index,
    columns=X_exog.columns
)

# Train-test split
split_idx = int(len(df) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X_exog_scaled[:split_idx], X_exog_scaled[split_idx:]

print(f"✓ Eğitim boyutu: {len(y_train)} ay")
print(f"✓ Test boyutu: {len(y_test)} ay")

# ========== SARIMA PARAMETRE OPTİMİZASYONU ==========
print("\n" + "=" * 100)
print("SARIMA PARAMETRE OPTİMİZASYONU")
print("=" * 100)

# Grid search için parametre aralıkları
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
P = range(0, 2)
D = range(0, 2)
Q = range(0, 2)
s = [12]  # Yıllık mevsimsellik

pdq = list(product(p, d, q))
seasonal_pdq = list(product(P, D, Q, s))

best_aic = float('inf')
best_params = None
best_seasonal = None

print("⏳ En iyi parametreler aranıyor (bu biraz zaman alabilir)...")
print(f"   Toplam {len(pdq) * len(seasonal_pdq)} kombinasyon test edilecek...")

tested = 0
for param in pdq[:9]:  # İlk 9 kombinasyonu test et (hız için)
    for param_seasonal in seasonal_pdq[:4]:  # İlk 4 mevsimsel kombinasyonu
        try:
            tested += 1
            model = SARIMAX(y_train,
                            exog=X_train,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

            results = model.fit(disp=False, maxiter=100)

            if results.aic < best_aic:
                best_aic = results.aic
                best_params = param
                best_seasonal = param_seasonal

            if tested % 10 == 0:
                print(f"   {tested} kombinasyon test edildi... En iyi AIC: {best_aic:.2f}")
        except:
            continue

print(f"\n✓ Optimizasyon tamamlandı! ({tested} kombinasyon test edildi)")
print(f"✓ En iyi parametreler: {best_params}")
print(f"✓ En iyi mevsimsel parametreler: {best_seasonal}")
print(f"✓ En iyi AIC: {best_aic:.2f}")

# ========== SARIMA MODEL EĞİTİMİ ==========
print("\n" + "=" * 100)
print("SARIMA MODEL EĞİTİMİ")
print("=" * 100)

# Eğer optimizasyon başarısız olduysa default değerler kullan
if best_params is None:
    best_params = (1, 1, 1)
    best_seasonal = (1, 1, 1, 12)
    print("⚠ Varsayılan parametreler kullanılıyor")

print(f"✓ SARIMA{best_params}x{best_seasonal} modeli oluşturuluyor...")

model = SARIMAX(y_train,
                exog=X_train,
                order=best_params,
                seasonal_order=best_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False)

print("⏳ Model eğitiliyor...")
results = model.fit(disp=False, maxiter=200)
print("✓ Eğitim tamamlandı!")

# ========== TEST PERFORMANSI ==========
print("\n" + "=" * 100)
print("TEST PERFORMANSI")
print("=" * 100)

# Tahmin
forecast = results.forecast(steps=len(y_test), exog=X_test)
y_pred = forecast.values
y_true = y_test.values

# Metrikler
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred)
accuracy = (1 - mape) * 100

print(f"\n📊 PERFORMANS METRİKLERİ:")
print(f"   R² Score:  {r2:.4f}")
print(f"   Doğruluk:  %{accuracy:.2f}")
print(f"   MAE:       {mae:.2f}")
print(f"   RMSE:      {rmse:.2f}")
print(f"   MAPE:      %{mape * 100:.2f}")

# ========== GÖRSELLEŞTİRME ==========
print("\n" + "=" * 100)
print("GRAFİKLER OLUŞTURULUYOR")
print("=" * 100)

# Test Tahminleri
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.plot(y_test.index, y_true, 'o-', label='Gerçek Değer', linewidth=3, markersize=10, color='#2E86AB')
ax.plot(y_test.index, y_pred, 's--', label='SARIMA Tahmini', linewidth=3, markersize=10, color='#06A77D')

ax.set_title(
    f"SARIMA{best_params}x{best_seasonal} MODEL - TEST SONUÇLARI\n"
    f"R² = {r2:.4f} | Doğruluk = %{accuracy:.2f} | MAE = {mae:.2f}",
    fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Tarih', fontsize=13, fontweight='bold')
ax.set_ylabel('Su Tüketimi (W-Water)', fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.4, linestyle='--')
ax.tick_params(axis='x', rotation=45, labelsize=10)

# Performans metrik kutusu
textstr = f'Performans Metrikleri:\nR² Score: {r2:.4f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: %{mape * 100:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('sarima_test_predictions.png', dpi=300, bbox_inches='tight')
print("✓ SARIMA test tahminleri grafiği: sarima_test_predictions.png")
plt.close()

print("\n" + "=" * 100)
print("ANALİZ TAMAMLANDI!".center(100))
print("=" * 100)
print("\n📁 Oluşturulan Dosya:")
print("   • sarima_test_predictions.png    - Test sonuçları grafiği")
print("\n" + "=" * 100)
print(f"🥇 SARIMA MODEL PERFORMANSI:")
print(f"   • Model: SARIMA{best_params}x{best_seasonal}")
print(f"   • Özellikler: {', '.join(top_3_features)}")
print(f"   • R² Score: {r2:.4f}")
print(f"   • Doğruluk: %{accuracy:.2f}")
print(f"   • MAE: {mae:.2f}")
print(f"   • RMSE: {rmse:.2f}")
print("=" * 100)