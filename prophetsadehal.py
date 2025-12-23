import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("PROPHET MODEL - SU TÜKETİMİ TAHMİNİ (TOP 3 ÖZELLİK)".center(100))
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
print("ÖZELLİK KORELASYON ANALİZİ")
print("=" * 100)

feature_cols = [col for col in df.columns if col not in ['Year', 'Mont', 'date', 'W-Water']]
correlations = df[feature_cols + ['W-Water']].corr()['W-Water'].abs().sort_values(ascending=False)

print(f"\n📊 Toplam özellik sayısı: {len(feature_cols)}")
print("\nTop 15 En Yüksek Korelasyonlu Özellik:")
print(correlations.drop('W-Water').head(15).to_string())

# En iyi 3 özelliği seç
top_3_features = correlations.drop('W-Water').head(3).index.tolist()
print(f"\n✓ SEÇİLEN 3 ÖZELLİK: {top_3_features}")

# ========== MODEL EĞİTİMİ ==========
print("\n" + "=" * 100)
print("PROPHET MODEL EĞİTİMİ")
print("=" * 100)

# Veri hazırlama
prophet_df = df[['date', 'W-Water'] + top_3_features].copy()
prophet_df.columns = ['ds', 'y'] + [f'feat_{i}' for i in range(len(top_3_features))]
prophet_df = prophet_df.dropna()

# Normalizasyon
scaler = StandardScaler()
feat_cols = [f'feat_{i}' for i in range(len(top_3_features))]
prophet_df[feat_cols] = scaler.fit_transform(prophet_df[feat_cols])

# Train-test split
split_idx = int(len(prophet_df) * 0.8)
train = prophet_df[:split_idx]
test = prophet_df[split_idx:]

print(f"✓ Eğitim boyutu: {len(train)} ay")
print(f"✓ Test boyutu: {len(test)} ay")

# Model oluştur
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    seasonality_mode='additive'
)

# Regresörleri ekle
for i in range(len(top_3_features)):
    model.add_regressor(f'feat_{i}', prior_scale=10)

# Eğitim
print("\n⏳ Model eğitiliyor...")
model.fit(train, algorithm='LBFGS')
print("✓ Eğitim tamamlandı!")

# ========== TEST PERFORMANSI ==========
print("\n" + "=" * 100)
print("TEST PERFORMANSI")
print("=" * 100)

forecast = model.predict(test)
y_true = test['y'].values
y_pred = forecast['yhat'].values

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

dates = test['ds']
ax.plot(dates, y_true, 'o-', label='Gerçek Değer', linewidth=3, markersize=10, color='#2E86AB')
ax.plot(dates, y_pred, 's--', label='Prophet Tahmini', linewidth=3, markersize=10, color='#A23B72')

ax.set_title(
    f"PROPHET MODEL - TEST SONUÇLARI\n"
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
plt.savefig('prophet_test_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Test tahminleri grafiği: prophet_test_predictions.png")
plt.close()

print("\n" + "=" * 100)
print("ANALİZ TAMAMLANDI!".center(100))
print("=" * 100)
print("\n📁 Oluşturulan Dosya:")
print("   • prophet_test_predictions.png    - Test sonuçları grafiği")
print("\n" + "=" * 100)
print(f"🥇 PROPHET MODEL PERFORMANSI:")
print(f"   • Özellikler: {', '.join(top_3_features)}")
print(f"   • R² Score: {r2:.4f}")
print(f"   • Doğruluk: %{accuracy:.2f}")
print(f"   • MAE: {mae:.2f}")
print(f"   • RMSE: {rmse:.2f}")
print("=" * 100)