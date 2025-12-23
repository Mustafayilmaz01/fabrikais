import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("LSTM MODEL - SU TÜKETİMİ TAHMİNİ (TOP 3 ÖZELLİK)".center(100))
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

X = df[top_3_features].values
y = df['W-Water'].values.reshape(-1, 1)
dates = df['date'].values

# Train-test split (80/20) - ÖNCELİKLE!
split_idx = int(len(X) * 0.8)
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

# Ölçeklendirme (SADECE TRAIN'E FIT!)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_x.fit_transform(X_train_raw).reshape(len(X_train_raw), 1, 3)
X_test_scaled = scaler_x.transform(X_test_raw).reshape(len(X_test_raw), 1, 3)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

print(f"✓ Eğitim boyutu: {len(X_train_raw)} ay")
print(f"✓ Test boyutu: {len(X_test_raw)} ay")

# ========== LSTM MODEL EĞİTİMİ ==========
print("\n" + "=" * 100)
print("LSTM MODEL EĞİTİMİ")
print("=" * 100)

model = Sequential([
    LSTM(50, input_shape=(1, 3)),
    Dense(1)
])
model.compile(optimizer=Adam(0.01), loss='mse')

print("⏳ Model eğitiliyor...")
model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=16, verbose=0)
print("✓ Eğitim tamamlandı!")

# ========== TEST PERFORMANSI ==========
print("\n" + "=" * 100)
print("TEST PERFORMANSI")
print("=" * 100)

y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_test = y_test_raw.flatten()

# Metrikler
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
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

ax.plot(dates_test, y_test, 'o-', label='Gerçek Değer', linewidth=3, markersize=10, color='#2E86AB')
ax.plot(dates_test, y_pred, 's--', label='LSTM Tahmini', linewidth=3, markersize=10, color='#E63946')

ax.set_title(
    f"LSTM MODEL - TEST SONUÇLARI\n"
    f"R² = {r2:.4f} | Doğruluk = %{accuracy:.2f} | MAE = {mae:.2f}",
    fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Tarih', fontsize=13, fontweight='bold')
ax.set_ylabel('Su Tüketimi (W-Water)', fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.4, linestyle='--')
ax.tick_params(axis='x', rotation=45, labelsize=10)

# Performans kutusu
textstr = f'Performans:\nR² Score: {r2:.4f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: %{mape * 100:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('lstm_test_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Test tahminleri: lstm_test_predictions.png")
plt.close()

print("\n" + "=" * 100)
print("ANALİZ TAMAMLANDI!".center(100))
print("=" * 100)
print("\n📁 Oluşturulan Dosya:")
print("   • lstm_test_predictions.png     - Test sonuçları grafiği")
print("\n" + "=" * 100)
print(f"🥇 LSTM MODEL PERFORMANSI:")
print(f"   • Özellikler: {', '.join(top_3_features)}")
print(f"   • R² Score: {r2:.4f}")
print(f"   • Doğruluk: %{accuracy:.2f}")
print(f"   • MAE: {mae:.2f}")
print(f"   • RMSE: {rmse:.2f}")
print("=" * 100)