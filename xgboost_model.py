import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("XGBoost MODEL - SU TÜKETİMİ TAHMİNİ (TOP 3 ÖZELLİK)".center(100))
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

top_3_features = correlations.drop('W-Water').head(3).index.tolist()
print(f"\n✓ SEÇİLEN 3 ÖZELLİK: {top_3_features}")

# ========== VERİ HAZIRLAMA ==========
print("\n" + "=" * 100)
print("VERİ HAZIRLAMA")
print("=" * 100)

X = df[top_3_features].values
y = df['W-Water'].values
dates = df['date'].values

# Normalizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80/20)
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

print(f"✓ Eğitim boyutu: {len(X_train)} ay")
print(f"✓ Test boyutu: {len(X_test)} ay")

# ========== XGBoost MODEL EĞİTİMİ ==========
print("\n" + "=" * 100)
print("XGBoost MODEL EĞİTİMİ")
print("=" * 100)

params = {
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

print("\n⏳ Model eğitiliyor...")
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
print("✓ Eğitim tamamlandı!")

# ========== TEST PERFORMANSI ==========
print("\n" + "=" * 100)
print("TEST PERFORMANSI")
print("=" * 100)

y_pred = model.predict(X_test)

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

# ========== FEATURE IMPORTANCE ==========
print("\n" + "=" * 100)
print("ÖZELLİK ÖNEMLERİ")
print("=" * 100)

importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': top_3_features,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n🏆 3 ÖZELLİĞİN ÖNEMİ:")
for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

# ========== GÖRSELLEŞTİRME ==========
print("\n" + "=" * 100)
print("GRAFİKLER OLUŞTURULUYOR")
print("=" * 100)

# Test Tahminleri
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.plot(dates_test, y_test, 'o-', label='Gerçek Değer', linewidth=3, markersize=10, color='#2E86AB')
ax.plot(dates_test, y_pred, 's--', label='XGBoost Tahmini', linewidth=3, markersize=10, color='#F4A460')

ax.set_title(
    f"XGBoost MODEL - TEST SONUÇLARI\n"
    f"R² = {r2:.4f} | Doğruluk = %{accuracy:.2f} | MAE = {mae:.2f}",
    fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Tarih', fontsize=13, fontweight='bold')
ax.set_ylabel('Su Tüketimi (W-Water)', fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.4, linestyle='--')
ax.tick_params(axis='x', rotation=45, labelsize=10)

textstr = f'Performans:\nR² Score: {r2:.4f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: %{mape * 100:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('xgboost_test_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Test tahminleri: xgboost_test_predictions.png")
plt.close()

print("\n" + "=" * 100)
print("ANALİZ TAMAMLANDI!".center(100))
print("=" * 100)
print("\n📁 Oluşturulan Dosya:")
print("   • xgboost_test_predictions.png    - Test sonuçları grafiği")
print("\n" + "=" * 100)
print(f"🥇 XGBoost MODEL PERFORMANSI:")
print(f"   • Özellikler: {', '.join(top_3_features)}")
print(f"   • R² Score: {r2:.4f}")
print(f"   • Doğruluk: %{accuracy:.2f}")
print(f"   • MAE: {mae:.2f}")
print(f"   • RMSE: {rmse:.2f}")
print("=" * 100)