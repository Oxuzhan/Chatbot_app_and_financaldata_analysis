import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Veri setini okuma
print("Veri seti yükleniyor...")
df = pd.read_csv('data.csv')

print("\nVeri Seti Bilgileri:")
print(f"Boyut: {df.shape}")
print(f"Eksik Veri Sayısı: {df.isnull().sum().sum()}")

# Sayısal değişkenleri seçme
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
numeric_columns = [col for col in numeric_columns if col not in ['Attrition', 'EmployeeNumber']]

print(f"\nAnaliz Edilecek Sayısal Değişken Sayısı: {len(numeric_columns)}")

# Aykırı veri analizi için fonksiyon
def outlier_analysis(df, columns):
    outlier_stats = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_percentage = (len(outliers) / len(df)) * 100
        
        outlier_stats[col] = {
            'Alt Sınır': lower_bound,
            'Üst Sınır': upper_bound,
            'Aykırı Veri Sayısı': len(outliers),
            'Aykırı Veri Yüzdesi': outlier_percentage,
            'Min Değer': df[col].min(),
            'Max Değer': df[col].max(),
            'Ortalama': df[col].mean(),
            'Medyan': df[col].median(),
            'Standart Sapma': df[col].std()
        }
    
    return pd.DataFrame(outlier_stats).T

# Aykırı veri istatistiklerini hesaplama
print("\nAykırı veri analizi yapılıyor...")
outlier_stats_df = outlier_analysis(df, numeric_columns)

# Sonuçları yazdırma
print("\nAykırı Veri İstatistikleri:")
print(outlier_stats_df.round(2))

# En çok aykırı veri içeren değişkenleri listele
print("\nEn Çok Aykırı Veri İçeren Değişkenler (Top 5):")
top_outliers = outlier_stats_df.sort_values('Aykırı Veri Yüzdesi', ascending=False).head()
print(top_outliers[['Aykırı Veri Sayısı', 'Aykırı Veri Yüzdesi']].round(2))

# Görselleştirme
print("\nGörselleştirmeler oluşturuluyor...")

# Box plot'lar
n_cols = 3
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numeric_columns):
    # Box plot
    sns.boxplot(data=df, y=col, ax=axes[idx])
    axes[idx].set_title(f'{col} Box Plot')
    axes[idx].set_ylabel('')
    
    # Aykırı veri sayısını başlığa ekle
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
    axes[idx].set_title(f'{col}\nAykırı Veri: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)')

# Kullanılmayan subplot'ları kaldır
for idx in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('outlier_analysis_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# Violin plot'lar
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numeric_columns):
    # Violin plot
    sns.violinplot(data=df, y=col, ax=axes[idx])
    axes[idx].set_title(f'{col} Violin Plot')
    axes[idx].set_ylabel('')
    
    # Aykırı veri sayısını başlığa ekle
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
    axes[idx].set_title(f'{col}\nAykırı Veri: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)')

# Kullanılmayan subplot'ları kaldır
for idx in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('outlier_analysis_violinplots.png', dpi=300, bbox_inches='tight')
plt.close()

# Sonuçları CSV'ye kaydet
outlier_stats_df.to_csv('outlier_analysis_results.csv')
print("\nAykırı veri analizi sonuçları 'outlier_analysis_results.csv' dosyasına kaydedildi.")
print("Görselleştirmeler 'outlier_analysis_boxplots.png' ve 'outlier_analysis_violinplots.png' dosyalarına kaydedildi.") 