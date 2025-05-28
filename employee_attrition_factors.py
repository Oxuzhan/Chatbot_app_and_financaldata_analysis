import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

# Görselleştirme için stil ayarları
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8')  # seaborn stilini güncel versiyona uygun olarak değiştirdim

# Veri setini okuma
df = pd.read_csv('data.csv')

# Attrition sütununu sayısal değerlere dönüştür
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print("="*50)
print("ÇALIŞAN KAYBI ETKİLEYEN TEMEL FAKTÖRLER ANALİZİ")
print("="*50)

# 1. İş Tatmini ve Çalışan Kaybı İlişkisi
plt.figure(figsize=(15, 10))

# İş Tatmini dağılımı
plt.subplot(2, 2, 1)
sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
plt.title('İş Tatmini ve Çalışan Kaybı İlişkisi')
plt.xlabel('Çalışan Kaybı')
plt.ylabel('İş Tatmini')

# Çalışma-Yaşam Dengesi
plt.subplot(2, 2, 2)
sns.boxplot(x='Attrition', y='WorkLifeBalance', data=df)
plt.title('Çalışma-Yaşam Dengesi ve Çalışan Kaybı İlişkisi')
plt.xlabel('Çalışan Kaybı')
plt.ylabel('Çalışma-Yaşam Dengesi')

# Maaş Artışı
plt.subplot(2, 2, 3)
sns.boxplot(x='Attrition', y='PercentSalaryHike', data=df)
plt.title('Maaş Artışı ve Çalışan Kaybı İlişkisi')
plt.xlabel('Çalışan Kaybı')
plt.ylabel('Maaş Artış Oranı (%)')

# Şirkette Geçirilen Süre
plt.subplot(2, 2, 4)
sns.boxplot(x='Attrition', y='YearsAtCompany', data=df)
plt.title('Şirkette Geçirilen Süre ve Çalışan Kaybı İlişkisi')
plt.xlabel('Çalışan Kaybı')
plt.ylabel('Şirkette Geçirilen Yıl')

plt.tight_layout()
plt.savefig('temel_faktorler_1.png')
plt.close()

# 2. Kategorik Değişkenler Analizi
plt.figure(figsize=(15, 10))

# Departman
plt.subplot(2, 2, 1)
dept_attrition = pd.crosstab(df['Department'], df['Attrition'], normalize='index') * 100
dept_attrition.plot(kind='bar', stacked=True)
plt.title('Departman ve Çalışan Kaybı İlişkisi')
plt.xlabel('Departman')
plt.ylabel('Oran (%)')
plt.xticks(rotation=45)

# İş Rolü
plt.subplot(2, 2, 2)
role_attrition = pd.crosstab(df['JobRole'], df['Attrition'], normalize='index') * 100
role_attrition.plot(kind='bar', stacked=True)
plt.title('İş Rolü ve Çalışan Kaybı İlişkisi')
plt.xlabel('İş Rolü')
plt.ylabel('Oran (%)')
plt.xticks(rotation=45)

# Eğitim Seviyesi
plt.subplot(2, 2, 3)
edu_attrition = pd.crosstab(df['Education'], df['Attrition'], normalize='index') * 100
edu_attrition.plot(kind='bar', stacked=True)
plt.title('Eğitim Seviyesi ve Çalışan Kaybı İlişkisi')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Oran (%)')

# Evlilik Durumu
plt.subplot(2, 2, 4)
marital_attrition = pd.crosstab(df['MaritalStatus'], df['Attrition'], normalize='index') * 100
marital_attrition.plot(kind='bar', stacked=True)
plt.title('Evlilik Durumu ve Çalışan Kaybı İlişkisi')
plt.xlabel('Evlilik Durumu')
plt.ylabel('Oran (%)')

plt.tight_layout()
plt.savefig('temel_faktorler_2.png')
plt.close()

# 3. İstatistiksel Analiz ve Korelasyonlar
numeric_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Korelasyon analizi
correlation_matrix = df[numeric_cols + ['Attrition']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[['Attrition']].sort_values('Attrition', ascending=False),
            annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Çalışan Kaybı ile Sayısal Değişkenler Arasındaki Korelasyonlar')
plt.tight_layout()
plt.savefig('korelasyonlar.png')
plt.close()

# 4. Detaylı İstatistiksel Analiz
print("\nÇALIŞAN KAYBI ETKİLEYEN TEMEL FAKTÖRLER - İSTATİSTİKSEL ANALİZ")
print("-"*50)

# İş Tatmini Analizi
job_sat_yes = df[df['Attrition'] == 1]['JobSatisfaction']
job_sat_no = df[df['Attrition'] == 0]['JobSatisfaction']
t_stat, p_value = stats.ttest_ind(job_sat_yes, job_sat_no)
print(f"\n1. İş Tatmini:")
print(f"İşten Ayrılanların Ortalama İş Tatmini: {job_sat_yes.mean():.2f}")
print(f"Kalanların Ortalama İş Tatmini: {job_sat_no.mean():.2f}")
print(f"İstatistiksel Anlamlılık (p-değeri): {p_value:.4f}")

# Maaş Analizi
salary_yes = df[df['Attrition'] == 1]['MonthlyIncome']
salary_no = df[df['Attrition'] == 0]['MonthlyIncome']
t_stat, p_value = stats.ttest_ind(salary_yes, salary_no)
print(f"\n2. Aylık Maaş:")
print(f"İşten Ayrılanların Ortalama Maaşı: {salary_yes.mean():.2f}")
print(f"Kalanların Ortalama Maaşı: {salary_no.mean():.2f}")
print(f"İstatistiksel Anlamlılık (p-değeri): {p_value:.4f}")

# Şirkette Geçirilen Süre Analizi
years_yes = df[df['Attrition'] == 1]['YearsAtCompany']
years_no = df[df['Attrition'] == 0]['YearsAtCompany']
t_stat, p_value = stats.ttest_ind(years_yes, years_no)
print(f"\n3. Şirkette Geçirilen Süre:")
print(f"İşten Ayrılanların Ortalama Şirket Süresi: {years_yes.mean():.2f} yıl")
print(f"Kalanların Ortalama Şirket Süresi: {years_no.mean():.2f} yıl")
print(f"İstatistiksel Anlamlılık (p-değeri): {p_value:.4f}")

# Çalışma-Yaşam Dengesi Analizi
wlb_yes = df[df['Attrition'] == 1]['WorkLifeBalance']
wlb_no = df[df['Attrition'] == 0]['WorkLifeBalance']
t_stat, p_value = stats.ttest_ind(wlb_yes, wlb_no)
print(f"\n4. Çalışma-Yaşam Dengesi:")
print(f"İşten Ayrılanların Ortalama Denge Skoru: {wlb_yes.mean():.2f}")
print(f"Kalanların Ortalama Denge Skoru: {wlb_no.mean():.2f}")
print(f"İstatistiksel Anlamlılık (p-değeri): {p_value:.4f}")

# Departman Bazlı Analiz
print("\n5. Departman Bazlı Çalışan Kaybı Oranları:")
dept_analysis = pd.crosstab(df['Department'], df['Attrition'], normalize='index') * 100
dept_analysis.columns = ['Kalanlar (%)', 'Ayrılanlar (%)']
print(dept_analysis)

# İş Rolü Bazlı Analiz
print("\n6. İş Rolü Bazlı Çalışan Kaybı Oranları:")
role_analysis = pd.crosstab(df['JobRole'], df['Attrition'], normalize='index') * 100
role_analysis.columns = ['Kalanlar (%)', 'Ayrılanlar (%)']
print(role_analysis)

# Önemli Bulgular ve Öneriler
print("\nÖNEMLİ BULGULAR VE ÖNERİLER")
print("-"*50)
print("""
1. İş Tatmini:
   - İş tatmini düşük olan çalışanlarda ayrılma oranı daha yüksektir
   - Öneri: Düzenli iş tatmini anketleri ve geri bildirim mekanizmaları kurulmalı

2. Maaş ve Kariyer Gelişimi:
   - Maaş artışı ve kariyer gelişimi fırsatları çalışan bağlılığını etkilemektedir
   - Öneri: Şeffaf maaş politikaları ve kariyer gelişim planları oluşturulmalı

3. Çalışma-Yaşam Dengesi:
   - Denge skoru düşük olan çalışanlarda ayrılma eğilimi daha yüksektir
   - Öneri: Esnek çalışma saatleri ve uzaktan çalışma seçenekleri sunulmalı

4. Departman ve İş Rolü:
   - Bazı departman ve iş rollerinde ayrılma oranları daha yüksektir
   - Öneri: Yüksek ayrılma oranı olan departmanlarda özel retansiyon stratejileri geliştirilmeli

5. Şirkette Geçirilen Süre:
   - Yeni çalışanlarda ayrılma oranı daha yüksektir
   - Öneri: Yeni çalışan oryantasyon programları güçlendirilmeli
""") 