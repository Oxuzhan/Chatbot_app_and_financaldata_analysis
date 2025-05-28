import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

# Veri setini okuma
df = pd.read_csv('data.csv')

print("="*50)
print("YAPISAL VERİ ARTIRMA (STRUCTURAL DATA AUGMENTATION)")
print("="*50)

# Hedef değişkeni ayırma
y = df['Attrition'].map({'No': 0, 'Yes': 1})
X = df.drop(['Attrition', 'EmployeeNumber'], axis=1)

# Kategorik değişkenleri one-hot encode et
X_encoded = pd.get_dummies(X, drop_first=True)

# Yapısal veri artırma fonksiyonları
def feature_swap(X, y, n_new=200):
    """Özellik değişimi ile yeni örnekler oluştur"""
    X_minority = X[y == 1]
    new_samples = []
    for _ in range(n_new):
        rows = X_minority.sample(2, random_state=np.random.randint(0, 10000))
        new_row = rows.iloc[0].copy()
        swap_cols = np.random.choice(X.columns, size=int(len(X.columns)/3), replace=False)
        for col in swap_cols:
            new_row[col] = rows.iloc[1][col]
        new_samples.append(new_row)
    return pd.DataFrame(new_samples)

def feature_mix(X, y, n_new=200):
    """Özellik karıştırma ile yeni örnekler oluştur"""
    X_minority = X[y == 1]
    new_samples = []
    for _ in range(n_new):
        rows = X_minority.sample(3, random_state=np.random.randint(0, 10000))
        new_row = rows.iloc[0].copy()
        for col in X.columns:
            weights = np.random.dirichlet(np.ones(3))
            new_row[col] = np.sum([rows.iloc[i][col] * weights[i] for i in range(3)])
        new_samples.append(new_row)
    return pd.DataFrame(new_samples)

def feature_noise(X, y, n_new=200, noise_level=0.1):
    """Gürültü ekleyerek yeni örnekler oluştur"""
    X_minority = X[y == 1]
    new_samples = []
    for _ in range(n_new):
        row = X_minority.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]
        noise = np.random.normal(0, noise_level, size=len(X.columns))
        new_row = row + noise
        new_samples.append(new_row)
    return pd.DataFrame(new_samples)

# SMOTE ile artırılmış veri
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_smote, y_smote = smote.fit_resample(X_encoded, y)

# Yapısal veri artırma uygula
X_smote_df = pd.DataFrame(X_smote, columns=X_encoded.columns)
X_swap = feature_swap(X_smote_df, y_smote, n_new=200)
X_mix = feature_mix(X_smote_df, y_smote, n_new=200)
X_noise = feature_noise(X_smote_df, y_smote, n_new=200)

# Tüm artırılmış verileri birleştir
X_aug = pd.concat([X_smote_df, X_swap, X_mix, X_noise], ignore_index=True)
y_aug = pd.concat([pd.Series(y_smote), pd.Series([1]*600)], ignore_index=True)

# Özellik mühendisliği
X_aug['WorkLifeBalanceScore'] = X_aug['WorkLifeBalance'] * X_aug['JobSatisfaction']
X_aug['TotalSatisfaction'] = X_aug['EnvironmentSatisfaction'] + X_aug['JobSatisfaction'] + X_aug['RelationshipSatisfaction']
X_aug['SalaryHikeRatio'] = X_aug['PercentSalaryHike'] / X_aug['MonthlyIncome']
X_aug['ExperienceLevel'] = X_aug['TotalWorkingYears'] / X_aug['Age']
X_aug['TenureRatio'] = X_aug['YearsAtCompany'] / X_aug['TotalWorkingYears']
X_aug['PromotionSpeed'] = X_aug['YearsAtCompany'] / (X_aug['YearsSinceLastPromotion'] + 1)

# Özellik listelerini güncelle
numerical_features = X_aug.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_aug.select_dtypes(include=['object']).columns.tolist()

# Veri ön işleme pipeline'ları
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Eksik değerleri medyan ile doldur
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Eksik değerleri en sık değer ile doldur
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Preprocessor tanımla
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug)

# Model pipeline'ları
models = {
    'Logistic Regression L1': ImbPipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            penalty='l1',
            solver='liblinear',
            class_weight={0: 1, 1: 7},  # Sınıf 1'e daha fazla ağırlık ver
            random_state=42,
            max_iter=2000
        ))
    ]),
    'Logistic Regression L2': ImbPipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            penalty='l2',
            solver='liblinear',
            class_weight={0: 1, 1: 7},  # Sınıf 1'e daha fazla ağırlık ver
            random_state=42,
            max_iter=2000
        ))
    ]),
    'Random Forest': ImbPipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=500,  # Ağaç sayısını artır
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight={0: 1, 1: 7},  # Sınıf 1'e daha fazla ağırlık ver
            random_state=42
        ))
    ])
}

# Hiperparametre ızgaraları
param_grids = {
    'Logistic Regression L1': {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__class_weight': [{0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 10}]
    },
    'Logistic Regression L2': {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__class_weight': [{0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 10}]
    },
    'Random Forest': {
        'classifier__n_estimators': [300, 400, 500],
        'classifier__max_depth': [10, 15, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__class_weight': [{0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 10}]
    }
}

# Model eğitimi ve değerlendirme
results = {}
best_model = None
best_f1 = 0

# Performans metriklerini saklamak için listeler
model_names = []
f1_scores = []
precision_scores = []
recall_scores = []
roc_auc_scores = []

for name, model in models.items():
    print(f"\n{name} Modeli Hiperparametre Optimizasyonu ve Değerlendirmesi")
    print("-" * 50)
    
    grid = GridSearchCV(
        model,
        param_grids[name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    # En iyi modeli al
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Performans metriklerini hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Metrikleri listelere ekle
    model_names.append(name)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    roc_auc_scores.append(roc_auc)
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Best Parameters': grid.best_params_
    }
    
    # Sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen Değer')
    plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    
    # ROC eğrisi
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Eğrisi')
    plt.legend()
    plt.savefig(f'{name.lower().replace(" ", "_")}_roc_curve.png')
    plt.close()
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = best_model

# Performans metriklerini görselleştir
plt.figure(figsize=(15, 10))

# F1 Skorları
plt.subplot(2, 2, 1)
plt.bar(model_names, f1_scores, color='skyblue')
plt.title('F1 Skorları')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# Precision Skorları
plt.subplot(2, 2, 2)
plt.bar(model_names, precision_scores, color='lightgreen')
plt.title('Precision Skorları')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# Recall Skorları
plt.subplot(2, 2, 3)
plt.bar(model_names, recall_scores, color='salmon')
plt.title('Recall Skorları')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# ROC AUC Skorları
plt.subplot(2, 2, 4)
plt.bar(model_names, roc_auc_scores, color='orange')
plt.title('ROC AUC Skorları')
plt.xticks(rotation=45)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.close()

# Sonuçları karşılaştırma
results_df = pd.DataFrame(results).T
print("\nModel Performans Karşılaştırması:")
print(results_df)

# En iyi modeli belirleme
best_model_name = results_df['F1 Score'].idxmax()
print(f"\nEn İyi Model: {best_model_name}")
print(f"En İyi F1 Skoru: {results_df.loc[best_model_name, 'F1 Score']:.3f}")

# Özellik önemliliklerini görselleştirme (Random Forest için)
if 'Random Forest' in models:
    rf_model = models['Random Forest'].named_steps['classifier']
    feature_importances = pd.DataFrame({
        'feature': numerical_features + [f"{col}_{val}" for col, vals in 
                                       zip(categorical_features, 
                                           models['Random Forest'].named_steps['preprocessor']
                                           .named_transformers_['cat'].named_steps['onehot']
                                           .categories_) 
                                       for val in vals],
        'importance': rf_model.feature_importances_
    })
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
    plt.title('En Önemli 10 Özellik (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nEn Önemli 10 Özellik:")
    print(feature_importances.head(10)) 