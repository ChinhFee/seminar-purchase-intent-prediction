import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "shopping.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. CHUẨN BỊ DỮ LIỆU
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['Revenue'] = df['Revenue'].astype(int)

cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
if 'Revenue' in cat_cols: cat_cols.remove('Revenue')
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=['Revenue'])
y = df_encoded['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chuẩn hóa cho Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 2. VẼ FEATURE IMPORTANCE CHO RANDOM FOREST ---
print("Đang vẽ Feature Importance cho Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Lấy top 10 đặc trưng
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=rf_importances, y=rf_importances.index, palette='viridis')
plt.title('Top 10 Đặc trưng quan trọng nhất - Random Forest')
plt.xlabel('Mức độ quan trọng (Feature Importance)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '4_feature_importance_random_forest.png')
plt.close()

# --- 3. VẼ FEATURE IMPORTANCE CHO LOGISTIC REGRESSION ---
print("Đang vẽ Feature Importance cho Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

# Với Logistic Regression, dùng giá trị tuyệt đối của các hệ số (coef_)
lr_importances = pd.Series(np.abs(lr.coef_), index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=lr_importances, y=lr_importances.index, palette='magma')
plt.title('Top 10 Đặc trưng quan trọng nhất - Logistic Regression')
plt.xlabel('Mức độ quan trọng (Trị tuyệt đối hệ số)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '4_feature_importance_logistic_regression.png')
plt.close()

print("✅ Đã xuất xong 2 biểu đồ Feature Importance!")
