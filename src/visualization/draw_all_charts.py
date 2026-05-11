import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
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

# Chuẩn hóa cho KNN và Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ratio = float((y_train == 0).sum()) / (y_train == 1).sum()

# Danh sách 4 mô hình
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random_Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', use_label_encoder=False, random_state=42)
}

# 2. VẼ MA TRẬN NHẦM LẪN CHO TỪNG THUẬT TOÁN
for name, model in models.items():
    print(f"Đang tạo biểu đồ cho: {name}...")
    
    # Huấn luyện và dự đoán với dữ liệu phù hợp
    if name in ["Logistic_Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    # Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Không Mua', 'Mua'], 
                yticklabels=['Không Mua', 'Mua'])
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.title(f'Ma trận nhầm lẫn - {name.replace("_", " ")}')
    plt.tight_layout()
    
    # Tự động lưu file theo tên thuật toán
    plt.savefig(OUTPUT_DIR / f'confusion_matrix_{name.lower()}.png')
    plt.close() # Đóng bộ nhớ đồ họa để vẽ hình tiếp theo

print("✅ Đã xuất xong 4 biểu đồ Ma trận nhầm lẫn!")
