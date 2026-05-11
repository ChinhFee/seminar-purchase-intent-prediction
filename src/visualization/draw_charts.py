import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "shopping.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_xgb_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. TẢI DỮ LIỆU ĐỂ LẤY TẬP TEST
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['Revenue'] = df['Revenue'].astype(int)
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
if 'Revenue' in cat_cols: cat_cols.remove('Revenue')
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=['Revenue'])
y = df_encoded['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. TẢI MÔ HÌNH ĐÃ TRAIN TỪ FILE TRƯỚC ĐÓ
print("Đang tải mô hình đã lưu...")
with MODEL_PATH.open('rb') as f:
    xgb_model = pickle.load(f) # Đọc model không cần train lại [1]

# Dự đoán luôn bằng model vừa tải
y_pred = xgb_model.predict(X_test)

# --- BIỂU ĐỒ 2: CONFUSION MATRIX ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Không Mua', 'Mua'], yticklabels=['Không Mua', 'Mua'])
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.title('Ma trận nhầm lẫn - XGBoost')
plt.savefig(OUTPUT_DIR / '2_confusion_matrix.png')

# --- BIỂU ĐỒ 3: FEATURE IMPORTANCE ---
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10, height=0.8, color='#C44E52')
plt.title('Top 10 Yếu tố ảnh hưởng đến quyết định mua hàng')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3_feature_importance.png')

print("✅ Đã tạo biểu đồ thành công từ mô hình lưu trữ!")
