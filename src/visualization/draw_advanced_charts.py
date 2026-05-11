import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
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

# --- 1. CHUẨN BỊ DỮ LIỆU ---
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['Revenue'] = df['Revenue'].astype(int)

cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
if 'Revenue' in cat_cols: cat_cols.remove('Revenue')
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=['Revenue'])
y = df_encoded['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ratio = float((y_train == 0).sum()) / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', use_label_encoder=False, random_state=42)
}

# --- 2. SETUP BIỂU ĐỒ CHUNG ---
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    print(f"Đang xử lý và vẽ: {name}...")
    
    # Chọn đúng dữ liệu (Đã chuẩn hóa cho Logistic và KNN)
    X_tr = X_train_scaled if name in ["Logistic Regression", "KNN"] else X_train
    X_te = X_test_scaled if name in ["Logistic Regression", "KNN"] else X_test
    
    # Huấn luyện và lấy xác suất dự báo (Probability) thay vì chỉ lấy nhãn 0/1
    model.fit(X_tr, y_train)
    y_prob = model.predict_proba(X_te)[:, 1] 
    
    # --- VẼ ROC CURVE (Gộp chung) ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # --- VẼ PRECISION-RECALL CURVE (Gộp chung) ---
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax_pr.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.2f})')

    # --- VẼ LEARNING CURVE (Tách riêng từng thuật toán) ---
    plt.figure(figsize=(6, 4))
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_tr, y_train, cv=5, scoring='f1', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Điểm Train (F1)")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Điểm Validation (F1)")
    plt.title(f'Learning Curve - {name}')
    plt.xlabel('Số lượng mẫu huấn luyện')
    plt.ylabel('F1-Score')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'learning_curve_{name.replace(" ", "_").lower()}.png')
    plt.close()

# --- 3. LƯU CÁC BIỂU ĐỒ CHUNG ---
ax_roc.plot([1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlabel('False Positive Rate (Báo nhầm)')
ax_roc.set_ylabel('True Positive Rate (Đoán trúng)')
ax_roc.set_title('Biểu đồ ROC (So sánh 4 thuật toán)')
ax_roc.legend(loc="lower right")
fig_roc.tight_layout()
fig_roc.savefig(OUTPUT_DIR / 'compare_roc_curve.png')

ax_pr.set_xlabel('Recall (Độ phủ)')
ax_pr.set_ylabel('Precision (Độ chính xác)')
ax_pr.set_title('Biểu đồ Precision-Recall (So sánh 4 thuật toán)')
ax_pr.legend(loc="lower left")
fig_pr.tight_layout()
fig_pr.savefig(OUTPUT_DIR / 'compare_pr_curve.png')

print("\n✅ Đã xuất xong tất cả biểu đồ chuyên sâu! Hãy kiểm tra thư mục của bạn.")
