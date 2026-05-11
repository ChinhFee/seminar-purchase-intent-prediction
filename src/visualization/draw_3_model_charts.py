import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random_Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

# 2. VẼ VÀ XUẤT 9 FILE ẢNH
for name, model in models.items():
    print(f"Đang vẽ 3 biểu đồ cho: {name}...")
    X_tr, X_te = (X_train_scaled, X_test_scaled) if name in ["Logistic_Regression", "KNN"] else (X_train, X_test)
        
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    # 1. Ma trận nhầm lẫn
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Không Mua', 'Mua'], yticklabels=['Không Mua', 'Mua'])
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.title(f'Ma trận nhầm lẫn - {name}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name.lower()}_1_confusion_matrix.png')
    plt.close()
    
    # 2. Biểu đồ ROC
    plt.figure(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Biểu đồ ROC - {name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name.lower()}_2_roc.png')
    plt.close()
    
    # 3. Biểu đồ Precision-Recall
    plt.figure(figsize=(5, 4))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, color='purple', lw=2, label=f'AP = {average_precision_score(y_test, y_prob):.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Biểu đồ PR - {name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name.lower()}_3_pr.png')
    plt.close()

print("✅ Đã tạo xong 9 file ảnh riêng biệt!")
