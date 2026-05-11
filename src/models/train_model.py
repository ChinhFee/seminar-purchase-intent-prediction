import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
import xgboost as xgb
import warnings
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "shopping.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_xgb_model.pkl"

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# --- 1. XỬ LÝ DỮ LIỆU ---
def load_data(filepath):
    """Đọc và tiền xử lý dữ liệu: Làm sạch tên cột, mã hóa One-Hot"""
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip() # Xóa khoảng trắng thừa
        
        target_col = 'Revenue'
        # Chuyển đổi nhãn True/False sang 1/0
        if df[target_col].dtype == 'bool':
            df[target_col] = df[target_col].astype(int)
        
        # Tự động chọn và mã hóa các cột phân loại (categorical)
        cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        if target_col in cat_cols: cat_cols.remove(target_col)
        
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print("✅ Đã xử lý dữ liệu thành công!")
        return df_encoded, target_col
    except Exception as e:
        print(f"❌ Lỗi đọc dữ liệu: {e}")
        return None, None

# --- 2. HÀM ĐÁNH GIÁ MÔ HÌNH ---
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n>>> Đang chạy: {name}...")
    
    start = time.time()
    model.fit(X_train, y_train) # Huấn luyện
    y_pred = model.predict(X_test) # Dự báo
    end = time.time()
    
    # Tính toán các chỉ số
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"⏱️  Thời gian: {(end-start):.3f}s")
    print(f"📊 Kết quả: Accuracy={acc:.1%} | F1-Score={f1:.1%} | Recall={recall:.1%}")
    
    return [acc, f1, recall]

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    file_path = DATA_PATH
    df, target_col = load_data(file_path)
    
    if df is not None:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Chia dữ liệu 80/20 (Stratified để giữ tỷ lệ mất cân bằng)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # --- CHUẨN HÓA DỮ LIỆU (SCALING) ---
        # Bắt buộc cho KNN và giúp Logistic Regression hội tụ nhanh hơn
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Tính tỷ lệ mất cân bằng để hỗ trợ XGBoost
        # Tỷ lệ = Số lượng mẫu Âm / Số lượng mẫu Dương
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

        # Danh sách 4 thuật toán (Đã loại bỏ ANN)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "XGBoost": xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', use_label_encoder=False, random_state=42)
        }

        # Lưu kết quả
        summary = {}
        
        print(f"\n--- BẮT ĐẦU SO SÁNH 4 THUẬT TOÁN ---")
        for name, model in models.items():
            # Lưu ý: Logistic và KNN dùng dữ liệu đã Scaled (chuẩn hóa)
            if name in ["Logistic Regression", "KNN (K-Nearest Neighbors)"]:
                stats = evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test)
            else:
                # Random Forest và XGBoost dùng dữ liệu gốc (không cần chuẩn hóa)
                stats = evaluate_model(name, model, X_train, y_train, X_test, y_test)
            
            summary[name] = stats

        # --- BẢNG XẾP HẠNG ---
        print("\n🏆 BẢNG XẾP HẠNG F1-SCORE (Càng cao càng tốt):")
        # Sắp xếp dựa trên F1-Score (vị trí index 1 trong list [acc, f1, recall])
        # summary.items() -> (name, stats), stats = [acc, f1, recall]
        sorted_models = sorted(summary.items(), key=lambda item: item[1][1], reverse=True)
        for name, stats in sorted_models:
            print(f"{name}: {stats[1]:.2%} (Recall: {stats[2]:.2%})")
            
        # --- LƯU MÔ HÌNH XGBOOST TỐT NHẤT ---
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open('wb') as f:
            pickle.dump(models["XGBoost"], f) # Lưu model XGBoost vào file [1]
        print(f"\n💾 Đã lưu mô hình XGBoost thành file '{MODEL_PATH}'")
