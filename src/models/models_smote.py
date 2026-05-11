import csv
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

FEATURE_NAMES = [
    "Administrative", "Administrative_Duration", "Informational", 
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration", 
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", 
    "Month", "OperatingSystems", "Browser", "Region", 
    "TrafficType", "VisitorType", "Weekend"
]

def load_data(filename):
    evidence, labels = [], []
    month_mapping = {"Jan":0,"Feb":1,"Mar":2,"Apr":3,"May":4,"June":5,"Jul":6,"Aug":7,"Sep":8,"Oct":9,"Nov":10,"Dec":11}
    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row["Administrative"]), float(row["Administrative_Duration"]),
                int(row["Informational"]), float(row["Informational_Duration"]),
                int(row["ProductRelated"]), float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]), float(row["ExitRates"]),
                float(row["PageValues"]), float(row["SpecialDay"]),
                month_mapping[row["Month"]], int(row["OperatingSystems"]),
                int(row["Browser"]), int(row["Region"]), int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0,
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)
    return np.array(evidence), np.array(labels)

def evaluate_detailed(model_name, model, X_train, y_train, X_test, y_test, feature_importances=None):
    """Hàm in Báo cáo Đánh giá 5 Tiêu chí cho TỪNG thuật toán"""
    print(f"\n{'='*65}")
    print(f" ĐÁNH GIÁ CHI TIẾT MÔ HÌNH: {model_name.upper()}")
    print(f"{'='*65}")
    
    # Inference Speed
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time_ms = (time.time() - start_time) * 1000
    
    # Ma trận
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    acc = accuracy_score(y_test, predictions)
    f1_test = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    # Overfitting
    train_preds = model.predict(X_train)
    f1_train = f1_score(y_train, train_preds)
    
    print("[1] HIỆU SUẤT DỰ BÁO TỔNG THỂ")
    print(f"    - Tổng mẫu Test: {len(y_test)} | Đoán đúng: {tp + tn} | Đoán sai: {fp + fn}")
    print(f"    - Độ chính xác (Accuracy): {acc*100:.2f}%")
    print(f"    - Độ nhạy / TPR (Recall) : {tpr*100:.2f}% -> (Tỷ lệ bắt trúng khách CÓ MUA)")
    print(f"    - Độ đặc hiệu (TNR)      : {tnr*100:.2f}% -> (Tỷ lệ lọc nhiễu khách XEM DẠO)")
    print(f"    - Điểm cân bằng F1-Score : {f1_test*100:.2f}%")
    
    print(f"\n[2] KIỂM TRA QUÁ KHỚP (OVERFITTING)")
    print(f"    - F1-Score (Tập Train) : {f1_train*100:.2f}%")
    print(f"    - F1-Score (Tập Test)  : {f1_test*100:.2f}%")
    delta = abs(f1_train - f1_test) * 100
    status = "Ổn định, tổng quát hóa tốt" if delta < 5.0 else "Có dấu hiệu Overfitting"
    print(f"    - Độ chênh lệch (Delta): {delta:.2f}% -> ({status})")
    
    print(f"\n[3] TỐC ĐỘ XỬ LÝ (INFERENCE SPEED)")
    print(f"    - Tổng thời gian dự báo {len(y_test)} mẫu: {inference_time_ms:.2f} ms")
    
    print(f"\n[4] KHẢ NĂNG GIẢI THÍCH (INTERPRETABILITY)")
    if feature_importances is not None:
        importance_list = [(FEATURE_NAMES[i], feature_importances[i]) for i in range(len(FEATURE_NAMES))]
        importance_list.sort(key=lambda x: abs(x[1]), reverse=True)
        print("    - Top 3 yếu tố tác động mạnh nhất đến quyết định mua:")
        for i in range(3):
            feat, weight = importance_list[i]
            if model_name == "Logistic Regression":
                direction = "(+ Tác động Tích cực)" if weight > 0 else "(- Tác động Tiêu cực)"
            else:
                direction = "(Mức độ quan trọng)"
            print(f"      * {feat}: {weight:.4f} {direction}")
    else:
        print("    - Thuật toán dựa trên khoảng cách (Black-box), không trích xuất được trọng số biến.")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python models_detailed.py shopping.csv")

    print("--- BẮT ĐẦU ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU ---")
    X, y = load_data(sys.argv[1])
    
    # Phân chia dữ liệu 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print("-> Đã chia 80/20, chuẩn hóa Z-score và áp dụng SMOTE thành công!\n")
    print("--- ĐANG TIẾN HÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ 4 MÔ HÌNH ---")
    
    # XGBoost
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    evaluate_detailed("XGBoost", xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, feature_importances=xgb_model.feature_importances_)
    # Logistic Regression
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    evaluate_detailed("Logistic Regression", lr_model, X_train_scaled, y_train, X_test_scaled, y_test, feature_importances=lr_model.coef_[0])
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_smote, y_train_smote)
    evaluate_detailed("Random Forest", rf_model, X_train_smote, y_train_smote, X_test_scaled, y_test, feature_importances=rf_model.feature_importances_)
    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_smote, y_train_smote)
    evaluate_detailed("K-Nearest Neighbors (K-NN)", knn_model, X_train_smote, y_train_smote, X_test_scaled, y_test, feature_importances=None)
if __name__ == "__main__":
    main()