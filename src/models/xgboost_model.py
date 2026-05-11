import csv
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def evaluate_and_save_all(model, X_train, y_train, X_test, y_test, feature_importances):
    """Hàm All-in-One: In báo cáo Text ra Terminal + Lưu 4 hình HD"""
    
    # ==========================================
    # PHẦN 1: TÍNH TOÁN VÀ IN BÁO CÁO TEXT
    # ==========================================
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time_ms = (time.time() - start_time) * 1000
    y_probs = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    acc = accuracy_score(y_test, predictions)
    f1_test = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    f1_train = f1_score(y_train, model.predict(X_train))
    
    print(f"\n{'='*65}")
    print(f" ĐÁNH GIÁ CHI TIẾT MÔ HÌNH: XGBOOST (ALL-IN-ONE)")
    print(f"{'='*65}")
    print("[1] HIỆU SUẤT DỰ BÁO TỔNG THỂ")
    print(f"    - Tổng mẫu Test: {len(y_test)} | Đoán đúng: {tp + tn} | Đoán sai: {fp + fn}")
    print(f"    - Độ chính xác (Accuracy): {acc*100:.2f}%")
    print(f"    - Recall / TPR           : {tpr*100:.2f}% -> (Bắt trúng khách CÓ MUA)")
    print(f"    - Độ đặc hiệu (TNR)      : {tnr*100:.2f}% -> (Lọc khách XEM DẠO)")
    print(f"    - Điểm cân bằng F1-Score : {f1_test*100:.2f}%")
    
    print(f"\n[2] KIỂM TRA QUÁ KHỚP (OVERFITTING)")
    delta = abs(f1_train - f1_test) * 100
    status = "Ổn định" if delta < 5.0 else "Có dấu hiệu Overfitting"
    print(f"    - F1 Train: {f1_train*100:.2f}% | F1 Test: {f1_test*100:.2f}% -> Delta: {delta:.2f}% ({status})")
    
    print(f"\n[3] TỐC ĐỘ XỬ LÝ: {inference_time_ms:.2f} ms")

    # ==========================================
    # PHẦN 2: LƯU TỰ ĐỘNG 4 BIỂU ĐỒ HD (300dpi)
    # ==========================================
    print("\n[4] XUẤT FILE BIỂU ĐỒ (Chuẩn HD 300dpi):")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Ma trận nhầm lẫn
    plt.figure(figsize=(9, 7)) 
    row_totals = cm.sum(axis=1)
    percentages = [
        cm[0,0]/row_totals[0]*100, cm[0,1]/row_totals[0]*100,
        cm[1,0]/row_totals[1]*100, cm[1,1]/row_totals[1]*100
    ]
    group_names = [
        'Lọc khách dạo\n(True Negative)', 
        'Dự báo nhầm có mua\n(False Positive)', 
        'Bỏ sót khách mua\n(False Negative)', 
        'Dự đoán đúng khách mua\n(True Positive)'
    ]
    group_counts = [f"{value} phiên" for value in cm.ravel()]
    group_percentages = [f"({pct:.1f}%)" for pct in percentages]

    labels = [f"{v1}\n\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', linewidths=2, linecolor='white',
                annot_kws={"size": 14, "weight": "bold"}, 
                xticklabels=['Không Mua (0)', 'Có Mua (1)'], 
                yticklabels=['Không Mua (0)', 'Có Mua (1)'])
                
    plt.title('Confusion Matrix', fontsize=16, pad=20, fontweight='bold', color='#333333')
    plt.xlabel('Dự báo của hệ thống', fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel('Hành vi của khách hàng', fontsize=14, fontweight='bold', labelpad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_xgboost_confusion_matrix.png', dpi=300)
    plt.close()
    print("   [+] Đã lưu: 1_xgboost_confusion_matrix.png")

    # 2. Đường cong ROC
    plt.figure(figsize=(7, 6))
    fpr, tpr_roc, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr_roc)
    plt.plot(fpr, tpr_roc, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Đường cong ROC - XGBoost', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Tỷ lệ Âm tính giả (FPR)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính thật (TPR)', fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_xgboost_roc_curve.png', dpi=300)
    plt.close()
    print("   [+] Đã lưu: 2_xgboost_roc_curve.png")

    # 3. Đường cong Precision-Recall
    plt.figure(figsize=(7, 6))
    precision, recall_curve_vals, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    plt.plot(recall_curve_vals, precision, color='purple', lw=2.5, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Đường cong Precision-Recall - XGBoost', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Độ nhạy (Recall)', fontsize=12)
    plt.ylabel('Độ chính xác dự báo Dương (Precision)', fontsize=12)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_xgboost_pr_curve.png', dpi=300)
    plt.close()
    print("   [+] Đã lưu: 3_xgboost_pr_curve.png")

    # 4. Feature Importance
    plt.figure(figsize=(9, 6))
    importance_list = [(FEATURE_NAMES[i], feature_importances[i]) for i in range(len(FEATURE_NAMES))]
    importance_list.sort(key=lambda x: x[1], reverse=True)
    top_10 = importance_list[:10]
    
    names = [x[0] for x in top_10][::-1]
    vals = [x[1] for x in top_10][::-1]

    bars = plt.barh(names, vals, color='teal')
    plt.title('Top 10 Yếu tố quan trọng nhất - XGBoost', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Mức độ quan trọng (Feature Importance)', fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_xgboost_feature_importance.png', dpi=300)
    plt.close()
    print("   [+] Đã lưu: 4_xgboost_feature_importance.png\n")
    print("=> Hoàn tất 100%! Ông copy text phía trên dán vào báo cáo và lấy 4 hình chèn vào Word nhé!")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/models/xgboost_model.py data/shopping.csv")

    X, y = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    evaluate_and_save_all(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, xgb_model.feature_importances_)

if __name__ == "__main__":
    main()
