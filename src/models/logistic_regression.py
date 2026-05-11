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
from sklearn.linear_model import LogisticRegression

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

def evaluate_and_save_all_logistic(model, X_train, y_train, X_test, y_test, feature_importances):
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
    print(f" ĐÁNH GIÁ CHI TIẾT MÔ HÌNH: LOGISTIC REGRESSION (BẢN ĐẸP)")
    print(f"{'='*65}")
    print("[1] HIỆU SUẤT DỰ BÁO TỔNG THỂ")
    print(f"    - Tổng mẫu Test: {len(y_test)} | Đoán đúng: {tp + tn} | Đoán sai: {fp + fn}")
    print(f"    - Độ chính xác (Accuracy): {acc*100:.2f}%")
    print(f"    - Recall / TPR           : {tpr*100:.2f}%")
    print(f"    - Độ đặc hiệu (TNR)      : {tnr*100:.2f}%")
    print(f"    - Điểm cân bằng F1-Score : {f1_test*100:.2f}%")
    
    print(f"\n[2] KIỂM TRA QUÁ KHỚP (OVERFITTING)")
    delta = abs(f1_train - f1_test) * 100
    status = "Cực kỳ ổn định" if delta < 5.0 else "Có dấu hiệu Overfitting"
    print(f"    - F1 Train: {f1_train*100:.2f}% | F1 Test: {f1_test*100:.2f}% -> Delta: {delta:.2f}% ({status})")
    print(f"\n[3] TỐC ĐỘ XỬ LÝ: {inference_time_ms:.2f} ms")

    print("\n[4] XUẤT FILE BIỂU ĐỒ (Đã fix lỗi đè chữ):")
    sns.set_theme(style="ticks", font_scale=1.1)
    
    # 1. Ma trận nhầm lẫn
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', linewidths=2, linecolor='white',
                annot_kws={"size": 18, "weight": "bold"}, xticklabels=['Không Mua (0)', 'Có Mua (1)'], yticklabels=['Không Mua (0)', 'Có Mua (1)'])
    plt.title('Ma Trận Nhầm Lẫn - Logistic Regression', fontsize=16, pad=20, fontweight='bold', color='#333333')
    plt.xlabel('Dự báo của mô hình', fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel('Dữ liệu thực tế', fontsize=13, fontweight='bold', labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_logistic_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Đường cong ROC
    plt.figure(figsize=(7, 6))
    fpr, tpr_roc, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr_roc)
    plt.plot(fpr, tpr_roc, color='#d62728', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.fill_between(fpr, tpr_roc, alpha=0.15, color='#d62728')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Đường Cong ROC - Logistic Regression', fontsize=16, pad=20, fontweight='bold', color='#333333')
    plt.xlabel('Tỷ lệ Âm tính giả (FPR)', fontsize=13, fontweight='bold')
    plt.ylabel('Tỷ lệ Dương tính thật (TPR)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, shadow=True, fancybox=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_logistic_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Đường cong Precision-Recall
    plt.figure(figsize=(7, 6))
    precision, recall_curve_vals, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    plt.plot(recall_curve_vals, precision, color='#1f77b4', lw=3, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.fill_between(recall_curve_vals, precision, alpha=0.15, color='#1f77b4')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Đường Cong Precision-Recall', fontsize=16, pad=20, fontweight='bold', color='#333333')
    plt.xlabel('Độ nhạy (Recall)', fontsize=13, fontweight='bold')
    plt.ylabel('Độ chính xác dự báo Dương (Precision)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower left", frameon=True, shadow=True, fancybox=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_logistic_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Feature Importance (ĐÃ FIX LỖI OVERLAP)
    plt.figure(figsize=(10, 6.5))
    importance_list = [(FEATURE_NAMES[i], feature_importances[i]) for i in range(len(FEATURE_NAMES))]
    importance_list.sort(key=lambda x: abs(x[1]), reverse=True)
    top_10 = importance_list[:10]
    
    names = [x[0] for x in top_10][::-1]
    vals = [x[1] for x in top_10][::-1]
    colors = ['#ff6347' if val < 0 else '#3cb371' for val in vals]

    bars = plt.barh(names, vals, color=colors, edgecolor='black', linewidth=0.5)
    plt.title('Top 10 Yếu Tố Tác Động - Logistic Regression', fontsize=16, pad=20, fontweight='bold', color='#333333')
    plt.xlabel('Trọng số hồi quy (Coefficients)', fontsize=13, fontweight='bold', labelpad=10)
    
    # CHIÊU CHỐT: Mở rộng trục X để tên biến và text số không tông vào nhau
    min_val = min(vals)
    max_val = max(vals)
    plt.xlim(min_val - 0.6, max_val + 0.6) 

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    sns.despine(left=True, bottom=True)
    
    # In số liệu lên biểu đồ
    for index, value in enumerate(vals):
        offset = 0.05 if value > 0 else -0.05
        ha = 'left' if value > 0 else 'right'
        plt.text(value + offset, index, f'{value:.3f}', va='center', ha=ha, 
                 fontsize=11, fontweight='bold', color='#333333',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_logistic_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [+] Đã lưu toàn bộ hình ảnh mới!")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/models/logistic_regression.py data/shopping.csv")

    X, y = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    evaluate_and_save_all_logistic(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, lr_model.coef_[0])

if __name__ == "__main__":
    main()
