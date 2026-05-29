"""
Các Ví Dụ Khác của Accuracy Paradox
Other Examples of Accuracy Paradox
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("=" * 80)
print("CÁC VÍ DỤ KHÁC CỦA ACCURACY PARADOX")
print("Other Examples of Accuracy Paradox")
print("=" * 80)

# ===== VÍ DỤ 1: PHÁT HIỆN BỆNH HIẾM GẶP =====
print("\n1️⃣ VÍ DỤ 1: PHÁT HIỆN BỆNH HIẾM GẶP (Rare Disease Detection)")
print("-" * 80)

# Dữ liệu: 10,000 bệnh nhân, chỉ 1% mắc bệnh
np.random.seed(42)
n_samples = 10000
disease_rate = 0.01

y_true = np.random.binomial(1, disease_rate, n_samples)
# Baseline: dự đoán luôn "không bệnh"
y_pred_baseline = np.zeros(n_samples)
# Model: dự đoán cơ bản
y_pred_model = np.random.binomial(1, 0.15, n_samples)

acc_baseline = accuracy_score(y_true, y_pred_baseline)
acc_model = accuracy_score(y_true, y_pred_model)
rec_baseline = recall_score(y_true, y_pred_baseline, zero_division=0)
rec_model = recall_score(y_true, y_pred_model, zero_division=0)

print(f"Tổng bệnh nhân: {n_samples}")
print(f"Tỷ lệ mắc bệnh: {disease_rate*100:.1f}%")
print()
print(f"📊 Baseline (Dự đoán luôn 'không bệnh'):")
print(f"   - Accuracy: {acc_baseline:.4f} ⚠️ Rất cao!")
print(f"   - Recall:   {rec_baseline:.4f} ❌ Không phát hiện bệnh nào")
print()
print(f"📊 Model Thực Tế:")
print(f"   - Accuracy: {acc_model:.4f} (thấp hơn)")
print(f"   - Recall:   {rec_model:.4f} ✅ Phát hiện được bệnh")
print()
print("⚠️ ACCURACY PARADOX:")
print("   - Baseline tưởng chừng tốt hơn (cao 99.1%)")
print("   - Nhưng không bao giờ phát hiện bệnh!")
print("   - KHÔNG thể dùng Accuracy cho trường hợp này")
print("   - Phải dùng Recall (hoặc F1) làm metric chính")

# ===== VÍ DỤ 2: PHÁT HIỆN GỬI LẠM =====
print("\n\n2️⃣ VÍ DỤ 2: PHÁT HIỆN GIA LẬN (Fraud Detection)")
print("-" * 80)

# Dữ liệu: 100,000 giao dịch, 0.5% là gian lận
n_transactions = 100000
fraud_rate = 0.005

y_fraud = np.random.binomial(1, fraud_rate, n_transactions)
# Baseline: dự đoán luôn "hợp lệ"
y_pred_fraud_baseline = np.zeros(n_transactions)
# Model: phát hiện được một số gian lận (nhưng cũng có false alarms)
y_pred_fraud_model = np.random.binomial(1, 0.08, n_transactions)

acc_fraud_baseline = accuracy_score(y_fraud, y_pred_fraud_baseline)
acc_fraud_model = accuracy_score(y_fraud, y_pred_fraud_model)
prec_fraud_baseline = precision_score(y_fraud, y_pred_fraud_baseline, zero_division=0)
prec_fraud_model = precision_score(y_fraud, y_pred_fraud_model, zero_division=0)
rec_fraud_baseline = recall_score(y_fraud, y_pred_fraud_baseline, zero_division=0)
rec_fraud_model = recall_score(y_fraud, y_pred_fraud_model, zero_division=0)

print(f"Tổng giao dịch: {n_transactions}")
print(f"Tỷ lệ gian lận: {fraud_rate*100:.2f}%")
print()
print(f"📊 Baseline (Dự đoán luôn 'hợp lệ'):")
print(f"   - Accuracy:  {acc_fraud_baseline:.4f} ⚠️ 99.5%!")
print(f"   - Precision: {prec_fraud_baseline:.4f} ❌")
print(f"   - Recall:    {rec_fraud_baseline:.4f} ❌")
print()
print(f"📊 Model Thực Tế:")
print(f"   - Accuracy:  {acc_fraud_model:.4f}")
print(f"   - Precision: {prec_fraud_model:.4f}")
print(f"   - Recall:    {rec_fraud_model:.4f}")
print()
print("⚠️ ACCURACY PARADOX:")
print("   - Baseline accuracy cao nhất, nhưng KHÔNG DÙNG ĐƯỢC")
print("   - Phải ưu tiên Recall hoặc F1-Score")
print("   - Lỡ bỏ 1 giao dịch gian lận có thể chi phí triệu đô")

# ===== VÍ DỤ 3: PHÂN LOẠI ĐỒ VẬT QUÝ HIẾM =====
print("\n\n3️⃣ VÍ DỤ 3: PHÂN LOẠI ĐỒ VẬT QUÝ HIẾM (Rare Item Classification)")
print("-" * 80)

# Dữ liệu: 50,000 hình ảnh, chỉ 2% là đồ vật quý hiếm
n_images = 50000
rare_rate = 0.02

y_rare = np.random.binomial(1, rare_rate, n_images)
# Baseline: dự đoán luôn "không quý hiếm"
y_pred_rare_baseline = np.zeros(n_images)
# Model: tốt hơn một chút
y_pred_rare_model = np.random.binomial(1, 0.05, n_images)

acc_rare_baseline = accuracy_score(y_rare, y_pred_rare_baseline)
acc_rare_model = accuracy_score(y_rare, y_pred_rare_model)

print(f"Tổng hình ảnh: {n_images}")
print(f"Tỷ lệ quý hiếm: {rare_rate*100:.1f}%")
print()
print(f"📊 Baseline: Accuracy = {acc_rare_baseline:.4f} ⚠️ 98%!")
print(f"📊 Model:    Accuracy = {acc_rare_model:.4f}")
print()
print("✅ SOLUTION:")
print("   - Không bao giờ tin Accuracy khi dữ liệu mất cân bằng")
print("   - Luôn kiểm tra Precision, Recall, F1-Score")
print("   - Dùng Stratified Cross-Validation")
print("   - Cân nhắc xử lý Class Imbalance (SMOTE, etc.)")

# ===== TẠO BIỂU ĐỒ TỔNG HỢP =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Accuracy Paradox: Các Ví Dụ Thực Tế', fontsize=14, fontweight='bold')

# VÍ DỤ 1: Phát hiện bệnh
ax1 = axes[0]
labels = ['Accuracy', 'Recall']
baseline_vals = [acc_baseline, rec_baseline]
model_vals = [acc_model, rec_model]

x = np.arange(len(labels))
width = 0.35
bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, model_vals, width, label='Model', color='#4ECDC4', alpha=0.8)

ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('1. Phát Hiện Bệnh Hiếm Gặp\n(Rare Disease)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 1)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# VÍ DỤ 2: Phát hiện gian lận
ax2 = axes[1]
labels2 = ['Accuracy', 'Precision', 'Recall']
baseline_vals2 = [acc_fraud_baseline, prec_fraud_baseline, rec_fraud_baseline]
model_vals2 = [acc_fraud_model, prec_fraud_model, rec_fraud_model]

x2 = np.arange(len(labels2))
bars3 = ax2.bar(x2 - width/2, baseline_vals2, width, label='Baseline', color='#FF6B6B', alpha=0.8)
bars4 = ax2.bar(x2 + width/2, model_vals2, width, label='Model', color='#4ECDC4', alpha=0.8)

ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('2. Phát Hiện Gian Lận\n(Fraud Detection)', fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(labels2, fontsize=9)
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# VÍ DỤ 3: Tỷ lệ class imbalance so sánh
ax3 = axes[2]
examples = ['Bệnh\n(1%)', 'Gian Lận\n(0.5%)', 'Đồ Quý\n(2%)']
imbalance_ratios = [99/1, 199.5/0.5, 98/2]
colors_pie = ['#FF6B6B', '#4ECDC4', '#95E1D3']
bars5 = ax3.bar(examples, imbalance_ratios, color=colors_pie, alpha=0.8, edgecolor='black', linewidth=2)

ax3.set_ylabel('Tỷ Lệ Class Imbalance', fontweight='bold')
ax3.set_title('3. Độ Mất Cân Bằng\nCác Ví Dụ', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar in bars5:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}:1', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figures/accuracy_paradox_examples.png', dpi=300, bbox_inches='tight')
print("\n✅ Hình ảnh đã được lưu: outputs/figures/accuracy_paradox_examples.png")

# ===== TẠO BẢNG SO SÁNH CHI TIẾT =====
print("\n" + "=" * 80)
print("📋 BẢNG SO SÁNH CHI TIẾT (Detailed Comparison Table)")
print("=" * 80)

comparison_data = {
    'Ví Dụ': [
        'Phát Hiện Bệnh',
        'Phát Hiện Gian Lận',
        'Phân Loại Đồ Quý',
        'Shopping Data'
    ],
    'Class Ratio': ['99:1', '199.5:0.5', '98:2', '5.46:1'],
    'Baseline Acc': ['99.00%', '99.50%', '98.00%', '84.54%'],
    'Model Acc': ['~98.50%', '~99.20%', '~97.00%', '88.32%'],
    'Key Metric': ['Recall', 'Precision/Recall', 'Recall', 'F1-Score'],
    'Why?': [
        'Miss 1 disease = tệ hại',
        'Miss fraud = tổn thất lớn',
        'Miss rare item = tổn thất',
        'Balance both metrics'
    ]
}

import pandas as pd
df_comp = pd.DataFrame(comparison_data)
print(df_comp.to_string(index=False))

print("\n" + "=" * 80)
print("🎓 KẾT LUẬN (Conclusion)")
print("=" * 80)
print("""
1️⃣ ACCURACY PARADOX xuất hiện khi:
   - Dữ liệu có sự mất cân bằng lớp rõ rệt (>5:1 hoặc lớn hơn)
   - Chỉ dùng Accuracy để đánh giá
   - Lớp thiểu số có giá trị cao (disease, fraud, etc.)

2️⃣ VẤN ĐỀ:
   - Baseline model có accuracy cao nhất
   - Nhưng hoàn toàn KHÔNG DÙNG ĐƯỢC
   - Không phát hiện được lớp thiểu số quan trọng

3️⃣ GIẢI PHÁP:
   ✅ Kiểm tra class distribution ngay lập tức
   ✅ Chọn metric phù hợp (Precision/Recall/F1/ROC-AUC)
   ✅ Xử lý class imbalance (SMOTE, weights, etc.)
   ✅ Dùng stratified cross-validation
   ✅ So sánh với baseline model
   ✅ Visualize confusion matrix

4️⃣ HỌC THÊM:
   - ROC-AUC: Better for imbalanced data
   - SMOTE: Synthetic Minority Over-sampling Technique
   - Stratified K-Fold: Maintain class distribution in folds
   - Cost-Sensitive Learning: Penalize minority class errors more
""")

plt.show()
