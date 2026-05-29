"""
Biểu đồ phân phối mất cân bằng lớp và minh họa Accuracy Paradox
Illustration of Class Imbalance Distribution and Accuracy Paradox
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('data/shopping.csv')

# Convert Revenue to binary
df['Revenue'] = df['Revenue'].astype(int)

print("=" * 80)
print("PHÂN TÍCH MẤT CÂN BẰNG LỚP VÀ ACCURACY PARADOX")
print("Class Imbalance Analysis and Accuracy Paradox")
print("=" * 80)

# ===== 1. PHÂN TÍCH PHÂN PHỐI LỚP =====
print("\n1. PHÂN TÍCH PHÂN PHỐI LỚP (CLASS DISTRIBUTION ANALYSIS)")
print("-" * 80)

class_counts = df['Revenue'].value_counts().sort_index()
class_labels = ['Không mua (0)', 'Có mua (1)']
percentages = (class_counts / len(df) * 100).values

print(f"Tổng số mẫu: {len(df)}")
print(f"Lớp 0 (Không mua): {class_counts[0]} mẫu ({percentages[0]:.2f}%)")
print(f"Lớp 1 (Có mua):    {class_counts[1]} mẫu ({percentages[1]:.2f}%)")
print(f"Tỷ lệ mất cân bằng: {class_counts[0]/class_counts[1]:.2f}:1")

# ===== 2. TẠNG BIỂU ĐỒ PHÂN PHỐI LỚP =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ACCURACY PARADOX - Minh họa Nghịch lý Độ Chính Xác', 
             fontsize=16, fontweight='bold', y=0.995)

# Biểu đồ 1: Phân phối lớp (Column chart)
ax1 = axes[0, 0]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(class_labels, class_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Số lượng mẫu', fontsize=11, fontweight='bold')
ax1.set_title('1. Phân Phối Lớp (Class Distribution)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Thêm giá trị trên cột
for i, (bar, count, pct) in enumerate(zip(bars, class_counts, percentages)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}\n({pct:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Biểu đồ 2: Phân phối lớp (Pie chart)
ax2 = axes[0, 1]
explode = (0.05, 0.05)
wedges, texts, autotexts = ax2.pie(class_counts, labels=class_labels, autopct='%1.1f%%',
                                     colors=colors, explode=explode, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('2. Tỷ Lệ Phân Phối Lớp (Distribution Ratio)', fontsize=12, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# ===== 3. MÔ PHỎNG ACCURACY PARADOX =====
print("\n2. MÔ PHỎNG ACCURACY PARADOX (ACCURACY PARADOX SIMULATION)")
print("-" * 80)

# Tách train-test
X = df.drop('Revenue', axis=1).copy()
y = df['Revenue'].copy()

# Xử lý categorical features
X = pd.get_dummies(X, columns=['VisitorType', 'Month'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Baseline Model: Dự đoán luôn lớp đa số (Majority Class Predictor)
y_pred_baseline = np.zeros(len(y_test))  # Luôn dự đoán lớp 0

# Model thực tế: Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_pred_model = model.predict(X_test)

# Tính các metric
print("\n📊 Baseline Model (Dự đoán luôn lớp 0 - Không mua):")
print(f"   - Accuracy:  {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"   - Precision: {precision_score(y_test, y_pred_baseline, zero_division=0):.4f}")
print(f"   - Recall:    {recall_score(y_test, y_pred_baseline, zero_division=0):.4f}")
print(f"   - F1-Score:  {f1_score(y_test, y_pred_baseline, zero_division=0):.4f}")

print("\n📊 Logistic Regression Model (Mô hình thực tế):")
print(f"   - Accuracy:  {accuracy_score(y_test, y_pred_model):.4f}")
print(f"   - Precision: {precision_score(y_test, y_pred_model, zero_division=0):.4f}")
print(f"   - Recall:    {recall_score(y_test, y_pred_model, zero_division=0):.4f}")
print(f"   - F1-Score:  {f1_score(y_test, y_pred_model, zero_division=0):.4f}")

print("\n⚠️  ACCURACY PARADOX:")
baseline_acc = accuracy_score(y_test, y_pred_baseline)
model_acc = accuracy_score(y_test, y_pred_model)
print(f"   - Baseline model có accuracy cao ({baseline_acc:.4f})")
print(f"   - Nhưng nó chỉ đoán đúng lớp 0, không bao giờ phát hiện lớp 1!")
print(f"   - Model thực tế có accuracy thấp hơn nhưng thực sự hữu ích hơn")

# Biểu đồ 3: So sánh các metrics
ax3 = axes[1, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline_metrics = [
    accuracy_score(y_test, y_pred_baseline),
    precision_score(y_test, y_pred_baseline, zero_division=0),
    recall_score(y_test, y_pred_baseline, zero_division=0),
    f1_score(y_test, y_pred_baseline, zero_division=0)
]
model_metrics = [
    accuracy_score(y_test, y_pred_model),
    precision_score(y_test, y_pred_model, zero_division=0),
    recall_score(y_test, y_pred_model, zero_division=0),
    f1_score(y_test, y_pred_model, zero_division=0)
]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, baseline_metrics, width, label='Baseline (Predict 0)', 
                color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x_pos + width/2, model_metrics, width, label='Logistic Regression',
                color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('3. So Sánh Metrics: Accuracy Paradox', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics, fontsize=10)
ax3.set_ylim(0, 1)
ax3.legend(fontsize=10, loc='lower right')
ax3.grid(axis='y', alpha=0.3)

# Thêm giá trị trên các cột
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Biểu đồ 4: Confusion Matrix của Baseline
ax4 = axes[1, 1]
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', ax=ax4,
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
ax4.set_xlabel('Dự đoán (Predicted)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Thực tế (Actual)', fontsize=11, fontweight='bold')
ax4.set_title('4. Confusion Matrix: Baseline Model\n(Luôn dự đoán lớp 0)', 
              fontsize=12, fontweight='bold')
ax4.set_xticklabels(['0 (Không)', '1 (Có)'])
ax4.set_yticklabels(['0 (Không)', '1 (Có)'])

plt.tight_layout()
plt.savefig('outputs/figures/accuracy_paradox_illustration.png', dpi=300, bbox_inches='tight')
print("\n✅ Hình ảnh đã được lưu: outputs/figures/accuracy_paradox_illustration.png")

# ===== 4. HIỂN THỊ CÓ ĐỒI CHIẾU CONFUSION MATRIX =====
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('CONFUSION MATRIX - So Sánh Baseline vs Thực Tế', 
              fontsize=14, fontweight='bold')

# Confusion Matrix Baseline
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', ax=axes2[0],
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black', 
            vmin=0, vmax=max(cm_baseline.max(), confusion_matrix(y_test, y_pred_model).max()))
axes2[0].set_xlabel('Dự đoán (Predicted)', fontsize=11, fontweight='bold')
axes2[0].set_ylabel('Thực tế (Actual)', fontsize=11, fontweight='bold')
axes2[0].set_title('Baseline: Luôn Dự Đoán Lớp 0\nAccuracy: {:.4f}'.format(accuracy_score(y_test, y_pred_baseline)),
                   fontsize=11, fontweight='bold')
axes2[0].set_xticklabels(['0 (Không)', '1 (Có)'])
axes2[0].set_yticklabels(['0 (Không)', '1 (Có)'])

# Confusion Matrix Model
cm_model = confusion_matrix(y_test, y_pred_model)
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues', ax=axes2[1],
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black',
            vmin=0, vmax=max(cm_baseline.max(), cm_model.max()))
axes2[1].set_xlabel('Dự đoán (Predicted)', fontsize=11, fontweight='bold')
axes2[1].set_ylabel('Thực tế (Actual)', fontsize=11, fontweight='bold')
axes2[1].set_title('Logistic Regression: Mô Hình Thực Tế\nAccuracy: {:.4f}'.format(accuracy_score(y_test, y_pred_model)),
                   fontsize=11, fontweight='bold')
axes2[1].set_xticklabels(['0 (Không)', '1 (Có)'])
axes2[1].set_yticklabels(['0 (Không)', '1 (Có)'])

plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Hình ảnh đã được lưu: outputs/figures/confusion_matrix_comparison.png")

# ===== 5. GHI CHÚ GIẢI THÍCH ACCURACY PARADOX =====
print("\n" + "=" * 80)
print("📌 GIẢI THÍCH ACCURACY PARADOX (Explanation of Accuracy Paradox)")
print("=" * 80)
print("""
Accuracy Paradox (Nghịch lý Độ Chính Xác) là hiện tượng:

1️⃣ ĐỊNH NGHĨA:
   - Một mô hình với độ chính xác (Accuracy) cao có thể không thực sự hữu ích
   - Đặc biệt khi dữ liệu có sự mất cân bằng lớp rõ rệt

2️⃣ VÍ DỤ THỰC TẾ (từ dữ liệu này):
   - {:.1f}% khách hàng không mua hàng (Lớp 0)
   - {:.1f}% khách hàng mua hàng (Lớp 1)
   
   - Nếu ta xây dựng một mô hình "ngu" chỉ dự đoán "Không mua" cho mọi trường hợp:
     ✓ Accuracy = {:.4f} (rất cao!)
     ✗ Nhưng nó không bao giờ phát hiện ra khách hàng sẽ mua (Recall = 0)
     
   - Mô hình thực tế (Logistic Regression):
     ✓ Accuracy = {:.4f} (thấp hơn)
     ✓ Nhưng nó THỰC SỰ có thể phát hiện khách hàng sẽ mua (Recall = {:.4f})

3️⃣ GIẢI PHÁP:
   ❌ Không dùng Accuracy làm metric chính trong trường hợp dữ liệu mất cân bằng
   ✅ Sử dụng các metric khác:
      - Precision: Trong những gì dự đoán là "Có mua", bao nhiêu % thực sự mua?
      - Recall: Trong tất cả những người mua, bao nhiêu % được phát hiện?
      - F1-Score: Kết hợp Precision và Recall
      - ROC-AUC: Đánh giá toàn bộ ngưỡng phân loại
      - Confusion Matrix: Xem chi tiết True/False Positive/Negative

4️⃣ LÝ DO LỰA CHỌN METRIC:
   - Precision: Khi False Positive có chi phí cao (email spam filter)
   - Recall: Khi False Negative có chi phí cao (phát hiện bệnh, gian lận)
   - F1-Score: Khi cần cân bằng Precision và Recall
   - ROC-AUC: Đánh giá tổng thể hiệu suất trên tất cả ngưỡng
""".format(percentages[0], percentages[1], baseline_acc, model_acc, recall_score(y_test, y_pred_model, zero_division=0)))

print("=" * 80)
plt.show()
