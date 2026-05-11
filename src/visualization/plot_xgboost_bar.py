import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dữ liệu của XGBoost
metrics = ['Accuracy', 'Recall', 'F1-Score', 'AUC']
values = [87.80, 68.32, 63.43, 91.30]

# MÀU SẮC CHO XGBOOST (Trích từ Word): 
# Xanh mòng két/Teal (Feature Imp), Cam Nâu (CM), Tím Đậm (PR Curve), Xanh Navy (ROC Curve)
colors = ['#008080', '#d95f0e', '#800080', '#000080']

# STYLE PREMIUM XỊN XÒ (Giống hệt bản đẹp của LR)
fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='#F8F9FA')
ax.set_facecolor=('#F8F9FA')

bars = ax.bar(metrics, values, color=colors, edgecolor='#333333', linewidth=1.5, width=0.55, alpha=0.95)

plt.title('XGBOOST', 
          fontsize=16, pad=25, fontweight='bold', color='#2C3E50')
plt.ylabel('Tỷ lệ phần trăm (%)', fontsize=13, fontweight='bold', color='#34495E', labelpad=15)

plt.xticks(fontsize=12, fontweight='bold', color='#2C3E50')
plt.yticks(np.arange(0, 110, 20), fontsize=11, fontweight='bold', color='#7F8C8D')
plt.ylim(0, 108)

# CHIÊU CHỐT: Gắn label số có khung viền bo góc y hệt màu cột
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', 
             ha='center', va='bottom', fontsize=13, fontweight='bold', color=bar.get_facecolor(),
             bbox=dict(facecolor='white', edgecolor=bar.get_facecolor(), boxstyle='round,pad=0.4', lw=1.5))

# Đẩy lưới chìm xuống dưới
ax.grid(axis='y', linestyle='--', alpha=0.7, color='#BDC3C7')
ax.set_axisbelow(True)

# Khử viền cứng
sns.despine(left=True, bottom=False, right=True, top=True)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('#2C3E50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'xgb_bar.png', dpi=300, bbox_inches='tight')
print("   [+] Đã xuất file xgb_bar.png")
