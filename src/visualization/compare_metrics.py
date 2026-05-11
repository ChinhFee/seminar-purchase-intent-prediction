import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Danh sách 4 thuật toán
models = ['Logistic\nRegression', 'K-NN', 'Random\nForest', 'XGBoost']

# Dữ liệu trích xuất từ bảng tổng hợp đồ án
accuracy = [85.0, 86.9, 89.3, 87.8]
recall = [74.87, 35.34, 51.31, 68.32]
f1_score = [60.72, 45.45, 59.85, 63.43]

# Màu đặc trưng cho 4 mô hình: LR (Đỏ), KNN (Xám), RF (Xanh lá), XGB (Xanh dương)
colors = ['#c0392b', '#7f8c8d', '#27ae60', '#2980b9']

def draw_comparison_chart(metric_name, data, filename):
    # STYLE PREMIUM XỊN XÒ
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')
    ax.set_facecolor('#F8F9FA') # Đã sửa lỗi dư dấu ngoặc ở đây
    
    # Vẽ biểu đồ cột
    bars = ax.bar(models, data, color=colors, edgecolor='#333333', linewidth=1.5, width=0.55, alpha=0.95)
    
    # Căn chỉnh tiêu đề
    plt.title(f'SO SÁNH {metric_name.upper()}', 
              fontsize=15, pad=25, fontweight='bold', color='#2C3E50')
    plt.ylabel('Tỷ lệ phần trăm (%)', fontsize=12, fontweight='bold', color='#34495E', labelpad=15)
    
    # Format các trục
    plt.xticks(fontsize=11, fontweight='bold', color='#2C3E50')
    plt.yticks(np.arange(0, 110, 20), fontsize=10, fontweight='bold', color='#7F8C8D')
    plt.ylim(0, 108)
    
    # CHIÊU CHỐT: Gắn label số có khung viền bo góc y hệt màu cột
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color=bar.get_facecolor(),
                 bbox=dict(facecolor='white', edgecolor=bar.get_facecolor(), boxstyle='round,pad=0.4', lw=1.5))
                 
    # Đẩy lưới chìm xuống dưới
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # Khử viền cứng
    sns.despine(left=True, bottom=False, right=True, top=True)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('#2C3E50')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close() # Đóng plot để vẽ hình tiếp theo
    print(f"   [+] Đã xuất file: {filename}")

# ĐÃ SỬA LẠI TÊN HÀM CHO CHUẨN XÁC
draw_comparison_chart('Độ chính xác (Accuracy)', accuracy, '6a_compare_accuracy.png')
draw_comparison_chart('Độ nhạy (Recall)', recall, '6b_compare_recall.png')
draw_comparison_chart('Điểm cân bằng (F1-Score)', f1_score, '6c_compare_f1.png')

print("=> Xong! Ông kiểm tra thư mục để lấy 3 tấm hình nhé.")
