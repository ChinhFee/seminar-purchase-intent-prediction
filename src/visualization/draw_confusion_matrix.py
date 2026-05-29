"""
Hình ảnh minh họa Confusion Matrix cho phân lớp nhị phân
"""
import matplotlib
matplotlib.use('Agg')  # Không hiển thị GUI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def draw_confusion_matrix_structure():
    """
    Vẽ cấu trúc tổng quát của Confusion Matrix
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ============ HÌNH 1: Cấu trúc lý thuyết ============
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Cấu trúc Confusion Matrix\n(Binary Classification)', fontsize=14, fontweight='bold', pad=20)
    
    # Tạo bảng
    cell_size = 2
    x_start, y_start = 2, 4
    
    # Header hàng
    ax1.text(x_start - 1.5, y_start + 2.2, 'Thực tế →', fontsize=11, fontweight='bold', ha='center')
    ax1.text(x_start + 0.8, y_start + 2.8, 'Dương (Positive)', fontsize=10, ha='center')
    ax1.text(x_start + 3.2, y_start + 2.8, 'Âm (Negative)', fontsize=10, ha='center')
    
    # Header cột
    ax1.text(x_start - 2, y_start + 1.2, 'Dự\nđoán\n↓', fontsize=11, fontweight='bold', ha='center', va='center')
    ax1.text(x_start - 0.5, y_start + 0.5, 'Dương', fontsize=10, ha='center', va='center')
    ax1.text(x_start - 0.5, y_start - 2.3, 'Âm', fontsize=10, ha='center', va='center')
    
    # Ô TP (True Positive)
    rect1 = Rectangle((x_start, y_start), cell_size, cell_size, linewidth=2, edgecolor='black', facecolor='#90EE90')
    ax1.add_patch(rect1)
    ax1.text(x_start + cell_size/2, y_start + cell_size/2 + 0.3, 'TP', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.text(x_start + cell_size/2, y_start + cell_size/2 - 0.5, 'True Positive', fontsize=9, ha='center', va='center', style='italic')
    
    # Ô FN (False Negative)
    rect2 = Rectangle((x_start + 2.5, y_start), cell_size, cell_size, linewidth=2, edgecolor='black', facecolor='#FFB6C1')
    ax1.add_patch(rect2)
    ax1.text(x_start + 2.5 + cell_size/2, y_start + cell_size/2 + 0.3, 'FN', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.text(x_start + 2.5 + cell_size/2, y_start + cell_size/2 - 0.5, 'False Negative', fontsize=9, ha='center', va='center', style='italic')
    
    # Ô FP (False Positive)
    rect3 = Rectangle((x_start, y_start - 2.5), cell_size, cell_size, linewidth=2, edgecolor='black', facecolor='#FFB6C1')
    ax1.add_patch(rect3)
    ax1.text(x_start + cell_size/2, y_start - 2.5 + cell_size/2 + 0.3, 'FP', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.text(x_start + cell_size/2, y_start - 2.5 + cell_size/2 - 0.5, 'False Positive', fontsize=9, ha='center', va='center', style='italic')
    
    # Ô TN (True Negative)
    rect4 = Rectangle((x_start + 2.5, y_start - 2.5), cell_size, cell_size, linewidth=2, edgecolor='black', facecolor='#90EE90')
    ax1.add_patch(rect4)
    ax1.text(x_start + 2.5 + cell_size/2, y_start - 2.5 + cell_size/2 + 0.3, 'TN', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.text(x_start + 2.5 + cell_size/2, y_start - 2.5 + cell_size/2 - 0.5, 'True Negative', fontsize=9, ha='center', va='center', style='italic')
    
    # Thêm chú giải
    ax1.text(x_start + 1.25, 0.5, '✓ Correct', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.7))
    ax1.text(x_start + 3.75, 0.5, '✗ Error', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#FFB6C1', alpha=0.7))
    
    # ============ HÌNH 2: Ví dụ cụ thể ============
    ax2 = axes[1]
    
    # Dữ liệu ví dụ
    cm_data = np.array([[85, 10],   # TP=85, FN=10
                        [5, 100]])   # FP=5, TN=100
    
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=['Dương', 'Âm'], yticklabels=['Dương', 'Âm'],
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=2, linecolor='black')
    
    ax2.set_title('Ví dụ Confusion Matrix\n(với số liệu cụ thể)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Thực tế (Actual)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Dự đoán (Predicted)', fontsize=11, fontweight='bold')
    
    # Thêm thông tin
    info_text = (
        f"TP (True Positive) = 85\n"
        f"FN (False Negative) = 10\n"
        f"FP (False Positive) = 5\n"
        f"TN (True Negative) = 100\n"
        f"\nTổng: {cm_data.sum()}"
    )
    ax2.text(1.15, 0.5, info_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(r'c:\CODE\SEMINAR\outputs\figures\confusion_matrix_structure.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu: confusion_matrix_structure.png")
    plt.close()


def draw_confusion_matrix_meanings():
    """
    Vẽ giải thích chi tiết của các phần tử Confusion Matrix
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title = "Ý Nghĩa Chi Tiết Các Phần Tử Confusion Matrix"
    ax.text(5, 9.5, title, fontsize=16, fontweight='bold', ha='center')
    
    definitions = [
        {
            'title': 'TP (True Positive)',
            'color': '#90EE90',
            'definition': 'Dự đoán ĐÚNG là Dương\n(Thực tế cũng là Dương)\n→ Kết quả tích cực đúng',
            'y': 8
        },
        {
            'title': 'FN (False Negative)',
            'color': '#FFB6C1',
            'definition': 'Dự đoán SAI là Âm\n(Thực tế là Dương)\n→ Bỏ sót trường hợp dương',
            'y': 6
        },
        {
            'title': 'FP (False Positive)',
            'color': '#FFB6C1',
            'definition': 'Dự đoán SAI là Dương\n(Thực tế là Âm)\n→ Cảnh báo sai',
            'y': 4
        },
        {
            'title': 'TN (True Negative)',
            'color': '#90EE90',
            'definition': 'Dự đoán ĐÚNG là Âm\n(Thực tế cũng là Âm)\n→ Kết quả tiêu cực đúng',
            'y': 2
        }
    ]
    
    for item in definitions:
        # Hộp màu
        rect = Rectangle((0.5, item['y'] - 0.4), 1.5, 0.8, linewidth=2, 
                         edgecolor='black', facecolor=item['color'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(1.25, item['y'], item['title'], fontsize=11, fontweight='bold', 
               ha='center', va='center')
        
        # Giải thích
        ax.text(2.5, item['y'], item['definition'], fontsize=10, va='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(r'c:\CODE\SEMINAR\outputs\figures\confusion_matrix_meanings.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu: confusion_matrix_meanings.png")
    plt.close()


def draw_metrics_from_cm():
    """
    Vẽ các metrics có thể tính từ Confusion Matrix
    """
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    title = "Các Metrics (Chỉ số) Tính Từ Confusion Matrix"
    ax.text(5, 9.5, title, fontsize=16, fontweight='bold', ha='center')
    
    metrics = [
        {
            'name': 'Accuracy (Độ chính xác)',
            'formula': '(TP + TN) / (TP + TN + FP + FN)',
            'meaning': 'Tỉ lệ dự đoán đúng trên tổng',
            'y': 8.5
        },
        {
            'name': 'Precision (Độ chính xác lớp Dương)',
            'formula': 'TP / (TP + FP)',
            'meaning': 'Trong những cái dự đoán là Dương, bao nhiêu % là đúng',
            'y': 7.2
        },
        {
            'name': 'Recall / Sensitivity (Độ nhạy)',
            'formula': 'TP / (TP + FN)',
            'meaning': 'Trong những cái thực tế là Dương, bao nhiêu % được phát hiện',
            'y': 5.9
        },
        {
            'name': 'Specificity (Độ đặc thù)',
            'formula': 'TN / (TN + FP)',
            'meaning': 'Trong những cái thực tế là Âm, bao nhiêu % được phát hiện đúng',
            'y': 4.6
        },
        {
            'name': 'F1-Score',
            'formula': '2 × (Precision × Recall) / (Precision + Recall)',
            'meaning': 'Trung bình điều hòa của Precision và Recall',
            'y': 3.3
        }
    ]
    
    for metric in metrics:
        # Tên metrics
        ax.text(0.3, metric['y'], f"• {metric['name']}", fontsize=11, fontweight='bold', va='top')
        
        # Công thức
        ax.text(0.5, metric['y'] - 0.4, f"Công thức: {metric['formula']}", 
               fontsize=9, style='italic', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        # Ý nghĩa
        ax.text(0.5, metric['y'] - 0.8, f"Ý nghĩa: {metric['meaning']}", 
               fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(r'c:\CODE\SEMINAR\outputs\figures\confusion_matrix_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu: confusion_matrix_metrics.png")
    plt.close()


if __name__ == '__main__':
    print("Tạo hình ảnh minh họa Confusion Matrix...\n")
    
    draw_confusion_matrix_structure()
    print()
    
    draw_confusion_matrix_meanings()
    print()
    
    draw_metrics_from_cm()
    print()
    
    print("✅ Hoàn thành! Tất cả các hình ảnh đã được lưu vào: outputs/figures/")
