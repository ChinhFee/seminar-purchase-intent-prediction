# Seminar - Online Shoppers Purchasing Intention

Repo này chứa mã nguồn, dữ liệu, model đã huấn luyện, hình kết quả và tài liệu báo cáo cho bài seminar Machine Learning dự đoán khả năng mua hàng trực tuyến.

## Cấu trúc thư mục

```text
SEMINAR/
├── data/                  # Dataset CSV
├── models/                # Model đã huấn luyện
├── outputs/figures/       # Biểu đồ và hình kết quả
├── src/
│   ├── app/               # Ứng dụng demo PyQt5
│   ├── models/            # Script huấn luyện/đánh giá từng mô hình
│   ├── visualization/     # Script vẽ biểu đồ
│   └── legacy/            # Code cũ còn giữ để tham khảo
└── docs/
    ├── assignments/       # Đề bài
    ├── lectures/          # Slide bài giảng
    ├── references/        # Tài liệu tham khảo
    └── reports/           # Báo cáo Word
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy demo

```bash
python src/app/demo_pyqt_app.py
```

Ứng dụng sử dụng:

- Dataset: `data/shopping.csv`
- Model: `models/best_xgb_model.pkl`

## Huấn luyện và tạo biểu đồ

Huấn luyện lại model XGBoost:

```bash
python src/models/train_model.py
```

Chạy từng mô hình và xuất hình vào `outputs/figures`:

```bash
python src/models/logistic_regression.py data/shopping.csv
python src/models/knn.py data/shopping.csv
python src/models/random_forest.py data/shopping.csv
python src/models/xgboost_model.py data/shopping.csv
```

Tạo biểu đồ tổng hợp:

```bash
python src/visualization/compare_metrics.py
```
