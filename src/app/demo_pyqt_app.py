import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "shopping.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_xgb_model.pkl"

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


NUMERIC_FIELDS = [
    ("Administrative", "Số trang quản trị", int, 0, 30, 0),
    ("Administrative_Duration", "Thời gian trang quản trị", float, 0, 4000, 0),
    ("Informational", "Số trang thông tin", int, 0, 30, 0),
    ("Informational_Duration", "Thời gian trang thông tin", float, 0, 4000, 0),
    ("ProductRelated", "Số trang sản phẩm", int, 0, 800, 8),
    ("ProductRelated_Duration", "Thời gian xem sản phẩm", float, 0, 20000, 250),
    ("BounceRates", "Tỷ lệ thoát", float, 0, 1, 0.03),
    ("ExitRates", "Tỷ lệ rời trang", float, 0, 1, 0.05),
    ("PageValues", "Giá trị trang", float, 0, 400, 0),
    ("SpecialDay", "Gần ngày khuyến mãi", float, 0, 1, 0),
    ("OperatingSystems", "Hệ điều hành", int, 1, 8, 2),
    ("Browser", "Trình duyệt", int, 1, 13, 2),
    ("Region", "Khu vực", int, 1, 9, 1),
    ("TrafficType", "Nguồn truy cập", int, 1, 20, 2),
]


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#ffffff")
        super().__init__(self.fig)


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo dự đoán hành vi mua hàng trực tuyến")
        self.resize(1360, 820)

        self.raw_df = pd.read_csv(DATA_PATH)
        self.raw_df.columns = self.raw_df.columns.str.strip()
        saved_xgb = self.load_model()
        self.feature_names = list(getattr(saved_xgb, "feature_names_in_", []))
        self.encoded_df = self.encode_dataframe(self.raw_df)
        self.prepare_train_test()
        self.models = self.build_models(saved_xgb)
        self.metrics_by_model = self.evaluate_all_models()
        self.selected_model_name = "XGBoost"
        self.model = self.models[self.selected_model_name]
        self.metrics = self.metrics_by_model[self.selected_model_name]
        self.inputs = {}
        self.model_combo = None
        self.chart_layout = None
        self.result_label = None
        self.prob_label = None
        self.progress = None
        self.recommendation = None

        self.setStyleSheet(STYLE)
        self.setCentralWidget(self.build_ui())

    def load_model(self):
        if not MODEL_PATH.exists():
            QMessageBox.critical(self, "Thiếu model", f"Không tìm thấy {MODEL_PATH}")
            raise FileNotFoundError(MODEL_PATH)
        with MODEL_PATH.open("rb") as f:
            return pickle.load(f)

    def encode_dataframe(self, df):
        encoded = df.copy()
        encoded["Revenue"] = encoded["Revenue"].astype(bool).astype(int)
        categorical_cols = encoded.select_dtypes(include=["object", "bool"]).columns.tolist()
        if "Revenue" in categorical_cols:
            categorical_cols.remove("Revenue")
        encoded = pd.get_dummies(encoded, columns=categorical_cols, drop_first=True)
        for col in self.feature_names:
            if col not in encoded.columns:
                encoded[col] = 0
        return encoded[self.feature_names + ["Revenue"]]

    def prepare_train_test(self):
        X = self.encoded_df[self.feature_names]
        y = self.encoded_df["Revenue"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def build_models(self, saved_xgb):
        ratio = float(np.sum(self.y_train == 0)) / max(np.sum(self.y_train == 1), 1)
        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=160,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": saved_xgb,
        }

        for name, model in models.items():
            if name != "XGBoost":
                model.fit(self.X_train, self.y_train)

        # Nếu file pkl không khớp dữ liệu hiện tại, train lại XGBoost để UI vẫn chạy được.
        if getattr(models["XGBoost"], "n_features_in_", None) != len(self.feature_names):
            models["XGBoost"] = xgb.XGBClassifier(
                scale_pos_weight=ratio,
                eval_metric="logloss",
                random_state=42,
            )
            models["XGBoost"].fit(self.X_train, self.y_train)
        return models

    def evaluate_all_models(self):
        return {
            name: self.evaluate_model(model)
            for name, model in self.models.items()
        }

    def evaluate_model(self, model):
        started = time.perf_counter()
        preds = model.predict(self.X_test)
        elapsed_ms = (time.perf_counter() - started) * 1000
        probs = model.predict_proba(self.X_test)[:, 1]
        cm = confusion_matrix(self.y_test, preds)
        return {
            "accuracy": accuracy_score(self.y_test, preds),
            "f1": f1_score(self.y_test, preds),
            "recall": recall_score(self.y_test, preds),
            "cm": cm,
            "probs": probs,
            "y_test": self.y_test.to_numpy(),
            "elapsed_ms": elapsed_ms,
        }

    def select_model(self, name):
        self.selected_model_name = name
        self.model = self.models[name]
        self.metrics = self.metrics_by_model[name]

    def build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = self.build_header()
        tabs = QTabWidget()
        tabs.addTab(self.build_dashboard_tab(), "Tổng quan")
        tabs.addTab(self.build_predict_tab(), "Dự đoán trực tiếp")
        tabs.addTab(self.build_charts_tab(), "Biểu đồ mô hình")
        tabs.addTab(self.build_data_tab(), "Dữ liệu mẫu")

        layout.addWidget(header)
        layout.addWidget(tabs, 1)
        return root

    def build_header(self):
        box = QFrame()
        box.setObjectName("header")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(22, 18, 22, 18)

        title = QLabel("Hệ thống dự đoán khách hàng có mua hàng hay không")
        title.setObjectName("title")
        subtitle = QLabel(
            "Demo đồ án Machine Learning trên bộ dữ liệu Online Shoppers Purchasing Intention"
        )
        subtitle.setObjectName("subtitle")
        left = QVBoxLayout()
        left.addWidget(title)
        left.addWidget(subtitle)

        badge = QLabel("4 thuật toán | 26 features | PyQt5")
        badge.setObjectName("badge")
        badge.setAlignment(Qt.AlignCenter)

        layout.addLayout(left, 1)
        layout.addWidget(badge)
        return box

    def build_dashboard_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        cards = QGridLayout()
        total = len(self.raw_df)
        buy_rate = self.raw_df["Revenue"].astype(bool).mean()
        cards.addWidget(self.metric_card("Số phiên truy cập", f"{total:,}", "Bộ dữ liệu huấn luyện"), 0, 0)
        cards.addWidget(self.metric_card("Tỷ lệ mua hàng", f"{buy_rate:.1%}", "Nhãn Revenue = TRUE"), 0, 1)
        best_name = max(self.metrics_by_model, key=lambda name: self.metrics_by_model[name]["f1"])
        cards.addWidget(self.metric_card("Model tốt nhất", best_name, "Xếp theo F1-score"), 0, 2)
        cards.addWidget(self.metric_card("F1 tốt nhất", f"{self.metrics_by_model[best_name]['f1']:.1%}", "Trên cùng tập kiểm thử"), 0, 3)
        layout.addLayout(cards)

        split = QSplitter(Qt.Horizontal)
        story = QTextEdit()
        story.setReadOnly(True)
        story.setObjectName("story")
        story.setText(
            "Luồng trình bày gợi ý:\n\n"
            "1. Bài toán: dự đoán khả năng khách mua hàng trong một phiên truy cập.\n"
            "2. Dữ liệu đầu vào: số trang đã xem, thời gian xem, tỷ lệ thoát, nguồn truy cập, tháng, loại khách và cuối tuần.\n"
            "3. So sánh thuật toán: Logistic Regression làm baseline tuyến tính, KNN dựa trên láng giềng gần, Random Forest dùng nhiều cây quyết định, XGBoost tối ưu boosting.\n"
            "4. Demo: chọn từng thuật toán, nhập cùng một phiên truy cập và quan sát xác suất/nhãn dự đoán thay đổi.\n\n"
            "Điểm nhấn khi bảo vệ: các mô hình khác nhau có trade-off khác nhau. Accuracy cao chưa chắc tốt nếu bỏ sót nhiều khách có mua, vì vậy nên nhìn thêm F1-score và Recall."
        )
        split.addWidget(story)

        canvas = MplCanvas(width=6, height=4)
        ax = canvas.fig.add_subplot(111)
        labels = list(self.metrics_by_model.keys())
        vals = [self.metrics_by_model[name]["f1"] for name in labels]
        colors = ["#2563eb", "#7c3aed", "#16a34a", "#f59e0b"]
        bars = ax.bar(labels, vals, color=colors, width=0.58)
        ax.set_ylim(0, 1)
        ax.set_title("So sánh F1-score giữa các thuật toán", fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=12)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}", ha="center", weight="bold")
        canvas.fig.tight_layout()
        split.addWidget(canvas)
        split.setSizes([520, 620])
        layout.addWidget(split, 1)
        return page

    def metric_card(self, title, value, note):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        label = QLabel(title)
        label.setObjectName("cardTitle")
        number = QLabel(value)
        number.setObjectName("cardValue")
        desc = QLabel(note)
        desc.setObjectName("cardNote")
        layout.addWidget(label)
        layout.addWidget(number)
        layout.addWidget(desc)
        return card

    def build_predict_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(16)

        form_group = QGroupBox("Thông tin phiên truy cập")
        form_layout = QGridLayout(form_group)
        form_layout.setHorizontalSpacing(14)
        form_layout.setVerticalSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.models.keys()))
        self.model_combo.setCurrentText(self.selected_model_name)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        form_layout.addWidget(QLabel("Thuật toán"), 0, 0)
        form_layout.addWidget(self.model_combo, 0, 1)

        for row, (key, label, kind, min_val, max_val, default) in enumerate(NUMERIC_FIELDS, start=1):
            form_layout.addWidget(QLabel(label), row, 0)
            widget = QSpinBox() if kind is int else QDoubleSpinBox()
            widget.setRange(min_val, max_val)
            widget.setValue(default)
            if kind is float:
                widget.setDecimals(4)
                widget.setSingleStep(0.01)
            form_layout.addWidget(widget, row, 1)
            self.inputs[key] = widget

        row = len(NUMERIC_FIELDS) + 1
        self.inputs["Month"] = QComboBox()
        self.inputs["Month"].addItems(["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        self.inputs["VisitorType"] = QComboBox()
        self.inputs["VisitorType"].addItems(["New_Visitor", "Returning_Visitor", "Other"])
        self.inputs["Weekend"] = QCheckBox("Cuối tuần")
        form_layout.addWidget(QLabel("Tháng"), row, 0)
        form_layout.addWidget(self.inputs["Month"], row, 1)
        form_layout.addWidget(QLabel("Loại khách"), row + 1, 0)
        form_layout.addWidget(self.inputs["VisitorType"], row + 1, 1)
        form_layout.addWidget(QLabel("Thời điểm"), row + 2, 0)
        form_layout.addWidget(self.inputs["Weekend"], row + 2, 1)

        presets = QHBoxLayout()
        low_btn = QPushButton("Kịch bản khách xem dạo")
        high_btn = QPushButton("Kịch bản khách có ý định mua")
        low_btn.clicked.connect(self.apply_low_intent)
        high_btn.clicked.connect(self.apply_high_intent)
        presets.addWidget(low_btn)
        presets.addWidget(high_btn)
        form_layout.addLayout(presets, row + 3, 0, 1, 2)

        predict_btn = QPushButton("Chạy dự đoán")
        predict_btn.setObjectName("primaryButton")
        predict_btn.clicked.connect(self.predict_current_input)
        form_layout.addWidget(predict_btn, row + 4, 0, 1, 2)

        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(26, 26, 26, 26)
        self.result_label = QLabel("Nhập dữ liệu và bấm Chạy dự đoán")
        self.result_label.setObjectName("resultTitle")
        self.prob_label = QLabel("Xác suất mua hàng: --")
        self.prob_label.setObjectName("probability")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.recommendation = QTextEdit()
        self.recommendation.setReadOnly(True)
        self.recommendation.setObjectName("recommendation")
        self.recommendation.setText(
            "Kết quả sẽ hiển thị xác suất khách mua hàng, nhãn dự đoán và gợi ý hành động phù hợp để nhóm trình bày trước hội đồng."
        )
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.prob_label)
        result_layout.addWidget(self.progress)
        result_layout.addWidget(self.recommendation, 1)

        layout.addWidget(form_group, 0)
        layout.addWidget(result_panel, 1)
        return page

    def on_model_changed(self, name):
        self.select_model(name)
        if self.prob_label is not None:
            self.predict_current_input()
        self.refresh_charts()

    def current_raw_row(self):
        row = {}
        for key, *_ in NUMERIC_FIELDS:
            row[key] = self.inputs[key].value()
        row["Month"] = self.inputs["Month"].currentText()
        row["VisitorType"] = self.inputs["VisitorType"].currentText()
        row["Weekend"] = bool(self.inputs["Weekend"].isChecked())
        row["Revenue"] = False
        return row

    def encode_single_row(self, raw_row):
        encoded = {col: 0 for col in self.feature_names}
        for key, *_ in NUMERIC_FIELDS:
            if key in encoded:
                encoded[key] = raw_row[key]

        month_col = f"Month_{raw_row['Month']}"
        visitor_col = f"VisitorType_{raw_row['VisitorType']}"
        if month_col in encoded:
            encoded[month_col] = 1
        if visitor_col in encoded:
            encoded[visitor_col] = 1
        if raw_row["Weekend"] and "Weekend_True" in encoded:
            encoded["Weekend_True"] = 1
        return pd.DataFrame([encoded], columns=self.feature_names)

    def predict_current_input(self):
        X = self.encode_single_row(self.current_raw_row())
        started = time.perf_counter()
        probability = float(self.model.predict_proba(X)[0, 1])
        elapsed_ms = (time.perf_counter() - started) * 1000
        prediction = probability >= 0.5
        self.progress.setValue(round(probability * 100))
        self.prob_label.setText(f"Xác suất mua hàng: {probability:.1%}")
        if prediction:
            self.result_label.setText("Dự đoán: CÓ KHẢ NĂNG MUA")
            action = (
                "Gợi ý demo: hệ thống nên ưu tiên hiển thị mã giảm giá, nhắc giỏ hàng, "
                "hoặc chuyển khách vào nhóm remarketing giá trị cao."
            )
        else:
            self.result_label.setText("Dự đoán: CHƯA CÓ Ý ĐỊNH MUA RÕ")
            action = (
                "Gợi ý demo: nên tối ưu trải nghiệm duyệt sản phẩm, giảm tỷ lệ thoát, "
                "hoặc đề xuất nội dung giúp khách ở lại lâu hơn."
            )
        self.recommendation.setText(
            f"{action}\n\n"
            f"Thuật toán đang chọn: {self.selected_model_name}.\n"
            f"Thời gian suy luận mẫu này: {elapsed_ms:.3f} ms.\n"
            "Ngưỡng phân loại đang dùng: 50%.\n\n"
            f"{self.compare_current_input(X)}"
        )

    def compare_current_input(self, X):
        lines = ["So sánh xác suất của cùng một phiên truy cập:"]
        for name, model in self.models.items():
            prob = float(model.predict_proba(X)[0, 1])
            label = "Có mua" if prob >= 0.5 else "Không mua"
            lines.append(f"- {name}: {prob:.1%} ({label})")
        return "\n".join(lines)

    def apply_low_intent(self):
        values = {
            "Administrative": 0,
            "Administrative_Duration": 0,
            "Informational": 0,
            "Informational_Duration": 0,
            "ProductRelated": 3,
            "ProductRelated_Duration": 60,
            "BounceRates": 0.18,
            "ExitRates": 0.16,
            "PageValues": 0,
            "SpecialDay": 0,
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 2,
        }
        self.apply_values(values, "Feb", "Returning_Visitor", False)

    def apply_high_intent(self):
        values = {
            "Administrative": 4,
            "Administrative_Duration": 120,
            "Informational": 2,
            "Informational_Duration": 80,
            "ProductRelated": 75,
            "ProductRelated_Duration": 3200,
            "BounceRates": 0.005,
            "ExitRates": 0.018,
            "PageValues": 38,
            "SpecialDay": 0,
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 3,
            "TrafficType": 2,
        }
        self.apply_values(values, "Nov", "Returning_Visitor", True)

    def apply_values(self, values, month, visitor, weekend):
        for key, value in values.items():
            self.inputs[key].setValue(value)
        self.inputs["Month"].setCurrentText(month)
        self.inputs["VisitorType"].setCurrentText(visitor)
        self.inputs["Weekend"].setChecked(weekend)
        self.predict_current_input()

    def build_charts_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(QLabel("Biểu đồ đang hiển thị theo thuật toán đã chọn ở tab Dự đoán trực tiếp."))
        top.addStretch(1)
        self.chart_model_label = QLabel(f"Đang xem: {self.selected_model_name}")
        self.chart_model_label.setObjectName("badgeSmall")
        top.addWidget(self.chart_model_label)
        outer.addLayout(top)

        self.chart_layout = QGridLayout()
        self.chart_layout.setSpacing(14)
        outer.addLayout(self.chart_layout, 1)
        self.refresh_charts()
        return page

    def refresh_charts(self):
        if self.chart_layout is None:
            return
        while self.chart_layout.count():
            item = self.chart_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if hasattr(self, "chart_model_label"):
            self.chart_model_label.setText(f"Đang xem: {self.selected_model_name}")
        self.chart_layout.addWidget(self.chart_confusion_matrix(), 0, 0)
        self.chart_layout.addWidget(self.chart_feature_importance(), 0, 1)
        self.chart_layout.addWidget(self.chart_probability_distribution(), 1, 0)
        self.chart_layout.addWidget(self.chart_algorithm_comparison(), 1, 1)

    def chart_confusion_matrix(self):
        canvas = MplCanvas(width=5.6, height=3.7)
        ax = canvas.fig.add_subplot(111)
        cm = self.metrics["cm"]
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title("Ma trận nhầm lẫn", fontsize=11, weight="bold")
        ax.set_xticks([0, 1], ["Không mua", "Có mua"])
        ax.set_yticks([0, 1], ["Không mua", "Có mua"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "#111827"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, weight="bold")
        canvas.fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        canvas.fig.tight_layout()
        return canvas

    def chart_feature_importance(self):
        canvas = MplCanvas(width=5.6, height=3.7)
        ax = canvas.fig.add_subplot(111)
        importances = self.get_feature_importance()
        if importances is None:
            ax.text(
                0.5,
                0.55,
                "KNN không có feature importance trực tiếp",
                ha="center",
                va="center",
                fontsize=12,
                weight="bold",
            )
            ax.text(
                0.5,
                0.42,
                "Giải thích khi thuyết trình: KNN dự đoán bằng khoảng cách tới các mẫu gần nhất.",
                ha="center",
                va="center",
                wrap=True,
                color="#475569",
            )
            ax.axis("off")
        else:
            top = sorted(zip(self.feature_names, importances), key=lambda item: item[1], reverse=True)[:10]
            names = [name for name, _ in top][::-1]
            vals = [val for _, val in top][::-1]
            ax.barh(names, vals, color="#0f766e")
            ax.set_title(f"Top 10 feature quan trọng - {self.selected_model_name}", fontsize=11, weight="bold")
            ax.grid(axis="x", alpha=0.18)
        canvas.fig.tight_layout()
        return canvas

    def get_feature_importance(self):
        model = self.model
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if isinstance(model, Pipeline):
            estimator = model.named_steps.get("model")
            if hasattr(estimator, "coef_"):
                return np.abs(estimator.coef_[0])
            if hasattr(estimator, "feature_importances_"):
                return estimator.feature_importances_
        return None

    def chart_probability_distribution(self):
        canvas = MplCanvas(width=5.6, height=3.7)
        ax = canvas.fig.add_subplot(111)
        probs = self.metrics["probs"]
        y_test = self.metrics["y_test"]
        ax.hist(probs[y_test == 0], bins=24, alpha=0.72, label="Không mua", color="#64748b")
        ax.hist(probs[y_test == 1], bins=24, alpha=0.72, label="Có mua", color="#f97316")
        ax.axvline(0.5, color="#dc2626", linestyle="--", linewidth=1.5, label="Ngưỡng 50%")
        ax.set_title("Phân bố xác suất dự đoán", fontsize=11, weight="bold")
        ax.set_xlabel("Xác suất mua hàng")
        ax.set_ylabel("Số phiên")
        ax.legend()
        canvas.fig.tight_layout()
        return canvas

    def chart_month_revenue(self):
        canvas = MplCanvas(width=5.6, height=3.7)
        ax = canvas.fig.add_subplot(111)
        order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_rate = self.raw_df.groupby("Month")["Revenue"].apply(lambda s: s.astype(bool).mean())
        vals = [month_rate.get(month, 0) for month in order]
        ax.plot(order, vals, marker="o", linewidth=2.2, color="#2563eb")
        ax.fill_between(order, vals, alpha=0.12, color="#2563eb")
        ax.set_title("Tỷ lệ mua hàng theo tháng", fontsize=11, weight="bold")
        ax.set_ylabel("Revenue rate")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
        canvas.fig.tight_layout()
        return canvas

    def chart_algorithm_comparison(self):
        canvas = MplCanvas(width=5.6, height=3.7)
        ax = canvas.fig.add_subplot(111)
        names = list(self.metrics_by_model.keys())
        x = np.arange(len(names))
        width = 0.25
        acc = [self.metrics_by_model[name]["accuracy"] for name in names]
        f1 = [self.metrics_by_model[name]["f1"] for name in names]
        rec = [self.metrics_by_model[name]["recall"] for name in names]
        ax.bar(x - width, acc, width, label="Accuracy", color="#2563eb")
        ax.bar(x, f1, width, label="F1", color="#16a34a")
        ax.bar(x + width, rec, width, label="Recall", color="#f59e0b")
        ax.set_ylim(0, 1)
        ax.set_xticks(x, names, rotation=12)
        ax.set_title("So sánh Accuracy, F1, Recall", fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.legend()
        canvas.fig.tight_layout()
        return canvas

    def build_data_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        info = QLabel("Xem nhanh 200 dòng đầu của dataset để giải thích các feature đầu vào.")
        export_btn = QPushButton("Mở file CSV khác")
        export_btn.clicked.connect(self.open_csv_dialog)
        toolbar.addWidget(info, 1)
        toolbar.addWidget(export_btn)
        layout.addLayout(toolbar)

        table = QTableWidget()
        sample = self.raw_df.head(200)
        table.setRowCount(len(sample))
        table.setColumnCount(len(sample.columns))
        table.setHorizontalHeaderLabels(sample.columns.tolist())
        for r, (_, row) in enumerate(sample.iterrows()):
            for c, col in enumerate(sample.columns):
                item = QTableWidgetItem(str(row[col]))
                if col == "Revenue":
                    item.setBackground(QColor("#dcfce7") if str(row[col]).upper() == "TRUE" else QColor("#fee2e2"))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)
        return page

    def open_csv_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file CSV", str(PROJECT_ROOT / "data"), "CSV Files (*.csv)")
        if path:
            QMessageBox.information(
                self,
                "Gợi ý",
                "UI demo hiện đang cố định theo model đã huấn luyện từ shopping.csv. "
                f"Bạn vừa chọn: {path}",
            )


STYLE = """
QWidget {
    font-family: Segoe UI, Arial;
    font-size: 10.5pt;
    color: #111827;
}
QMainWindow, QTabWidget::pane {
    background: #f4f6f8;
}
#header {
    background: #0f172a;
    border-radius: 8px;
}
#title {
    color: #ffffff;
    font-size: 20pt;
    font-weight: 700;
}
#subtitle {
    color: #cbd5e1;
    font-size: 10.5pt;
}
#badge {
    background: #e0f2fe;
    color: #075985;
    border-radius: 6px;
    padding: 10px 14px;
    font-weight: 700;
}
#badgeSmall {
    background: #e0f2fe;
    color: #075985;
    border-radius: 6px;
    padding: 6px 10px;
    font-weight: 700;
}
#card, QGroupBox, #resultPanel, #story {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
#cardTitle {
    color: #64748b;
    font-weight: 600;
}
#cardValue {
    color: #0f172a;
    font-size: 24pt;
    font-weight: 800;
}
#cardNote {
    color: #6b7280;
}
QGroupBox {
    font-weight: 700;
    margin-top: 10px;
    padding: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background: #e5e7eb;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #dbeafe;
    border-color: #93c5fd;
}
#primaryButton {
    background: #2563eb;
    color: white;
    border-color: #1d4ed8;
    padding: 11px 12px;
}
#resultTitle {
    font-size: 20pt;
    font-weight: 800;
    color: #0f172a;
}
#probability {
    font-size: 16pt;
    font-weight: 700;
    color: #2563eb;
}
#recommendation {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px;
}
QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    height: 24px;
    text-align: center;
    background: #f8fafc;
    font-weight: 700;
}
QProgressBar::chunk {
    background: #22c55e;
    border-radius: 6px;
}
QTabBar::tab {
    background: #e5e7eb;
    border: 1px solid #cbd5e1;
    padding: 9px 16px;
    margin-right: 3px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 650;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1d4ed8;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 8px;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = DemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
