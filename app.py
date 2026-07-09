# =============================================================================
# ỨNG DỤNG STREAMLIT: DỰ BÁO RỦI RO TÍN DỤNG (PD) BẰNG LOGISTIC REGRESSION
# Tái hiện pipeline từ notebook mohinh.ipynb trên dữ liệu khung 5C (5c.csv)
# =============================================================================

import streamlit as st

# ---- 1) set_page_config: LỆNH STREAMLIT ĐẦU TIÊN ----------------------------
st.set_page_config(
    layout="wide",
    page_title="Mô hình 5C",
    page_icon="💕",
)

# ---- 2) Import & hàm dùng chung ---------------------------------------------
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Tập biến trích xuất CHÍNH XÁC từ notebook
FEATURES = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]
TARGET = "PD"

NHOM_BIEN = {
    "TC": "Tư cách (Character)",
    "NL": "Năng lực (Capacity)",
    "DK": "Điều kiện (Conditions)",
    "V": "Vốn (Capital)",
    "TS": "Tài sản đảm bảo (Collateral)",
}

CHART_HEIGHT = 340


@st.cache_data(show_spinner="Đang nạp dữ liệu...")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Hàm nạp dữ liệu DÙNG CHUNG: nhận bytes (hashable) → DataFrame.

    Notebook không tạo biến phái sinh, chỉ đọc CSV nguyên trạng.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df


def validate_columns(df: pd.DataFrame, required: list) -> list:
    """Trả về danh sách cột bị thiếu so với yêu cầu."""
    return [c for c in required if c not in df.columns]


def plot_for_variable(df: pd.DataFrame, col: str):
    """Tự chọn loại biểu đồ theo kiểu biến (suy từ dtype & bản chất dữ liệu)."""
    s = df[col].dropna()
    if col == TARGET:
        # Mục tiêu phân loại nhị phân → bar phân phối lớp
        vc = s.map({0: "0 - Không rủi ro", 1: "1 - Có rủi ro"}).value_counts()
        fig = px.bar(
            x=vc.index, y=vc.values,
            labels={"x": "Nhóm PD", "y": "Số quan sát"},
            title=f"Phân phối biến mục tiêu {col}",
            color=vc.index,
            color_discrete_sequence=["#2E86AB", "#E4572E"],
        )
        fig.update_layout(showlegend=False, height=CHART_HEIGHT)
        return fig
    if pd.api.types.is_numeric_dtype(s):
        nunique = s.nunique()
        if nunique <= 10:
            # Rời rạc (thang Likert 1-5) → bar theo value_counts
            vc = s.value_counts().sort_index()
            fig = px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": col, "y": "Số quan sát"},
                title=f"Phân phối {col}",
                color_discrete_sequence=["#2E86AB"],
            )
        else:
            fig = px.histogram(s, x=col, title=f"Phân phối {col}",
                               color_discrete_sequence=["#2E86AB"])
        fig.update_layout(height=CHART_HEIGHT, showlegend=False)
        return fig
    # Phân loại dạng chuỗi → bar
    vc = s.value_counts().head(20)
    fig = px.bar(x=vc.index.astype(str), y=vc.values,
                 labels={"x": col, "y": "Số quan sát"},
                 title=f"Phân phối {col}")
    fig.update_layout(height=CHART_HEIGHT, showlegend=False)
    return fig


# ---- 3) SIDEBAR — VÙNG CẤU HÌNH ---------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải tệp dữ liệu (.csv)",
        type=["csv"],
        help="Tệp CSV có cấu trúc như 5c.csv: 24 biến khung 5C (TC1–TC5, NL1–NL4, "
             "DK1–DK5, V1–V6, TS1–TS4) và biến mục tiêu PD (0/1).",
    )

    st.subheader("Tham số mô hình AI")
    # Notebook dùng LogisticRegression() với siêu tham số MẶC ĐỊNH của sklearn
    c_value = st.slider(
        "C (nghịch đảo cường độ hiệu chỉnh)",
        min_value=0.01, max_value=10.0, value=1.0, step=0.01,
        help="Giá trị C nhỏ → hiệu chỉnh (regularization) mạnh hơn. "
             "Mặc định của notebook: 1.0.",
    )
    max_iter = st.slider(
        "Số vòng lặp tối đa (max_iter)",
        min_value=100, max_value=2000, value=100, step=100,
        help="Số vòng lặp tối đa của thuật toán tối ưu. Mặc định của notebook: 100. "
             "Tăng lên nếu gặp cảnh báo không hội tụ.",
    )
    with st.expander("Tham số nâng cao"):
        solver = st.selectbox(
            "Thuật toán tối ưu (solver)",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Mặc định của notebook: lbfgs.",
        )
        test_size = st.slider(
            "Tỷ lệ tập kiểm định (test_size)",
            min_value=0.1, max_value=0.4, value=0.2, step=0.05,
            help="Tỷ lệ dữ liệu dành cho tập kiểm định. Notebook dùng 0.2.",
        )
        random_state = st.number_input(
            "Random state",
            min_value=0, max_value=9999, value=32, step=1,
            help="Hạt giống ngẫu nhiên khi chia dữ liệu. Notebook dùng 32.",
        )

    st.divider()
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
        help="Chia dữ liệu train/test và huấn luyện Logistic Regression.",
    )

# ---- 4) HEADER — VÙNG ĐỊNH HƯỚNG --------------------------------------------
st.title("🏦 Dự báo rủi ro tín dụng khách hàng theo khung 5C")
st.caption(
    "Ứng dụng huấn luyện mô hình **Logistic Regression** dự báo xác suất vỡ nợ (PD) "
    "của khách hàng dựa trên 24 tiêu chí thuộc 5 nhóm: Tư cách (TC), Năng lực (NL), "
    "Điều kiện (DK), Vốn (V) và Tài sản đảm bảo (TS) — thang đo Likert 1–5. "
    "Đầu vào kỳ vọng: tệp CSV có cấu trúc như 5c.csv."
)

if uploaded_file is None:
    st.info(
        "👈 Vui lòng tải tệp dữ liệu **.csv** ở thanh bên trái để bắt đầu. "
        "Tệp cần chứa 24 biến đầu vào (TC1–TC5, NL1–NL4, DK1–DK5, V1–V6, TS1–TS4) "
        "và biến mục tiêu **PD** (0 = không rủi ro, 1 = có rủi ro)."
    )
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    df = load_data(file_bytes)
except Exception as e:
    st.error(f"❌ Không đọc được tệp dữ liệu. Vui lòng kiểm tra định dạng CSV. Chi tiết: {e}")
    st.stop()

if df.empty:
    st.error("❌ Tệp dữ liệu rỗng. Vui lòng kiểm tra lại.")
    st.stop()

missing_cols = validate_columns(df, FEATURES + [TARGET])
if missing_cols:
    st.error(
        "❌ Tệp dữ liệu thiếu các cột bắt buộc: **" + ", ".join(missing_cols) + "**. "
        "Vui lòng kiểm tra lại cấu trúc tệp (tham chiếu 5c.csv)."
    )
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(
    f"Tóm tắt nhanh: {df.shape[0]:,} quan sát × {df.shape[1]} cột · "
    f"Biến mục tiêu PD: {int((df[TARGET] == 0).sum())} không rủi ro / "
    f"{int((df[TARGET] == 1).sum())} có rủi ro."
)
st.divider()

# ---- 5) KHỐI TRAIN (chạy khi bấm nút, lưu session_state) ---------------------
if train_clicked:
    try:
        X = df[FEATURES]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        model = LogisticRegression(C=c_value, max_iter=int(max_iter), solver=solver)
        model.fit(X_train, y_train)

        yhat_test = model.predict(X_test)
        yproba_test = model.predict_proba(X_test)[:, 1]

        # Bảng kết quả đã chấm điểm trên tập test
        scored = X_test.copy()
        scored["PD_thực_tế"] = y_test.values
        scored["PD_dự_báo"] = yhat_test
        scored["Xác_suất_rủi_ro"] = yproba_test

        st.session_state["model"] = model                 # (1) mô hình đã fit
        st.session_state["preprocessor"] = None           # (2) notebook KHÔNG dùng scaler/encoder
        st.session_state["scored_df"] = scored            # (3) bảng kết quả đã chấm điểm
        st.session_state["eval"] = {
            "y_test": y_test, "yhat_test": yhat_test, "yproba_test": yproba_test,
            "n_train": len(X_train), "n_test": len(X_test),
            "params": {"C": c_value, "max_iter": int(max_iter), "solver": solver,
                       "test_size": test_size, "random_state": int(random_state)},
        }
        st.success(
            f"✅ Huấn luyện thành công! Tập huấn luyện: {len(X_train)} quan sát · "
            f"Tập kiểm định: {len(X_test)} quan sát. Xem chi tiết ở tab "
            f"**Kết quả huấn luyện & Kiểm định**."
        )
    except Exception as e:
        st.error(f"❌ Lỗi khi huấn luyện mô hình: {e}")

# ---- 6) CÁC TAB NỘI DUNG ------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tổng quan dữ liệu",
    "📊 Trực quan hóa dữ liệu",
    "🎯 Kết quả huấn luyện & Kiểm định",
    "🔮 Sử dụng mô hình",
])

# ========== TAB 1: TỔNG QUAN DỮ LIỆU ==========
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{df.shape[0]:,}")
    c2.metric("Số cột", f"{df.shape[1]:,}")
    c3.metric("Dung lượng tệp", f"{uploaded_file.size / (1024 * 1024):.3f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=320):
        st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Thống kê mô tả các biến của mô hình")
    st.caption("Chỉ mô tả 24 biến đầu vào (X) và biến mục tiêu PD (y) được đưa vào mô hình.")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# ========== TAB 2: TRỰC QUAN HÓA DỮ LIỆU ==========
with tab2:
    all_vars = [TARGET] + FEATURES  # mục tiêu ưu tiên trước
    default_vars = [TARGET, "TC1", "NL1", "TS1"]  # 4 biến ưu tiên: y + đại diện các nhóm
    selected = st.multiselect(
        "Chọn biến cần trực quan hóa (tối đa nên chọn 4 để bố cục cân đối)",
        options=all_vars,
        default=default_vars,
        help="PD là biến mục tiêu; các biến còn lại là tiêu chí 5C thang Likert 1–5.",
    )
    if not selected:
        st.info("Vui lòng chọn ít nhất một biến để vẽ biểu đồ.")
    else:
        # Lưới 2 cột, mỗi hàng 2 biểu đồ
        for i in range(0, len(selected), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(selected[i:i + 2]):
                with cols[j]:
                    st.plotly_chart(
                        plot_for_variable(df, col_name),
                        use_container_width=True,
                    )

# ========== TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH ==========
with tab3:
    if "model" not in st.session_state:
        st.info("⏳ Mô hình chưa được huấn luyện. Vui lòng bấm nút "
                "**🚀 Huấn luyện mô hình** ở thanh bên trái.")
    else:
        ev = st.session_state["eval"]
        y_test, yhat_test, yproba = ev["y_test"], ev["yhat_test"], ev["yproba_test"]

        st.caption(
            f"Mô hình: Logistic Regression · C={ev['params']['C']} · "
            f"max_iter={ev['params']['max_iter']} · solver={ev['params']['solver']} · "
            f"test_size={ev['params']['test_size']} · random_state={ev['params']['random_state']}"
        )

        # Chỉ tiêu vô hướng
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{accuracy_score(y_test, yhat_test):.4f}")
        m2.metric("Precision", f"{precision_score(y_test, yhat_test, zero_division=0):.4f}")
        m3.metric("Recall", f"{recall_score(y_test, yhat_test, zero_division=0):.4f}")
        m4.metric("F1-score", f"{f1_score(y_test, yhat_test, zero_division=0):.4f}")
        try:
            auc_val = roc_auc_score(y_test, yproba)
            m5.metric("ROC-AUC", f"{auc_val:.4f}")
        except ValueError:
            m5.metric("ROC-AUC", "N/A")
            auc_val = None

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, yhat_test)
            labels = ["0 - Không rủi ro", "1 - Có rủi ro"]
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=labels, y=labels,
                labels={"x": "Dự báo", "y": "Thực tế", "color": "Số quan sát"},
            )
            fig_cm.update_layout(height=CHART_HEIGHT + 40)
            st.plotly_chart(fig_cm, use_container_width=True)
        with colB:
            st.subheader("Đường cong ROC")
            if auc_val is not None:
                fpr, tpr, _ = roc_curve(y_test, yproba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"ROC (AUC = {auc_val:.4f})",
                    line=dict(color="#2E86AB", width=3),
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Ngẫu nhiên", line=dict(dash="dash", color="gray"),
                ))
                fig_roc.update_layout(
                    xaxis_title="Tỷ lệ dương tính giả (FPR)",
                    yaxis_title="Tỷ lệ dương tính thật (TPR)",
                    height=CHART_HEIGHT + 40,
                    legend=dict(x=0.55, y=0.05),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Không tính được ROC do tập kiểm định chỉ có một lớp.")

        st.subheader("Classification report")
        report = classification_report(
            y_test, yhat_test, output_dict=True, zero_division=0
        )
        st.dataframe(
            pd.DataFrame(report).T.round(4), use_container_width=True
        )

        st.subheader("Bảng kết quả chấm điểm trên tập kiểm định")
        with st.container(height=320):
            st.dataframe(st.session_state["scored_df"], use_container_width=True)

# ========== TAB 4: SỬ DỤNG MÔ HÌNH ==========
with tab4:
    if "model" not in st.session_state:
        st.info("⏳ Mô hình chưa được huấn luyện. Vui lòng bấm nút "
                "**🚀 Huấn luyện mô hình** ở thanh bên trái.")
    else:
        model = st.session_state["model"]
        # Notebook không dùng scaler/encoder → dữ liệu mới đưa thẳng vào mô hình
        mode = st.radio(
            "Chọn chế độ dự báo",
            ["✍️ Nhập trực tiếp một khách hàng", "📂 Tải tệp dự báo hàng loạt"],
            horizontal=True,
        )

        # ----- CHẾ ĐỘ 1: NHẬP TRỰC TIẾP -----
        if mode.startswith("✍️"):
            st.caption(
                "Nhập điểm đánh giá (thang 1–5) cho 24 tiêu chí 5C của khách hàng, "
                "sau đó bấm **Dự báo**. Giá trị mặc định là trung vị của dữ liệu."
            )
            medians = df[FEATURES].median()
            mins = df[FEATURES].min()
            maxs = df[FEATURES].max()

            with st.form("form_du_bao"):
                inputs = {}
                # Gom theo nhóm 5C cho dễ nhập
                for prefix, group_name in NHOM_BIEN.items():
                    group_feats = [f for f in FEATURES if f.rstrip("0123456789") == prefix]
                    st.markdown(f"**{group_name}**")
                    cols = st.columns(len(group_feats))
                    for k, feat in enumerate(group_feats):
                        with cols[k]:
                            inputs[feat] = st.number_input(
                                feat,
                                min_value=int(mins[feat]),
                                max_value=int(maxs[feat]),
                                value=int(medians[feat]),
                                step=1,
                                help=f"Điểm {feat} (khoảng {int(mins[feat])}–{int(maxs[feat])} theo dữ liệu).",
                            )
                submitted = st.form_submit_button("🔮 Dự báo", type="primary",
                                                  use_container_width=True)

            if submitted:
                try:
                    X_new = pd.DataFrame([inputs])[FEATURES]
                    pred = int(model.predict(X_new)[0])
                    proba = model.predict_proba(X_new)[0]
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Kết quả dự báo (PD)", pred)
                    r2.metric("Xác suất KHÔNG rủi ro", f"{proba[0] * 100:.2f}%")
                    r3.metric("Xác suất CÓ rủi ro", f"{proba[1] * 100:.2f}%")
                    if pred == 1:
                        st.error("⚠️ Khách hàng được dự báo **CÓ rủi ro tín dụng** (PD = 1).")
                    else:
                        st.success("✅ Khách hàng được dự báo **KHÔNG có rủi ro tín dụng** (PD = 0).")
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự báo: {e}")

        # ----- CHẾ ĐỘ 2: DỰ BÁO HÀNG LOẠT -----
        else:
            st.caption(
                "Tải tệp CSV chứa **đúng 24 cột biến đầu vào**: "
                + ", ".join(FEATURES) + ". Các cột khác (nếu có) sẽ được bỏ qua."
            )
            batch_file = st.file_uploader(
                "Tệp dữ liệu cần dự báo (.csv)",
                type=["csv"],
                key="batch_uploader",
                help="Cấu trúc cột giống X_test: 24 biến TC/NL/DK/V/TS.",
            )
            if batch_file is not None:
                try:
                    df_new = pd.read_csv(batch_file)
                except Exception as e:
                    st.error(f"❌ Không đọc được tệp: {e}")
                    df_new = None

                if df_new is not None:
                    if df_new.empty:
                        st.error("❌ Tệp dự báo rỗng.")
                    else:
                        missing = validate_columns(df_new, FEATURES)
                        if missing:
                            st.error(
                                "❌ Tệp thiếu các cột bắt buộc: **"
                                + ", ".join(missing) + "**."
                            )
                        else:
                            try:
                                X_batch = df_new[FEATURES]
                                preds = model.predict(X_batch)
                                probas = model.predict_proba(X_batch)
                                result = df_new.copy()
                                result["PD_dự_báo"] = preds
                                result["Xác_suất_không_rủi_ro"] = probas[:, 0]
                                result["Xác_suất_có_rủi_ro"] = probas[:, 1]

                                n_risk = int((preds == 1).sum())
                                b1, b2, b3 = st.columns(3)
                                b1.metric("Tổng số khách hàng", len(result))
                                b2.metric("Dự báo CÓ rủi ro", n_risk)
                                b3.metric("Tỷ lệ rủi ro", f"{n_risk / len(result) * 100:.1f}%")

                                with st.container(height=320):
                                    st.dataframe(result, use_container_width=True)

                                csv_out = result.to_csv(index=False).encode("utf-8-sig")
                                st.download_button(
                                    "⬇️ Tải kết quả dự báo (CSV)",
                                    data=csv_out,
                                    file_name="ket_qua_du_bao_PD.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.error(f"❌ Lỗi khi dự báo hàng loạt: {e}")
