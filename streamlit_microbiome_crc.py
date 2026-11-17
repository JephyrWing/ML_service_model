from pathlib import Path

import joblib
import matplotlib

# 웹 페이지에서는 화면 창 없이 그래프를 생성합니다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# 페이지 설정
st.set_page_config(page_title="장내 미생물 조성 예측", layout="centered")
st.title("장내 미생물 조성 예측")
st.caption("Normal / CRC")
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 7:3 분할로 학습한 실험 1번 모델 사용
selected_experiment = "1"


# 선택한 모델·Scaler·서열-학명 대응표 로딩
project_dir = Path(__file__).resolve().parent
output_dir = project_dir / "outputs" / "simple_oob"
model_path = output_dir / (str(selected_experiment) + "_model.joblib")
if not model_path.exists():
    st.error(f"{model_path.name} 파일이 필요합니다.")
    st.stop()

bundle = joblib.load(model_path)
model = bundle["model"]
scaler = bundle["scaler"]
normalization_columns = bundle["scaler_input_columns"]
model_features = bundle["model_input_columns"]
sequence_to_name = bundle["sequence_scientific_name_mapping"].set_index("sequence")["subject_scientific_name"]


# 한 샘플의 CSV 입력 형식
st.subheader("1. ASV read CSV 업로드")
st.write("한 샘플의 전체 ASV read를 UTF-8 CSV로 올려 주세요.")
example_sequence = sequence_to_name[sequence_to_name == model_features[0]].index[0]
st.code(f"sequence,read_count\n{example_sequence},120", language="csv")
st.caption(
    "sequence 열에는 DNA 서열을, read_count 열에는 해당 서열의 read 수를 넣어 주세요. "
    "같은 참조 학명의 ASV read는 합산하며, CSV에 없는 서열은 미검출(0)로 처리합니다.")
uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])
if uploaded_file is None:
    st.stop()


# CSV 형식·read count 확인
try:
    input_df = pd.read_csv(uploaded_file, encoding="utf-8")
except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
    st.error("UTF-8로 저장된 쉼표 구분 CSV 파일을 확인해 주세요.")
    st.stop()

if set(input_df.columns) != {"sequence", "read_count"}:
    st.error("열 이름은 sequence,read_count로 작성해 주세요.")
    st.stop()

input_df["sequence"] = input_df["sequence"].astype("string").str.strip().str.upper()
input_df["read_count"] = pd.to_numeric(input_df["read_count"], errors="coerce")
if input_df.empty or input_df["sequence"].isna().any() or input_df["sequence"].eq("").any():
    st.error("DNA 서열이 비어 있습니다.")
    st.stop()
if (not np.isfinite(input_df["read_count"]).all()
        or (input_df["read_count"] < 0).any()
        or (input_df["read_count"] % 1 != 0).any()):
    st.error("read_count에는 결측 없이 0 이상의 정수를 넣어 주세요.")
    st.stop()

total_reads = float(input_df["read_count"].sum())
if not np.isfinite(total_reads) or total_reads <= 0:
    st.error("전체 read 합은 0보다 커야 합니다.")
    st.stop()


# DNA 서열을 참조 학명에 연결하고 같은 학명의 read 합산
input_df["scientific_name"] = input_df["sequence"].map(sequence_to_name)
counts = input_df.groupby("scientific_name")["read_count"].sum()
if not counts.reindex(model_features, fill_value=0).gt(0).any():
    st.error("모델에 사용된 DNA 서열과 일치하는 양수 read가 없습니다. 서열을 확인해 주세요.")
    st.stop()


# 미매칭 read를 분모에 포함해 정규화한 뒤 선택 학명만 모델에 입력
X_count = counts.reindex(normalization_columns, fill_value=0).to_frame().T
X_count["unmatched_reads"] = input_df.loc[input_df["scientific_name"].isna(), "read_count"].sum()
X_all = pd.DataFrame(scaler.transform(X_count), columns=normalization_columns)
X_input = X_all[model_features]
matched_count = int(counts.reindex(model_features, fill_value=0).gt(0).sum())
unmatched_count = int(input_df["scientific_name"].isna().sum())
st.caption(
    f"전체 read {total_reads:,.0f} · 모델 학명 중 검출 {matched_count}/{len(model_features)}개 · "
    f"서열 미매칭 {unmatched_count}행")

if not st.button("예측하기", type="primary"):
    st.stop()


# 분류 결과와 클래스별 predict_proba 표시
prediction = model.predict(X_input)[0]
probabilities = model.predict_proba(X_input)[0]
classes = model.classes_.tolist()
st.subheader("2. 예측 결과")
if prediction == "CRC":
    st.info("대장암 환자들의 장내 미생물 조성과 비슷합니다.")
else:
    st.info("정상군(Healthy·용종 포함)의 장내 미생물 조성과 비슷합니다.")

crc_column, normal_column = st.columns(2)
crc_column.metric("CRC 분류 확률", f"{probabilities[classes.index('CRC')]:.1%}")
normal_column.metric("Normal 분류 확률", f"{probabilities[classes.index('Normal')]:.1%}")
st.caption("학습용 모델의 분류 결과이며, 대장암 진단이나 실제 발병 확률을 뜻하지 않습니다.")


# 이번 입력의 예측 클래스에 대한 SHAP 기여도 계산
rf = model.named_steps["rf"]
explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
explanation = explainer(X_input)
class_index = classes.index(prediction)
contribution_df = pd.DataFrame({
    "scientific_name": model_features,
    "contribution": explanation.values[0, :, class_index]
})
contribution_df["absolute_contribution"] = contribution_df["contribution"].abs()
top_features = contribution_df.sort_values("absolute_contribution", ascending=False).head(10)
top_features = top_features.iloc[::-1]


# 학명으로 상위 10개 기여도 그래프 표시
st.subheader("3. 이번 예측에 영향을 준 학명별 특성")
labels = top_features["scientific_name"]
colors = np.where(top_features["contribution"] >= 0, "#db6b59", "#4a84b5")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(labels, top_features["contribution"] * 100, color=colors)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel(f"{prediction} 분류 확률에 대한 SHAP 기여도 (%p)")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(
    f"절댓값이 큰 순서로 최대 10개를 표시합니다. 빨강은 {prediction} 분류 확률을 높이는 방향, "
    "파랑은 낮추는 방향입니다. 학명은 BLAST 참조 학명입니다.")
