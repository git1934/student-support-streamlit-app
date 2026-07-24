# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "student_support_dummy_summary_30students_202604.csv"

ID_COL = "児童ID"
GRADE_COL = "学年"
CLASS_COL = "クラス"
INTERNAL_GRADE_COL = "_grade"
INTERNAL_CLASS_COL = "_class"
RAW_COLUMNS = [
    "学年",
    "クラス",
    "児童ID",
    "病気欠席数",
    "事故欠席数",
    "遅刻数",
    "早退数",
    "忌引等数",
    "出席停止数",
    "保健室利用数",
    "心の天気晴れ数",
    "心の天気曇り数",
    "心の天気雨数",
]

DEFAULT_FEATURES = [
    "病気欠席数",
    "事故欠席数",
    "遅刻数",
    "早退数",
    "保健室利用数",
    "心の天気曇り率",
    "心の天気雨率",
    "心の天気晴れ率",
]

DEFAULT_OFF_FEATURES = {"忌引等数", "出席停止数"}
LOW_IS_CONCERNING_DEFAULT = {"心の天気晴れ数", "心の天気晴れ率"}

LABEL_ORDER_3 = ["低", "中", "高"]
FLAG_ORDER = ["0", "1"]
LABEL_COLORS = {"低": "#2E7D32", "中": "#ED6C02", "高": "#C62828", "0": "#2E7D32", "1": "#C62828"}


st.set_page_config(
    page_title="要サポート児童ラベル付け",
    page_icon="🏫",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the bundled dummy data. No upload feature is intentionally provided."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    if GRADE_COL not in df.columns:
        df.insert(0, GRADE_COL, "5年")
    if CLASS_COL not in df.columns:
        df.insert(1, CLASS_COL, "1組")
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"必要なカラムが不足しています: {missing}")
    out = df[RAW_COLUMNS].copy()
    out[INTERNAL_GRADE_COL] = out[GRADE_COL].astype(str)
    out[INTERNAL_CLASS_COL] = out[CLASS_COL].astype(str)
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = out["心の天気晴れ数"] + out["心の天気曇り数"] + out["心の天気雨数"]
    safe_total = total.replace(0, np.nan)

    out["心の天気入力数"] = total
    out["心の天気晴れ率"] = (out["心の天気晴れ数"] / safe_total * 100).fillna(0).round(1)
    out["心の天気曇り率"] = (out["心の天気曇り数"] / safe_total * 100).fillna(0).round(1)
    out["心の天気雨率"] = (out["心の天気雨数"] / safe_total * 100).fillna(0).round(1)

    out["欠席合計（病気＋事故）"] = out["病気欠席数"] + out["事故欠席数"]
    out["遅刻早退合計"] = out["遅刻数"] + out["早退数"]
    return out


def minmax_to_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    min_v = float(s.min())
    max_v = float(s.max())
    if np.isclose(max_v, min_v):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v) * 100


def build_risk_matrix(
    df: pd.DataFrame,
    selected_features: List[str],
    directions: Dict[str, str],
    comparison_basis: str,
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=df.index)
    group_cols = (
        [INTERNAL_GRADE_COL, INTERNAL_CLASS_COL]
        if comparison_basis == "クラス内比較"
        else [INTERNAL_GRADE_COL]
    )
    for feature in selected_features:
        normalized = df.groupby(group_cols, dropna=False)[feature].transform(minmax_to_100)
        if directions.get(feature) == "低いほど気になる":
            normalized = 100 - normalized
        matrix[feature] = normalized.clip(0, 100)
    return matrix


def calculate_score(risk_matrix: pd.DataFrame, weights: Dict[str, int]) -> pd.Series:
    if risk_matrix.empty:
        return pd.Series(np.zeros(len(risk_matrix)), index=risk_matrix.index)

    weight_values = np.array([weights.get(c, 1.0) for c in risk_matrix.columns], dtype=float)
    weight_values = np.maximum(weight_values, 0)
    if np.isclose(weight_values.sum(), 0):
        weight_values = np.ones_like(weight_values)
    weighted = risk_matrix.to_numpy(dtype=float) * weight_values
    score = weighted.sum(axis=1) / weight_values.sum()
    return pd.Series(score, index=risk_matrix.index).round(1)


def label_three_levels(score: pd.Series, low_mid: int, mid_high: int) -> pd.Series:
    return pd.Series(
        np.select(
            [score < low_mid, score < mid_high],
            ["低", "中"],
            default="高",
        ),
        index=score.index,
    )


def flag_binary(score: pd.Series, threshold: int) -> pd.Series:
    return (score >= threshold).astype(int).astype(str)


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numeric_cols].agg(["count", "mean", "min", "max"]).T.reset_index()
    summary.columns = ["変数", "件数", "平均", "最小", "最大"]
    summary["欠損数"] = df[numeric_cols].isna().sum().values
    summary["平均"] = summary["平均"].round(2)
    return summary[["変数", "件数", "欠損数", "平均", "最小", "最大"]]


def summarize_main_factors(
    risk_matrix: pd.DataFrame,
    weights: Dict[str, int],
    top_n: int = 3,
) -> pd.Series:
    if risk_matrix.empty:
        return pd.Series([""] * len(risk_matrix), index=risk_matrix.index)

    positive_weights = {feature: max(int(weights.get(feature, 0)), 0) for feature in risk_matrix.columns}
    if sum(positive_weights.values()) == 0:
        positive_weights = {feature: 1 for feature in risk_matrix.columns}

    def row_factors(row: pd.Series) -> str:
        contributions = [
            (feature, float(row[feature]) * positive_weights[feature])
            for feature in risk_matrix.columns
            if positive_weights[feature] > 0
        ]
        contributions = sorted(contributions, key=lambda item: item[1], reverse=True)
        main_features = [feature for feature, value in contributions if value > 0][:top_n]
        return "、".join(main_features) if main_features else "該当なし"

    return risk_matrix.apply(row_factors, axis=1)


def display_result_table(table_df: pd.DataFrame) -> None:
    st.dataframe(
        table_df,
        column_config={
            "サポート必要度スコア": st.column_config.NumberColumn(format="%.1f"),
        },
        use_container_width=True,
        hide_index=True,
    )


def label_badge(label: str) -> str:
    color = {
        "低": "#E8F5E9",
        "中": "#FFF3E0",
        "高": "#FFEBEE",
        "0": "#E8F5E9",
        "1": "#FFEBEE",
    }.get(label, "#F5F5F5")
    text_color = {
        "低": "#1B5E20",
        "中": "#E65100",
        "高": "#B71C1C",
        "0": "#1B5E20",
        "1": "#B71C1C",
    }.get(label, "#111827")
    display = {"0": "0：通常", "1": "1：要サポート候補"}.get(label, label)
    return (
        f"<span style='background:{color}; color:{text_color}; padding:6px 10px; "
        f"border-radius:999px; font-weight:700;'>{display}</span>"
    )


def build_function_requirements_markdown(
    main_output: str,
    threshold_low_mid: int,
    threshold_mid_high: int,
    flag_threshold: int,
    comparison_basis: str,
    selected_features: List[str],
    directions: Dict[str, str],
    weights: Dict[str, int],
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    main_label_col: str,
    main_order: List[str],
) -> str:
    """現在の設定値・利用データ・出力結果を含む機能要件書をMarkdown形式で生成する。"""

    generated_at = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M")
    data_start = "不明"
    data_end = "不明"

    # このサンプルデータは集計済みデータのため、日付列がない場合は画面仕様の対象期間を記載する。
    if "日付" in raw_df.columns:
        parsed_dates = pd.to_datetime(raw_df["日付"], errors="coerce").dropna()
        if not parsed_dates.empty:
            data_start = parsed_dates.min().strftime("%Y/%m/%d")
            data_end = parsed_dates.max().strftime("%Y/%m/%d")
    else:
        data_start = "2026/04/01"
        data_end = "2026/04/30"

    feature_rows = []
    for feature in selected_features:
        feature_rows.append(
            f"| {feature} | {directions.get(feature, '未設定')} | {weights.get(feature, 0)} |"
        )
    feature_table = "\n".join(feature_rows) or "| 選択なし | - | - |"

    label_counts = result_df[main_label_col].value_counts().reindex(main_order, fill_value=0)
    label_count_lines = "\n".join(
        f"- {label}：{int(label_counts[label])}名" for label in main_order
    )

    top_priority = result_df.sort_values(
        "サポート必要度スコア", ascending=False
    )[[GRADE_COL, CLASS_COL, ID_COL, "サポート必要度スコア", main_label_col, "主な要因"]].head(10)

    priority_rows = []
    for _, row in top_priority.iterrows():
        priority_rows.append(
            f"| {row[GRADE_COL]} | {row[CLASS_COL]} | {row[ID_COL]} | "
            f"{row['サポート必要度スコア']:.1f} | {row[main_label_col]} | {row['主な要因']} |"
        )
    priority_table = "\n".join(priority_rows)

    numeric_summary = summarize_numeric(df)
    summary_rows = []
    for _, row in numeric_summary.iterrows():
        summary_rows.append(
            f"| {row['変数']} | {int(row['件数'])} | {int(row['欠損数'])} | "
            f"{row['平均']} | {row['最小']} | {row['最大']} |"
        )
    numeric_summary_table = "\n".join(summary_rows)

    if main_output == "低／中／高ラベル":
        threshold_description = (
            f"- 低：スコア {threshold_low_mid} 未満\n"
            f"- 中：スコア {threshold_low_mid} 以上 {threshold_mid_high} 未満\n"
            f"- 高：スコア {threshold_mid_high} 以上"
        )
    else:
        threshold_description = (
            f"- 0：スコア {flag_threshold} 未満\n"
            f"- 1：スコア {flag_threshold} 以上"
        )

    return f"""# 児童サポート必要度ラベリング 機能要件書

## 1. ドキュメント情報

- 出力日時：{generated_at}
- アプリ名：要サポート児童ラベル付け
- 出力形式：Markdown
- 利用データ：`data/student_support_dummy_summary_30students_202604.csv`

## 2. 機能概要

児童ごとの生活面データを使用し、支援が必要そうな児童を確認するための
「サポート必要度スコア」とラベルを算出する。

本機能は文部科学省定義の不登校判定を行うものではなく、
教員や利用者が確認対象を絞り込むための補助的な目安として使用する。

## 3. 現在の画面設定

| 設定項目 | 設定値 |
|---|---|
| 分類手法 | {main_output} |
| 比較基準 | {comparison_basis} |
| 低・中の境界 | {threshold_low_mid} |
| 中・高の境界 | {threshold_mid_high} |
| 0/1フラグの境界 | {flag_threshold} |
| 選択変数数 | {len(selected_features)} |
| データ件数 | {len(raw_df)}件 |
| 学年数 | {raw_df[GRADE_COL].nunique()} |
| クラス数 | {raw_df[[GRADE_COL, CLASS_COL]].drop_duplicates().shape[0]} |
| 対象期間 | {data_start} 〜 {data_end} |

## 4. 選択変数・向き・重み

| 変数 | スコア方向 | 重み |
|---|---|---:|
{feature_table}

### 重みの意味

- 0：スコアに反映しない
- 1：標準
- 2〜5：重視
- 選択変数の重みがすべて0の場合は、計算継続のため全変数を重み1として扱う

## 5. スコア計算要件

1. 選択された変数だけをスコア計算に使用する。
2. 比較基準が「クラス内比較」の場合は、学年・クラス単位で正規化する。
3. 比較基準が「同学年内比較」の場合は、学年単位で正規化する。
4. 各変数をグループ内で0〜100点にMin-Max正規化する。
5. 「低いほど気になる」の変数は、正規化後に `100 - スコア` で反転する。
6. 各変数の正規化スコアに整数重みを掛ける。
7. 加重合計を重み合計で割り、0〜100のサポート必要度スコアを算出する。
8. 最終スコアは小数第1位に丸める。
9. 主な要因は、各変数の正規化スコア×重みの寄与度が高い順に最大3件表示する。

## 6. ラベル判定要件

現在選択されている分類手法：**{main_output}**

{threshold_description}

## 7. 現在の出力結果

### 7.1 ラベル分布

{label_count_lines}

### 7.2 サポート優先上位10名

| 学年 | クラス | 児童ID | スコア | ラベル | 主な要因 |
|---|---|---|---:|---|---|
{priority_table}

## 8. 利用データ要件

### 必須列

- 学年
- クラス
- 児童ID
- 病気欠席数
- 事故欠席数
- 遅刻数
- 早退数
- 忌引等数
- 出席停止数
- 保健室利用数
- 心の天気晴れ数
- 心の天気曇り数
- 心の天気雨数

### 自動生成する派生変数

- 心の天気入力数
- 心の天気晴れ率
- 心の天気曇り率
- 心の天気雨率
- 欠席合計（病気＋事故）
- 遅刻早退合計

### 利用データ概要

| 変数 | 件数 | 欠損数 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
{numeric_summary_table}

"""


raw_df = load_data()
df = add_derived_features(raw_df)

st.title("要サポート児童ラベル付け")
st.caption("2026年4月1日〜4月30日、5年3クラス90名のテストデータを内蔵しています。")
st.info(
    "サポート必要度スコアは、選択したカラムを比較基準ごとに0〜100へ換算し、整数重みを反映して集約した確認用の目安です。"
    "比較基準を変更すると、同じ児童でもスコアやラベルが変わることがあります。"
)

with st.expander("このアプリで行うこと", expanded=False):
    st.markdown(
        """
- 文科省定義の不登校判定ではなく、**支援が必要そうな児童を見つけるためのサポート必要度ラベル**を付けます。
- 複数の変数を選択し、変数ごとの向き・整数の重みを調整して、0〜100のサポート必要度スコアを作ります。
- 同じスコアから、**低／中／高ラベル** と **0/1フラグ** の両方を作れます。
- 心の天気は全児童が毎日入力する前提のため、晴れ率・曇り率・雨率を自動計算しています。
        """
    )

numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COL]

st.sidebar.header("ラベル設定")
main_output = st.sidebar.radio(
    "分類手法",
    ["低／中／高ラベル", "0/1フラグ"],
    index=0,
)

threshold_low_mid, threshold_mid_high = st.sidebar.slider(
    "低・中・高に分けるスコア基準",
    min_value=0,
    max_value=100,
    value=(35, 70),
    step=1,
    help="スコアが低いほど『低』、高いほど『高』になります。",
)
flag_threshold = st.sidebar.slider(
    "0/1フラグの境界",
    min_value=0,
    max_value=100,
    value=70,
    step=1,
    help="この値以上の児童を 1：要サポート候補 とします。",
)

st.sidebar.header("比較基準")
comparison_basis = st.sidebar.radio(
    "比較基準",
    ["クラス内比較", "同学年内比較"],
    index=0,
)

st.sidebar.header("予測に使うカラム")
selected_features = st.sidebar.multiselect(
    "予測に使うカラムを選択",
    options=numeric_features,
    default=[c for c in DEFAULT_FEATURES if c in numeric_features],
    help="忌引等数・出席停止数は初期設定では外していますが、必要に応じて追加できます。",
)

if not selected_features:
    st.warning("予測に使うカラムを1つ以上選択してください。")
    st.stop()

st.sidebar.header("向き・重み")
st.sidebar.caption("『低いほど気になる』を選ぶと、スコア計算時に値を反転します。重みは整数です。0はスコアに反映しない、1は標準、2以上は重視として扱います。")

directions: Dict[str, str] = {}
weights: Dict[str, int] = {}

for feature in selected_features:
    with st.sidebar.expander(feature, expanded=False):
        default_direction = "低いほど気になる" if feature in LOW_IS_CONCERNING_DEFAULT else "高いほど気になる"
        directions[feature] = st.radio(
            "向き",
            ["高いほど気になる", "低いほど気になる"],
            index=0 if default_direction == "高いほど気になる" else 1,
            key=f"direction_{feature}",
            horizontal=False,
        )
        default_weight = 1
        if feature in ["心の天気雨率", "心の天気雨数"]:
            default_weight = 2
        elif feature in DEFAULT_OFF_FEATURES:
            default_weight = 0
        weights[feature] = st.slider(
            "重み（整数）",
            min_value=0,
            max_value=5,
            value=int(default_weight),
            step=1,
            key=f"weight_{feature}",
            help="0=スコアに反映しない、1=標準、2〜5=重視",
        )

risk_matrix = build_risk_matrix(df, selected_features, directions, comparison_basis)
score = calculate_score(risk_matrix, weights)
three_label = label_three_levels(score, threshold_low_mid, threshold_mid_high)
binary_flag = flag_binary(score, flag_threshold)
main_factors = summarize_main_factors(risk_matrix, weights)

result_df = df.copy()
result_df.insert(1, "サポート必要度スコア", score)
result_df.insert(2, "サポート必要度ラベル", three_label)
result_df.insert(3, "0/1フラグ", binary_flag)
result_df.insert(4, "0/1フラグ表示", result_df["0/1フラグ"].map({"0": "0：通常", "1": "1：要サポート候補"}))
result_df.insert(5, "主な要因", main_factors)

main_label_col = "サポート必要度ラベル" if main_output == "低／中／高ラベル" else "0/1フラグ"
main_order = LABEL_ORDER_3 if main_label_col == "サポート必要度ラベル" else FLAG_ORDER

requirements_markdown = build_function_requirements_markdown(
    main_output=main_output,
    threshold_low_mid=threshold_low_mid,
    threshold_mid_high=threshold_mid_high,
    flag_threshold=flag_threshold,
    comparison_basis=comparison_basis,
    selected_features=selected_features,
    directions=directions,
    weights=weights,
    raw_df=raw_df,
    df=df,
    result_df=result_df,
    main_label_col=main_label_col,
    main_order=main_order,
)

st.sidebar.markdown("---")
st.sidebar.header("機能要件")
st.sidebar.download_button(
    label="機能要件一覧を出力する",
    data=requirements_markdown,
    file_name="student_support_function_requirements.md",
    mime="text/markdown",
    use_container_width=True,
)


summary_tab, result_tab, data_tab, settings_tab = st.tabs([
    "結果サマリー",
    "児童一覧",
    "元データ",
    "初期設定",
])

with summary_tab:
    st.subheader("ラベル分布")
    counts = result_df[main_label_col].value_counts().reindex(main_order, fill_value=0)
    kpi_cols = st.columns(len(main_order))
    for col, label in zip(kpi_cols, main_order):
        display_label = {"0": "0：通常", "1": "1：要サポート候補"}.get(label, label)
        col.metric(display_label, f"{int(counts[label])}名")

    left, right = st.columns([1, 1])
    with left:
        plot_counts = counts.reset_index()
        plot_counts.columns = ["ラベル", "人数"]
        fig_bar = px.bar(
            plot_counts,
            x="ラベル",
            y="人数",
            text="人数",
            color="ラベル",
            color_discrete_map=LABEL_COLORS,
            title="ラベル別人数",
        )
        fig_bar.update_layout(showlegend=False, yaxis_title="人数", xaxis_title="")
        fig_bar.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        fig_hist = px.histogram(
            result_df,
            x="サポート必要度スコア",
            nbins=12,
            color=main_label_col,
            category_orders={main_label_col: main_order},
            color_discrete_map=LABEL_COLORS,
            title="サポート必要度スコア分布",
        )
        fig_hist.update_layout(yaxis_title="人数", xaxis_title="スコア")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("クラス別のラベル分布")
    class_counts = (
        result_df.groupby([GRADE_COL, CLASS_COL, main_label_col], dropna=False)
        .size()
        .reset_index(name="人数")
    )
    class_counts["学年・クラス"] = class_counts[GRADE_COL].astype(str) + class_counts[CLASS_COL].astype(str)
    class_fig = px.bar(
        class_counts,
        x="学年・クラス",
        y="人数",
        color=main_label_col,
        text="人数",
        category_orders={main_label_col: main_order},
        color_discrete_map=LABEL_COLORS,
        title="クラス別ラベル分布",
    )
    class_fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="人数")
    st.plotly_chart(class_fig, use_container_width=True)
    class_table = (
        class_counts.pivot_table(
            index=["学年", "クラス"],
            columns=main_label_col,
            values="人数",
            fill_value=0,
            aggfunc="sum",
        )
        .reindex(columns=main_order, fill_value=0)
        .reset_index()
    )
    st.dataframe(class_table, use_container_width=True, hide_index=True)

    st.subheader("サポート優先一覧")
    priority_cols = [
        GRADE_COL,
        CLASS_COL,
        ID_COL,
        "サポート必要度スコア",
        "サポート必要度ラベル",
        "0/1フラグ",
        "主な要因",
    ]
    priority_df = result_df.sort_values("サポート必要度スコア", ascending=False)[priority_cols].head(10)
    display_result_table(priority_df)

with result_tab:
    st.subheader("児童ごとのラベル")
    filter_options = ["すべて"] + main_order
    selected_filter = st.selectbox("表示するラベル", filter_options)
    table_df = result_df.copy()
    if selected_filter != "すべて":
        table_df = table_df[table_df[main_label_col] == selected_filter]

    base_cols = [
        GRADE_COL,
        CLASS_COL,
        ID_COL,
        "サポート必要度スコア",
        "サポート必要度ラベル",
        "0/1フラグ",
        "主な要因",
    ]
    display_cols = base_cols + [c for c in RAW_COLUMNS if c not in base_cols] + [
        "心の天気晴れ率",
        "心の天気曇り率",
        "心の天気雨率",
        "欠席合計（病気＋事故）",
        "遅刻早退合計",
    ]
    display_cols = [c for c in display_cols if c in table_df.columns]

    display_result_table(table_df.sort_values("サポート必要度スコア", ascending=False)[display_cols])

with data_tab:
    st.subheader("搭載データ")
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    st.subheader("変数の概要")
    st.dataframe(summarize_numeric(df), use_container_width=True, hide_index=True)

with settings_tab:
    st.subheader("初期設定の考え方")
    st.markdown(
        """
| 変数 | 初期設定 |
|---|---|
| 病気欠席数・事故欠席数・遅刻数・早退数・保健室利用数 | 高いほど気になる |
| 心の天気曇り率・心の天気雨率 | 高いほど気になる |
| 心の天気晴れ率 | 低いほど気になる |
| 忌引等数・出席停止数 | 初期設定では使用しない候補 |

サポート必要度スコアは、各変数を比較基準ごとに0〜100点へ正規化し、向きの反転と整数の重み付けを行ったうえで平均した値です。重み0の変数はスコアに反映されません。
        """
    )
