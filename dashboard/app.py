"""Live AIoT meeting dashboard: reads Core2 prediction JSON from S3 and charts the conversation.

Run:  S3_BUCKET=<bucket> streamlit run dashboard/app.py   (AWS credentials via environment / IAM role)
"""
import json
import os
from datetime import datetime

import altair as alt
import boto3
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================
# Config
# =========================

BUCKET_NAME = os.getenv("S3_BUCKET", "<your-bucket>")
PREFIX = ""          
REGION = os.getenv("AWS_REGION", "us-east-1")
MAX_FILES = 300   

s3 = boto3.client("s3", region_name=REGION)


# =========================
# Page Settings
# =========================

st.set_page_config(
    page_title="AIoT Meeting Dashboard",
    page_icon="🎙️",
    layout="wide"
)

st_autorefresh(interval=2000, key="live_refresh")


# =========================
# Load Data from S3
# =========================

@st.cache_data(ttl=2)
def load_json_files_from_s3(bucket_name, prefix):
    try:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
    except Exception as e:
        st.error(f"Cannot read from S3: {e}")
        return pd.DataFrame()

    objects = response.get("Contents", [])

    if not objects:
        return pd.DataFrame()

    objects = sorted(objects, key=lambda x: x["LastModified"])[-MAX_FILES:]

    rows = []

    for obj in objects:
        key = obj["Key"]

        if not key.endswith(".json"):
            continue

        try:
            file_obj = s3.get_object(Bucket=bucket_name, Key=key)
            content = file_obj["Body"].read().decode("utf-8")
            data = json.loads(content)

            data["s3_key"] = key
            data["s3_last_modified"] = obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S")

            rows.append(data)

        except Exception:
            continue

    return pd.DataFrame(rows)


def clean_data(df):
    if df.empty:
        return df

    if "predicted_class" in df.columns:
        df["prediction"] = df["predicted_class"]

    if "prediction" not in df.columns:
        return pd.DataFrame()

    df["prediction"] = df["prediction"].replace({
        "multiple_speakers": "multi_speaker",
        "multi_speaker": "multi_speaker",
        "single_speaker": "single_speaker",
        "silence": "silence"
    })

    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["time"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    if df["time"].isna().all():
        df["time"] = pd.date_range(
            start=datetime.now(),
            periods=len(df),
            freq="1s"
        )

    df = df.sort_values("time").reset_index(drop=True)
    df["window_id"] = df.index + 1

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    else:
        df["confidence"] = None

    if "probabilities" in df.columns:
        df["prob_single_speaker"] = df["probabilities"].apply(
            lambda x: x.get("single_speaker") if isinstance(x, dict) else None
        )
        df["prob_multi_speaker"] = df["probabilities"].apply(
            lambda x: x.get("multiple_speakers") if isinstance(x, dict) else None
        )
        df["prob_silence"] = df["probabilities"].apply(
            lambda x: x.get("silence") if isinstance(x, dict) else None
        )

    return df


# =========================
# Sidebar
# =========================

st.sidebar.title("⚙️ Settings")

if st.sidebar.button("Manual Refresh"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.write("S3 Bucket")
st.sidebar.code(BUCKET_NAME)

st.sidebar.write("S3 Prefix")
st.sidebar.code("Root bucket / no folder")

st.sidebar.write("Running On")
st.sidebar.code("EC2")

st.sidebar.write("Auto Refresh")
st.sidebar.code("Every 2 seconds")


# =========================
# Main
# =========================

st.title("🎙️ AIoT Conversation Atmosphere Dashboard")
st.caption("Core2 M5Stack → AWS IoT Core → S3 → EC2 Streamlit")

raw_df = load_json_files_from_s3(BUCKET_NAME, PREFIX)
df = clean_data(raw_df)

if df.empty:
    st.warning("No valid JSON data found in S3.")
    st.stop()


# =========================
# KPI Summary
# =========================

total_windows = len(df)
single_pct = (df["prediction"] == "single_speaker").mean()
multi_pct = (df["prediction"] == "multi_speaker").mean()
silence_pct = (df["prediction"] == "silence").mean()
avg_confidence = df["confidence"].mean()

latest_state = df.iloc[-1]["prediction"]
latest_confidence = df.iloc[-1]["confidence"]

st.subheader("📊 Live Meeting Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Windows", total_windows)
col2.metric("Single Speaker", f"{single_pct:.1%}")
col3.metric("Multi Speaker", f"{multi_pct:.1%}")
col4.metric("Silence", f"{silence_pct:.1%}")
col5.metric("Avg Confidence", f"{avg_confidence:.2f}")


# =========================
# Current State
# =========================

st.subheader("🔴 Current Conversation State")

if latest_state == "multi_speaker":
    st.success(f"Multiple Speakers | Confidence: {latest_confidence:.2f}")
elif latest_state == "single_speaker":
    st.info(f"Single Speaker | Confidence: {latest_confidence:.2f}")
else:
    st.warning(f"Silence | Confidence: {latest_confidence:.2f}")


# =========================
# Engagement Analysis
# =========================

st.subheader("🧠 Engagement Analysis")

if single_pct > 0.70:
    st.error("Meeting is dominated by one speaker. Single-speaker time is above 70%.")
elif single_pct > 0.50:
    st.warning("Single-speaker time is relatively high.")
else:
    st.success("Single-speaker dominance is under control.")

if silence_pct > 0.30:
    st.warning("High silence detected. Silence time is above 30%.")
else:
    st.success("Silence level is acceptable.")

if multi_pct >= 0.30:
    st.success("Good interaction level. Multi-speaker activity is strong.")
else:
    st.info("Multi-speaker interaction is limited.")


# =========================
# Color Timeline
# =========================

st.subheader("🕒 Conversation State Timeline")

state_color_scale = alt.Scale(
    domain=["silence", "single_speaker", "multi_speaker"],
    range=["#9CA3AF", "#3B82F6", "#EF4444"]
)

timeline_chart = (
    alt.Chart(df)
    .mark_bar(size=18)
    .encode(
        x=alt.X("window_id:O", title="1-Second Window"),
        y=alt.Y("prediction:N", title="Conversation State"),
        color=alt.Color(
            "prediction:N",
            scale=state_color_scale,
            legend=alt.Legend(title="State")
        ),
        tooltip=[
            alt.Tooltip("window_id:O", title="Window"),
            alt.Tooltip("prediction:N", title="State"),
            alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
            alt.Tooltip("time:T", title="Time"),
            alt.Tooltip("s3_key:N", title="S3 File")
        ]
    )
    .properties(height=260)
)

st.altair_chart(timeline_chart, use_container_width=True)


# =========================
# State Distribution
# =========================

st.subheader("📌 State Distribution")

dist_df = df["prediction"].value_counts(normalize=True).reset_index()
dist_df.columns = ["prediction", "percentage"]
dist_df["percentage"] = dist_df["percentage"] * 100

dist_chart = (
    alt.Chart(dist_df)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("prediction:N", title="State"),
        y=alt.Y("percentage:Q", title="Percentage of Meeting Time"),
        color=alt.Color(
            "prediction:N",
            scale=state_color_scale,
            legend=None
        ),
        tooltip=[
            alt.Tooltip("prediction:N", title="State"),
            alt.Tooltip("percentage:Q", title="Percentage", format=".1f")
        ]
    )
    .properties(height=300)
)

st.altair_chart(dist_chart, use_container_width=True)


# =========================
# Confidence Over Time
# =========================

st.subheader("📈 Confidence Over Time")

confidence_chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("window_id:O", title="1-Second Window"),
        y=alt.Y("confidence:Q", title="Confidence"),
        color=alt.Color(
            "prediction:N",
            scale=state_color_scale,
            title="State"
        ),
        tooltip=[
            alt.Tooltip("window_id:O", title="Window"),
            alt.Tooltip("prediction:N", title="State"),
            alt.Tooltip("confidence:Q", title="Confidence", format=".2f")
        ]
    )
    .properties(height=320)
)

st.altair_chart(confidence_chart, use_container_width=True)


# =========================
# Probability Trends
# =========================

prob_cols = [
    "prob_single_speaker",
    "prob_multi_speaker",
    "prob_silence"
]

available_prob_cols = [col for col in prob_cols if col in df.columns]

if available_prob_cols:
    st.subheader("📉 Class Probability Trends")

    prob_df = df[["window_id"] + available_prob_cols].melt(
        id_vars="window_id",
        var_name="class",
        value_name="probability"
    )

    prob_df["class"] = prob_df["class"].replace({
        "prob_single_speaker": "single_speaker",
        "prob_multi_speaker": "multi_speaker",
        "prob_silence": "silence"
    })

    prob_chart = (
        alt.Chart(prob_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("window_id:O", title="1-Second Window"),
            y=alt.Y("probability:Q", title="Probability"),
            color=alt.Color("class:N", title="Class"),
            tooltip=[
                alt.Tooltip("window_id:O", title="Window"),
                alt.Tooltip("class:N", title="Class"),
                alt.Tooltip("probability:Q", title="Probability", format=".2f")
            ]
        )
        .properties(height=320)
    )

    st.altair_chart(prob_chart, use_container_width=True)


# =========================
# Raw Data
# =========================

st.subheader("🧾 Raw S3 Records")

with st.expander("Show raw data"):
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Live dashboard running on EC2 and reading JSON records from S3.")