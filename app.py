import streamlit as st
import pandas as pd

st.title("📊 AI-Based Performance Analytics Dashboard")

# Load Data
df = pd.read_csv("data/student_data.csv")

st.subheader("Raw Data")
st.write(df.head())

# -----------------------
# Subject-wise Report
# -----------------------
st.subheader("📘 Subject-wise Report")

subject_report = df.groupby("subject").agg(
    total_questions=("correct", "count"),
    total_correct=("correct", "sum"),
    avg_time=("time_taken", "mean")
)

subject_report["accuracy"] = (
    subject_report["total_correct"] / subject_report["total_questions"]
) * 100

st.write(subject_report)

# -----------------------
# Topic-wise Report
# -----------------------
st.subheader("📚 Topic-wise Report")

topic_report = df.groupby(["subject", "topic"]).agg(
    total_questions=("correct", "count"),
    total_correct=("correct", "sum"),
)

topic_report["accuracy"] = (
    topic_report["total_correct"] / topic_report["total_questions"]
) * 100

st.write(topic_report)

# -----------------------
# Question Type Analysis
# -----------------------
st.subheader("🧠 Question-Type Analysis")

type_report = df.groupby(["subject", "topic", "question_type"]).agg(
    total_questions=("correct", "count"),
    total_correct=("correct", "sum")
)

type_report["accuracy"] = (
    type_report["total_correct"] / type_report["total_questions"]
) * 100

st.write(type_report)


# -----------------------
# Concept vs Application Gap Detection
# -----------------------
st.subheader("⚠️ Concept vs Application Gap Detection")

type_df = type_report.reset_index()

direct_df = type_df[type_df["question_type"] == "Direct"]
twisted_df = type_df[type_df["question_type"] == "Twisted"]

gap_df = pd.merge(
    direct_df,
    twisted_df,
    on=["subject", "topic"],
    suffixes=("_direct", "_twisted")
)

gap_df["accuracy_gap"] = (
    gap_df["accuracy_direct"] - gap_df["accuracy_twisted"]
)

gap_df["application_weak"] = gap_df["accuracy_gap"] > 50

st.write(gap_df[[
    "subject",
    "topic",
    "accuracy_direct",
    "accuracy_twisted",
    "accuracy_gap",
    "application_weak"
]])

# Show warning message
weak_topics = gap_df[gap_df["application_weak"] == True]

if not weak_topics.empty:
    st.error("🚨 Application Weak Detected in These Topics:")
    st.write(weak_topics[["subject", "topic"]])
else:
    st.success("✅ No major concept-application gaps detected.")
    


# -----------------------
# Recommendation Engine
# -----------------------
st.subheader("📌 Recommended Action Plan")

if not weak_topics.empty:
    for index, row in weak_topics.iterrows():
        st.warning(f"""
        🔹 **{row['subject']} - {row['topic']}**
        - Practice 20 twisted/application-based questions  
        - Focus on translating word problems carefully  
        - Attempt timed mini-mock  
        """)
else:
    st.success("Great job! Maintain revision consistency.")

st.subheader("📊 Subject Accuracy Visualization")
st.bar_chart(subject_report["accuracy"])

st.title("🎓 AI-Based Concept vs Application Gap Detection System")