import pandas as pd

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Show first 5 rows
print("First 5 rows of data:")
print(df.head())

# Show basic info
print("\nDataset Info:")
print(df.info())

print("\n=== Subject-wise Report ===")

subject_report = df.groupby("subject").agg(
    total_questions=("correct", "count"),
    total_correct=("correct", "sum"),
    avg_time=("time_taken", "mean")
)

subject_report["accuracy"] = (
    subject_report["total_correct"] / subject_report["total_questions"]
) * 100

print(subject_report)

print("\n=== Topic-wise Report (Inside Each Subject) ===")

topic_report = df.groupby(["subject", "topic"]).agg(
    total_questions=("correct", "count"),
    total_correct=("correct", "sum"),
    avg_time=("time_taken", "mean")
)

topic_report["accuracy"] = (
    topic_report["total_correct"] / topic_report["total_questions"]
) * 100

print(topic_report)

print("\n=== Question Type-wise Accuracy ===")

type_report = df.groupby(["subject", "topic", "question_type"]).agg(
    total_questions=("correct", "count"),
    total_correct=("correct", "sum")
)

type_report["accuracy"] = (
    type_report["total_correct"] / type_report["total_questions"]
) * 100

print(type_report)


print("\n=== Concept vs Application Gap Detection ===")

# Reset index to make easier filtering
type_df = type_report.reset_index()

# Separate direct and twisted
direct_df = type_df[type_df["question_type"] == "Direct"]
twisted_df = type_df[type_df["question_type"] == "Twisted"]

# Merge direct and twisted on subject + topic
gap_df = pd.merge(
    direct_df,
    twisted_df,
    on=["subject", "topic"],
    suffixes=("_direct", "_twisted")
)

# Calculate gap
gap_df["accuracy_gap"] = gap_df["accuracy_direct"] - gap_df["accuracy_twisted"]

# Detect if gap > 30%
gap_df["application_weak"] = gap_df["accuracy_gap"] > 50

print(gap_df[[
    "subject",
    "topic",
    "accuracy_direct",
    "accuracy_twisted",
    "accuracy_gap",
    "application_weak"
]])

print("\n=== Creating ML Dataset ===")

# Create a simple ML dataset at topic level
ml_df = gap_df.copy()

# Target: 1 if application weak, else 0
ml_df["target"] = ml_df["application_weak"].astype(int)

# Features
features = ml_df[["accuracy_direct", "accuracy_twisted", "accuracy_gap"]]
target = ml_df["target"]

print(features.head())
print(target.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\n=== Model Accuracy ===")
print(accuracy_score(y_test, y_pred))
