"""
STUDENT PERFORMANCE ANALYZER
=============================
A data analysis project using Python, NumPy, and Pandas.

Phases covered:
1. Data Generation / Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Visualization
6. Insights & Reporting

Author: <your name>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
np.random.seed(42)

# Plot styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# =====================================================================
# PHASE 1: DATA GENERATION (replace this with pd.read_csv('your_file.csv')
#           once you have a real dataset, e.g. from Kaggle)
# =====================================================================

def generate_student_data(n_students=500):
    """Generate a synthetic student performance dataset."""

    student_ids = [f"S{1000 + i}" for i in range(n_students)]
    genders = np.random.choice(["Male", "Female"], size=n_students)
    classes = np.random.choice(["10A", "10B", "10C", "10D"], size=n_students)

    # Study hours per week (0-20), attendance % (40-100)
    study_hours = np.round(np.random.normal(10, 4, n_students).clip(0, 20), 1)
    attendance = np.round(np.random.normal(80, 12, n_students).clip(40, 100), 1)

    # Scores are influenced by study hours + attendance + random noise
    # This creates realistic correlations for the EDA phase later
    base_score = (
        study_hours * 2.5
        + attendance * 0.4
        + np.random.normal(0, 8, n_students)
    )

    def make_subject_scores(base):
        return np.round(np.clip(base + np.random.normal(0, 6, n_students), 0, 100), 1)

    math = make_subject_scores(base_score)
    science = make_subject_scores(base_score)
    english = make_subject_scores(base_score)

    parent_education = np.random.choice(
        ["High School", "Bachelors", "Masters", "PhD"],
        size=n_students,
        p=[0.35, 0.35, 0.22, 0.08],
    )

    df = pd.DataFrame({
        "student_id": student_ids,
        "gender": genders,
        "class": classes,
        "study_hours_per_week": study_hours,
        "attendance_percent": attendance,
        "math_score": math,
        "science_score": science,
        "english_score": english,
        "parent_education": parent_education,
    })

    # --- Inject some realistic "dirty data" issues on purpose ---
    # This lets your cleaning phase (Phase 2) have real work to do
    dirty_idx = np.random.choice(n_students, size=15, replace=False)
    df.loc[dirty_idx[:5], "math_score"] = np.nan            # missing values
    df.loc[dirty_idx[5:8], "attendance_percent"] = 105      # invalid outlier
    df.loc[dirty_idx[8:11], "gender"] = "M"                 # inconsistent category
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # duplicate rows

    return df


# =====================================================================
# PHASE 2: DATA CLEANING
# =====================================================================

def clean_data(df):
    """Clean the raw dataset: handle missing values, duplicates, outliers,
    and inconsistent categorical labels."""

    df = df.copy()

    # 1. Remove exact duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Standardize categorical values
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})

    # 3. Fix invalid values (attendance can't exceed 100%)
    df["attendance_percent"] = df["attendance_percent"].clip(upper=100)

    # 4. Handle missing scores -> fill with the median of that subject
    score_cols = ["math_score", "science_score", "english_score"]
    for col in score_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # 5. Ensure correct data types
    df["student_id"] = df["student_id"].astype(str)
    for col in score_cols + ["study_hours_per_week", "attendance_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.reset_index(drop=True, inplace=True)
    return df


# =====================================================================
# PHASE 3: FEATURE ENGINEERING
# =====================================================================

def add_features(df):
    """Add derived columns: total/average score, grade, performance category."""

    df = df.copy()
    score_cols = ["math_score", "science_score", "english_score"]

    # Total & average marks
    df["total_score"] = df[score_cols].sum(axis=1)
    df["average_score"] = np.round(df[score_cols].mean(axis=1), 2)

    # Letter grade based on average score
    grade_bins = [0, 40, 60, 75, 90, 101]
    grade_labels = ["F", "D", "C", "B", "A"]
    df["grade"] = pd.cut(df["average_score"], bins=grade_bins, labels=grade_labels, right=False)

    # Performance category using NumPy's np.select
    conditions = [
        df["average_score"] >= 85,
        (df["average_score"] >= 60) & (df["average_score"] < 85),
        df["average_score"] < 60,
    ]
    choices = ["Excellent", "Average", "Needs Improvement"]
    df["performance_category"] = np.select(conditions, choices, default="Unknown")

    # Pass/Fail flag (pass = average >= 40)
    df["result"] = np.where(df["average_score"] >= 40, "Pass", "Fail")

    # At-risk flag: low score AND low attendance
    df["at_risk"] = np.where(
        (df["average_score"] < 50) & (df["attendance_percent"] < 70), True, False
    )

    return df


# =====================================================================
# PHASE 4: EXPLORATORY DATA ANALYSIS
# =====================================================================

def run_eda(df):
    """Print key descriptive statistics and groupwise summaries."""

    print("\n" + "=" * 60)
    print("BASIC DATASET INFO")
    print("=" * 60)
    print(df.info())
    print("\nShape:", df.shape)

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df[["math_score", "science_score", "english_score",
              "average_score", "attendance_percent", "study_hours_per_week"]].describe())

    print("\n" + "=" * 60)
    print("AVERAGE SCORE BY CLASS")
    print("=" * 60)
    print(df.groupby("class")["average_score"].mean().round(2).sort_values(ascending=False))

    print("\n" + "=" * 60)
    print("AVERAGE SCORE BY GENDER")
    print("=" * 60)
    print(df.groupby("gender")["average_score"].mean().round(2))

    print("\n" + "=" * 60)
    print("PASS/FAIL COUNT")
    print("=" * 60)
    print(df["result"].value_counts())
    print(df["result"].value_counts(normalize=True).round(3) * 100, "% ")

    print("\n" + "=" * 60)
    print("PERFORMANCE CATEGORY DISTRIBUTION")
    print("=" * 60)
    print(df["performance_category"].value_counts())

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX (numeric columns)")
    print("=" * 60)
    numeric_cols = ["study_hours_per_week", "attendance_percent",
                     "math_score", "science_score", "english_score", "average_score"]
    print(df[numeric_cols].corr().round(2))

    print("\n" + "=" * 60)
    print("TOP 5 PERFORMERS")
    print("=" * 60)
    print(df.sort_values("average_score", ascending=False)
          [["student_id", "class", "average_score", "grade"]].head(5))

    print("\n" + "=" * 60)
    print("AT-RISK STUDENTS (low score + low attendance)")
    print("=" * 60)
    at_risk_df = df[df["at_risk"]][["student_id", "class", "average_score", "attendance_percent"]]
    print(at_risk_df if not at_risk_df.empty else "None found")


# =====================================================================
# PHASE 5: VISUALIZATION
# =====================================================================

def create_visualizations(df, save_dir="."):
    """Generate and save key charts as PNG files."""

    # 1. Distribution of average scores
    plt.figure()
    sns.histplot(df["average_score"], bins=20, kde=True, color="steelblue")
    plt.title("Distribution of Average Scores")
    plt.xlabel("Average Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/score_distribution.png", dpi=150)
    plt.close()

    # 2. Average score by class (bar chart)
    plt.figure()
    class_avg = df.groupby("class")["average_score"].mean().sort_values(ascending=False)
    sns.barplot(x=class_avg.index, y=class_avg.values, hue=class_avg.index,
                palette="viridis", legend=False)
    plt.title("Average Score by Class")
    plt.xlabel("Class")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_score_by_class.png", dpi=150)
    plt.close()

    # 3. Boxplot: score spread & outliers by subject
    plt.figure()
    subject_df = df[["math_score", "science_score", "english_score"]]
    sns.boxplot(data=subject_df, palette="Set2")
    plt.title("Score Distribution & Outliers by Subject")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/subject_boxplot.png", dpi=150)
    plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(8, 6))
    numeric_cols = ["study_hours_per_week", "attendance_percent",
                     "math_score", "science_score", "english_score", "average_score"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=150)
    plt.close()

    # 5. Scatter: study hours vs average score
    plt.figure()
    sns.scatterplot(data=df, x="study_hours_per_week", y="average_score",
                     hue="performance_category", palette="deep", alpha=0.7)
    plt.title("Study Hours vs Average Score")
    plt.xlabel("Study Hours per Week")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/study_hours_vs_score.png", dpi=150)
    plt.close()

    # 6. Performance category pie chart
    plt.figure()
    df["performance_category"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", colors=sns.color_palette("pastel")
    )
    plt.title("Performance Category Breakdown")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_category_pie.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to '{save_dir}/' as PNG files.")


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    print("Generating synthetic student dataset...")
    raw_df = generate_student_data(n_students=500)
    raw_df.to_csv("raw_student_data.csv", index=False)
    print(f"Raw data saved -> raw_student_data.csv  (shape: {raw_df.shape})")

    print("\nCleaning data...")
    clean_df = clean_data(raw_df)
    print(f"Cleaned data shape: {clean_df.shape}")

    print("\nEngineering features...")
    final_df = add_features(clean_df)
    final_df.to_csv("processed_student_data.csv", index=False)
    print("Processed data saved -> processed_student_data.csv")

    run_eda(final_df)

    print("\nCreating visualizations...")
    create_visualizations(final_df)

    print("\nDone! Explore 'processed_student_data.csv' and the generated charts.")


if __name__ == "__main__":
    main()
