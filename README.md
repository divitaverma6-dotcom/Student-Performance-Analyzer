# Student Performance Analyzer

A data analysis project built with **Python, NumPy, and Pandas** to explore, clean, and analyze student academic performance — identifying trends, correlations, and at-risk students.

## 📌 Project Overview

This project analyzes a dataset of 500 students across 4 classes, examining how factors like **study hours** and **attendance** relate to academic performance in Math, Science, and English. The pipeline covers the full data analysis lifecycle: data generation/loading, cleaning, feature engineering, exploratory data analysis (EDA), and visualization.

## 🛠️ Tools & Libraries

- **Python 3**
- **NumPy** — numerical computations, conditional feature creation
- **Pandas** — data loading, cleaning, grouping, aggregation
- **Matplotlib / Seaborn** — data visualization

## 📁 Repository Structure

```
├── student_performance_analyzer.py   # Main analysis script
├── raw_student_data.csv              # Raw dataset (before cleaning)
├── processed_student_data.csv        # Cleaned dataset with engineered features
├── score_distribution.png            # Chart: distribution of average scores
├── avg_score_by_class.png            # Chart: average score by class
├── subject_boxplot.png               # Chart: score spread/outliers by subject
├── correlation_heatmap.png           # Chart: correlation matrix
├── study_hours_vs_score.png          # Chart: study hours vs average score
├── performance_category_pie.png      # Chart: performance category breakdown
└── README.md                         # This report
```

## 🔄 Methodology

1. **Data Collection** — Dataset includes student ID, gender, class, study hours/week, attendance %, subject scores (Math/Science/English), and parental education.
2. **Data Cleaning** — Removed duplicate records, standardized inconsistent category labels (e.g. "M" → "Male"), capped invalid attendance values (>100%), and filled missing scores with the subject median.
3. **Feature Engineering** — Derived `total_score`, `average_score`, letter `grade` (A–F), `performance_category` (Excellent / Average / Needs Improvement), `result` (Pass/Fail), and an `at_risk` flag (low score + low attendance).
4. **Exploratory Data Analysis** — Computed descriptive statistics, groupwise comparisons (by class/gender), and correlation analysis.
5. **Visualization** — Generated 6 charts to visually communicate the findings.

## 📊 Key Findings

- **Dataset size:** 500 students, 15 columns after processing.
- **Overall pass rate:** 91.8% passed (459 students), 8.2% failed (41 students).
- **Performance breakdown:** 295 students "Needs Improvement," 189 "Average," and only 16 "Excellent" — indicating room for improvement across most of the cohort.
- **Class comparison:** Class 10B had the highest average score (58.99), followed closely by 10A and 10D (57.63) and 10C (57.33) — differences between classes are small.
- **Gender comparison:** Female students averaged slightly higher (58.29) than male students (57.48).
- **Study hours matter:** Study hours per week showed a strong positive correlation with average score (**r = 0.72**) — one of the strongest predictors in the dataset.
- **Attendance matters less, but still helps:** Attendance showed a moderate positive correlation with average score (**r = 0.35**).
- **At-risk students:** 39 students were flagged as at-risk (average score below 50 *and* attendance below 70%) — these students may benefit from early intervention.
- **Top performers:** The five highest-scoring students all achieved an 'A' grade, led by student S1426 with an average score of 97.5.

## 📈 Visualizations

**Score Distribution**
![Score Distribution](score_distribution.png)

**Average Score by Class**
![Average Score by Class](avg_score_by_class.png)

**Subject-wise Score Spread**
![Subject Boxplot](subject_boxplot.png)

**Correlation Heatmap**
![Correlation Heatmap](correlation_heatmap.png)

**Study Hours vs Average Score**
![Study Hours vs Score](study_hours_vs_score.png)

**Performance Category Breakdown**
![Performance Category Pie](performance_category_pie.png)

## 💡 Conclusions & Recommendations

- **Study hours are the single strongest lever for improving scores** — students should be encouraged to establish consistent weekly study routines rather than relying on attendance alone.
- **Attendance alone is not enough** — its correlation with scores, while positive, is weaker than study hours, suggesting that simply being present doesn't guarantee engagement or learning.
- **Early intervention target group identified** — the 39 at-risk students (low scores + low attendance) represent a clear, actionable list for teachers/mentors to prioritize support.
- **Class-level differences are minor**, so performance gaps are more likely driven by individual study habits than by which class/section a student is in.

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn
python student_performance_analyzer.py
```

This regenerates the dataset, runs the full cleaning/EDA/feature engineering pipeline, prints statistics to the console, and saves the processed CSV and all charts.

## 🔮 Future Scope

- Replace the synthetic dataset with a real-world dataset (e.g. Kaggle's Student Performance dataset).
- Add a predictive model (e.g. linear regression) to forecast final scores from study hours and attendance.
- Build an interactive dashboard (e.g. with Streamlit) for live exploration of the data.

---
*Project built as part of an internship submission.*
