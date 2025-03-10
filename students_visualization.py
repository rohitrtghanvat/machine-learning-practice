# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Style for Better Visuals
sns.set_style("whitegrid")

# Step 2: Create a Simple Dataset (Students Performance)
students = pd.DataFrame({
    'Student': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Math_Score': [78, 85, 90, 72, 88, 95, 60, 83, 91, 87],
    'Science_Score': [82, 79, 85, 70, 89, 92, 67, 81, 90, 86],
    'English_Score': [75, 80, 82, 78, 85, 88, 65, 79, 84, 83]
})

# ----------------------------------------
# 📌 Bar Plot (Students' Scores)
students.set_index('Student').plot(kind='bar', colormap='coolwarm', width=0.7)
plt.title("Students' Scores in Different Subjects")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# ----------------------------------------
# 📌 Stacked Bar Plot (Students' Scores)
students.set_index('Student').plot(kind='bar', stacked=True, colormap='viridis', width=0.7)
plt.title("Stacked Bar Plot: Students' Scores in Different Subjects")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# ----------------------------------------
# 📌 Box Plot (Score Distribution)
sns.boxplot(data=students.iloc[:, 1:], palette="pastel")
plt.title("Boxplot: Distribution of Student Scores")
plt.ylabel("Score")
plt.show()

# ----------------------------------------
# 📌 Violin Plot (Subject-Wise Score Distribution)
sns.violinplot(data=students.iloc[:, 1:], palette="muted")
plt.title("Violin Plot: Score Distribution Across Subjects")
plt.ylabel("Score")
plt.show()

# ----------------------------------------
# 📌 Swarm Plot (Student Score Spread)
melted_data = students.melt(id_vars=["Student"], var_name="Subject", value_name="Score")
sns.swarmplot(x="Subject", y="Score", data=melted_data, palette="coolwarm", size=8)
plt.title("Swarm Plot: Score Spread Across Subjects")
plt.show()

# ----------------------------------------
# 📌 Scatter Plot (Math vs Science Scores)
sns.scatterplot(x='Math_Score', y='Science_Score', data=students, color="blue", s=100)
plt.title("Scatter Plot: Math vs Science Scores")
plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.show()

# ----------------------------------------
# 📌 Line Plot (Trends in Student Performance)
sns.lineplot(x="Student", y="Math_Score", data=students, marker="o", label="Math", color="red")
sns.lineplot(x="Student", y="Science_Score", data=students, marker="o", label="Science", color="blue")
sns.lineplot(x="Student", y="English_Score", data=students, marker="o", label="English", color="green")
plt.title("Line Plot: Student Performance Trends")
plt.xlabel("Student")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend()
plt.show()

# ----------------------------------------
# 📌 Heatmap (Correlation Between Subjects)
sns.heatmap(students.iloc[:, 1:].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap: Correlation Between Subjects")
plt.show()

# ----------------------------------------
# 📌 Histogram (Math Score Distribution)
sns.histplot(students['Math_Score'], bins=5, kde=True, color="green")
plt.title("Histogram: Math Score Distribution")
plt.xlabel("Math Score")
plt.show()

# ----------------------------------------
# 📌 KDE Plot (Science Score Distribution)
sns.kdeplot(students['Science_Score'], fill=True, color="purple", linewidth=2)
plt.title("KDE Plot: Science Score Distribution")
plt.xlabel("Science Score")
plt.show()

# ----------------------------------------
# 📌 Pie Chart (Average Subject Performance)
avg_scores = students.iloc[:, 1:].mean()
plt.pie(avg_scores, labels=avg_scores.index, autopct='%1.1f%%', colors=['red', 'blue', 'green'], shadow=True)
plt.title("Pie Chart: Average Subject Performance")
plt.show()
