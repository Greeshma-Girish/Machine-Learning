import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\HP\diabetes.csv')

# Displaying the first and last few rows of the dataframe
print("First 5 rows of the dataset:")
print(df.head())
print("\nLast 5 rows of the dataset:")
print(df.tail())

# Displaying general information about the dataframe
print("\nDataFrame Information:")
print(df.info())

# Descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Number of unique values in each column
print("\nNumber of Unique Values in Each Column:")
print(df.nunique())

# Check for missing values in the dataset
print("\nMissing Values in Each Column:")
missing_values = df.isnull().sum()
print(missing_values)

# Percentage of missing values in each column
print("\nPercentage of Missing Values in Each Column:")
print((df.isnull().sum() / len(df)) * 100)

# Plot the bar graph for missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette='pastel')
plt.title('Missing Values per Column', fontsize=16)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Number of Missing Values', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Impute missing values with the mean for numerical columns and mode for categorical columns
for col in df.columns:
    if df[col].dtype == 'object':  # For categorical columns
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # For numerical columns
        df[col].fillna(df[col].mean(), inplace=True)

# Display the updated dataset to verify
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Show the specific row where imputation occurred
print("\nUpdated Rows with Previously Missing Values:")
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows)

# Remove the column 'sl.no'
df.drop(columns=['slno'], inplace=True)
print("\nColumns after removing 'sl.no':")
print(df.columns)

# Create new columns: obesity (1 if bmi > 30) and age_HbA1c_ratio
df['obesity'] = np.where(df['bmi'] > 30, 1, 0)  # 1 indicates obese, 0 otherwise
df['age_HbA1c_ratio'] = df['age'] / df['HbA1c_level']
print("\nNew Columns: 'obesity' and 'age_HbA1c_ratio'")
print(df[['obesity', 'age_HbA1c_ratio']].head())


# mean, median, mode
print("\nDescriptive Statistics:")
desc_stats = df.describe().T
desc_stats['range'] = desc_stats['max'] - desc_stats['min']  # Calculating the range
desc_stats['mode'] = df.mode().iloc[0]  # Calculating the mode
print(desc_stats)

# Univariate Analysis - Histograms for numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Plotting histograms for all numerical columns in one figure
plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)  
    df[column].hist(bins=20, alpha=0.7, color='blue')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



# Get all numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
rows = (len(numerical_columns) // 3) + (1 if len(numerical_columns) % 3 != 0 else 0)
cols = min(len(numerical_columns), 3)

# Plot histograms and check skewness for all numerical columns
plt.figure(figsize=(15, rows * 5))
for i, col in enumerate(numerical_columns, 1):
    # Mean, Median, and Skewness Calculation
    mean = df[col].mean()
    median = df[col].median()
    skewness = df[col].skew()

    # Print Mean, Median, and Skewness
    print(f'{col} - Mean: {mean}, Median: {median}, Skewness: {skewness}')

    # Determine skewness type
    if skewness > 0:
        print(f"{col} is positively skewed (Right Skew).\n")
    elif skewness < 0:
        print(f"{col} is negatively skewed (Left Skew).\n")
    else:
        print(f"{col} is symmetric (No Skew).\n")

    # Plot histogram with KDE for better visualization
    plt.subplot(rows, cols, i)  # Dynamically adjusting the grid size
    
    # Plot histogram and KDE
    sns.histplot(df[col], kde=True, bins=20, color='blue', alpha=0.7)
    
    # Add mean, median, and mode lines
    mode_val = df[col].mode()[0]
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    plt.axvline(mode_val, color='purple', linestyle='--', label=f'Mode: {mode_val:.2f}')
    
    # Titles and labels
    plt.title(f"Histogram & KDE of {col}")
    plt.xlabel(col)
    plt.ylabel("Density / Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()




# Univariate Analysis - Count plots for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
# Plotting count plots for all categorical columns in one figure
plt.figure(figsize=(15, 10))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i) 
    sns.countplot(x=column, data=df, palette='pastel')
    plt.title(f"Count Plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Box plot for age and diabetes level
print("Unique values in 'diabetes':", df['diabetes'].unique())
df = df[df['diabetes'].isin([0, 1])]

print("\nBox Plot of Age by Diabetes Level:")
sns.boxplot(x='diabetes', y='age', data=df, palette='pastel')
plt.title('Box Plot of Age by Diabetes Level (0 = Non-Diabetic, 1 = Diabetic)')
plt.xlabel('Diabetes (0 = Non-Diabetic, 1 = Diabetic)')
plt.ylabel('Age')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Bivariate Analysis
# Scatter plot for two continuous variables (e.g., bmi and blood glucose level)
print("\nScatter Plot of BMI vs Blood Glucose Level:")
sns.scatterplot(x='bmi', y='blood_glucose_level', data=df)
plt.title('Scatter Plot of BMI vs Blood Glucose Level')
plt.show()

# Pair plot to explore relationships between multiple continuous features
print("\nPair Plot of Continuous Features:")
sns.pairplot(df[['bmi', 'HbA1c_level', 'blood_glucose_level', 'age']])
plt.title('Pair Plot of Continuous Features:')
plt.show()

fig, axarr = plt.subplots(4, 2, figsize=(15, 20))

# Plotting grouped bar charts for categorical columns against the mean of 'diabetes'
df.groupby('gender')['diabetes'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12, color='skyblue')
axarr[0][0].set_title("Gender Vs Diabetes", fontsize=18)
axarr[0][0].set_ylabel("Mean Diabetes Value")

df.groupby('hypertension')['diabetes'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12, color='orange')
axarr[0][1].set_title("Hypertension Vs Diabetes", fontsize=18)
axarr[0][1].set_ylabel("Mean Diabetes Value")

# Adjust spacing
plt.subplots_adjust(hspace=1.0, wspace=0.5)
sns.despine()
plt.show()



# Positive Correlation (e.g., age vs. HbA1c_level)
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x='age', y='HbA1c_level', data=df, color='green')
plt.title('Positive Correlation: Age vs HbA1c Level', fontsize=14)
plt.xlabel('Age')
plt.ylabel('HbA1c Level')

# Negative Correlation (e.g., age vs. height)
plt.subplot(1, 3, 2)
sns.scatterplot(x='age', y='height', data=df, color='red')
plt.title('Negative Correlation: Age vs Height', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Height')

# No Correlation (e.g., gender vs. BMI as an example if categorical, but can be extended)
plt.subplot(1, 3, 3)
sns.scatterplot(x='bmi', y='blood_glucose_level', data=df, color='blue')
plt.title('No Correlation: BMI vs Blood Glucose Level', fontsize=14)
plt.xlabel('BMI')
plt.ylabel('Blood Glucose Level')

plt.tight_layout()
plt.show()

# Correlation heatmap for numerical columns in the dataset
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_columns].corr()

# Multivariate - Heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.show()

# Count the number of diabetic and non-diabetic individuals
diabetes_counts = df['diabetes'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    diabetes_counts, 
    labels=['Non-Diabetic', 'Diabetic'], 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['lightblue', 'lightcoral'], 
    explode=[0.05, 0]  
)
plt.title('Diabetic vs Non-Diabetic Individuals')
plt.show()

