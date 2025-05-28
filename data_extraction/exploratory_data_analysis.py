import pandas as pd

from data_extraction.utils import load_excel


def full_eda(df):
    print("Shape:", df.shape)
    print("Info:")
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDescriptive statistics:\n", df.describe())

    # Value counts for categoricals
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        print(f"\nValue counts for {col}:\n", df[col].value_counts())

    # Correlation heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

df= load_excel('../data/fm_dem_merged.xlsx')
df = df.drop(df.columns[0], axis=1)
full_eda(df)