"""
Use a stratified train/validation/test split (for example, 60%/20%/20%) with a fixed random
seed, so that the minority damage classes are represented in every split. Inspect class counts in
each split. Fit all learned preprocessing only on training data; never fit imputation, encoding,
rare-category grouping, scaling, PCA or feature selection on validation/test data. As a robustness check, optionally evaluate a time-based split (train on earlier years, test on the most recent
years).
1. For numerical predictors, compare median and mean imputation. Consider adding missingness indicators and explain their interpretation.
2. For categorical predictors, impute missing values with an explicit Missing category or the
most frequent category. Group rare levels of high-cardinality features into Other. Then use
one-hot encoding with handling for unseen categories.
3. Standardize numerical predictors for logistic regression and PCA. Use scikit-learn
ColumnTransformer and Pipeline so that preprocessing is refitted correctly during crossvalidation.
4. Apply PCA to appropriately scaled/encoded training features. Plot explained variance and
inspect principal-component loadings. Compare models with and without PCA, and discuss
why PCA may retain less useful structure here than for strongly correlated indicators.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def stratified_split(df, random_state=67):
    train_val, test = train_test_split(df, test_size=0.2, stratify=df['damage_class'], random_state=random_state)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['damage_class'], random_state=random_state)
    return train, val, test

def preprocess_data(train, val, test, numerical_features, categorical_features):
    ...

def main():
    ...

if __name__ == "__main__":
    main()