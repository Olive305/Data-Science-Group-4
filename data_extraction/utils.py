import os
import pandas as pd
import os
import sys

from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_excel(relative_path: str, sheet_name: str = None, header=0,**kwargs) -> pd.DataFrame:
    """
    loads an excel file into a pandas dataframe

    :param header:
    :param relative_path: path to the file
    :param sheet_name: (optional) sheet name
    :param kwargs: additional arguments
    :return: dataframe
    """
    abs_path = os.path.join(os.path.dirname(__file__), relative_path)
    if sheet_name:
        return pd.read_excel(abs_path, sheet_name=sheet_name, header=header)
    else:
        return pd.read_excel(abs_path, header=header)

def write_excel(df: pd.DataFrame, relative_path: str, **kwargs) -> None:
    """
    writes an excel file into a pandas dataframe

    :param df: dataframe
    :param relative_path: relative path
    :param kwargs: additional arguments
    """
    abs_path = os.path.join(os.path.dirname(__file__), relative_path)
    df.to_excel(abs_path, **kwargs)


def basic_data_cleanup(df: pd.DataFrame, column: str = 'Krankenkasse') -> pd.DataFrame:
    """
    Cleans all of the data from one row of the df
    :param df: pandas DataFrame
    :param column: column to clean
    :return: cleaned DataFrame
    """
    df[column] = (
        df[column]
        .str.lower()
        .str.replace('-', '', regex=True)
        .str.replace('–', '', regex=True)
        .str.strip()
        .str.replace(r'\s+', '', regex=True)
    )
    return df
def column_name_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.replace(r'[^A-Za-z0-9_]', '', regex=True)
    )
    return df

def normalize_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_norm = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_norm, scaler