import os
import pandas as pd
import os
import sys
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


def basic_data_cleanup(df, column = 'Krankenkasse'):
    """
    Cleans all of the data from one row of the df
    :param df:
    :return:
    """
    df[column]= (
        df[column]
        .str.lower()
        .str.replace('-', '', regex=True)
        .str.replace('–', '', regex=True)
        .str.strip()
        .str.replace(r'\s+', '', regex=True)
    )
    return df
