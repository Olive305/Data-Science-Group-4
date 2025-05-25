import pandas as pd


def data_cleanup(df):
    """
    Cleans all of the data from one row of the df
    :param df:
    :return:
    """
    df= (
        df
        .str.lower()
        .str.replace('-', '', regex=True)
        .str.replace('–', '', regex=True)
        .str.strip()
        .str.replace(r'\s+', '', regex=True)
    )
    return df
