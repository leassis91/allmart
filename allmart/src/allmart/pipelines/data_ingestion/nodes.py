"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.11
"""


import pandas as pd
import io
import chardet


def just_reading(df: pd.DataFrame) -> pd.DataFrame:
    """
    Just reading the input data to the output.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Output DataFrame
    """
    return df