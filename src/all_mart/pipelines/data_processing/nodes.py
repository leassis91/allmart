"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

"""
Aqui entrarão as funções de pré-processamento do nosso dataset.
Funções como "Rename", "Resizing", "Cleaning NA"...todas estarão aqui.
"""

import pandas as pd
import numpy as np


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"

def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x

def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x




def preprocess_ecommerce(Ecommerce: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for Ecommerce.

    Args:
        Ecommerce: Raw data.
    Returns:
        Dataframe preprocessado.
    """
    
    # Dropping unwanted columns
    Ecommerce.drop(columns=['Unnamed: 8'], axis=1, inplace=True)
    
    # Rename columns
    cols_new = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date', 'unit_price', 'customer_id', 'country']
    Ecommerce.columns = cols_new
    
    
    
    df_missing = Ecommerce.loc[Ecommerce['customer_id'].isna(), :]
    df_not_missing = Ecommerce.loc[~Ecommerce['customer_id'].isna(), :]
    
    
    # create reference
    df_backup = pd.DataFrame(df_missing['invoice_no'].drop_duplicates())
    df_backup['customer_id'] = np.arange(19000, 19000+len(df_backup), 1)


    # merge original with reference dataframe
    df2 = pd.merge(df2, df_backup, on='invoice_no', how='left')
    # coalesce
    df2['customer_id'] = df2['customer_id_x'].combine_first(df2['customer_id_y'])
    # drop extra columns
    df2 = df2.drop(columns=['customer_id_x', 'customer_id_y'], axis=1)
    # drop na
    df2.dropna(subset=['description', 'customer_id'], inplace=True)

    # Change Dtypes
    df2['invoice_date'] = pd.to_datetime(df2['invoice_date'], format='%d-%b-%y')
    df2['customer_id'] = df2['customer_id'].astype(int)

    # Create DataFrame Reference
    df_ref = df2.drop(['invoice_no', 'stock_code', 'description', 
                    'quantity', 'invoice_date', 'unit_price', 
                    'country'], axis=1).drop_duplicates(ignore_index=True)
    
    return df_ref


    
    




# def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for shuttles.

#     Args:
#         shuttles: Raw data.
#     Returns:
#         Preprocessed data, with `price` converted to a float and `d_check_complete`,
#         `moon_clearance_complete` converted to boolean.
#     """
#     shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
#     shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
#     shuttles["price"] = _parse_money(shuttles["price"])
#     return shuttles





# # Merge DFs
# from functools import reduce
# dfs
# df_fengi = reduce(lambda left, right: pd.merge(left, right, on='customer_id', how='left'))