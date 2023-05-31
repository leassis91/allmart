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
    df2 = pd.merge(Ecommerce, df_backup, on='invoice_no', how='left')
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


def feature_filtering(df_ref: pd.DataFrame) -> pd.DataFrame:    
    """Filter features from the data.

    Args:
        df_ref: Reference data processed.
    Returns:
        Dataframe preprocessado.
    """
    ###### Numerical Cols ######
    df_ref = df_ref.loc[df_ref['unit_price'] > 0.04, :]
    ###### Categorical Cols ######
    df_ref = df_ref.loc[~df_ref['stock_code'].isin(['POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY','DCGSSGIRL', 'PADS', 'B', 'CRUK'])]


    # Description
    df_ref = df_ref.drop(columns='description', axis=1)

    # Map
    df_ref = df_ref[~df_ref['country'].isin(['European Community', 'Unspecified'])]

    # Bad Users - Atualizado após Análise Univariada
    df_filtered = df_ref[~df_ref['customer_id'].isin([16464])]

    # Quantity - negative numbers means product returns
    df_returns = df_ref.loc[df_ref['quantity'] < 0, :]
    df_purchases = df_ref.loc[df_ref['quantity'] > 0, :]
    

    
    # When returning many variables, it is a good practice to give them names
    return dict(
        filtered_Ecommerce = df_filtered,
        returns = df_returns,
        purchases = df_purchases
        )
# Aqui eu termino a seção "3.0 - Feature Filtering", retorno o DF principal (df_filtered) 
# e tb retorno os df returns e purchases, que serão utilizados na seção "4.0 - Feature Engineering"



