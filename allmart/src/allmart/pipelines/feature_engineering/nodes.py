"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.11
"""


import pandas as pd


def feature_engineering(df_preprocessed: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for Ecommerce.

    Args:
        preprocessed_Ecommerce: Preprocessed data.
    Returns:
        Tuple containing RFM dataframe and another dataframe with derivative features.
    """
    
    df_preprocessed['gross_revenue'] = df_preprocessed['quantity'] * df_preprocessed['unit_price']
    df_preprocessed['invoice_date'] = pd.to_datetime(df_preprocessed['invoice_date'])
    # Monetary
    df_monetary = df_preprocessed.loc[:, ['customer_id', 'gross_revenue']].groupby('customer_id').sum().reset_index().copy()
    
    # Recency: Dia da última compra
    df_recency = df_preprocessed.groupby('customer_id')['invoice_date'].max().reset_index()
    df_recency['recency'] = (df_preprocessed['invoice_date'].max() - df_recency['invoice_date']).dt.days
    df_recency = df_recency[['customer_id', 'recency']].copy()
    
    
    # Frequency - Contagem do número de compras feitas pelo cliente
    df_freq = (df_preprocessed[['customer_id', 'invoice_no']].drop_duplicates()
                                                          .groupby('customer_id')
                                                          .count()
                                                          .reset_index()
                                                          .astype(int)
                                                          .rename(columns={'invoice_no':'frequency'}))

    df_rfm = pd.merge(df_recency, df_freq, on='customer_id', how='left')
    df_rfm = pd.merge(df_rfm, df_monetary, on='customer_id', how='left')
    df_rfm.rename(columns={'invoice_date': 'recency',
                           'invoice_no': 'frequency',
                           'gross_revenue': 'monetary'},
                  inplace=True)
    
    
    df_engineered = df_preprocessed.drop(['invoice_no', 'stock_code', 'quantity', 'invoice_date', 'unit_price', 'country'], axis=1).drop_duplicates(ignore_index=True)
    
    df_preprocessed['gross_revenue'] = df_preprocessed['quantity'] * df_preprocessed['unit_price']    
    
    # Freq Quantity items
    df_freq2 = (df_preprocessed[['customer_id', 'quantity']].groupby('customer_id').sum()
                                                    .reset_index()
                                                    .rename(columns={'quantity':'qtde_items'}))
    
    df_freq3 = (df_preprocessed[['customer_id', 'stock_code']].groupby('customer_id').count()
                                                .reset_index()
                                                .rename(columns={'stock_code':'qtde_products'}))
    
    # Avg Ticket - Ticket Médio
    df_avg_ticket = (df_preprocessed[['customer_id', 'gross_revenue']].groupby('customer_id')
                                                         .mean()
                                                         .reset_index()
                                                         .rename(columns={'gross_revenue':'avg_ticket'}))


    # Avg Basket Size
    df_avg_basket_size = (df_preprocessed.loc[:, ['customer_id', 'invoice_no', 'quantity']].groupby('customer_id') \
                                                                              .agg(n_purchase=('invoice_no', 'nunique'),
                                                                               n_products=('quantity', 'sum')) \
                                                                              .reset_index())
    avg_basket_size = df_avg_basket_size['n_products'] / df_avg_basket_size['n_purchase']
    df_avg_basket_size['avg_basket_size'] = avg_basket_size

    # Avg Unique Basket Size
    df_avg_unique_basket_size = (df_preprocessed.loc[:, ['customer_id', 'invoice_no', 'stock_code']].groupby('customer_id') \
                                                                          .agg(n_purchase=('invoice_no', 'nunique'),
                                                                               n_products=('stock_code', 'nunique')) \
                                                                          .reset_index())
    unique_basket_size = df_avg_unique_basket_size['n_products'] / df_avg_unique_basket_size['n_purchase']
    df_avg_unique_basket_size['avg_unique_basket_size'] = unique_basket_size

    
    df_engineered = pd.merge(df_engineered, df_rfm, on='customer_id', how='left')
    df_engineered = pd.merge(df_engineered, df_freq2, how='left', on='customer_id')
    df_engineered = pd.merge(df_engineered, df_freq3, how='left', on='customer_id')
    df_engineered = pd.merge(df_engineered, df_avg_ticket, on='customer_id', how='left')
    df_engineered = pd.merge(df_engineered, df_avg_basket_size[['customer_id', 'avg_basket_size']], how='left', on='customer_id')
    df_engineered = pd.merge(df_engineered, df_avg_unique_basket_size[['customer_id', 'avg_unique_basket_size']], how='left', on='customer_id')
    
    df_engineered.rename(columns={'gross_revenue':'monetary',
                          },
                    inplace=True)
    
    return df_engineered, df_rfm