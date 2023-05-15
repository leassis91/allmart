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
    Ecommerce.drop(columns=['Unnamed: 8'], axis=1, inplace=True)
    cols_new = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date', 'unit_price', 'customer_id', 'country']
    Ecommerce.columns = cols_new

    Ecommerce
    Ecommerce["iata_approved"] = _is_true(Ecommerce["iata_approved"])
    Ecommerce["company_rating"] = _parse_percentage(Ecommerce["company_rating"])
    return Ecommerce


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles