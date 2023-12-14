import pytest
from tabulate import tabulate
from functools import partial
from blendsql import SQLiteDBConnector, blend
from tests.utils import assert_equality, starts_with, get_length, select_first_sorted
from tests.constants import MULTI_TABLE_DB_PATH

tabulate = partial(tabulate, headers="keys", showindex="never")


@pytest.fixture
def db() -> SQLiteDBConnector:
    return SQLiteDBConnector(db_path=MULTI_TABLE_DB_PATH)


@pytest.fixture
def ingredients() -> set:
    return {starts_with, get_length, select_first_sorted}


def test_simple_multi_exec(db, ingredients):
    """Test with multiple tables.
    Also ensures we only pass what is neccessary to the external ingredient F().
    "Show me the price of tech stocks in my portfolio that start with 'A'"
    """
    blendsql = """
    SELECT FPS_SYMBOL, TRADE_DT, OPEN_PRICE_A FROM price_history
        WHERE price_history.FPS_SYMBOL IN (
            SELECT Symbol FROM portfolio
            WHERE {{starts_with('A', 'portfolio::Description')}} = 1
            AND portfolio.Symbol in (
                SELECT Symbol FROM constituents 
                WHERE constituents.Sector = 'Information Technology'
            )
        )
    """
    sql = """
    SELECT FPS_SYMBOL, TRADE_DT, OPEN_PRICE_A FROM price_history
        WHERE price_history.FPS_SYMBOL IN (
            SELECT Symbol FROM portfolio
            WHERE portfolio.Description LIKE "A%"
            AND Symbol in (
                SELECT Symbol FROM constituents 
                WHERE constituents.Sector = 'Information Technology'
            )
        )
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE Symbol in
    (
        SELECT Symbol FROM constituents
                WHERE sector = 'Information Technology'
    )
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_join_multi_exec(db, ingredients):
    blendsql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = 'Information Technology'
        AND {{starts_with('A', 'constituents::Name')}} = 1
        AND lower(account_history.Action) like "%dividend%"
        """
    sql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = 'Information Technology'
        AND constituents.Name LIKE "A%"
        AND lower(account_history.Action) like "%dividend%"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT Name) FROM constituents WHERE Sector = 'Information Technology'
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_select_multi_exec(db, ingredients):
    blendsql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name 
        FROM account_history 
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol 
        WHERE {{select_first_sorted('_', 'constituents::Sector')}}
        AND lower(account_history.Action) like "%dividend%"
    """
    sql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = (
            SELECT Sector FROM constituents 
            ORDER BY Sector LIMIT 1
        )
        AND lower(account_history.Action) like "%dividend%"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


def test_complex_multi_exec(db, ingredients):
    blendsql = """
    SELECT DISTINCT constituents.Symbol as Symbol FROM constituents
    LEFT JOIN fundamentals ON constituents.Symbol = fundamentals.FPS_SYMBOL
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE fundamentals.MARKET_DAY_DT > '2023-02-23'
    AND ({{get_length('n_length', 'constituents::Name')}} > 3 OR {{starts_with('A', 'portfolio::Symbol')}})
    AND portfolio.Symbol IS NOT NULL
    ORDER BY {{get_length('n_length', 'constituents::Name')}} LIMIT 1
    """
    sql = """
    SELECT DISTINCT constituents.Symbol as Symbol FROM constituents
    LEFT JOIN fundamentals ON constituents.Symbol = fundamentals.FPS_SYMBOL
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE fundamentals.MARKET_DAY_DT > '2023-02-23'
    AND (LENGTH(constituents.Name) > 3 OR portfolio.Symbol LIKE "A%")
    AND portfolio.Symbol IS NOT NULL
    ORDER BY LENGTH(constituents.Name) LIMIT 1
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
