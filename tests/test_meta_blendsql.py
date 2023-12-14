import pytest
from tabulate import tabulate
from functools import partial
from blendsql import SQLiteDBConnector, blend
from pathlib import Path
from tests.utils import starts_with, get_length
from tests.constants import SINGLE_TABLE_DB_PATH

tabulate = partial(tabulate, headers="keys", showindex="never")


@pytest.fixture
def db() -> SQLiteDBConnector:
    return SQLiteDBConnector(db_path=SINGLE_TABLE_DB_PATH)


@pytest.fixture
def ingredients() -> set:
    return {starts_with, get_length}


def test_save_recipe(db, ingredients):
    blendsql = """
    SELECT * FROM transactions WHERE {{starts_with('Z', 'transactions::merchant')}}
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    assert not smoothie.df.empty
    save_path = "test_saved_recipe"
    filenames = [save_path + ext for ext in [".ipynb", ".html"]]
    smoothie.save_recipe(save_path + ".ipynb", "My Test", as_html=True)
    for filename in filenames:
        p = Path(filename)
        assert p.exists()
        p.unlink()
