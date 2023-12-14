from attr import attrs, attrib
from abc import abstractmethod, ABC
import pandas as pd
from typing import Any, Iterable, Union
from sqlglot import parse_one

from .. import utils
from ..constants import IngredientType
from ..db.sqlite_db_connector import SQLiteDBConnector
from .._sqlglot import MODIFIERS


class IngredientException(ValueError):
    pass


def map_base_unpack_default_kwargs(**kwargs):
    """Unpack default kwargs for all ingredients where
    we get some list of column values as default kwargs.
    Includes MapIngredient, SelectIngredient.
    """
    return (
        kwargs.get("values"),
        kwargs.get("original_table"),
        kwargs.get("tablename"),
        kwargs.get("colname"),
    )


def align_to_real_columns(db: SQLiteDBConnector, colname: str, tablename: str) -> str:
    table_columns = db.execute_query(f'SELECT * FROM "{tablename}" LIMIT 1').columns
    if colname not in table_columns:
        # Try to align with column, according to some normalization rules
        cleaned_to_original = {
            col.replace("\\n", " ").replace("\xa0", " "): col for col in table_columns
        }
        colname = cleaned_to_original[colname]
    return colname


def string_base_unpack_default_kwargs(**kwargs):
    """Here we don't need values or table,
    so we just get basic tablename, colname.
    """
    return (
        kwargs.get("tablename"),
        kwargs.get("colname"),
    )


@attrs
class Ingredient(ABC):
    name: str = attrib()
    db: SQLiteDBConnector = attrib(init=False)
    session_uuid: str = attrib(init=False)
    ingredient_type: str = attrib(init=False)

    def __repr__(self):
        return f"{self.ingredient_type} {self.name}"

    def __str__(self):
        return f"{self.ingredient_type} {self.name}"

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        ...


@attrs
class MapIngredient(Ingredient):
    """For a given table/column pair, maps an external function
    to each of the given values, creating a new column."""

    ingredient_type: str = IngredientType.MAP.value
    num_values_passed: int = 0

    def unpack_default_kwargs(self, **kwargs):
        return map_base_unpack_default_kwargs(**kwargs)

    def __call__(self, arg: str, identifier: str, *args, **kwargs) -> tuple:
        """Returns tuple with format (arg, tablename, colname, new_table)"""
        tablename, colname = utils.get_tablename_colname(identifier)
        kwargs["tablename"] = tablename
        kwargs["colname"] = colname
        temp_subquery_table = kwargs.get("get_temp_subquery_table")(tablename)
        temp_session_table = kwargs.get("get_temp_session_table")(tablename)
        value_source_tablename = tablename
        if self.db.has_table(temp_subquery_table):
            # We've already applied some qa operation to this table
            # We want to use this as our base
            value_source_tablename = temp_subquery_table
        temp_session_tablename = None
        if self.db.has_table(temp_session_table):
            temp_session_tablename = temp_session_table

        # Need to be sure the new column doesn't already exist here
        new_arg_column = (
            f"___{arg}" if arg in set(self.db.iter_columns(tablename)) else arg
        )

        original_table = self.db.execute_query(f"SELECT * FROM '{tablename}'")

        # Get a list of values to map
        # First, check if we've already dumped some `MapIngredient` output to the main session table
        if temp_session_tablename:
            temp_session_table = self.db.execute_query(
                f"SELECT * FROM '{temp_session_tablename}'"
            )
            if new_arg_column in temp_session_table.columns:
                # We don't need to run this function on everything,
                #   if a previous subquery already got to certain values
                colname = align_to_real_columns(
                    db=self.db, colname=colname, tablename=temp_session_tablename
                )
                values = self.db.execute_query(
                    f'SELECT DISTINCT "{colname}" FROM "{temp_session_tablename}" WHERE "{new_arg_column}" IS NULL'
                )[colname].tolist()
        else:
            colname = align_to_real_columns(
                db=self.db, colname=colname, tablename=value_source_tablename
            )
            values = self.db.execute_query(
                f'SELECT DISTINCT "{colname}" FROM "{value_source_tablename}"'
            )[colname].tolist()

        if len(values) == 0:
            original_table[new_arg_column] = None
            return (new_arg_column, tablename, colname, original_table)

        kwargs["values"] = values
        kwargs["original_table"] = original_table
        mapped_values: Iterable[Any] = self.run(arg, *args, **kwargs)
        if not isinstance(mapped_values, Iterable):
            raise IngredientException(
                f"{self.name}.run() should return Iterable!\nGot{type(mapped_values)}"
            )
        self.num_values_passed += len(mapped_values)
        df_as_dict = {colname: [], new_arg_column: []}
        for value, mapped_value in zip(values, mapped_values):
            df_as_dict[colname].append(value)
            df_as_dict[new_arg_column].append(mapped_value)
        subtable = pd.DataFrame(df_as_dict)
        if all(isinstance(x, (int, type(None))) for x in mapped_values):
            subtable[new_arg_column] = subtable[new_arg_column].astype("Int64")
        # Add new_table to original table
        new_table = original_table.merge(subtable, how="left", on=colname)
        if new_table.shape[0] != original_table.shape[0]:
            raise IngredientException(
                f"subtable from run() needs same length as # rows from original\nOriginal has {original_table.shape[0]}, new_table has {new_table.shape[0]}"
            )
        # Now, new table has original columns + column with the name of the question we answered
        return (new_arg_column, tablename, colname, new_table)

    @abstractmethod
    def run(self, *args, **kwargs) -> Iterable[Any]:
        ...


@attrs
class SelectIngredient(Ingredient):
    """Similar to the Map ingredient, but instead chooses
    a single value from column values and asserts identity
    (column = chosen_value)."""

    ingredient_type: str = IngredientType.SELECT.value
    num_values_passed: int = 0

    def unpack_default_kwargs(self, **kwargs):
        return map_base_unpack_default_kwargs(**kwargs)

    def __call__(self, arg: str, identifier: str, *args, **kwargs) -> tuple:
        """Returns tuple with format (arg, tablename, colname, chosen_value)
        TODO: can probably abstract out the initial fetching of values to a separate
            function, normalize with MapIngredient
        """
        tablename, colname = utils.get_tablename_colname(identifier)
        kwargs["tablename"] = tablename
        kwargs["colname"] = colname
        temp_table = kwargs.get("get_temp_subquery_table")(tablename)
        value_source_table = tablename
        if self.db.has_table(temp_table):
            # We've already applied some qa operation to this table
            # We want to use this as our base
            value_source_table = temp_table

        # Get a list of values to map
        values = self.db.execute_query(
            f'SELECT DISTINCT "{colname}" FROM "{value_source_table}"'
        )[colname].tolist()

        if len(values) == 0:
            return (arg, tablename, None)

        kwargs["values"] = values

        chosen_value = self.run(arg, *args, **kwargs)
        if not isinstance(chosen_value, (str, int, float)):
            raise IngredientException(
                f"{self.name}.run() should return one of (str, int, float)!\nGot{type(chosen_value)}"
            )
        self.num_values_passed += len(values)
        # Now, new table has original columns + column with the name of the question we answered
        return (arg, tablename, colname, chosen_value)


@attrs
class QAIngredient(Ingredient):
    ingredient_type: str = IngredientType.QA.value

    def unpack_default_kwargs(self, **kwargs):
        return kwargs.get("subtable")

    def __call__(
        self, arg: str, identifier: str, *args, **kwargs
    ) -> Union[str, int, float]:
        subtable: pd.DataFrame = None
        try:
            tablename, colname = utils.get_tablename_colname(identifier)
            subtable = self.db.execute_query(f'SELECT "{colname}" FROM "{tablename}"')
        except ValueError:
            # Instead, we have a SQL subquery we need to execute
            # First check to see we're not modifying database state
            _query = parse_one(identifier)
            if _query.find(MODIFIERS):
                raise ValueError(
                    "BlendSQL query cannot have `DELETE` clause!"
                ) from None
            subtable = self.db.execute_query(_query.sql())
        kwargs["subtable"] = subtable
        response: Union[str, int, float] = self.run(arg, *args, **kwargs)
        return response


@attrs
class StringIngredient(Ingredient):
    """Outputs a string to be placed directly into the SQL query."""

    ingredient_type: str = IngredientType.STRING.value

    def unpack_default_kwargs(self, **kwargs):
        return string_base_unpack_default_kwargs(**kwargs)

    def __call__(self, identifier: str, *args, **kwargs) -> str:
        tablename, colname = utils.get_tablename_colname(identifier)
        kwargs["tablename"] = tablename
        kwargs["colname"] = colname
        # Don't pass identifier arg, we don't need it anymore
        args = tuple()
        new_str = self.run(*args, **kwargs)
        if not isinstance(new_str, str):
            raise IngredientException(
                f"{self.name}.run() should return str\nGot{type(new_str)}"
            )
        return new_str

    @abstractmethod
    def run(self, *args, **kwargs) -> str:
        ...
