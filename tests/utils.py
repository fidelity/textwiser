import pandas as pd
from typing import List
from typing import Iterable, Any
from blendsql.ingredients import MapIngredient, SelectIngredient


class starts_with(MapIngredient):
    def run(self, arg: str, **kwargs) -> Iterable[Any]:
        """Simple test function, equivalent to the following in SQL:
            `LIKE '{arg}%`
        This allows us to compare the output of a BlendSQL script with a SQL script easily.
        """
        # Unpack default kwargs
        values, original_table, tablename, colname = self.unpack_default_kwargs(
            **kwargs
        )

        mapped_values = [int(i.startswith(arg)) for i in values]

        return mapped_values


class get_length(MapIngredient):
    def run(self, arg: str, **kwargs) -> Iterable[Any]:
        """Simple test function, equivalent to the following in SQL:
            `LENGTH '{arg}%`
        This allows us to compare the output of a BlendSQL script with a SQL script easily.
        """
        # Unpack default kwargs
        values, original_table, tablename, colname = self.unpack_default_kwargs(
            **kwargs
        )
        mapped_values = [len(i) for i in values]
        return mapped_values


class select_first_sorted(SelectIngredient):
    def run(self, arg: str, **kwargs) -> Iterable[Any]:
        """Simple test function, equivalent to the following in SQL:
        `ORDER BY {colname} LIMIT 1`
        """
        # Unpack default kwargs
        values, original_table, tablename, colname = self.unpack_default_kwargs(
            **kwargs
        )
        chosen_value = sorted(values)[0]
        return chosen_value


def assert_equality(smoothie, sql_df: pd.DataFrame, args: List[str] = None):
    blendsql_df = smoothie.df
    if args is not None:
        arg_overlap = blendsql_df.columns.intersection(args).tolist()
        if len(arg_overlap) > 0:
            blendsql_df = blendsql_df.drop(arg_overlap, axis=1)
    pd.testing.assert_frame_equal(blendsql_df, sql_df)
