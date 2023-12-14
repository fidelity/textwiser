import copy
import logging
import time
import uuid


import pandas as pd
import pyparsing as pp
import re
from typing import Dict, Iterable, List, Set, Union
from sqlite3 import OperationalError
import sqlglot.expressions
from attr import attrs, attrib
from functools import partial
from sqlglot import parse_one, exp
from colorama import Fore

pp.ParserElement.enable_packrat()

from .utils import (
    sub_tablename,
    get_temp_session_table,
    get_temp_subquery_table,
    delete_session_tables,
    recover_blendsql,
)
from .db.sqlite_db_connector import SQLiteDBConnector
from ._sqlglot import (
    construct_abstracted_selects,
    MODIFIERS,
    get_singleton_child,
    get_reversed_subqueries,
    maybe_set_subqueries_to_true,
)
from ._sqlglot import infer_map_constraints as _infer_map_constraints
from .ingredients.ingredient import Ingredient
from .smoothie import Smoothie, SmoothieMeta
from .constants import IngredientType, DEFAULT_ENDPOINT_NAME
from .ingredients.builtin.llm.endpoint import Endpoint
from .ingredients.builtin.llm.utils import initialize_endpoint


def construct_function_call_grammar(function_names: Set[str]) -> pp.core.Forward:
    """Creates pyparsing grammar to extract positional + named args from function."""
    function_list = pp.MatchFirst([pp.CaselessKeyword(i) for i in function_names])
    nums = pp.Word(pp.nums)
    str_arg = pp.Regex(r"(\').*?(\')").setParseAction(
        lambda x: re.sub(r"(\')", "", x[0])
    )
    int_arg = nums.setParseAction(lambda x: int(x[0]))
    float_arg = pp.Combine(pp.Optional("-") + nums + "." + nums).setParseAction(
        lambda x: float(x[0])
    )
    arg = str_arg | float_arg | int_arg

    positional_command_arg = arg + ~pp.FollowedBy(pp.Char("=")) | pp.Suppress(
        ","
    ).leave_whitespace() + arg + ~pp.FollowedBy(pp.Char("="))
    named_command_arg = pp.Group(
        pp.Word(pp.alphas) + pp.Char("=") + arg
        | pp.Suppress(",").leave_whitespace()
        + pp.Word(pp.alphas + "_")
        + pp.Char("=")
        + arg
    )

    function_call_start = pp.Suppress(pp.Literal("{{"))
    function_call_end = pp.Suppress(pp.Literal("}}"))
    function_call = pp.Forward()
    function_call <<= (
        function_call_start
        + function_list("function")
        + pp.Suppress("(").leave_whitespace()
        - pp.ZeroOrMore(positional_command_arg)("args")
        - pp.ZeroOrMore(named_command_arg)("kwargs")
        + pp.Suppress(")")
        + function_call_end
    )
    return function_call


@attrs
class Kitchen(list):
    """Superset of list. A collection of ingredients."""

    db: SQLiteDBConnector = attrib()
    session_uuid: str = attrib()

    def names(self):
        return [i.name for i in self]

    def get_from_name(self, name: str):
        for f in self:
            if f.name == name:
                return f
        raise ValueError(f"Function with name {name} not found")

    def extend(self, functions: Iterable[Ingredient]) -> None:
        assert all(
            issubclass(x, Ingredient) for x in functions
        ), "All arguments passed to `Kitchen` must be ingredients!"
        for function in functions:
            function = function(function.__name__)
            # Add db and session_uuid as default kwargs
            # This way, they are able to interact with data
            function.db = self.db
            function.session_uuid = self.session_uuid
            self.append(function)


# @profile
def blend(
    query: str,
    db: SQLiteDBConnector,
    ingredients: Iterable[Ingredient] = None,
    verbose: bool = False,
    use_endpoint: Union[str, Endpoint] = None,
    infer_map_constraints: bool = False,
    table_to_title: Dict[str, str] = None,
    silence_db_exec_errors: bool = True,
) -> Smoothie:
    """Executes a BlendSQL query on a database given an ingredient context.

    Args:
        query: The BlendSQL query to execute
        db: Database connector object
        ingredients: List of ingredient objects, to use in interpreting BlendSQL query
        verbose: Boolean defining whether to run in logging.debug mode
        use_endpoint: Optionally override whatever endpoint_name argument we pass to LLM ingredient.
            Useful for research applications, where we don't (necessarily) want the parser to choose endpoints.
        infer_map_constraints: Optionally infer the output format of an `IngredientMap` call, given the predicate context
            For example, in `{{LLMMap('convert to date', 'w::listing date')}} <= '1960-12-31'`
            We can infer the output format should look like '1960-12-31'
                and put this in the `example_outputs` kwarg
        table_to_title: Optional mapping from table name to title of table.
            Useful for datasets like WikiTableQuestions, where relevant info is stored in table title.

    Returns:
        smoothie: Smoothie dataclass containing pd.DataFrame output and execution metadata
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        start = time.time()
        example_map_outputs = []
        naive_execution = False
        session_uuid = str(uuid.uuid4())[:4]
        if ingredients is None:
            ingredients = []
        _query: sqlglot.expressions.Select = parse_one(re.sub(r"\s+", " ", query))
        original_query = copy.deepcopy(recover_blendsql(_query.sql(pretty=False)))
        query = recover_blendsql(_query.sql(pretty=False))

        # Preliminary check - we can't have anything that modifies database state
        if _query.find(MODIFIERS):
            raise ValueError("BlendSQL query cannot have `DELETE` clause!")

        # Create our Kitchen
        kitchen = Kitchen(db=db, session_uuid=session_uuid)
        kitchen.extend(ingredients)
        grammar = construct_function_call_grammar(kitchen.names())
        function_re = re.compile(
            r"({})".format(
                "|".join(
                    [
                        r"{{\s*" + name + r"\(.*?\)" + r"\s*}}"
                        for name in list(kitchen.names())
                    ]
                )
            ),
            flags=re.IGNORECASE,
        )

        # If there's no `SELECT` and just a QAIngredient, wrap it in a `SELECT CASE` query
        if _query.find(exp.Select) is None:
            for m in function_re.finditer(query):
                function_str = m.group(0)
                parsed_function = grammar.parseString(function_str).as_dict()
                _function: Ingredient = kitchen.get_from_name(
                    parsed_function["function"]
                )
                if _function.ingredient_type != IngredientType.QA:
                    raise ValueError(
                        "Invalid BlendSQL - needs to be QA ingredient for auto-wrapping with `SELECT CASE`"
                    )
                query = query.replace(
                    function_str,
                    f"""
                SELECT CASE WHEN FALSE THEN FALSE WHEN TRUE THEN {function_str} END
                """,
                )
            # Now re-parse with sqlglot
            _query: sqlglot.expressions.Select = parse_one(re.sub(r"\s+", " ", query))
            original_query = copy.deepcopy(recover_blendsql(_query.sql(pretty=False)))

        # If we don't have any ingredient calls, execute as normal SQL
        if len(ingredients) == 0 or function_re.search(query) is None:
            return Smoothie(
                df=db.execute_query(query, silence_errors=silence_db_exec_errors),
                meta=SmoothieMeta(
                    process_time_seconds=time.time() - start,
                    num_values_passed=0,
                    example_map_outputs=example_map_outputs,
                    ingredients=[],
                    query=original_query,
                    db_path=db.db_path,
                    contains_ingredient=False,
                ),
            )

        _get_temp_session_table = partial(get_temp_session_table, session_uuid)
        # Mapping from {"QA('does this company...', 'constituents::Name')": 'does this company'...})
        function_call_to_res: Dict[str, str] = {}
        modified_tables = set()
        # TODO: Currently, as we traverse upwards from deepest subquery,
        #   if any lower subqueries have an ingredient, we deem the current
        #   as inelligible for optimization. Maybe this can be improved in the future.
        prev_subquery_has_ingredient = False
        for subquery_idx, subquery in enumerate(get_reversed_subqueries(_query)):
            # # Only cache executed_ingredients within the same subquery
            # The same ingredient may have different results within a different subquery context
            executed_subquery_ingredients: Set[str] = set()
            _get_temp_subquery_table = partial(
                get_temp_subquery_table, session_uuid, subquery_idx
            )
            if not isinstance(subquery, exp.Select):
                # We need to create a select query from this subquery
                # So we find the parent select, and grab that table
                parent_select_tablenames = [
                    i.name
                    for i in subquery.find_ancestor(exp.Select).find_all(exp.Table)
                ]
                if len(parent_select_tablenames) == 1:
                    subquery_str = recover_blendsql(
                        f"SELECT * FROM {parent_select_tablenames[0]} WHERE "
                        + get_singleton_child(subquery).sql()
                    )
                else:
                    logging.debug(
                        Fore.YELLOW
                        + "Encountered subquery without `SELECT`, and more than 1 table!\nCannot optimize yet, skipping this step."
                    )
                    continue
            else:
                subquery_str = recover_blendsql(subquery.sql())
            subquery = parse_one(
                subquery_str
            )  # Need to do this so we don't track parents into construct_abstracted_selects
            for tablename, abstracted_query in construct_abstracted_selects(
                subquery, prev_subquery_has_ingredient=prev_subquery_has_ingredient
            ):
                logging.debug(
                    Fore.CYAN
                    + f"Executing `{abstracted_query}` and setting to `{_get_temp_subquery_table(tablename)}`..."
                    + Fore.RESET
                )
                try:
                    db.execute_query(abstracted_query).to_sql(
                        _get_temp_subquery_table(tablename),
                        db.con,
                        if_exists="replace",
                        index=False,
                    )
                except OperationalError:
                    # Fallback to naive execution
                    subquery = _query
                    subquery_str = recover_blendsql(subquery.sql())
                    naive_execution = True
            if prev_subquery_has_ingredient:
                subquery = maybe_set_subqueries_to_true(subquery)
                subquery_str = recover_blendsql(subquery.sql())
            # Find all ingredients to execute (e.g. '{{f(a, b, c)}}')
            # Iterate with regex, then parse with pyparsing grammar
            # Track when we've created a new table from a MapIngredient call
            # only at the end of parsing a subquery, we can merge to the original session_uuid table
            tablename_to_map_out: Dict[str, List[pd.DataFrame]] = {}
            for m in function_re.finditer(subquery_str):
                prev_subquery_has_ingredient = True
                function_str = m.group(0)
                if function_str in executed_subquery_ingredients:
                    # Don't execute same ingredient twice
                    continue
                executed_subquery_ingredients.add(function_str)
                parsed_function = grammar.parseString(function_str).as_dict()
                _function: Ingredient = kitchen.get_from_name(
                    parsed_function["function"]
                )
                if _function.ingredient_type not in IngredientType:
                    raise ValueError(
                        f"Not sure what to do with ingredient_type '{_function.ingredient_type}' yet"
                    )
                # kwargs gets returned as ['limit', '=', 10] sort of list
                # So we need to parse by indices in dict expression
                # maybe if I was better at pp.Suppress we wouldn't need this
                kwargs_dict = {x[0]: x[-1] for x in parsed_function["kwargs"]}

                # Optionally modify kwargs dict, depending on blend() args
                if use_endpoint is not None:
                    if "endpoint_name" in kwargs_dict:
                        logging.debug(
                            Fore.YELLOW
                            + "Overriding passed arg for 'endpoint_name'!"
                            + Fore.RESET
                        )
                    kwargs_dict["endpoint_name"] = None
                    if isinstance(use_endpoint, Endpoint):
                        kwargs_dict["endpoint"] = use_endpoint
                    else:
                        kwargs_dict["endpoint"] = initialize_endpoint(use_endpoint)
                else:
                    kwargs_dict["endpoint"] = initialize_endpoint(
                        kwargs_dict.get("endpoint_name", DEFAULT_ENDPOINT_NAME)
                    )
                if _function.ingredient_type == IngredientType.MAP:
                    # Latter is the winner.
                    # So if we already define something in kwargs_dict,
                    #   It's not overriden here
                    kwargs_dict = (
                        _infer_map_constraints(
                            subquery=subquery,
                            subquery_str=subquery_str,
                            start=m.start(),
                            end=m.end(),
                        )
                        | kwargs_dict
                        if infer_map_constraints
                        else kwargs_dict
                    )
                if table_to_title is not None:
                    kwargs_dict["table_to_title"] = table_to_title

                # Execute our ingredient function
                function_out = _function(
                    *parsed_function["args"],
                    **kwargs_dict
                    | {
                        "get_temp_subquery_table": _get_temp_subquery_table,
                        "get_temp_session_table": _get_temp_session_table,
                    },
                )
                # Check how to handle output, depending on ingredient type
                if _function.ingredient_type == IngredientType.MAP:
                    # Parse so we replace this function in blendsql with 1st arg
                    #   (new_col, which is the question we asked)
                    #  But also update our underlying table, so we can execute correctly at the end
                    (new_col, tablename, colname, new_table) = function_out
                    non_null_subset = new_table[new_table[new_col].notnull()]
                    # These are just for logging + debugging purposes
                    example_map_outputs.append(
                        tuple(zip(non_null_subset[colname], non_null_subset[new_col]))
                    )
                    if tablename in tablename_to_map_out:
                        tablename_to_map_out[tablename].append(new_table)
                    else:
                        tablename_to_map_out[tablename] = [new_table]
                    modified_tables.add(tablename)
                    function_call_to_res[function_str] = f'"{new_col}"'
                elif _function.ingredient_type == IngredientType.SELECT:
                    (choose_q, tablename, colname, chosen_value) = function_out
                    example_map_outputs.append((choose_q, chosen_value))
                    chosen_value = (
                        f"'{chosen_value}'"
                        if isinstance(chosen_value, str)
                        else chosen_value
                    )
                    function_call_to_res[
                        function_str
                    ] = f'"{tablename}"."{colname}" = {chosen_value}'
                elif _function.ingredient_type in (
                    IngredientType.STRING,
                    IngredientType.QA,
                ):
                    # Here, we can simply insert the function's output
                    function_call_to_res[function_str] = function_out
                else:
                    raise ValueError(
                        f"Not sure what to do with ingredient_type '{_function.ingredient_type}' yet\n(Also, we should have never hit this error....)"
                    )
                if naive_execution:
                    break
            # Combine all the retrieved ingredient outputs
            for tablename, llm_outs in tablename_to_map_out.items():
                if len(llm_outs) > 0:
                    # Once we finish parsing this subquery, write to our session_uuid table
                    # Below, we differ from Binder, which seems to replace the old table
                    # On their left join merge command: https://github.com/HKUNLP/Binder/blob/9eede69186ef3f621d2a50572e1696bc418c0e77/nsql/database.py#L196
                    # We create a new temp table to avoid a potentially self-destructive operation
                    base_tablename = tablename
                    _base_table: pd.DataFrame = db.execute_query(
                        f"SELECT * FROM '{base_tablename}'"
                    )
                    base_table = _base_table
                    if db.has_table(_get_temp_session_table(tablename)):
                        base_tablename = _get_temp_session_table(tablename)
                        base_table: pd.DataFrame = db.execute_query(
                            f"SELECT * FROM '{base_tablename}'"
                        )
                    previously_added_columns = base_table.columns.difference(
                        _base_table.columns
                    )
                    assert len(set([len(x) for x in llm_outs])) == 1
                    llm_out_df = pd.concat(llm_outs, axis=1)
                    llm_out_df = llm_out_df.loc[:, ~llm_out_df.columns.duplicated()]
                    # Handle duplicate columns, e.g. in test_nested_duplicate_ingredient_calls()
                    for column in previously_added_columns:
                        if all(
                            column in x
                            for x in [llm_out_df.columns, base_table.columns]
                        ):
                            # Fill nan in llm_out_df with those values in base_table
                            pd.testing.assert_index_equal(
                                base_table.index, llm_out_df.index
                            )
                            llm_out_df[column] = llm_out_df[column].fillna(
                                base_table[column]
                            )
                            base_table = base_table.drop(columns=column)
                    llm_out_df = llm_out_df[
                        llm_out_df.columns.difference(base_table.columns)
                    ]
                    pd.testing.assert_index_equal(base_table.index, llm_out_df.index)
                    merged = base_table.merge(
                        llm_out_df, how="left", right_index=True, left_index=True
                    )
                    merged.to_sql(
                        name=_get_temp_session_table(tablename),
                        con=db.con,
                        if_exists="replace",
                        index=False,
                    )
                    modified_tables.add(tablename)

        # Now insert the function outputs to the original query
        for function_str, res in function_call_to_res.items():
            query = query.replace(function_str, res)
        for t in modified_tables:
            query = sub_tablename(t, f"'{_get_temp_session_table(t)}'", query)
        logging.debug("")
        logging.debug(
            "**********************************************************************************"
        )
        logging.debug(Fore.GREEN + f"Final Query:\n{query}" + Fore.RESET)
        logging.debug(
            "**********************************************************************************"
        )
        logging.debug("")
        df = db.execute_query(query, silence_errors=silence_db_exec_errors)
        return Smoothie(
            df=df,
            meta=SmoothieMeta(
                process_time_seconds=time.time() - start,
                num_values_passed=sum(
                    [
                        i.num_values_passed
                        for i in kitchen
                        if hasattr(i, "num_values_passed")
                    ]
                ),
                example_map_outputs=example_map_outputs,
                ingredients=ingredients,
                query=original_query,
                db_path=db.db_path,
            ),
        )

    except Exception:
        raise
    finally:
        delete_session_tables(db, session_uuid)
