from sqlglot import exp, parse_one
from typing import Generator, Tuple, Union, Set, List, Dict, Any
import re
from ast import literal_eval

from .utils import recover_blendsql
from .constants import DEFAULT_ANS_SEP, DEFAULT_NAN_ANS

"""
Defines a set of transformations on the SQL AST, to be used with sqlglot.
https://github.com/tobymao/sqlglot

sqlglot.optimizer.simplify looks interesting
"""

SUBQUERY_EXP = (exp.Select,)
CONDITIONS = (
    exp.Where,
    exp.Group,
    # IMPORTANT: If we uncomment limit, then `test_limit` in `test_single_table_blendsql.py` will not pass
    # exp.Limit,
    exp.Except,
    exp.Order,
)
MODIFIERS = (exp.Delete, exp.AlterColumn, exp.AlterTable, exp.Drop)


def is_in_subquery(node):
    return node.find_ancestor(SUBQUERY_EXP).find_ancestor(SUBQUERY_EXP) is not None


def set_subqueries_to_true(node) -> Union[exp.Expression, None]:
    """For all subqueries (i.e. children exp.Select statements)
    set all these to TRUE abstractions.

    Used with node.transform().
    """
    if isinstance(node, exp.Predicate):
        if all(x in node.args for x in {"this", "expression"}):
            if node.args["expression"].find(exp.Subquery):
                if node.args["this"].find(exp.Select) is None:
                    return node.args["this"]
                else:
                    return None
        if is_in_subquery(node):
            return None
    if isinstance(node, SUBQUERY_EXP + (exp.Paren,)) and node.parent is not None:
        return parse_one("true")
    parent_select = node.find_ancestor(SUBQUERY_EXP)
    if parent_select and parent_select.parent is not None:
        return None
    return node


def prune_empty_where(node) -> Union[exp.Expression, None]:
    """
    Removes any `exp.Where` clause without any values.

    Used with node.transform()
    """
    if isinstance(node, exp.Where):
        if set(node.args.values()) == {None}:
            return None
        elif "this" in node.args:
            where_arg = node.args["this"]
            if "query" in where_arg.args and isinstance(
                where_arg.args["query"], exp.Boolean
            ):
                return None
    return node


def prune_alias_conditions(
    node, table_aliases: Set[str]
) -> Union[exp.Expression, None]:
    """
    If a node contains a table name appearing as an alias, we remove that node.
    This is only run on the `conditions` we extract.

    Used with node.transform()
    """
    if node is not None:
        if any(i.name in table_aliases for i in node.find_all(exp.Identifier)):
            return None
    return node


def set_structs_to_true(node) -> Union[exp.Expression, None]:
    """Prunes all nodes with an exp.Struct parent.
    Turns the exp.Struct node itself to a TRUE.

    Used with node.transform()
    """
    if isinstance(node, exp.Struct):
        return exp.TRUE
    parent_struct = node.find_ancestor(exp.Struct)
    if parent_struct and parent_struct.parent is not None:
        return None
    # Check to see if this is a literal or boolean, and we have a TRUE as a sibling
    if isinstance(node, (exp.Literal, exp.Boolean)):
        parent_pred = node.find_ancestor(exp.Predicate)
        if parent_pred is not None:
            child_bool = parent_pred.find(exp.Boolean)
            if child_bool is not None and child_bool.sql() == exp.TRUE.sql():
                return None
    return node


def prune_true_operators(node) -> Union[exp.Expression, None]:
    """

    Used with node.transform()
    """
    # Check to see if this is an exp.EQ, and we have a TRUE as a child
    if isinstance(node, exp.Condition):
        child_bool = node.find(exp.Boolean)
        if child_bool is not None:
            if len([i for i in node.walk(bfs=True)]) == 2:  # 1 child
                return child_bool
    return node


def extract_multi_table_predicates(
    node: exp.Where, tablename: str
) -> Union[exp.Expression, None]:
    """Extracts all non-Column predicates acting on a given tablename.
    non-Column since we want to exclude JOIN's (e.g. JOIN on A.col = B.col)

    Requirements to keep:
        - Must be a predicate node with an expression arg containing column associated with tablename, and some other non-column arg
        - Or, we're in a subquery
    If we have a predicate not meeting these conditions, set to exp.TRUE
        - This is much simpler than doing surgery on `WHERE AND ...` sort of relics

    Used with node.transform()

    Args:
        node: The exp.Where clause we're extracting predicates from
        tablename: The name of the table whose predicates we keep
    """
    if isinstance(node, exp.Where):
        return node
    # Don't abstract to `TRUE` if we're in a subquery
    # This is important!!! Without this, test_multi_table_blendsql.test_simple_multi_exec will fail
    # Causes difference between `SELECT * FROM portfolio WHERE TRUE AND portfolio.Symbol IN (SELECT Symbol FROM constituents WHERE constituents.Sector = 'Information Technology')`
    #   and `SELECT * FROM portfolio WHERE TRUE AND portfolio.Symbol IN (SELECT Symbol FROM constituents WHERE TRUE)`
    if is_in_subquery(node):
        return node
    if isinstance(node, exp.Predicate):
        if "this" in node.args:
            # This is true if we have a subquery
            # Just leave this as-is
            if "query" in node.args:
                return node
            # Need to apply `find` here in case of `LOWER` arg getting in our way
            this_column = node.args["this"].find(exp.Column)
            expression_column = node.args["expression"].find(exp.Column)
            if this_column is None:
                return node
            if this_column.table == tablename:
                if expression_column is None:
                    return node
                # This is False if we have a `JOIN` (a.colname = b.colname)
                elif expression_column.table == "":
                    return node
        return exp.TRUE
    return node


def get_singleton_child(node):
    """
    Helper function to get child of a node with exactly one child.
    """
    gen = node.walk()
    _ = next(gen)
    return next(gen)[0]


def get_alias_identifiers(node) -> Set[str]:
    """Given a SQL statement, returns defined aliases.
    Example:
        >>> get_alias_identifiers(parse_one("SELECT {{LLMMap('year from date', 'w::date')}} AS year FROM w")
        Returns:
            >>> ['year']
    """
    return set([i.find(exp.Identifier).name for i in node.find_all(exp.Alias)])


def get_predicate_literals(node) -> List[str]:
    """From a given SQL clause, gets all literals appearing as object of predicate.
    We treat booleans as literals here, which might be a misuse of terminology.
    Example:
        >>> get_predicate_literals(parse_one("{{LLM('year', 'w::year')}} IN ('2010', '2011', '2012')"))
        Returns:
            >>> ['2010', '2011', '2012']
    """
    literals = set()
    gen = node.walk()
    _ = next(gen)
    if isinstance(node.parent, exp.Select):
        return []
    for child, _, _ in gen:
        if child.find_ancestor(exp.Struct) or isinstance(child, exp.Struct):
            continue
        if isinstance(child, exp.Literal):
            literals.add(
                literal_eval(child.name) if not child.is_string else child.name
            )
            continue
        elif isinstance(child, exp.Boolean):
            literals.add(child.args["this"])
            continue
        for i in child.find_all(exp.Literal):
            if i.find_ancestor(exp.Struct):
                continue
            literals.add(literal_eval(i.name) if not i.is_string else i.name)
    return list(literals)


def get_reversed_subqueries(node):
    # Iterate through all subqueries (either parentheses or select)
    reversed_subqueries = [i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,))][::-1]
    return reversed_subqueries


def maybe_set_subqueries_to_true(node):
    if len([i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,))]) == 1:
        return node
    return node.transform(set_subqueries_to_true).transform(prune_empty_where)


def infer_map_constraints(
    subquery: exp.Expression, subquery_str: str, start: int, end: int
) -> Dict[str, str]:
    added_kwargs = {}
    ingredient_node = parse_one(subquery_str[start:end])
    child = None
    for child, _, _ in subquery.walk():
        if child == ingredient_node:
            break
    if child is None:
        raise ValueError
    ingredient_node_in_context = child
    start_node = ingredient_node_in_context.parent
    # Below handles when we're in a function
    # Example: CAST({{LLMMap('jump distance', 'w::notes')}} AS FLOAT)
    while isinstance(start_node, exp.Func) and start_node is not None:
        start_node = start_node.parent
    predicate_literals = []
    if start_node is not None:
        predicate_literals: List[Any] = get_predicate_literals(start_node)
    if len(predicate_literals) > 0:
        if all(isinstance(x, bool) for x in predicate_literals):
            # Add our bool pattern
            added_kwargs["pattern"] = f"(t|f|{DEFAULT_ANS_SEP}|{DEFAULT_NAN_ANS})+"
            added_kwargs["output_type"] = "boolean"
        elif all(isinstance(x, (int, float)) for x in predicate_literals):
            # Add our int/float pattern
            added_kwargs[
                "pattern"
            ] = f"(([0-9]|\.)+\{DEFAULT_ANS_SEP}|\{DEFAULT_NAN_ANS}\{DEFAULT_ANS_SEP})+"
            added_kwargs["output_type"] = "numeric"
        else:
            predicate_literals = [str(i) for i in predicate_literals]
            added_kwargs["output_type"] = "string"
            if len(predicate_literals) > 1:
                added_kwargs["example_outputs"] = DEFAULT_ANS_SEP.join(
                    predicate_literals
                )
            else:
                added_kwargs[
                    "example_outputs"
                ] = f"{predicate_literals[0]}{DEFAULT_ANS_SEP}{DEFAULT_NAN_ANS}"
    elif isinstance(
        ingredient_node_in_context.parent, (exp.Order, exp.Ordered, exp.AggFunc)
    ):
        # Add our int/float pattern
        added_kwargs[
            "pattern"
        ] = f"(([0-9]|\.)+\{DEFAULT_ANS_SEP}|\{DEFAULT_NAN_ANS}\{DEFAULT_ANS_SEP})+"
        added_kwargs["output_type"] = "numeric"
    return added_kwargs


def table_star_queries(
    node: exp.Select,
) -> Generator[Tuple[str, exp.Select], None, None]:
    """For each table in the select query, generates a new query
    selecting all columns with the given predicates (Relationships like x = y, x > 1, x >= y).

    Args:
        node: The exp.Select node containing the query to extract table_star queries for

    Returns:
        table_star_queries: Generator with (tablename, exp.Select). The exp.Select is the table_star query

    Example:
        SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
          FROM account_history
          LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
          WHERE constituents.Sector = 'Information Technology'
          AND lower(Action) like "%dividend%"

          Returns:
            >>> ('account_history', 'SELECT * FROM account_history WHERE lower(Action) like "%dividend%')
            >>> ('constituents', 'SELECT * FROM constituents WHERE sector = \'Information Technology\'')
    """
    # Check if this subquery has more than 1 table
    # In this case, we isolate the conditions for each table and abstract
    num_tables = len(list(node.find_all(exp.Table)))
    alias_identifiers: Set[str] = get_alias_identifiers(node)
    if num_tables > 1:
        for tablenode in node.find_all(exp.Table):
            # Check to be sure this is in the top-level `SELECT`
            if is_in_subquery(tablenode):
                continue
            table_conditions = node.find(exp.Where).transform(
                extract_multi_table_predicates, tablename=tablenode.name
            )
            table_conditions = prune_alias_conditions(
                table_conditions, table_aliases=alias_identifiers
            )
            table_conditions_str = table_conditions.sql() if table_conditions else ""
            yield (
                tablenode.name,
                parse_one(f"SELECT * FROM {tablenode.name} " + table_conditions_str),
            )
    else:
        table = node.find(exp.Table)
        if table is not None:
            tablename = table.name
            conditions = node.find(CONDITIONS)
            # Remove any conditions that include an alias
            #   as these are undeclared at this point
            conditions = prune_alias_conditions(
                conditions, table_aliases=alias_identifiers
            )
            conditions_str = conditions.sql() if conditions else ""
            yield (
                tablename,
                parse_one(f"SELECT * FROM {tablename} " + conditions_str),
            )


def construct_abstracted_selects(
    node: exp.Select, prev_subquery_has_ingredient: bool = False
) -> Generator[Tuple[str, exp.Select], None, None]:
    """For each table in a given query, generates a `SELECT *` query where all unneeded predicates
    are set to `TRUE`.
    We say `unneeded` in the sense that to minimize the data that gets passed to an ingredient,
    we don't need to factor in this operation at the moment.

    Example:
        {{LLM('is this an italian restaurant?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1
        AND child_category = 'Restaurants & Dining'
        Returns:
            >>> ('transactions', 'SELECT * FROM transactions WHERE TRUE AND child_category = \'Restaurants & Dining\'')
    Args:
        node: exp.Select node from which to construct abstracted versions of queries for each table.
    Returns:
        abstracted_queries: Generator with (tablename, exp.Select). The exp.Select is the abstracted query.
    """
    # TODO: don't really know how to optimize with 'CASE' queries right now
    if node.find(exp.Case):
        return
    for tablename, table_star_query in table_star_queries(node):
        # If our previous subquery has an ingredient, we can't optimize with subquery condition
        # So, remove this subquery constraint and run
        if prev_subquery_has_ingredient:
            table_star_query = table_star_query.transform(maybe_set_subqueries_to_true)
        # Substitute all ingredients with 'TRUE'
        abstracted_query = table_star_query.transform(set_structs_to_true).transform(
            prune_true_operators
        )
        abstracted_query_str = recover_blendsql(abstracted_query.sql())
        # We also need to change whatever we're selecting to '*', since we don't know
        #   at this time what the function needs
        # 'or true' will break things, e.g. "sector = 'Energy' OR True" will return all
        # TODO: should the below be done with sqlglot.transform also?
        abstracted_query_str = re.sub(
            r"or true",
            "and true",
            re.sub(
                r"(?<=^select).*?(?=from)",
                " * ",
                abstracted_query_str,
                flags=re.IGNORECASE,
            ),
            flags=re.IGNORECASE,
        )
        yield (tablename, abstracted_query_str)
