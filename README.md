<div align="center">
  <a href="https://lmql.ai">
    <img src="./img/logo_as_circle.png" alt="Logo" width="250" height="250">
  </a>

  <h3 align="center">BlendSQL</h3>

  <p align="center">
    <i> SQL 🤝 LLMs </i>
  </p>
</div>


## Intro
BlendSQL is a *superset of SQLite*. It allows the user to leverage external functions (LLMs, Python code, APIs) within the expressive structure of SQL.

It can be viewed as an inversion of the typical text-to-SQL paradigm, where a user calls a LLM, and the LLM calls a SQL program.
Here, the user is given the control to oversee all calls (LLM + SQL) within a unified query language.

![comparison](./img/comparison.jpg)

For example, imagine we have the following table.

| **description**                   | **amount** | **merchant**             | **parent_category** | **child_category**   | **date**   |
|-----------------------------------|------------|--------------------------|---------------------|----------------------|------------|
| PARKING METERS (POM) MONTEREY CA  | -0.75      | Keep The Change          | Fees & Charges      | Service Fee          | 2022-07-19 |
| CHECKCARD WELCOME TO LAS VEGAS NV | -31.6      | Welcome To Las Vegas     | Gifts               | Gifts                | 2022-10-25 |
| UWM PLAZA GIFT ESP                | -6.62      | Uwmc Gift Shop           | Gifts               | Gifts                | 2022-05-10 |
| MINERS DELIGHT ROCK ELKO NV       | -42.91     | Miners Delight Rock Shop | Gifts               | Gifts                | 2022-06-20 |
| OLIVE GARDEN TRACY, CA, US | -50.41     | Olive Garden             | Food                | Restaurants & Dining | 2022-04-25 |

BlendSQL allows us to ask the following questions by injecting "ingredients", which are callable functions denoted by double curly brackets (`{{`, `}}`).
The below examples work out of the box, but you are able to design your own ingredients as well! 

*Show me some gifts appropriate for a child.*
```sql
SELECT DISTINCT description, merchant FROM transactions WHERE
      {{LLMMap('would this gift be appropriate for a child?', 'transactions::description')}} = TRUE
      AND child_category = 'Gifts'
```

*Ok, show me some gifts not appropriate for a child I bought in q2 this year*
```sql
SELECT DISTINCT description, merchant FROM transactions WHERE
      {{LLMMap('would this gift be appropriate for a child?', 'transactions::description')}} = TRUE
      AND {{DT('transactions::date', start='q2')}}
      AND child_category = 'Gifts'
```

*Forget gifts, I'm hungry. What Italian restaurants have I been to in California?*
```sql
SELECT DISTINCT description, merchant FROM transactions WHERE
   {{LLMMap('is this an italian restaurant?', 'transactions::merchant')}} = TRUE
   AND {{
       LLMMap(
           'what state is this transaction from? Choose -1 when N.A.',
           'transactions::description',
           example_outputs='TX;CA;MA;-1;',
       )
   }} = 'CA'
   AND child_category = 'Restaurants & Dining'
```

*Show me a place where I can buy a burger.*
```sql
SELECT merchant FROM transactions
  WHERE {{
      LLMSelect('most likely to sell burgers?', 'transactions::merchant')
  }} 
```

*Summarize my spending from Coffee shops.*
```sql
{{
    LLMQA(
        'Summarize my coffee spending in 10 words.', 
        'SELECT * FROM transactions WHERE child_category = "Coffee"'
    )
}} 
```

### Features 
- Smart parsing optimizes what is passed to external functions 🧠
  - Traverses AST to minimize external function calls
- Accelerated LLM calls and caching 🚀
  - Enabled with [gptcache](https://github.com/zilliztech/GPTCache) via [guidance](https://github.com/guidance-ai/guidance)
- Easy logging of execution environment with `smoothie.save_recipe()` 🖥️
  - Enables reproducibility across machines

### Benchmarks
The below benchmarks were done on my local M1 Macbook Pro. by running the scripts found in `examples/benchmarks`. 
'Lines of Code' is a rough estimate of the user-written code for each usecase.

| **Name**                     | **Description**                                                 | **Runtime/s (Across 10 runs)** | **Lines of Code** |
|------------------------------|-----------------------------------------------------------------|-------------------------------------------------|-------------------|
| BlendSQL                     |                                                     |5.685 +/- 0.930                                 | 9                 |
| SQL + LLM Calls       | Filtering what we can with SQL, then running LLM calls.         | 9.083 +/- 2.061                                 | 106               |
| Naive SQL + LLM Calls | Runing LLM calls on entire table, regardless of SQL conditions. | 64.809 +/- 6.225                                | 106               |

<hr>

For a technical walkthrough of how a BlendSQL query is executed, check out [technical_walkthrough.md](./technical_walkthrough.md).
## Setup
### Prep Env
To set up a `blendsql` conda environment, run the following command.

```
conda env create && conda activate blendsql && pre-commit install
```


### Run Examples
`python -m examples.example`

### Run Line Profiling 
First uncomment `@profile` above `blend()` in `grammar.py`.
Make sure you've run `pip install line_profiler` first. This installs the tool here: https://github.com/pyutils/line_profiler

`PYTHONPATH=$PWD:$PYTHONPATH kernprof -lv examples/benchmarks/with_blendsql.py`

## Example Usage

```python
from blendsql import blend, SQLiteDBConnector, init_secrets
# Import our pre-built ingredients
from blendsql.ingredients import LLMMap, LLMSelect, DT

# Initialize our Azure OpenAI secrets, so we can use LLM() calls
init_secrets("secrets.json")
db_path = ""
db = SQLiteDBConnector(db_path=db_path)
blendsql = """
SELECT merchant FROM transactions WHERE 
     {{LLMMap('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = TRUE
     AND parent_category = 'Food'
"""
# Make our smoothie - the executed BlendSQL script
smoothie = blend(
    query=blendsql,
    db=db,
    ingredients={LLMMap, DT},
    verbose=True
)

```

# Documentation

> [!WARNING]
> WIP, will be updated

## Ingredients 
Ingredients are at the core of a BlendSQL script. 

They are callable functions that perform one the task paradigms defined in [ingredient.py](./blendsql/ingredients/ingredient.py).

Ingredient calls are denoted by wrapping them in double curly brackets, `{{ingredient}}`.

The following ingredient types are valid.

### `MapIngredient`
This type of ingredient applies a function on a given table/column pair to create a new column containing the function output.

For example, take the following query.

```sql 
SELECT merchant FROM transactions
    WHERE {{LLMMap('Is this a pizza shop?', 'transactions::merchant')}} = TRUE
```

`LLMMap` is one of our builtin MapIngredients. For each of the distinct values in the "merchant" column of the "transactions" table, it will create a column containing the function output.

| merchant | Is this a pizza shop? |
|----------|-----------------------|
| Domino's | 1                     |
| Safeway  | 0                     |
| Target   | 0                     |

The temporary table shown above is then combined with the original "transactions" table with a `LEFT JOIN` on the "merchant" column.


### `SelectIngredient`
This ingredient type functions similarly to the `MapIngredient`, with the distinction that it selects exclusively **one** value.

For example, take the following query.
```sql 
SELECT Name, Symbol from constituents
    WHERE {{LLMSelect('Which company was lead by Steve Jobs?', 'constituents::Name')}}
```

In this case, we call a LLM to select a single value from the `Name` column that best answers the given question.

For example, if we have `["Amazon", "Apple", "Google"]` as distinct values in our `Name` column, this BlendSQL query would effectively get interpreted as:

```sql 
SELECT Name, Symbol from constituents
    WHERE Name = 'Apple'
```

### `QAIngredient`
Sometimes, simply selecting data from a given database is not enough to sufficiently answer a user's question.

The `QAIngredient` is designed to return data of variable types, and is best used in cases when we either need:
1) Unstructured, free-text responses ("Give me a summary of all my spending in coffe")
2) Complex, unintuitive relationships extracted from table subsets ("How many consecutive days did I spend in coffee?")

The following query demonstrates usage of the builtin `LLMQA` ingredient.

```sql
{{
    LLMQA(
        'How many consecutive days did I buy stocks in Financials?', 
        'SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
            FROM account_history
            LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
            WHERE Sector = "Financials"
            ORDER BY "Run Date" LIMIT 5'
    )
}} 
```
This is slightly more complicated than the rest of the ingredients. 

Behind the scenes, we wrap the call to `LLMQA` in a trivial `CASE` clause, ensuring that the ingredient's output gets returned.
```sql 
SELECT CASE WHEN FALSE THEN FALSE 
  WHEN TRUE then {{QAIngredient}}
  END
```
The LLM gets both the question asked, alongside the subset of the SQL database fetched by our subquery.

| **"Run Date"** | **Symbol** | **Sector** |
|----------------|------------|------------|
| 2022-01-14     | HBAN       | Financials |
| 2022-01-20     | AIG        | Financials |
| 2022-01-24     | AIG        | Financials |
| 2022-01-24     | NTRS       | Financials |
| 2022-01-25     | HBAN       | Financials |


From examining this table, we see that we bought stocks in the Financials sector 2 consecutive days (2022-01-24, and 2022-01-25).
The LLM answers the question in an end-to-end manner, returning the result `2`.

The `QAIngredient` can be used as a standalone end-to-end QA tool, or as a component within a larger BlendSQL query.

For example, the BlendSQL query below translates to the valid (but rather confusing) question: 

"Show me stocks in my portfolio, whose price is greater than the number of consecutive days I bought Financial stocks multiplied by 10. Only display those companies which offer a media streaming service."
```sql
 SELECT Symbol, "Last Price" FROM portfolio WHERE "Last Price" > {{
  LLMQA(
        'How many consecutive days did I buy stocks in Financials?', 
        'SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
            FROM account_history
            LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
            WHERE Sector = "Financials"
            ORDER BY "Run Date" LIMIT 5'
    )
  }} * 10
  AND {{LLMMap('Offers a media streaming service?', 'portfolio::Description')}} = 1
```
#### Changing QA Output with `options`
Perhaps we want the answer to the above question in a different format. We can combine the best of the `Select` ingredient with the `QA` ingredient by passing a `options` argument, where we provide semicolon-separated options.

```sql
{{
    LLMQA(
        'How many consecutive days did I buy stocks in Financials?', 
        'SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
            FROM account_history
            LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
            WHERE Sector = "Financials"
            ORDER BY "Run Date" LIMIT 5',
        options='one consecutive day!;two consecutive days!;three consecutive days!'
    )
}}
```

Running the above BlendSQL query, we get the output `two consecutive days!`.

This `options` argument can also be a reference to a given column.
These options will be restricted to only the values exposed via the subquery (2nd arg in `LLMQA`).

```sql
{{
  LLMQA(
      'Which did i buy the most?',
      'SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
          FROM account_history
          LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
          WHERE Sector = "Financials"
          ORDER BY "Run Date" LIMIT 5',
      options='account_history::Symbol'
  )
}}
```

The above BlendSQL will yield the result `AIG`.

### `StringIngredient`
This is the simplest type of ingredient. This will output a string to be placed directly into the SQL query.

We have the `DT` function as a builtin StringIngredient.

```sql 
SELECT merchant FROM transactions
    WHERE {{DT('transactions::date', start='q2')}}
```

This will call a Python function that uses `datetime` to interpret the absolute dates which the relative phrase "q2" most likely refers to.

We do not create any new tables or perform any joins with a StringIngredient; instead, we simply get the following SQL query.

> [!NOTE]
> The below SQL interpretation of the `DT` function assumes we're calling it in December, 2022. The phrase 'q2' will be interpreted differently in, say, March 1998.

```sql 
SELECT merchant FROM transactions
    WHERE date > '2022-09-30' AND date < '2022-12-01'
```

### Smoothie 
The [smoothie.py](./blendsql/smoothie.py) object defines the output of an executed BlendSQL script.

```python
@dataclass
class SmoothieMeta:
    process_time_seconds: float
    num_values_passed: int  # Number of values passed to a Map ingredient
    example_map_outputs: List[Any] # Outputs from a Map ingredient, for debugging
    ingredients: List[Ingredient]
    query: str
    db_path: str
    contains_ingredient: bool = True


@dataclass
class Smoothie:
    df: pd.DataFrame
    meta: SmoothieMeta

def blend(*args, **kwargs) -> Smoothie:
  ... 
```