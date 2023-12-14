import logging
from typing import Dict, Union
from colorama import Fore
from tqdm import tqdm

from blendsql.ingredients.builtin.llm.utils import initialize_endpoint
from blendsql import programs
from blendsql import constants as CONST
from blendsql.ingredients.ingredient import SelectIngredient


class LLMSelect(SelectIngredient):
    def run(
        self,
        arg: str,
        value_limit: Union[int, None] = None,
        endpoint_name: str = "text-davinci-003",
        table_to_title: Dict[str, str] = None,
        **kwargs,
    ) -> Union[str, int, float]:
        """
        SELECT merchant FROM transactions
            WHERE {{
                LLMSelect('most likely to sell burgers?', 'transactions::merchant')
            }}
        """
        # Unpack default kwargs
        values, original_table, tablename, colname = self.unpack_default_kwargs(
            **kwargs
        )
        if kwargs.get("endpoint", None) is not None:
            endpoint = kwargs["endpoint"]
            endpoint_name = endpoint.endpoint_name
        else:
            endpoint = initialize_endpoint(endpoint_name)

        if value_limit is not None:
            values = values[:value_limit]

        # Pass in batches, until we have a single final answer
        next_values = []
        curr_values = values
        while len(curr_values) > 1:
            curr_values_dict = [
                {"value": value, "idx": idx} for idx, value in enumerate(curr_values)
            ]
            # Only use tqdm if we're in debug mode
            context_manager = (
                tqdm(
                    range(0, len(curr_values_dict), CONST.VALUE_BATCH_SIZE),
                    total=len(curr_values_dict) // CONST.VALUE_BATCH_SIZE,
                    desc=f"Making calls to LLM with batch_size {CONST.VALUE_BATCH_SIZE}",
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
                )
                if logging.DEBUG >= logging.root.level
                else range(0, len(curr_values_dict), CONST.VALUE_BATCH_SIZE)
            )
            for i in context_manager:
                # Allow the LLM to select 'N.A.' also
                curr_batch_options = curr_values[i : i + CONST.VALUE_BATCH_SIZE] + [
                    "N.A."
                ]
                program: str = (
                    programs.SELECT_PROGRAM_CHAT(curr_batch_options)
                    if endpoint_name in CONST.OPENAI_CHAT_LLM
                    else programs.SELECT_PROGRAM_COMPLETION(curr_batch_options)
                )
                res = endpoint.predict(
                    program=program,
                    question=arg,
                    values_dict=curr_values_dict[i : i + CONST.VALUE_BATCH_SIZE],
                    colname=colname,
                )
                if res["result"] != "N.A.":
                    next_values.append(res["result"])
            # Update our trackers
            curr_values = next_values
            next_values = []
        if len(curr_values) != 1:
            raise ValueError(f"{curr_values}")
        return curr_values[0]
