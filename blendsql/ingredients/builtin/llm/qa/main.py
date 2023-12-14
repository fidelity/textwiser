from typing import Dict, Union

from blendsql.ingredients.builtin.llm.utils import (
    construct_gen_clause,
)
from blendsql.ingredients.builtin.llm.endpoint import Endpoint
from blendsql import programs
from blendsql import constants as CONST
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.utils import get_tablename_colname


class LLMQA(QAIngredient):
    def run(
        self,
        arg: str,
        endpoint: Endpoint,
        value_limit: Union[int, None] = None,
        options: str = None,
        table_to_title: Dict[str, str] = None,
        **kwargs,
    ) -> Union[str, int, float]:
        # Unpack default kwargs
        subtable = self.unpack_default_kwargs(**kwargs)
        if value_limit is not None:
            subtable = subtable.iloc[:value_limit]
        if options is not None:
            try:
                tablename, colname = get_tablename_colname(options)
                options = subtable[colname].unique().tolist()
            except ValueError:
                options = options.split(";")

        gen_clause: str = construct_gen_clause(
            gen_type="select" if options else "gen",
            max_tokens=30,
            options=options,
            **endpoint.gen_kwargs,
        )
        program: str = (
            programs.QA_PROGRAM_CHAT(gen_clause)
            if endpoint.endpoint_name in CONST.OPENAI_CHAT_LLM
            else programs.QA_PROGRAM_COMPLETION(gen_clause)
        )
        res = endpoint.predict(
            program=program,
            question=arg,
            serialized_db=subtable.to_string(),
            table_title=None,
        )

        return '"{}"'.format(res["result"].strip())
