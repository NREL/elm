# -*- coding: utf-8 -*-
"""
ELM energy wizard
"""
import copy
import numpy as np

from elm.base import ApiBase
from elm.wizard import EnergyWizard


class DataBaseWizard(ApiBase):
    """Interface to ask OpenAI LLMs about energy research."""

    MODEL_ROLE = "You parse through articles to answer questions."
    """High level model role, somewhat redundant to MODEL_INSTRUCTION"""

    MODEL_INSTRUCTION = ('Use the information below to answer the subsequent '
                         'question. If the answer cannot be found in the '
                         'text, write "I could not find an answer."')
    """Prefix to the engineered prompt"""

    def __init__(self, corpus, model=None, token_budget=3500, ref_col=None):
        """
        Parameters
        ----------
        corpus : pd.DataFrame
            Corpus of text in dataframe format. Must have columns "text" and
            "embedding".
        model : str
            GPT model name, default is the DEFAULT_MODEL global var
        token_budget : int
            Number of tokens that can be embedded in the prompt. Note that the
            default budget for GPT-3.5-Turbo is 4096, but you want to subtract
            some tokens to account for the response budget.
        ref_col : None | str
            Optional column label in the corpus that provides a reference text
            string for each chunk of text.
        """

        super().__init__(model)

        self.corpus = self.preflight_corpus(corpus)
        self.token_budget = token_budget
        self.embedding_arr = np.vstack(self.corpus['embedding'].values)
        self.text_arr = self.corpus['text'].values
        self.ref_col = ref_col

    def get_sql_for(self, query):
        """Take the raw user query and ask the LLM for a SQL query that will
        get data to support a response
        """
        sql_role = ('You are a data engineer creating SQL queries that will '
                    'pull data for user requests. Return only the SQL query '
                    'as a single string that can be run directly without any '
                    'comments.')
        e_query = ('Here is a description of the SQL table: {}\n\n'
                   'Please create a SQL query that will pull data that can '
                   'answer this user question: {}'
                   .format(self.table_summary, query))
        out = self.generic_query(e_query, model_role=sql_role,
                                 temperature=0)
        return out

    def run_sql(self, sql):
        """Takes a SQL query that can support a user prompt, runs SQL query
        based on the db connection (self.connection), returns dataframe
        response."""

    def get_py_code(self, query, df):
        """"""
        sql_role = ('you make python code to support a user question based on available data')
        e_query = ('here is available data from a pandas dataframe: {}, write python code to answer this user question based on the data: {}')
        out = self.generic_query(e_query, model_role=sql_role,
                                 temperature=0)
        return out

    def run_py_code():
        """Jordan to write code that takes LLM response and generates plots"""

    def chat(self, query,
             debug=True,
             stream=True,
             temperature=0,
             convo=False,
             token_budget=None,
             new_info_threshold=0.7,
             print_references=False,
             return_chat_obj=False):
        """Answers a query by doing a semantic search of relevant text with
        embeddings and then sending engineered query to the LLM.

        Parameters
        ----------
        query : str
            Question being asked of EnergyWizard
        debug : bool
            Flag to return extra diagnostics on the engineered question.
        stream : bool
            Flag to print subsequent chunks of the response in a streaming
            fashion
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.
        convo : bool
            Flag to perform semantic search with full conversation history
            (True) or just the single query (False). Call EnergyWizard.clear()
            to reset the chat history.
        token_budget : int
            Option to override the class init token budget.
        new_info_threshold : float
            New text added to the engineered query must contain at least this
            much new information. This helps prevent (for example) the table of
            contents being added multiple times.
        print_references : bool
            Flag to print references if EnergyWizard is initialized with a
            valid ref_col.
        return_chat_obj : bool
            Flag to only return the ChatCompletion from OpenAI API.

        Returns
        -------
        response : str
            GPT output / answer.
        query : str
            If debug is True, the engineered query asked of GPT will also be
            returned here
        references : list
            If debug is True, the list of references (strs) used in the
            engineered prompt is returned here
        """

        self.sql = self.get_sql_for(query)  # this is 1 LLM query
        self.df = self.run_sql(self.sql)
        self.py = self.get_py_code(query, self.df)  # this is 1 LLM query
        self.run_py_code(self.py, self.df)
