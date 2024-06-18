# -*- coding: utf-8 -*-
"""
ELM energy wizard
"""
import os
import copy
import numpy as np
import json
import psycopg2
from datetime import date, datetime
import pandas as pd

from elm.base import ApiBase
from elm.wizard import EnergyWizard

class DataBaseWizard(ApiBase):
    """Interface to ask OpenAI LLMs about energy research."""

    MODEL_ROLE = ("You are a data engineer pulling data from a relational "
                  "database using SQL and writing python code to plot the "
                  "output based on user queries.")
    """High level model role, somewhat redundant to MODEL_INSTRUCTION"""

    def __init__(self, connection_string, model=None, token_budget=3500, ref_col=None):
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
        self.connection_string = connection_string
        self.connection = psycopg2.connect(self.connection_string)
        self.token_budget = token_budget

        fpcache = './database_manual.txt'

        if os.path.exists(fpcache):
            with open(fpcache, 'r') as f:
                self.database_describe = f.read()

        else:
            print('Error no expert database file')

    # Getting sql from a generic query
    def get_sql_for(self, query):
        """Take the raw user query and ask the LLM for a SQL query that will
        get data to support a response
        """
        e_query = ('{}\n\nPlease create a SQL query that will answer this '
                   'user question: "{}"\n\n'
                   'Return all columns from the database. '
                   'All the tables are in the schema "loads"'
                   'Please only return the SQL query with no commentary or preface.'
                   .format(self.database_describe, query))
        out = super().chat(e_query, temperature=0)
        print(out)
        return out

    def run_sql(self, sql):
        """Takes a SQL query that can support a user prompt, runs SQL query
        based on the db connection (self.connection), returns dataframe
        response."""
        query = sql
        print(query)
        # Move Connection or cursor to init and test so that you aren't re-intializing
        # it with each instance.
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=column_names)
        return df

    def get_py_code(self, query, df):
        """"""
        e_query = ('Great it worked! I have made a dataframe from the output '
                   'of the SQL query you gave me. '
                   'Here is the dataframe head: \n{}\n'
                   'Here is the dataframe tail: \n{}\n'
                   'Here is the dataframe description: \n{}\n'
                   'Here is the dataframe datatypes: \n{}\n'
                   'Now please write python code using matplotlib to plot '
                   'the data in the dataframe based on the original user query: "{}"'
                   .format(df.head(), df.tail(), df.describe(), df.dtypes, query))
        out = super().chat(e_query, temperature=0)

        ## get response from output
        # Need to fix full response
        full_response = out
        #print(full_response)
        ## get python code from response
        full_response = full_response[full_response.find('```python')+9:]
        full_response = full_response[:full_response.find('```')]
        py = full_response
        return py

    def run_py_code(self, py, df):
        try:
            exec(py)
            return plt
        except:
            print(py)

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

        self.query = query
        self.sql = self.get_sql_for(query)
        self.df = self.run_sql(self.sql)
        self.py = self.get_py_code(query = query, df = self.df)
        self.plt = self.run_py_code(self.py, self.df)
        return self.plt


