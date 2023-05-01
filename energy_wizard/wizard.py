# -*- coding: utf-8 -*-
"""
Energy Wizard
"""
import openai
import tiktoken
import textwrap

from energy_wizard.abs import ApiBase
from energy_wizard.dist import DistanceMetrics


class EnergyWizard(ApiBase):

    DEFAULT_MODEL = 'gpt-3.5-turbo'
    """Default model to answer energy questions."""

    def __init__(self, corpus, dist_fun=DistanceMetrics.cosine_dist,
                 model=None, token_budget=4096):
        """
        Parameters
        ----------
        corpus : pd.DataFrame
            Corpus of text in dataframe format. Must have columns "text" and
            "embedding".
        dist_fun : None | function
            Function to evaluate the distance between two 1D arrays.
        model : str
            GPT model name, default is the DEFAULT_MODEL global var
        token_budget : int
            Number of tokens that can be embedded in the prompt
        """

        super().__init__(model)
        self.corpus = self.parse_corpus(corpus)
        self.dist_fun = dist_fun
        self.token_budget = token_budget

    @staticmethod
    def parse_corpus(corpus):
        assert 'text' in corpus
        if 'embedding' not in corpus:
            msg = ('Text corpus must have "embedding" column but received '
                   'corpus with columns: {}'.format(list(corpus.columns)))
            raise KeyError(msg)
        return corpus

    def rank_strings(self, query, top_n=100):
        """Returns a list of strings and relatednesses, sorted from most
        related to least.

        Parameters
        ----------
        query : str
            Question being asked of GPT
        top_n : int
            Number of top results to return.

        Returns
        -------
        strings : list
            List of related strings
        score : list
            List of float scores of strings
        """
        embedding = self.get_embedding(query)
        strings_and_scores = [
            (row["text"], self.dist_fun(embedding, row["embedding"]))
            for i, row in self.corpus.iterrows()
        ]
        strings_and_scores.sort(key=lambda x: x[1], reverse=True)
        strings, score = zip(*strings_and_scores)
        return strings[:top_n], score[:top_n]

    def engineer_query(self, query):
        """Engineer a query for GPT using the corpus of information

        Parameters
        ----------
        query : str
            Question being asked of GPT

        Returns
        -------
        message : str
            Engineered question to GPT including information from corpus and
            the original query
        """

        strings, _ = self.rank_strings(query)

        message = ('Use the below articles to answer the subsequent question. '
                   'If the answer cannot be found in the articles, '
                   'write "I could not find an answer."')
        question = f"\n\nQuestion: {query}"

        for string in strings:
            next_article = (f'\n\nWikipedia article section:\n'
                            f'"""\n{string}\n"""')
            token_usage = self.num_tokens(message + next_article + question)
            if token_usage > self.token_budget:
                break
            else:
                message += next_article

        return message + question

    @staticmethod
    def pprint(text, width=90):
        """Print a string with column width limits"""
        for x in textwrap.wrap(text, width=width):
            print(x)

    def ask(self, query, debug=False, print=False):
        """Answers a query using GPT and a dataframe of relevant texts and
        embeddings.

        Parameters
        ----------
        query : str
            Question being asked of GPT
        debug : bool
            Flag to return extra diagnostics on the engineered question.

        Returns
        -------
        response : str
            GPT output / answer.
        """

        query = self.engineer_query(query)

        role_str = "You parse through articles to answer questions."
        messages = [{"role": "system", "content": role_str},
                    {"role": "user", "content": query}]

        response = openai.ChatCompletion.create(model=self.model,
                                                messages=messages,
                                                temperature=0)

        response_message = response["choices"][0]["message"]["content"]

        if print:
            self.pprint(response_message)

        if debug:
            return response_message, query
        else:
            return response_message
