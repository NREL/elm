# -*- coding: utf-8 -*-
"""
Energy Wizard
"""
import numpy as np
import openai

from energy_wizard.abs import ApiBase
from energy_wizard.dist import DistanceMetrics


class EnergyWizard(ApiBase):

    def __init__(self, corpus, dist_fun=DistanceMetrics.cosine_dist,
                 model=None, token_budget=4096-500):
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
            Number of tokens that can be embedded in the prompt. Note that the
            default budget for GPT-3.5-Turbo is 4096, but you want to subtract
            some tokens to account for the response budget.
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

        scores = np.zeros(len(self.corpus))
        for i, row in self.corpus.iterrows():
            scores[i] = self.dist_fun(embedding, row["embedding"])

        best = np.argsort(scores)[::-1][:top_n]
        strings = self.corpus.loc[best, 'text'].values.tolist()
        scores = scores[best]

        references = ['Unknown reference'] * top_n
        if 'reference' in self.corpus:
            references = self.corpus.loc[best, 'reference'].values.tolist()

        return strings, scores, references

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

        strings, _, references = self.rank_strings(query)

        message = ('Use the information below to answer the subsequent '
                   'question. Note that there may be additional data '
                   'on this research in the references provided. '
                   'If the answer cannot be found in the text, '
                   'write "I could not find an answer."')
        question = f"\n\nQuestion: {query}"

        for string, ref in zip(strings, references):
            next_str = (f'\n\n{ref}:\n"""\n{string}\n"""')
            token_usage = self.num_tokens(message + next_str + question)
            if token_usage > self.token_budget:
                break
            else:
                message += next_str

        return message + question

    def ask(self, query, debug=True, stream=True, temperature=0):
        """Answers a query using GPT and a dataframe of relevant texts and
        embeddings.

        Parameters
        ----------
        query : str
            Question being asked of EnergyWizard
        debug : bool
            Flag to return extra diagnostics on the engineered question.
        temperature : float
            GPT model temperature: 0 is more reliable and nearly deterministic,
            1 is more fluid and higher entropy

        Returns
        -------
        response : str
            GPT output / answer.
        query : str
            If debug is True, the engineered query asked of GPT will also be
            returned here
        """

        query = self.engineer_query(query)

        role_str = "You parse through articles to answer questions."
        messages = [{"role": "system", "content": role_str},
                    {"role": "user", "content": query}]

        if stream:
            response_message = ''
            response = openai.ChatCompletion.create(model=self.model,
                                                    messages=messages,
                                                    temperature=temperature,
                                                    stream=True)
            for chunk in response:
                chunk_msg = chunk['choices'][0]['delta']
                chunk_msg = chunk_msg.get('content', '')
                response_message += chunk_msg
                print(chunk_msg, end='')

        else:
            response = openai.ChatCompletion.create(model=self.model,
                                                    messages=messages,
                                                    temperature=temperature,
                                                    stream=False)
            response_message = response["choices"][0]["message"]["content"]

        if debug:
            return response_message, query
        else:
            return response_message
