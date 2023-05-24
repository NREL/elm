# -*- coding: utf-8 -*-
"""
Energy Wizard
"""
import copy
import numpy as np
import openai

from energy_wizard.abs import ApiBase


class EnergyWizard(ApiBase):
    """Interface to ask OpenAI LLMs about energy research."""

    MODEL_INSTRUCTION = ('Use the information below to answer the subsequent '
                         'question. If the answer cannot be found in the '
                         'text, write "I could not find an answer."')
    """Prefix to the engineered prompt"""

    def __init__(self, corpus, model=None, token_budget=3500):
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
        """

        super().__init__(model)
        self.corpus = self.preflight_corpus(corpus)
        self.token_budget = token_budget
        self.embedding_arr = np.vstack(self.corpus['embedding'].values)
        self.text_arr = self.corpus['text'].values

    @staticmethod
    def preflight_corpus(corpus, required=('text', 'embedding')):
        """Run preflight checks on the text corpus.

        Parameters
        ----------
        corpus : pd.DataFrame
            Corpus of text in dataframe format. Must have columns "text" and
            "embedding".
        required : list | tuple
            Column names required to be in the corpus df

        Returns
        -------
        corpus : pd.DataFrame
            Corpus of text in dataframe format. Must have columns "text" and
            "embedding".
        """
        missing = [col for col in required if col not in corpus]
        if any(missing):
            msg = ('Text corpus must have {} columns but received '
                   'corpus with columns: {}'
                   .format(missing, list(corpus.columns)))
            raise KeyError(msg)
        return corpus

    def cosine_dist(self, query_embedding):
        """Compute the cosine distance of the query embedding array vs. all of
        the embedding arrays of the full text corpus

        Parameters
        ----------
        query_embedding : np.ndarray
            1D array of the numerical embedding of the request query.

        Returns
        -------
        out : np.ndarray
            1D array with length equal to the number of entries in the text
            corpus. Each value is a distance score where smaller is closer
        """

        dot = np.dot(self.embedding_arr, query_embedding)
        norm1 = np.linalg.norm(query_embedding)
        norm2 = np.linalg.norm(self.embedding_arr, axis=1)

        out = 1 - (dot / (norm1 * norm2))

        return out

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
        strings : np.ndarray
            1D array of related strings
        score : np.ndarray
            1D array of float scores of strings
        idx : np.ndarray
            1D array of indices in the text corpus corresponding to the
            ranked strings/scores outputs.
        """

        embedding = self.get_embedding(query)
        scores = 1 - self.cosine_dist(embedding)
        best = np.argsort(scores)[::-1][:top_n]

        strings = self.text_arr[best]
        scores = scores[best]

        return strings, scores, best

    def engineer_query(self, query, token_budget=None, new_info_threshold=0.7):
        """Engineer a query for GPT using the corpus of information

        Parameters
        ----------
        query : str
            Question being asked of GPT
        token_budget : int
            Option to override the class init token budget.
        new_info_threshold : float
            New text added to the engineered query must contain at least this
            much new information. This helps prevent (for example) the table of
            contents being added multiple times.

        Returns
        -------
        message : str
            Engineered question to GPT including information from corpus and
            the original query
        """

        token_budget = token_budget or self.token_budget

        strings = self.rank_strings(query)[0]

        message = copy.deepcopy(self.MODEL_INSTRUCTION)
        question = f"\n\nQuestion: {query}"

        for string in strings:
            next_str = (f'\n\n"""\n{string}\n"""')
            token_usage = self.count_tokens(message + next_str + question)

            new_words = set(next_str.split(' '))
            additional_info = new_words - set(message.split(' '))
            new_info_frac = len(additional_info) / len(new_words)

            if new_info_frac > new_info_threshold:
                if token_usage > token_budget:
                    break
                else:
                    message += next_str

        return message + question

    def ask(self, query, debug=True, stream=True, temperature=0,
            token_budget=None, new_info_threshold=0.7):
        """Answers a query using GPT and a dataframe of relevant texts and
        embeddings.

        Parameters
        ----------
        query : str
            Question being asked of EnergyWizard
        debug : bool
            Flag to return extra diagnostics on the engineered question.
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.
        token_budget : int
            Option to override the class init token budget.
        new_info_threshold : float
            New text added to the engineered query must contain at least this
            much new information. This helps prevent (for example) the table of
            contents being added multiple times.

        Returns
        -------
        response : str
            GPT output / answer.
        query : str
            If debug is True, the engineered query asked of GPT will also be
            returned here
        """

        query = self.engineer_query(query, token_budget=token_budget,
                                    new_info_threshold=new_info_threshold)

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
