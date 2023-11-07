# -*- coding: utf-8 -*-
"""
ELM energy wizard
"""
import copy
import numpy as np

from elm.base import ApiBase


class EnergyWizard(ApiBase):
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

        if not isinstance(corpus.index.values[0], int):
            corpus['index'] = np.arange(len(corpus))
            corpus = corpus.set_index('index', drop=False)

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

    def engineer_query(self, query, token_budget=None, new_info_threshold=0.7,
                       convo=False):
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
        convo : bool
            Flag to perform semantic search with full conversation history
            (True) or just the single query (False). Call EnergyWizard.clear()
            to reset the chat history.
        Returns
        -------
        message : str
            Engineered question to GPT including information from corpus and
            the original query
        references : list
            The list of references (strs) used in the engineered prompt is
            returned here
        """

        self.messages.append({"role": "user", "content": query})

        if convo:
            # [1:] to not include the system role in the semantic search
            query = [f"{msg['role'].upper()}: {msg['content']}"
                     for msg in self.messages[1:]]
            query = '\n\n'.join(query)

        token_budget = token_budget or self.token_budget

        strings, _, idx = self.rank_strings(query)

        message = copy.deepcopy(self.MODEL_INSTRUCTION)
        question = f"\n\nQuestion: {query}"
        used_index = []

        for string, i in zip(strings, idx):
            next_str = (f'\n\n"""\n{string}\n"""')
            token_usage = self.count_tokens(message + next_str + question,
                                            self.model)

            new_words = set(next_str.split(' '))
            additional_info = new_words - set(message.split(' '))
            new_info_frac = len(additional_info) / len(new_words)

            if new_info_frac > new_info_threshold:
                if token_usage > token_budget:
                    break
                else:
                    message += next_str
                    used_index.append(i)

        message = message + question
        used_index = np.array(used_index)
        references = self.make_ref_list(used_index)

        return message, references

    def make_ref_list(self, idx):
        """Make a reference list

        Parameters
        ----------
        used_index : np.ndarray
            Indices of the used text from the text corpus

        Returns
        -------
        ref_list : list
            A list of references (strs) used.
        """
        ref_list = ''
        if self.ref_col is not None and self.ref_col in self.corpus:
            ref_list = list(self.corpus[self.ref_col].iloc[idx].unique())

        return ref_list

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

        out = self.engineer_query(query, token_budget=token_budget,
                                  new_info_threshold=new_info_threshold,
                                  convo=convo)
        query, references = out

        messages = [{"role": "system", "content": self.MODEL_ROLE},
                    {"role": "user", "content": query}]
        response_message = ''
        kwargs = dict(model=self.model,
                      messages=messages,
                      temperature=temperature,
                      stream=stream)

        response = self._client.chat.completions.create(**kwargs)

        if return_chat_obj:
            return response, query, references

        if stream:
            for chunk in response:
                chunk_msg = chunk.choices[0].delta.content or ""
                response_message += chunk_msg
                print(chunk_msg, end='')

        else:
            response_message = response["choices"][0]["message"]["content"]

        self.messages.append({'role': 'assistant',
                              'content': response_message})

        if any(references) and print_references:
            ref_msg = ('\n\nThe model was provided with the '
                       'following documents to support its answer:')
            ref_msg += '\n - ' + '\n - '.join(references)
            response_message += ref_msg
            if stream:
                print(ref_msg)

        if debug:
            return response_message, query, references
        else:
            return response_message
