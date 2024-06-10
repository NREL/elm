# -*- coding: utf-8 -*-
"""
ELM energy wizard
"""
from abc import ABC, abstractmethod
import copy
import os
import json
import numpy as np

from elm.base import ApiBase
from elm.utilities.try_import import try_import


class EnergyWizardBase(ApiBase, ABC):
    """Base interface to ask OpenAI LLMs about energy research."""

    MODEL_ROLE = "You parse through articles to answer questions."
    """High level model role, somewhat redundant to MODEL_INSTRUCTION"""

    MODEL_INSTRUCTION = ('Use the information below to answer the subsequent '
                         'question. If the answer cannot be found in the '
                         'text, write "I could not find an answer."')
    """Prefix to the engineered prompt"""

    def __init__(self, model=None, token_budget=3500):
        """
        Parameters
        ----------
        model : str
            GPT model name, default is the DEFAULT_MODEL global var
        token_budget : int
            Number of tokens that can be embedded in the prompt. Note that the
            default budget for GPT-3.5-Turbo is 4096, but you want to subtract
            some tokens to account for the response budget.
        """

        super().__init__(model)
        self.token_budget = token_budget

    @abstractmethod
    def query_vector_db(self, query, limit=100):
        """Returns a list of strings and relatednesses, sorted from most
        related to least.

        Parameters
        ----------
        query : str
            Question being asked of GPT
        limit : int
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

        strings, _, idx = self.query_vector_db(query)

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

        return message, references, used_index

    @abstractmethod
    def make_ref_list(self, idx):
        """Make a reference list

        Parameters
        ----------
        used_index : np.ndarray
            Indices of the used text from the text corpus

        Returns
        -------
        ref_list : list
            A list of references (strs) used. Ideally, this is something like:
            ["{ref_title} ({ref_url})"]
        """

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
        query, references, _ = out

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
            response_message = response.choices[0].message.content

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


class EnergyWizard(EnergyWizardBase):
    """Interface to ask OpenAI LLMs about energy research.

    This class is for execution on a local machine with a vector database in
    memory
    """

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

        super().__init__(model, token_budget=token_budget)

        self.corpus = self.preflight_corpus(corpus)
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

    def query_vector_db(self, query, limit=100):
        """Returns a list of strings and relatednesses, sorted from most
        related to least.

        Parameters
        ----------
        query : str
            Question being asked of GPT
        limit : int
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
        best = np.argsort(scores)[::-1][:limit]

        strings = self.text_arr[best]
        scores = scores[best]

        return strings, scores, best

    def make_ref_list(self, idx):
        """Make a reference list

        Parameters
        ----------
        used_index : np.ndarray
            Indices of the used text from the text corpus

        Returns
        -------
        ref_list : list
            A list of references (strs) used. This takes information straight
            from ``ref_col``. Ideally, this is something like:
            ["{ref_title} ({ref_url})"]
        """
        ref_list = ''
        if self.ref_col is not None and self.ref_col in self.corpus:
            ref_list = list(self.corpus[self.ref_col].iloc[idx].unique())

        return ref_list


class EnergyWizardPostgres(EnergyWizardBase):
    """Interface to ask OpenAI LLMs about energy research.

    This class is for execution with a postgres vector database.
    Querying the database requires the use of the psycopg2 and
    boto3 python packages, environment variables ('EWIZ_DB_USER'
    and 'EWIZ_DB_PASSWORD') storing the db user and password, and
    the specification of other connection paremeters such as host,
    port, and name. The database has the following columns: id,
    embedding, chunks, and metadata.

    This class is designed as follows:
    Vector database: PostgreSQL database accessed using psycopg2.
    Query Embedding: AWS titan using boto3
    LLM Application: GPT4 via Azure deployment
    """
    EMBEDDING_MODEL = 'amazon.titan-embed-text-v1'

    def __init__(self, db_host, db_port, db_name,
                 db_schema, db_table, cursor=None, boto_client=None,
                 model=None, token_budget=3500):
        """
        Parameters
        ----------
        db_host : str
            Host url for postgres database.
        db_port : str
            Port for postres database. ex: '5432'
        db_name : str
            Postgres database name.
        db_schema : str
            Schema name for postres database.
        db_table : str
            Table to query in Postgres database. Necessary columns: id,
            chunks, embedding, title, and url.
        cursor : psycopg2.extensions.cursor
            PostgreSQL database cursor used to execute queries.
        boto_client: botocore.client.BedrockRuntime
            AWS boto3 client used to access embedding model.
        model : str
            GPT model name, default is the DEFAULT_MODEL global var
        token_budget : int
            Number of tokens that can be embedded in the prompt. Note that the
            default budget for GPT-3.5-Turbo is 4096, but you want to subtract
            some tokens to account for the response budget.
        """
        boto3 = try_import('boto3')
        psycopg2 = try_import('psycopg2')

        self.db_schema = db_schema
        self.db_table = db_table

        if cursor is None:
            db_user = os.getenv("EWIZ_DB_USER")
            db_password = os.getenv('EWIZ_DB_PASSWORD')
            assert db_user is not None, "Must set EWIZ_DB_USER!"
            assert db_password is not None, "Must set EWIZ_DB_PASSWORD!"
            self.conn = psycopg2.connect(user=db_user,
                                         password=db_password,
                                         host=db_host,
                                         port=db_port,
                                         database=db_name)

            self.cursor = self.conn.cursor()
        else:
            self.cursor = cursor

        if boto_client is None:
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            session_token = os.getenv('AWS_SESSION_TOKEN')
            assert access_key is not None, "Must set AWS_ACCESS_KEY_ID!"
            assert secret_key is not None, ("Must set AWS_SECRET_ACCESS_KEY!")
            assert session_token is not None, "Must set AWS_SESSION_TOKEN!"
            self._aws_client = boto3.client(service_name='bedrock-runtime',
                                            region_name='us-west-2',
                                            aws_access_key_id=access_key,
                                            aws_secret_access_key=secret_key,
                                            aws_session_token=session_token)
        else:
            self._aws_client = boto_client

        super().__init__(model, token_budget=token_budget)

    def get_embedding(self, text):
        """Get the 1D array (list) embedding of a text string
        as generated by AWS Titan.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        embedding : list
            List of float that represents the numerical embedding of the text
        """

        body = json.dumps({"inputText": text, })

        model_id = self.EMBEDDING_MODEL
        accept = 'application/json'
        content_type = 'application/json'

        response = self._aws_client.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')

        return embedding

    def query_vector_db(self, query, limit=100):
        """Returns a list of strings and relatednesses, sorted from most
        related to least.

        Parameters
        ----------
        query : str
            Question being asked of GPT
        limit : int
            Number of top results to return.

        Returns
        -------
        strings : np.ndarray
            1D array of related strings
        score : np.ndarray
            1D array of float scores of strings
        ids : np.ndarray
            1D array of IDs in the text corpus corresponding to the
            ranked strings/scores outputs.
        """

        query_embedding = self.get_embedding(query)

        self.cursor.execute(f"SELECT {self.db_table}.id, "
                            f"{self.db_table}.chunks, "
                            f"{self.db_table}.embedding "
                            "<=> %s::vector as score "
                            f"FROM {self.db_schema}.{self.db_table} "
                            "ORDER BY embedding <=> %s::vector LIMIT %s;",
                            (query_embedding, query_embedding, limit,), )

        result = self.cursor.fetchall()

        strings = [s[1] for s in result]
        scores = [s[2] for s in result]
        best = [s[0] for s in result]

        return strings, scores, best

    def make_ref_list(self, ids):
        """Make a reference list

        Parameters
        ----------
        used_index : np.ndarray
            IDs of the used text from the text corpus

        Returns
        -------
        ref_list : list
            A list of references (strs) used. Ideally, this is something like:
            ["{ref_title} ({ref_url})"]
        """

        placeholders = ', '.join(['%s'] * len(ids))

        sql_query = (f"SELECT {self.db_table}.title, {self.db_table}.url "
                     f"FROM {self.db_schema}.{self.db_table} "
                     f"WHERE {self.db_table}.id IN (" + placeholders + ")")

        self.cursor.execute(sql_query, ids)

        refs = self.cursor.fetchall()

        ref_strs = (f"{{\"parentTitle\": \"{item[0]}\", "
                    f"\"parentUrl\": \"{item[1]}\"}} " for item in refs)

        unique_values = set(ref_strs)

        ref_list = list(unique_values)

        return ref_list
