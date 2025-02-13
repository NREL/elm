# -*- coding: utf-8 -*-
"""
ELM energy wizard
"""
from abc import ABC, abstractmethod
from time import perf_counter
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

    def engineer_query(self,
                       query,
                       token_budget=None,
                       new_info_threshold=0.7,
                       convo=False
                       ):
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
        used_index : list
            Shows the indices of the documents used in making a query to the
            vector database
        vector_query_time : float
            measures vector database query time
        """

        self.messages.append({"role": "user", "content": query})

        if convo:
            # [1:] to not include the system role in the semantic search
            query = [f"{msg['role'].upper()}: {msg['content']}"
                     for msg in self.messages[1:]]

            query = '\n\n'.join(query)

        token_budget = token_budget or self.token_budget
        start_time = perf_counter()
        strings, _, idx = self.query_vector_db(query)
        end_time = perf_counter()
        vector_query_time = end_time - start_time
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
        return message, references, used_index, vector_query_time

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

    def chat(self,
             query,
             stream=True,
             temperature=0,
             convo=False,
             token_budget=None,
             new_info_threshold=0.7,
             print_references=False,
             return_chat_obj=False
             ):
        """Answers a query by doing a semantic search of relevant text with
        embeddings and then sending engineered query to the LLM.

        Parameters
        ----------
        query : str
            Question being asked of EnergyWizard
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
            valid ``ref_col``.
        return_chat_obj : bool
            Flag to return the ChatCompletion object from OpenAI API instead of
            the message string.

        Returns
        -------
        response : str | ChatCompletion
            GPT response string ``if not return_chat_obj`` or OpenAI
            ChatCompletion object ``if return_chat_obj``
        query : str
            The engineered query asked of GPT including retrieved context
        references : list
            The list of references (strs) used in the engineered prompt is
            returned here
        performance : dict | None
            dictionary with keys of ``total_chat_time``,
            ``chat_completion_time`` and ``vectordb_query_time``. If
            return_chat_obj, this is None
        """
        start_chat_time = perf_counter()
        out = self.engineer_query(query, token_budget=token_budget,
                                  new_info_threshold=new_info_threshold,
                                  convo=convo)
        query, references, _, vector_query_time = out

        messages = [{"role": "system", "content": self.MODEL_ROLE},
                    {"role": "user", "content": query}]
        response_message = ''
        kwargs = dict(model=self.model,
                      messages=messages,
                      temperature=temperature,
                      stream=stream)

        start_completion_time = perf_counter()
        response = self._client.chat.completions.create(**kwargs)

        finish_completion_time = perf_counter()
        chat_completion_time = finish_completion_time - start_completion_time
        total_chat_time = finish_completion_time - start_chat_time

        performance = {
            "total_chat_time": total_chat_time,
            "chat_completion_time": chat_completion_time,
            "vectordb_query_time": vector_query_time
        }

        if return_chat_obj:
            return response, query, references, performance

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

        end_time = perf_counter()
        total_chat_time = end_time - start_chat_time
        performance = {
            "total_chat_time": total_chat_time,
            "chat_completion_time": chat_completion_time,
            "vectordb_query_time": vector_query_time
        }

        return response_message, query, references, performance


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
    Query Embedding: AWS using boto3
    LLM Application: GPT4 via Azure deployment
    """
    EMBEDDING_MODEL = 'amazon.titan-embed-text-v1'

    TOKENIZER_ALIASES = {**EnergyWizardBase.TOKENIZER_ALIASES,
                         'ewiz-gpt-4': 'gpt-4'
                         }
    """Optional mappings for weird azure names to tiktoken/openai names."""

    DEFAULT_META_COLS = ['title', 'url', 'nrel_id', 'id']
    """Default columns to retrieve for metadata"""

    def __init__(self, db_host, db_port, db_name,
                 db_schema, db_table, probes=25,
                 meta_columns=None, cursor=None,
                 boto_client=None, model=None,
                 token_budget=3500, tag=False):
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
        probes : int
            Number of lists to search in vector database. Recommended
            value is sqrt(n_lists).
        meta_columns : list
            List of metadata columns to retrieve from database. Default
            query returns title, url, nrel_id, and id. nrel_id and id are
            necessary to correctly format references.
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
        tag: bool
            Flag to add tag/metadata to text chunks before sending query to
            GPT.
        """
        boto3 = try_import('boto3')
        self.psycopg2 = try_import('psycopg2')

        if meta_columns is None:
            self.meta_columns = self.DEFAULT_META_COLS
        else:
            self.meta_columns = meta_columns

        assert 'id' in self.meta_columns, ("Please include the chunk id "
                                           "column: 'id'!")
        assert 'nrel_id' in self.meta_columns, ("Please include the document "
                                                "id column: 'nrel_id'!")

        if cursor is None:
            db_user = os.getenv("EWIZ_DB_USER")
            db_password = os.getenv('EWIZ_DB_PASSWORD')
            assert db_user is not None, "Must set EWIZ_DB_USER!"
            assert db_password is not None, "Must set EWIZ_DB_PASSWORD!"
            self.db_kwargs = dict(user=db_user, password=db_password,
                                  host=db_host, port=db_port,
                                  database=db_name)
            self.conn = self.psycopg2.connect(**self.db_kwargs)

            self.cursor = self.conn.cursor()
        else:
            self.cursor = cursor

        self.db_schema = db_schema
        self.db_table = db_table
        self.tag = tag
        self.probes = probes

        if boto_client is None:
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret = os.getenv('AWS_SECRET_ACCESS_KEY')
            session = os.getenv('AWS_SESSION_TOKEN')
            if access_key and secret and session:
                self._aws_client = boto3.client(service_name='bedrock-runtime',
                                                region_name='us-west-2',
                                                aws_access_key_id=access_key,
                                                aws_secret_access_key=secret,
                                                aws_session_token=session)
            else:
                self._aws_client = boto3.client(service_name='bedrock-runtime',
                                                region_name='us-west-2')
        else:
            self._aws_client = boto_client

        super().__init__(model, token_budget=token_budget)

    def get_embedding(self, text):
        """Get the 1D array (list) embedding of a text string
        as generated by specified AWS model.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        embedding : list
            List of float that represents the numerical embedding of the text
        """
        model_id = self.EMBEDDING_MODEL

        if 'cohere' in model_id:
            input_type = "search_query"

            body = json.dumps({"texts": [text],
                               "input_type": input_type})
        else:
            body = json.dumps({"inputText": text, })

        accept = 'application/json'
        content_type = 'application/json'

        response = self._aws_client.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response['body'].read())

        if 'cohere' in model_id:
            embedding = response_body.get('embeddings')[0]
        else:
            embedding = response_body.get('embedding')

        return embedding

    @staticmethod
    def _add_tag(meta):
        """Function to add tag/metadata to text strings before
        sending query to GPT.

        Parameters
        ----------
        meta : tuple
            Text values to include in tag (title, authors, year)

        Returns
        -------
        tag : str
            Text string containing provided metadata.
        """
        title, authors, year = meta
        if authors and year:
            tag = (f"Title: {title}\n"
                   f"Authors: {authors}\n"
                   f"Publication Year: {year}\n\n"
                   )
        else:
            tag = f"Title: {title}\n\n"

        return tag

    def query_vector_db(self, query, limit=100):
        """Returns a list of strings and relatednesses, sorted from most
        related to least. SQL query uses a context handler and rollback
        to ensure a failed query does not interupt future questions from
        the user. Ex: a user submitting a new question before the first
        one completes will close the cursor preventing future database
        access.

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

        with self.psycopg2.connect(**self.db_kwargs) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"SET LOCAL ivfflat.probes = {self.probes};"
                               f"SELECT {self.db_table}.id, "
                               f"{self.db_table}.chunks, "
                               f"{self.db_table}.embedding "
                               "<=> %s::vector as score, "
                               f"{self.db_table}.title, "
                               f"{self.db_table}.authors, "
                               f"{self.db_table}.year "
                               f"FROM {self.db_schema}.{self.db_table} "
                               "ORDER BY embedding <=> %s::vector LIMIT %s;",
                               (query_embedding, query_embedding, limit,), )
            except Exception as exc:
                conn.rollback()
                msg = (f'Received error when querying the postgres '
                       f'vector database: {exc}')
                raise RuntimeError(msg) from exc
            else:
                conn.commit()
                result = cursor.fetchall()

        if self.tag:
            strings = [self._add_tag(s[3:]) + s[1] for s in result]
        else:
            strings = [s[1] for s in result]

        scores = [s[2] for s in result]
        best = [s[0] for s in result]

        return strings, scores, best

    def _format_refs(self, refs, ids):
        """Parse and nicely format a reference dictionary into a list of well
        formatted string representations

        Parameters
        ----------
        refs : list
            List of references returned from the vector db
        ids : np.ndarray
            IDs of the used text from the text corpus sorted by embedding
            relevance.

        Returns
        -------
        out : list
            Unique ordered list of references (most relevant first)
        """

        ref_list = []
        for item in refs:
            ref_dict = {col: str(value).replace(chr(34), '')
                        for col, value in zip(self.meta_columns, item)}

            ref_list.append(ref_dict)

        assert len(ref_list) > 0, ("The Wizard did not return any "
                                   "references. Please check your database "
                                   "connection or query.")

        unique_ref_list = []
        unique_nrel_ids = set()
        for ref_dict in ref_list:
            if ref_dict['nrel_id'] in unique_nrel_ids:
                continue
            unique_ref_list.append(ref_dict)
            unique_nrel_ids.add(ref_dict['nrel_id'])

        ref_list = unique_ref_list

        if 'id' in ref_list[0]:
            ids_list = list(ids)
            sorted_ref_list = []
            for ref_id in ids_list:
                for ref_dict in ref_list:
                    if ref_dict['id'] == ref_id:
                        sorted_ref_list.append(ref_dict)
                        break
            ref_list = sorted_ref_list

        ref_list = [json.dumps(ref) for ref in ref_list]

        return ref_list

    def make_ref_list(self, ids):
        """Make a reference list. SQL query uses a context handler and
        rollback to ensure a failed query does not interupt future questions
        from the user. Ex: a user submitting a new question before the first
        one completes will close the cursor preventing future database
        access.

        Parameters
        ----------
        ids : np.ndarray
            IDs of the used text from the text corpus

        Returns
        -------
        ref_list : list
            A list of references (strs) used. Ideally, this is something like:
            ["{ref_title} ({ref_url})"]
        """

        placeholders = ', '.join(['%s'] * len(ids))
        columns_str = ', '.join([f"{self.db_table}.{c}"
                                 for c in self.meta_columns])

        sql_query = (f"SELECT {columns_str} "
                     f"FROM {self.db_schema}.{self.db_table} "
                     f"WHERE {self.db_table}.id IN (" + placeholders + ")")

        with self.psycopg2.connect(**self.db_kwargs) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql_query, ids)
            except Exception as exc:
                conn.rollback()
                msg = (f'Received error when querying the postgres '
                       f'vector database: {exc}')
                raise RuntimeError(msg) from exc
            else:
                conn.commit()
                refs = cursor.fetchall()

        ref_list = self._format_refs(refs, ids)

        return ref_list
