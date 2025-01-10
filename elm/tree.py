# -*- coding: utf-8 -*-
"""
ELM decision trees.
"""
import networkx as nx
import logging


logger = logging.getLogger(__name__)


class DecisionTree:
    """Class to traverse a directed graph of LLM prompts. Nodes are
    prompts and edges are transitions between prompts based on conditions
    being met in the LLM response."""

    def __init__(self, graph):
        """Class to traverse a directed graph of LLM prompts. Nodes are
        prompts and edges are transitions between prompts based on conditions
        being met in the LLM response.

        Examples
        --------
        Here's a simple example to setup a decision tree graph and run with the
        DecisionTree class:

        >>> import logging
        >>> import networkx as nx
        >>> from rex import init_logger
        >>> from elm.base import ApiBase
        >>> from elm.tree import DecisionTree
        >>>
        >>> init_logger('elm.tree')
        >>>
        >>> G = nx.DiGraph(text='hello', name='Grant',
                           api=ApiBase(model='gpt-35-turbo'))
        >>>
        >>> G.add_node('init', prompt='Say {text} to {name}')
        >>> G.add_edge('init', 'next', condition=lambda x: 'Grant' in x)
        >>> G.add_node('next', prompt='How are you?')
        >>>
        >>> tree = DecisionTree(G)
        >>> out = tree.run()
        >>>
        >>> print(tree.all_messages_txt)

        Parameters
        ----------
        graph : nx.DiGraph
            Directed acyclic graph where nodes are LLM prompts and edges are
            logical transitions based on the response. Must have high-level
            graph attribute "api" which is an ApiBase instance. Nodes should
            have attribute "prompt" which is a string that can have {format}
            named arguments that will be filled from the high-level graph
            attributes. Nodes can also have "callback" attributes that are
            callables that act on the LLM response in an arbitrary way. The
            function signature for a callback must be
            ``callback(llm_response, decision_tree, node_name)``.
            Edges can have attribute "condition" that is a callable to be
            executed on the LLM response text that determines the edge
            transition. An edge from a node without a condition acts as an
            "else" statement if no other edge conditions are satisfied. A
            single edge from node to node does not need a condition.
        """
        self._g = graph
        self._history = []
        assert isinstance(self.graph, nx.DiGraph)
        assert 'api' in self.graph.graph

    def __getitem__(self, key):
        """Retrieve a node by name (str) or edge by (node0, node1) tuple"""
        out = None
        if key in self.graph.nodes:
            out = self.graph.nodes[key]
        elif key in self.graph.edges:
            out = self.graph.edges[key]
        else:
            msg = (f'Could not find "{key}" in graph')
            logger.error(msg)
            raise KeyError(msg)
        return out

    @property
    def api(self):
        """Get the ApiBase object.

        Returns
        -------
        ApiBase
        """
        return self.graph.graph['api']

    @property
    def messages(self):
        """Get a list of the conversation messages with the LLM.

        Returns
        -------
        list
        """
        return self.api.messages

    @property
    def all_messages_txt(self):
        """Get a printout of the full conversation with the LLM

        Returns
        -------
        str
        """
        return self.api.all_messages_txt

    @property
    def history(self):
        """Get a record of the nodes traversed in the tree

        Returns
        -------
        list
        """
        return self._history

    @property
    def graph(self):
        """Get the networkx graph object

        Returns
        -------
        nx.DiGraph
        """
        return self._g

    def call_node(self, node_name):
        """Call the LLM with the prompt from the input node and search the
        successor edges for a valid transition condition

        Parameters
        ----------
        node_name : str
            Name of node being executed.

        Returns
        -------
        out : str
            Next node or LLM response if at a leaf node.
        """

        node = self[node_name]
        prompt = self._prepare_graph_call(node_name)
        out = self.api.chat(prompt)
        node['response'] = out

        if 'callback' in node:
            callback = node['callback']
            callback(out, self, node_name)

        return self._parse_graph_output(node_name, out)

    def _prepare_graph_call(self, node_name):
        """Prepare a graph call for given node."""
        prompt = self[node_name]['prompt']
        txt_fmt = {k: v for k, v in self.graph.graph.items() if k != 'api'}
        prompt = prompt.format(**txt_fmt)
        self._history.append(node_name)
        return prompt

    def _parse_graph_output(self, node0, out):
        """Parse graph output for given node and LLM call output. """
        successors = list(self.graph.successors(node0))
        edges = [self[(node0, node1)] for node1 in successors]
        conditions = [edge.get('condition', None) for edge in edges]

        if len(successors) == 0:
            logger.info(f'Reached leaf node "{node0}".')
            return out

        if len(successors) > 1 and all(c is None for c in conditions):
            msg = (f'At least one of the edges from "{node0}" should have '
                   f'a "condition": {edges}')
            logger.error(msg)
            raise AttributeError(msg)

        # prioritize callable conditions
        for i, condition in enumerate(conditions):
            if callable(condition) and condition(out):
                logger.info(f'Node transition: "{node0}" -> "{successors[i]}" '
                            '(satisfied by callable condition)')
                return successors[i]

        # None condition is basically "else" statement
        for i, condition in enumerate(conditions):
            if condition is None:
                logger.info(f'Node transition: "{node0}" -> "{successors[i]}" '
                            '(satisfied by None condition)')
                return successors[i]

        msg = (f'None of the edge conditions from "{node0}" '
               f'were satisfied: {edges}')
        logger.error(msg)
        raise AttributeError(msg)

    def run(self, node0='init'):
        """Traverse the decision tree starting at the input node.

        Parameters
        ----------
        node0 : str
            Name of starting node in the graph. This is typically called "init"

        Returns
        -------
        out : str
            Final response from LLM at the leaf node.
        """

        self._history = []

        while True:
            try:
                out = self.call_node(node0)
            except Exception as e:
                last_message = self.messages[-1]['content']
                msg = ('Ran into an exception when traversing tree. '
                       'Last message from LLM is printed below. '
                       'See debug logs for more detail. '
                       '\nLast message: \n'
                       f'"""\n{last_message}\n"""')
                logger.debug('Error traversing trees, heres the full '
                             'conversation printout:'
                             f'\n{self.all_messages_txt}')
                logger.error(msg)
                raise RuntimeError(msg) from e
            if out in self.graph:
                node0 = out
            else:
                break

        logger.info(f'Output: {out}')

        return out
