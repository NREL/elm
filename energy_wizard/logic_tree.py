# -*- coding: utf-8 -*-
"""
Energy Wizard logic trees.
"""
import networkx as nx
import logging


logger = logging.getLogger(__name__)


class LogicTree:
    """Class to traverse a directed graph."""

    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : nx.DiGraph
            Directed acyclic graph where nodes are LLM prompts and edges are
            logical transitions based on the response. Must have high-level
            graph attribute "api" which is an ApiBase instance. Nodes should
            have attribute "prompt" which can have {format} named arguments
            that will be filled from the high-level graph attributes. Edges can
            have attribute "condition" that is a callable to be executed on the
            LLM response text. No edge condition will signify a preferred
            transition.
        """
        self._g = graph
        assert isinstance(self.graph, nx.DiGraph)
        assert 'api' in self.graph.graph

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
        return self.api.chat_messages

    @property
    def all_messages_txt(self):
        """Get a printout of the full conversation with the LLM

        Returns
        -------
        str
        """
        messages = [f"{msg['role'].upper()}: {msg['content']}"
                    for msg in self.messages]
        messages = '\n\n'.join(messages)
        return messages

    @property
    def graph(self):
        """Get the networkx graph object

        Returns
        -------
        nx.DiGraph
        """
        return self._g

    def call_node(self, node0):
        """Call the LLM with the prompt from the input node and search the
        successor edges for a valid transition condition

        Parameters
        ----------
        node0 : str
            Name of node being executed.

        Returns
        -------
        out : str
            Next node or LLM response if at a leaf node.
        """

        prompt = self.graph.nodes[node0]['prompt']
        txt_fmt = {k: v for k, v in self.graph.graph.items() if k != 'api'}
        prompt = prompt.format(**txt_fmt)

        out = self.api.chat(prompt)

        successors = list(self.graph.successors(node0))
        edges = [self.graph.edges[(node0, node1)] for node1 in successors]
        conditions = [edge.get('condition', None) for edge in edges]

        if len(successors) == 0:
            logger.info(f'Reached leaf node "{node0}".')
            return out

        if len(successors) > 1 and all(c is None for c in conditions):
            msg = (f'None of the edges from "{node0}" have '
                   f'a "condition": {edges}')
            logger.error(msg)
            raise AttributeError(msg)

        for i, condition in enumerate(conditions):

            if condition is None:
                logger.info(f'Node transition: "{node0}" -> "{successors[i]}" '
                            '(satisfied by None condition)')
                return successors[i]

            elif callable(condition) and condition(out):
                logger.info(f'Node transition: "{node0}" -> "{successors[i]}" '
                            '(satisfied by callable condition)')
                return successors[i]

        msg = (f'None of the edge conditions from "{node0}" '
               f'were satisfied: {edges}')
        logger.error(msg)
        raise AttributeError(msg)

    def run(self, node0='init'):
        """Traverse the logic tree starting at the input node.

        Parameters
        ----------
        node0 : str
            Name of starting node in the graph. This is typically called "init"

        Returns
        -------
        out : str
            Final response from LLM at the leaf node.
        """
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
