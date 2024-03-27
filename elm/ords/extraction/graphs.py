# -*- coding: utf-8 -*-
"""ELM Ordinance Decision Tree Graph setup functions."""
import networkx as nx


_SECTION_PROMPT = (
    'The value of the "section" key should be a string representing the '
    "title of the section (including numerical labels), if it's given, "
    "and `null` otherwise."
)
_COMMENT_PROMPT = (
    'The value of the "comment" key should be a one-sentence explanation '
    "of how you determined the value, if you think it is necessary "
    "(`null` otherwise)."
)
EXTRACT_ORIGINAL_TEXT_PROMPT = (
    "Can you extract the raw text with original formatting "
    "that states how close I can site {wes_type} to {feature}? "
)


def _setup_graph_no_nodes(**kwargs):
    return nx.DiGraph(
        SECTION_PROMPT=_SECTION_PROMPT,
        COMMENT_PROMPT=_COMMENT_PROMPT,
        **kwargs
    )


def llm_response_starts_with_yes(response):
    """Check if LLM response begins with "yes" (case-insensitive)

    Parameters
    ----------
    response : str
        LLM response string.

    Returns
    -------
    bool
        `True` if LLM response begins with "Yes".
    """
    return response.lower().startswith("yes")


def llm_response_starts_with_no(response):
    """Check if LLM response begins with "no" (case-insensitive)

    Parameters
    ----------
    response : str
        LLM response string.

    Returns
    -------
    bool
        `True` if LLM response begins with "No".
    """
    return response.lower().startswith("no")


def llm_response_does_not_start_with_no(response):
    """Check if LLM response does not start with "no" (case-insensitive)

    Parameters
    ----------
    response : str
        LLM response string.

    Returns
    -------
    bool
        `True` if LLM response does not begin with "No".
    """
    return not llm_response_starts_with_no(response)


def setup_graph_wes_types(**kwargs):
    """Setup Graph to get the largest turbine size in the ordinance text.

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = _setup_graph_no_nodes(**kwargs)

    G.add_node(
        "init",
        prompt=(
            "Does the following text distinguish between multiple "
            "turbine sizes? Distinctions are often made as 'small' vs 'large' "
            "wind energy conversion systems or actual MW values. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge("init", "get_text", condition=llm_response_starts_with_yes)
    G.add_node(
        "get_text",
        prompt=(
            "What are the different turbine sizes this text mentions? "
            "List them in order of increasing size."
        ),
    )
    G.add_edge("get_text", "final")
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "largest_wes_type" and '
            '"explanation". The value of the "largest_wes_type" key should '
            "be a string that labels the largest wind energy conversion "
            'system mentioned in the text. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )
    return G


def setup_base_graph(**kwargs):
    """Setup Graph to get setback ordinance text for a particular feature.

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = _setup_graph_no_nodes(**kwargs)

    G.add_node(
        "init",
        prompt=(
            "Is there text in the following legal document that describes "
            "how close I can site or how far I have to setback "
            "{wes_type} to {feature}? {feature_clarifications}"
            "Pay extra attention to clarifying text found in parentheses "
            "and footnotes. Begin your response with either 'Yes' or 'No' "
            "and explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge(
        "init", "get_text", condition=llm_response_does_not_start_with_no
    )
    G.add_node("get_text", prompt=EXTRACT_ORIGINAL_TEXT_PROMPT)

    return G


def setup_participating_owner(**kwargs):
    """Setup Graph to check for participating vs non-participating owner
    setbacks for a feature.

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = _setup_graph_no_nodes(**kwargs)

    G.add_node(
        "init",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly specify "
            "a value that applies to participating owners? Occupying owners "
            "are not participating owners unless explicitly mentioned in the "
            "text. Justify your answer by quoting the raw text directly."
        ),
    )
    G.add_edge("init", "non_part")
    G.add_node(
        "non_part",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly specify "
            "a value that applies to non-participating owners? Non-occupying "
            "owners are not non-participating owners unless explicitly "
            "mentioned in the text. Justify your answer by quoting the raw "
            "text directly."
        ),
    )
    G.add_edge("non_part", "final")
    G.add_node(
        "final",
        prompt=(
            "Now we are ready to extract structured data. Respond based on "
            "our entire conversation so far. Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly two "
            'keys. The keys are "participating" and "non-participating". The '
            'value of the "participating" key should be a string containing '
            "the raw text with original formatting from the ordinance that "
            "applies to participating owners or `null` if there was no such "
            'text. The value of the "non-participating" key should be a '
            "string containing the raw text with original formatting from the "
            "ordinance that applies to non-participating owners or simply the "
            "full ordinance if the text did not make the distinction between "
            "participating and non-participating owners."
        ),
    )
    return G


def setup_multiplier(**kwargs):
    """Setup Graph to extract a setbacks multiplier values for a feature.

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = _setup_graph_no_nodes(**kwargs)

    G.add_node(
        "init",
        prompt=(
            "We will attempt to extract structured data for this ordinance. "
            "Let's think step by step. Does the text mention a multiplier "
            "that should be applied to a turbine dimension (e.g. height, "
            "rotor diameter, etc) to compute the setback distance from "
            "{feature}? Ignore any text related to {ignore_features}. "
            "Remember that 1 is a valid multiplier, and treat any mention of "
            "'fall zone' as a system height multiplier of 1. Begin your "
            "response with either 'Yes' or 'No' and explain your answer."
        ),
    )
    G.add_edge("init", "no_multiplier", condition=llm_response_starts_with_no)
    G.add_node(
        "no_multiplier",
        prompt=(
            "Does the ordinance give the setback from {feature} as a fixed "
            "distance value? Explain yourself."
        ),
    )
    G.add_edge("no_multiplier", "out_static")
    G.add_node(
        "out_static",
        prompt=(
            "Now we are ready to extract structured data. Respond based on "
            "our entire conversation so far. Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly "
            'four keys. The keys are "fixed_value", "units", "section", '
            '"comment". The value of the "fixed_value" key should be a '
            "numerical value corresponding to the setback distance value "
            "from {feature} or `null` if there was no such value. The value "
            'of the "units" key should be a string corresponding to the units '
            "of the setback distance value from {feature} or `null` if there "
            'was no such value. {SECTION_PROMPT} The value of the "comment" '
            "key should be a one-sentence explanation of how you determined "
            "the value, or a short description of the ordinance itself if no "
            "multiplier or static setback value was found."
        ),
    )
    G.add_edge("init", "mult_single", condition=llm_response_starts_with_yes)

    G.add_node(
        "mult_single",
        prompt=(
            "Are multiple values given for the multiplier used to "
            "compute the setback distance value from {feature}? If so, "
            "select and state the largest one. Otherwise, repeat the single "
            "multiplier value that was given in the text. "
        ),
    )
    G.add_edge("mult_single", "mult_type")
    G.add_node(
        "mult_type",
        prompt=(
            "What should the multiplier be applied to? Common acronyms "
            "include RD for rotor diameter and HH for hub height. Remember "
            "that system/total height is the tip-hight of the turbine. "
            "Select a value from the following list and explain yourself: "
            "['tip-height-multiplier', 'hub-height-multiplier', "
            "'rotor-diameter-multiplier]"
        ),
    )

    G.add_edge("mult_type", "adder")
    G.add_node(
        "adder",
        prompt=(
            "Does the ordinance include a static distance value that "
            "should be added to the result of the multiplication? Do not "
            "confuse this value with static setback requirements. Ignore text "
            "with clauses such as 'no lesser than', 'no greater than', "
            "'the lesser of', or 'the greater of'. Begin your response with "
            "either 'Yes' or 'No' and explain your answer, stating the adder "
            "value if it exists."
        ),
    )
    G.add_edge("adder", "out_mult", condition=llm_response_starts_with_no)
    G.add_edge("adder", "adder_eq", condition=llm_response_starts_with_yes)

    G.add_node(
        "adder_eq",
        prompt=(
            "We are only interested in adders that satisfy the following "
            "equation: 'multiplier * turbine_dimension + <adder>'. Does the "
            "adder value you identified satisfy this equation? Begin your "
            "response with either 'Yes' or 'No' and explain your answer."
        ),
    )
    G.add_edge("adder_eq", "out_mult", condition=llm_response_starts_with_no)
    G.add_edge(
        "adder_eq",
        "conversion",
        condition=llm_response_starts_with_yes,
    )
    G.add_node(
        "conversion",
        prompt=(
            "If the adder value is not given in feet, convert "
            "it to feet (remember that there are 3.28084 feet in one meter "
            "and 5280 feet in one mile). Show your work step-by-step "
            "if you had to perform a conversion."
        ),
    )
    G.add_edge("conversion", "out_mult")

    G.add_node(
        "out_mult",
        prompt=(
            "Now we are ready to extract structured data. Respond based on "
            "our entire conversation so far. Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly five "
            'keys. The keys are "mult_value", "mult_type", "adder", '
            '"section", "comment". The value of the "mult_value" key should '
            "be a numerical value corresponding to the multiplier value we "
            'determined earlier. The value of the "mult_type" key should be '
            "a string corresponding to the dimension that the multiplier "
            "should be applied to, as we determined earlier. The value of "
            'the "adder" key should be a numerical value corresponding to '
            "the static value to be added to the total setback distance after "
            "multiplication, as we determined earlier, or `null` if there is "
            "no such value. {SECTION_PROMPT} {COMMENT_PROMPT}"
        ),
    )

    return G


def setup_conditional(**kwargs):
    """Setup Graph to extract min/max setback values (after mult) for a
    feature. These are typically given within the context of
    'the greater of' or 'the lesser of' clauses.

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = _setup_graph_no_nodes(**kwargs)

    G.add_node(
        "init",
        prompt=(
            "We will attempt to extract structured data for this ordinance. "
            "Let's think step by step. Does the setback from {feature} "
            "mention a minimum or maximum static setback distance regardless "
            "of the outcome of the multiplier calculation? This is often "
            "phrased as 'the greater of' or 'the lesser of'. Do not confuse "
            "this value with static values to be added to multiplicative "
            "setbacks. Begin your response with either 'Yes' or 'No' and "
            "explain your answer."
        ),
    )

    G.add_edge("init", "conversions", condition=llm_response_starts_with_yes)
    G.add_node(
        "conversions",
        prompt=(
            "Tell me the minimum and/or maximum setback distances, "
            "converting to feet if necessary (remember that there are "
            "3.28084 feet in one meter and 5280 feet in one mile). "
            "Explain your answer and show your work if you had to perform "
            "a conversion."
        ),
    )

    G.add_edge("conversions", "out_condition")
    G.add_node(
        "out_condition",
        prompt=(
            "Now we are ready to extract structured data. Respond based "
            "on our entire conversation so far. Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly two "
            'keys. The keys are "min_dist" and "max_dist". The value of the '
            '"min_dist" key should be a numerical value corresponding to the '
            "minimum setback value from {feature} we determined earlier, or "
            '`null` if no such value exists. The value of the "max_dist" key '
            "should be a numerical value corresponding to the maximum setback "
            "value from {feature} we determined earlier, or `null` if no such "
            "value exists."
        ),
    )

    return G


def setup_graph_extra_restriction(**kwargs):
    """Setup Graph to extract non-setback ordinance values from text.

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = _setup_graph_no_nodes(**kwargs)

    G.add_node(
        "init",
        prompt=(
            "We will attempt to extract structured data for this "
            "ordinance. Let's think step by step. Does the following text "
            "explicitly limit the {restriction} allowed for {wes_type}? "
            "Do not infer based on other restrictions; if this particular "
            "restriction is not explicitly mentioned then say 'No'. Pay extra "
            "attention to clarifying text found in parentheses and footnotes. "
            "Begin your response with either 'Yes' or 'No' and explain "
            "your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )
    G.add_edge("init", "final", condition=llm_response_starts_with_yes)

    G.add_node(
        "final",
        prompt=(
            "Now we are ready to extract structured data. Respond based "
            "on our entire conversation so far. Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly four "
            'keys. The keys are "value", "units", "section", "comment". The '
            'value of the "value" key should be a numerical value '
            "corresponding to the {restriction} allowed for {wes_type}, or "
            "`null` if the text does not mention such a restriction. Use our "
            'conversation to fill out this value. The value of the "units" '
            "key should be a string corresponding to the units for the "
            "{restriction} allowed for {wes_type} by the text below, or "
            "`null` if the text does not mention such a restriction. Make "
            'sure to include any "per XXX" clauses in the units. '
            "{SECTION_PROMPT} {COMMENT_PROMPT}"
        ),
    )
    return G
