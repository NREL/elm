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


def setup_graph_fluids(**kwargs):
    """Setup Graph to get the fluids mentioned in the text

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
            "Does the following text mention water or other fluid resources? "
            "Some key words to look for are 'water', 'brines', 'groundwater', "
            "or 'fluids'."
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=("Does the following text mention water or other fluid resources? "
            "Some key words to look for are 'water', 'brines', 'groundwater', "
            "or 'fluids'."),
    )

    G.add_edge("init", "get_fluid", condition=llm_response_starts_with_yes)
    G.add_node(
        "get_fluid",
        prompt=(
            "What are the different fluids this text mentions? "
        ),
    )

    G.add_edge("get_fluid", "final")
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "fluid" and '
            '"explanation". The value of the "fluid" key should '
            "be a string that labels the fluids mentioned in the text. "
            'The value of the "explanation" key should be a string containing '
            "a short explanation for your choice."
        ),
    )
    return G

def setup_graph_temperature(**kwargs):
    """Setup Graph to get the temperature threshold

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
            "Does the following text mention a temperature threshold "
            "in relation to the definition of a geothermal resource? "
            "Thresholds are typically labeled in degrees Fahrenheit "
            "or celsius and they describe the temperature at which "
            "a resource is considered geothermal. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge("init", "get_temperature", condition=llm_response_starts_with_yes)
    G.add_node(
        "get_temperature",
        prompt=(
            "What is the temperature threshold the text mentions? "
        ),
    )

    G.add_edge("get_temperature", "get_units")
    G.add_node(
        "get_units",
        prompt=(
            "What are the units specified for the temperature threshold?"
        ),
    )

    G.add_edge("get_units", "final")
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "temperature", "units", '
            'and "explanation". '
            'The value of the "temperature" key should be the temperature value '
            'mentioned in the text, if temperature is not mentioned the value '
            'of this key should be null. The value of the "units" key should be '
            'the units for the temperature threshold and will likely be '
            '"celsius" or "fahrenheit", if temperature is not mentioned the value '
            'of this key should be nullThe value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )
    return G

def setup_graph_permits(**kwargs):
    """Setup Graph to get permit requirements 

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
        prompt=( ## TODO: how to phrase this question? additional deets? 
            "Does the following text mention water well permit requirements? "
            "Requirements should specify whether or not an application is required "
            "in order to drill a groundwater well."
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=("Does the following text mention water well permit requirements? "
            "Requirements should specify whether or not an application is required "
            "in order to drill a groundwater well."), 
    ) 

    G.add_edge("init", "get_reqs", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_reqs",
        prompt=(
            "What are the requirements the text mentions? "
        ),
    )

    G.add_edge("get_reqs", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "permit_required", "requirements", '
            'and "explanation". '
            'The value of the "permit_required" key should be either "True" or "False" '
            'based on whether or not the conservation district requires water well permits. '
            'of this key should be null. The value of the "units" key should be '
            'the units for the temperature threshold and will likely be '
            '"celsius" or "fahrenheit", if temperature is not mentioned the value '
            'of this key should be nullThe value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_daily_limits(**kwargs):
    """Setup Graph to get permit requirements 

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
            "Does the following text mention daily water well extraction limits? "
            "Extraction limits may be defined as an acre-foot "
            "or a gallon limit."
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    ) 

    G.add_edge("init", "get_daily", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_daily",
        prompt=(
            "What is the daily extraction limit mentioned in the text? "
            "Include the units associated with the limit (example: gallons or acre-feet). "
        ),
    )

    G.add_edge("get_daily", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "extraction_limit", "units", '
            'and "explanation". '
            'The value of the "extraction_limit" key should be the numerical value associated '
            'with the extraction limit the "units" key should describe the units associated with '
            'the extraction limit. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_annual_limits(**kwargs):
    """Setup Graph to get permit requirements 

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
            "Does the following text mention annual water well extraction limits? "
            "Extraction limits may be defined as an acre-foot "
            "or a gallon limit."
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    ) 

    G.add_edge("init", "get_annual", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_annual",
        prompt=(
            "What is the annual extraction limit mentioned in the text? "
            "Include the units associated with the limit (example: gallons or acre-feet). "
        ),
    )

    G.add_edge("get_daily", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "extraction_limit", "units", '
            'and "explanation". '
            'The value of the "extraction_limit" key should be the numerical value associated '
            'with the extraction limit the "units" key should describe the units associated with '
            'the extraction limit. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_well_spacing(**kwargs):
    """Setup Graph to get permit requirements 

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
            "Does the following text mention restrictions related to well spacing? "
            "Such information typically dictates how far apart two well must be and "
            "could prohibit an individual from drilling a well with a certain distance "
            "of another well. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    ) 

    G.add_edge("init", "get_spacing", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_spacing",
        prompt=(
            "What is the spacing limit mentioned in the text? "
            "Include the units associated with the limit (example: feet or yards). "
        ),
    )

    G.add_edge("get_spacing", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "spacing", "units", '
            'and "explanation". '
            'The value of the "spacing" key should be the numerical value associated '
            'with the spacing requirements the "units" key should describe the units associated with '
            'the spacing. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_time(**kwargs):
    """Setup Graph to get permit requirements 

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
        prompt=( ## TODO: better word than 'drilling window' -- keep the example?
            "Does the following text mention a drilling window? "
            "Such a value specifies how long after permit approval "
            "that drilling must commence. For example, 'Drilling of a "
            "permitted well must commence within 120 days of issuance of "
            "the permit application.' represents a 120 day window. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    ) 

    G.add_edge("init", "get_time", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_time",
        prompt=(
            "What is the drilling window mentioned in the text? "
            "Include the units associated with the limit (example: days or months). "
        ),
    )

    G.add_edge("get_time", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "time_period", "units", '
            'and "explanation". '
            'The value of the "time_period" key should be the numerical value associated '
            'with the drilling window the "units" key should describe the units associated with '
            'the drilling window. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_metering_device(**kwargs):
    """Setup Graph to get permit requirements 

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
            "Does the following text mention a metering device? "
            "Metering devices are typically utilized to monitor "
            "water usage and might be required by the conservation "
            "district. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    ) 

    G.add_edge("init", "get_device", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_device",
        prompt=(
            "What device is mentioned in the text? "
        ),
    )

    G.add_edge("get_device", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "device" and '
            'and "explanation". '
            'The value of the "device" key should be a boolean value with the '
            'the value "True" is a device is required. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_drought(**kwargs):
    """Setup Graph to get permit requirements 

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
            "Does the following text mention a drought management plan? "
            "Drought management plans might specify actions or policies "
            "that could be implemented in the event of a drought and "
            "could impose additional restrictions on well users. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    ) 

    G.add_edge("init", "get_plan", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_device",
        prompt=(
            "Summarize the drought management plan mentioned in the text? "
        ),
    )

    G.add_edge("get_plan", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "drought_plan" and '
            'and "explanation". '
            'The value of the "drought_plan" key should be a boolean value with the '
            'the value "True" if a drought management plan is in place. '
            'The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

def setup_graph_plugging_reqs(**kwargs):
    """Setup Graph to get permit requirements 

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
            "Does the following text mention plugging requirements specific "
            "to water wells? Plugging requirements generally detail the  "
            "steps an individual needs to take when they no longer intend to "
            "use the well. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    ) 

    G.add_edge("init", "get_plugging", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_plugging",
        prompt=(
            "What are the plugging requirements mentioned in the text? "
        ),
    )

    G.add_edge("get_plugging", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "plugging_requirements" and '
            'and "explanation". '
            'The value of the "plugging_requirements" key should be a boolean value with the '
            'the value "True" if the conservation district implements plugging requirements. '
            'The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )
# def setup_base_graph(**kwargs):
#     """Setup Graph to get setback ordinance text for a particular feature.

#     Parameters
#     ----------
#     **kwargs
#         Keyword-value pairs to add to graph.

#     Returns
#     -------
#     nx.DiGraph
#         Graph instance that can be used to initialize an
#         `elm.tree.DecisionTree`.
#     """
#     G = _setup_graph_no_nodes(**kwargs)

#     G.add_node(
#         "init",
#         prompt=(
#             "Is there text in the following legal document that describes "
#             "how close I can site or how far I have to setback "
#             "{wes_type} to {feature}? {feature_clarifications}"
#             "Pay extra attention to clarifying text found in parentheses "
#             "and footnotes. Begin your response with either 'Yes' or 'No' "
#             "and explain your answer."
#             '\n\n"""\n{text}\n"""'
#         ),
#     )

#     G.add_edge(
#         "init", "get_text", condition=llm_response_does_not_start_with_no
#     )
#     G.add_node("get_text", prompt=EXTRACT_ORIGINAL_TEXT_PROMPT)

#     return G


# def setup_participating_owner(**kwargs):
#     """Setup Graph to check for participating vs non-participating owner
#     setbacks for a feature.

#     Parameters
#     ----------
#     **kwargs
#         Keyword-value pairs to add to graph.

#     Returns
#     -------
#     nx.DiGraph
#         Graph instance that can be used to initialize an
#         `elm.tree.DecisionTree`.
#     """
#     G = _setup_graph_no_nodes(**kwargs)

#     G.add_node(
#         "init",
#         prompt=(
#             "Does the ordinance for {feature} setbacks explicitly specify "
#             "a value that applies to participating owners? Occupying owners "
#             "are not participating owners unless explicitly mentioned in the "
#             "text. Justify your answer by quoting the raw text directly."
#         ),
#     )
#     G.add_edge("init", "non_part")
#     G.add_node(
#         "non_part",
#         prompt=(
#             "Does the ordinance for {feature} setbacks explicitly specify "
#             "a value that applies to non-participating owners? Non-occupying "
#             "owners are not non-participating owners unless explicitly "
#             "mentioned in the text. Justify your answer by quoting the raw "
#             "text directly."
#         ),
#     )
#     G.add_edge("non_part", "final")
#     G.add_node(
#         "final",
#         prompt=(
#             "Now we are ready to extract structured data. Respond based on "
#             "our entire conversation so far. Return your answer in JSON "
#             "format (not markdown). Your JSON file must include exactly two "
#             'keys. The keys are "participating" and "non-participating". The '
#             'value of the "participating" key should be a string containing '
#             "the raw text with original formatting from the ordinance that "
#             "applies to participating owners or `null` if there was no such "
#             'text. The value of the "non-participating" key should be a '
#             "string containing the raw text with original formatting from the "
#             "ordinance that applies to non-participating owners or simply the "
#             "full ordinance if the text did not make the distinction between "
#             "participating and non-participating owners."
#         ),
#     )
#     return G


# def setup_multiplier(**kwargs):
#     """Setup Graph to extract a setbacks multiplier values for a feature.

#     Parameters
#     ----------
#     **kwargs
#         Keyword-value pairs to add to graph.

#     Returns
#     -------
#     nx.DiGraph
#         Graph instance that can be used to initialize an
#         `elm.tree.DecisionTree`.
#     """
#     G = _setup_graph_no_nodes(**kwargs)

#     G.add_node(
#         "init",
#         prompt=(
#             "We will attempt to extract structured data for this ordinance. "
#             "Let's think step by step. Does the text mention a multiplier "
#             "that should be applied to a turbine dimension (e.g. height, "
#             "rotor diameter, etc) to compute the setback distance from "
#             "{feature}? Ignore any text related to {ignore_features}. "
#             "Remember that 1 is a valid multiplier, and treat any mention of "
#             "'fall zone' as a system height multiplier of 1. Begin your "
#             "response with either 'Yes' or 'No' and explain your answer."
#         ),
#     )
#     G.add_edge("init", "no_multiplier", condition=llm_response_starts_with_no)
#     G.add_node(
#         "no_multiplier",
#         prompt=(
#             "Does the ordinance give the setback from {feature} as a fixed "
#             "distance value? Explain yourself."
#         ),
#     )
#     G.add_edge("no_multiplier", "out_static")
#     G.add_node(
#         "out_static",
#         prompt=(
#             "Now we are ready to extract structured data. Respond based on "
#             "our entire conversation so far. Return your answer in JSON "
#             "format (not markdown). Your JSON file must include exactly "
#             'four keys. The keys are "fixed_value", "units", "section", '
#             '"comment". The value of the "fixed_value" key should be a '
#             "numerical value corresponding to the setback distance value "
#             "from {feature} or `null` if there was no such value. The value "
#             'of the "units" key should be a string corresponding to the units '
#             "of the setback distance value from {feature} or `null` if there "
#             'was no such value. {SECTION_PROMPT} The value of the "comment" '
#             "key should be a one-sentence explanation of how you determined "
#             "the value, or a short description of the ordinance itself if no "
#             "multiplier or static setback value was found."
#         ),
#     )
#     G.add_edge("init", "mult_single", condition=llm_response_starts_with_yes)

#     G.add_node(
#         "mult_single",
#         prompt=(
#             "Are multiple values given for the multiplier used to "
#             "compute the setback distance value from {feature}? If so, "
#             "select and state the largest one. Otherwise, repeat the single "
#             "multiplier value that was given in the text. "
#         ),
#     )
#     G.add_edge("mult_single", "mult_type")
#     G.add_node(
#         "mult_type",
#         prompt=(
#             "What should the multiplier be applied to? Common acronyms "
#             "include RD for rotor diameter and HH for hub height. Remember "
#             "that system/total height is the tip-hight of the turbine. "
#             "Select a value from the following list and explain yourself: "
#             "['tip-height-multiplier', 'hub-height-multiplier', "
#             "'rotor-diameter-multiplier]"
#         ),
#     )

#     G.add_edge("mult_type", "adder")
#     G.add_node(
#         "adder",
#         prompt=(
#             "Does the ordinance include a static distance value that "
#             "should be added to the result of the multiplication? Do not "
#             "confuse this value with static setback requirements. Ignore text "
#             "with clauses such as 'no lesser than', 'no greater than', "
#             "'the lesser of', or 'the greater of'. Begin your response with "
#             "either 'Yes' or 'No' and explain your answer, stating the adder "
#             "value if it exists."
#         ),
#     )
#     G.add_edge("adder", "out_mult", condition=llm_response_starts_with_no)
#     G.add_edge("adder", "adder_eq", condition=llm_response_starts_with_yes)

#     G.add_node(
#         "adder_eq",
#         prompt=(
#             "We are only interested in adders that satisfy the following "
#             "equation: 'multiplier * turbine_dimension + <adder>'. Does the "
#             "adder value you identified satisfy this equation? Begin your "
#             "response with either 'Yes' or 'No' and explain your answer."
#         ),
#     )
#     G.add_edge("adder_eq", "out_mult", condition=llm_response_starts_with_no)
#     G.add_edge(
#         "adder_eq",
#         "conversion",
#         condition=llm_response_starts_with_yes,
#     )
#     G.add_node(
#         "conversion",
#         prompt=(
#             "If the adder value is not given in feet, convert "
#             "it to feet (remember that there are 3.28084 feet in one meter "
#             "and 5280 feet in one mile). Show your work step-by-step "
#             "if you had to perform a conversion."
#         ),
#     )
#     G.add_edge("conversion", "out_mult")

#     G.add_node(
#         "out_mult",
#         prompt=(
#             "Now we are ready to extract structured data. Respond based on "
#             "our entire conversation so far. Return your answer in JSON "
#             "format (not markdown). Your JSON file must include exactly five "
#             'keys. The keys are "mult_value", "mult_type", "adder", '
#             '"section", "comment". The value of the "mult_value" key should '
#             "be a numerical value corresponding to the multiplier value we "
#             'determined earlier. The value of the "mult_type" key should be '
#             "a string corresponding to the dimension that the multiplier "
#             "should be applied to, as we determined earlier. The value of "
#             'the "adder" key should be a numerical value corresponding to '
#             "the static value to be added to the total setback distance after "
#             "multiplication, as we determined earlier, or `null` if there is "
#             "no such value. {SECTION_PROMPT} {COMMENT_PROMPT}"
#         ),
#     )

#     return G


# def setup_conditional(**kwargs):
#     """Setup Graph to extract min/max setback values (after mult) for a
#     feature. These are typically given within the context of
#     'the greater of' or 'the lesser of' clauses.

#     Parameters
#     ----------
#     **kwargs
#         Keyword-value pairs to add to graph.

#     Returns
#     -------
#     nx.DiGraph
#         Graph instance that can be used to initialize an
#         `elm.tree.DecisionTree`.
#     """
#     G = _setup_graph_no_nodes(**kwargs)

#     G.add_node(
#         "init",
#         prompt=(
#             "We will attempt to extract structured data for this ordinance. "
#             "Let's think step by step. Does the setback from {feature} "
#             "mention a minimum or maximum static setback distance regardless "
#             "of the outcome of the multiplier calculation? This is often "
#             "phrased as 'the greater of' or 'the lesser of'. Do not confuse "
#             "this value with static values to be added to multiplicative "
#             "setbacks. Begin your response with either 'Yes' or 'No' and "
#             "explain your answer."
#         ),
#     )

#     G.add_edge("init", "conversions", condition=llm_response_starts_with_yes)
#     G.add_node(
#         "conversions",
#         prompt=(
#             "Tell me the minimum and/or maximum setback distances, "
#             "converting to feet if necessary (remember that there are "
#             "3.28084 feet in one meter and 5280 feet in one mile). "
#             "Explain your answer and show your work if you had to perform "
#             "a conversion."
#         ),
#     )

#     G.add_edge("conversions", "out_condition")
#     G.add_node(
#         "out_condition",
#         prompt=(
#             "Now we are ready to extract structured data. Respond based "
#             "on our entire conversation so far. Return your answer in JSON "
#             "format (not markdown). Your JSON file must include exactly two "
#             'keys. The keys are "min_dist" and "max_dist". The value of the '
#             '"min_dist" key should be a numerical value corresponding to the '
#             "minimum setback value from {feature} we determined earlier, or "
#             '`null` if no such value exists. The value of the "max_dist" key '
#             "should be a numerical value corresponding to the maximum setback "
#             "value from {feature} we determined earlier, or `null` if no such "
#             "value exists."
#         ),
#     )

#     return G


# def setup_graph_extra_restriction(**kwargs):
#     """Setup Graph to extract non-setback ordinance values from text.

#     Parameters
#     ----------
#     **kwargs
#         Keyword-value pairs to add to graph.

#     Returns
#     -------
#     nx.DiGraph
#         Graph instance that can be used to initialize an
#         `elm.tree.DecisionTree`.
#     """
#     G = _setup_graph_no_nodes(**kwargs)

#     G.add_node(
#         "init",
#         prompt=(
#             "We will attempt to extract structured data for this "
#             "ordinance. Let's think step by step. Does the following text "
#             "explicitly limit the {restriction} allowed for {wes_type}? "
#             "Do not infer based on other restrictions; if this particular "
#             "restriction is not explicitly mentioned then say 'No'. Pay extra "
#             "attention to clarifying text found in parentheses and footnotes. "
#             "Begin your response with either 'Yes' or 'No' and explain "
#             "your answer."
#             '\n\n"""\n{text}\n"""'
#         ),
#     )
#     G.add_edge("init", "final", condition=llm_response_starts_with_yes)

#     G.add_node(
#         "final",
#         prompt=(
#             "Now we are ready to extract structured data. Respond based "
#             "on our entire conversation so far. Return your answer in JSON "
#             "format (not markdown). Your JSON file must include exactly four "
#             'keys. The keys are "value", "units", "section", "comment". The '
#             'value of the "value" key should be a numerical value '
#             "corresponding to the {restriction} allowed for {wes_type}, or "
#             "`null` if the text does not mention such a restriction. Use our "
#             'conversation to fill out this value. The value of the "units" '
#             "key should be a string corresponding to the units for the "
#             "{restriction} allowed for {wes_type} by the text below, or "
#             "`null` if the text does not mention such a restriction. Make "
#             'sure to include any "per XXX" clauses in the units. '
#             "{SECTION_PROMPT} {COMMENT_PROMPT}"
#         ),
#     )
#     return G