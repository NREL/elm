# -*- coding: utf-8 -*-
"""ELM Ordinance Decision Tree Graph setup functions."""
import networkx as nx
from elm.ords.extraction.graphs import (llm_response_does_not_start_with_no, 
                                        llm_response_starts_with_no,
                                        llm_response_starts_with_yes)


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
        **kwargs)


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
        prompt=(
            "Does the following text mention a requirement for water well permits? "
            "Requirements should specify whether or not an application is required "
            "in order to drill a groundwater well."
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=("Is an application or permit required to drill a groundwater well in {DISTRICT_NAME}?"),
    ) 

    G.add_edge("init", "get_reqs", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_reqs", #TODO: maybe we don't need this portion
        prompt=(
            "What are the requirements the text mentions? "
        ),
    )

    G.add_edge("get_reqs", "get_exempt")

    G.add_node(
        "get_exempt",
        prompt=(
            "Are any wells exempt from the permitting process? "
        ),
    )
    
    G.add_edge("get_exempt", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly four keys. The keys are "permit_required", "requirements", '
            '"exemptions", and "explanation". '
            'The value of the "permit_required" key should be either "True" or "False" '
            'based on whether or not the conservation district requires water well permits. '
            'The value "requirements" should list the well permitting requirements if applicable. '
            'The value of the "exemptions" key should specify whether or not there are well types ' 
            'that are not subject to the permitting process. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_geothermal(**kwargs): ## TODO: finish this one --> water wells specifically for geothermal systems
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
            "Does the following text mention a requirement for geothermal systems specifically? "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        # db_query=("Does {DISTRICT_NAME} require a permit or application to drill a water well?"),
        # db_query=("Is there an application process to drill a water well in {DISTRICT_NAME}?"),
        db_query=("Is an application or permit required to drill a water well in {DISTRICT_NAME}?"),
    )

def setup_graph_gas(**kwargs): ## TODO: finish this one --> water wells specifically for oil and gas
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
            "Does the following text mention a requirement for oil and gas operations specifically? "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        # db_query=("Does {DISTRICT_NAME} require a permit or application to drill a water well?"),
        # db_query=("Is there an application process to drill a water well in {DISTRICT_NAME}?"),
        db_query=("Is an application or permit required to drill a water well in {DISTRICT_NAME}?"),
    ) 

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
        db_query=(
            "Does {DISTRICT_NAME} have daily water well extraction limits? "
            "Extraction limits might be defined as an acre-foot "
            "or a gallon limit."),
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

def setup_graph_monthly_limits(**kwargs):
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
            "Does the following text mention monthly water well extraction limits? "
            "Extraction limits may be defined as an acre-foot "
            "or a gallon limit."
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            "Does {DISTRICT_NAME} have monthly water well extraction limits? "
            "Extraction limits might be defined as an acre-foot "
            "or a gallon limit."),
    ) 

    G.add_edge("init", "get_monthly", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_monthly",
        prompt=(
            "What is the monthly extraction limit mentioned in the text? "
            "Include the units associated with the limit (example: gallons or acre-feet). "
        ),
    )

    G.add_edge("get_monthly", "final")

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
        db_query=(
            "Does {DISTRICT_NAME} have annual water well extraction limits? "
            "Extraction limits may be defined as an acre-foot "
            "or a gallon limit."
        )
    ) 

    G.add_edge("init", "get_annual", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_annual",
        prompt=(
            "What is the annual extraction limit mentioned in the text? "
            "Include the units associated with the limit (example: gallons or acre-feet). "
        ),
    )

    G.add_edge("get_annual", "final")

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
        db_query=(
            'Does {DISTRICT_NAME} have restrictions related to well spacing? ' # TODO: what exactly are we trying to capture? spacing to another well? spacing from property line?
            'Such information typically dictates how far apart two wells must be and '
            'could prohibit an individual from drilling a well with a certain distance '
            'of another well or feature. '
        )   
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
        db_query=(
            'Do the documents mention a drilling window?'
            'Such a value specifies how long after permit approval '
            'that drilling must commence. '
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
    """Setup Graph to get metering device requirements 

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
        db_query=(
            'Do the documents mention a requirement for a metering device? '
            'Metering devices are typically utilized to monitor '
            'water usage and might be required by the conservation '
            'district. '
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
        db_query=(
            'Do the documents mention a drought management plan? '
            'Drought management plans might specify actions or policies '
            'that could be implemented in the event of a drought and '
            'could impose additional restrictions on well users. '
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

    return G

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
        db_query=(
            'Do the documents mention plugging requirements specific '
            'to water wells? Plugging requirements generally detail the  '
            'steps an individual needs to take when they no longer intend to '
            'use the well. '
        )
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

    return G