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

def setup_graph_permits(**kwargs):
    """Setup Graph to get permit requirements.

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
        db_query=("Is an application or permit required to drill a groundwater "
                  "well in the {DISTRICT_NAME}?"),
    )

    G.add_edge("init", "get_reqs", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_reqs",
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

def setup_graph_extraction(**kwargs):
    """Setup Graph to get extraction requirements.

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
            "Does the following text mention a permit requirement to "
            "extract or produce water from a well? Begin your response "
            "with either 'Yes' or 'No' and explain your answer."
            "'Yes' or 'No' and explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=("Is an application or permit required to extract or "
                  "produce groundwater in the {DISTRICT_NAME}?"),
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
            'include exactly two keys. The keys are "permit_required" '
            'and "explanation". The value of the "permit_required" key '
            'should be either "True" or "False" based on whether or not '
            'the conservation district requires extraction permits. '
            'The value of the "explanation" key should be a string '
            'containing a short explanation for your choice.'
        ),
    )

    return G

def setup_graph_geothermal(**kwargs):
    """Setup Graph to get geothermal specific policies.

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
            "Does the following text mention provisions that apply to "
            "geothermal systems specifically? Begin your response with "
            "either 'Yes' or 'No' and explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=("Does {DISTRICT_NAME} implement policies that are specific "
                  "to geothermal systems?"),
    )

    G.add_edge("init", "get_reqs", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_reqs", 
        prompt=(
            "Summarize the requirements for geothermal systems mentioned in the text. "
        ),
    )

    G.add_edge("get_reqs", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "geothermal_requirements" '
            '"summary", and "explanation". The value of the "geothermal_requirements" '
            'key should be either "True" or "False" based on whether or not the '
            'conservation district has specific requirements for geothermal systems. '
            'The value "summary" should summarize the requirements mentioned above, '
            'if applicable. The value of the "explanation" key should be a string '
            'containing a short explanation for your choice.'
        ),
    )

    return G

def setup_graph_oil_and_gas(**kwargs):
    """Setup Graph to get oil and gas specific policies.

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
            "Does the following text mention provisions that apply to "
            "oil and gas operations specifically? Begin your response with "
            "either 'Yes' or 'No' and explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=("Does {DISTRICT_NAME} implement policies that are specific "
                  "to oil and gas operations?"),
    )

    G.add_edge("init", "get_reqs", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_reqs", 
        prompt=(
            "Summarize the requirements for oil and gas operations mentioned in the text. "
        ),
    )

    G.add_edge("get_reqs", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly three keys. The keys are "oil_and_gas_requirements" '
            '"summary", and "explanation". The value of the "oil_and_gas_requirements" '
            'key should be either "True" or "False" based on whether or not the '
            'conservation district has specific requirements for oil and gas operations. '
            'The value "summary" should summarize the requirements mentioned above, '
            'if applicable. The value of the "explanation" key should be a string '
            'containing a short explanation for your choice.'
        ),
    )

    return G

def setup_graph_limits(**kwargs):
    """Setup Graph to get extraction limits.

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
            "Does the following text mention {interval} water well production, "
            "extraction, or withdrawal limits? Limits may be defined as an "
            "acre-foot or a gallon limit. Ensure these limits are specific "
            "to {interval} production and disregard limits related to other "
            "time periods. Begin your response with either 'Yes' or 'No' and "
            "explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            "Does {DISTRICT_NAME} have {interval} water well production, "
            "extraction, or withdrawal limits? "
    ),
    )

    G.add_edge("init", "get_application", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_application",
        prompt=(
            "Are the limits mentioned in text specific to a permit type (such as "
            "limited production), well, or aquifer?"
        ),
    )

    G.add_edge("get_application", "get_type")


    G.add_node(
        "get_type",
        prompt=(
            "Does the text mention an explicit, numerical limit such as a gallons "
            "or acre-foot figure? Begin your response with either 'Yes' or 'No' "
            "and explain your answer. Some limits maybe permit specific in which "
            "case your answer should be 'No'."
        ),
    )

    G.add_edge("get_type", "get_limit", condition=llm_response_starts_with_yes)
    G.add_edge("get_type", "permit_final", condition=llm_response_starts_with_no)


    G.add_node(
        "get_limit",
        prompt=(
            "What is the {interval} extraction limit mentioned in the text? "
            "Include the units associated with the limit (example: gallons or acre-feet)."
        ),
    )

    G.add_node(
        "permit_final",
        prompt=(
            'Summarize the production limit described in the text and '
            'respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly two keys. The keys are "limit_type" '
            'and "explanation". Based on the conversation so far, the '
            'value of the "limit_type" key should be "permit specific". '
            'The value of the "explanation" key should be a string containing '
            'a short explanation for your choice.'
        ),
    )

    G.add_edge("get_limit", "final")

    G.add_node(
        "final",
        prompt=(
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly five keys. The keys are "limit_type", '
            '"extraction_limit", "units", "application", and "explanation". '
            'The value of the "limit_type" key should be either "explicit" '
            'or "permit specific". The value of the "extraction_limit" key '
            'should be the numerical value(s) associated with the extraction '
            'limit, if there are multiple values based on permit, well, or aquifer, '
            'include all the values. The "units" key should describe the units '
            'associated with the extraction limit. The "application" key should '
            'describe whether the limit is specific to a permit type, well, or '
            'aquifer. The value of the "explanation" key should be a string '
            'containing a short explanation for your choice.'
        ),
    )

    return G

def setup_graph_well_spacing(**kwargs):
    """Setup Graph to get well spacing requirements. 

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
            "Does the following text mention how far a new well must be from "
            "another well? Well spacing refers to how far apart two wells must be and "
            "could prohibit an individual from drilling a well within a certain distance "
            "of a different well. Focus only on spacing requirements between wells and "
            "ignore spacing that is specific to other features such as property lines "
            "or septic systems. Begin your response with either 'Yes' or 'No' and "
            "explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            'Does {DISTRICT_NAME} have restrictions related to well spacing or '
            'a required distance between wells? '
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

    G.add_edge("get_spacing", "get_wells")

    G.add_node(
        "get_wells",
        prompt=(
            "Do the spacing requirements mentioned apply specifically to "
            "the required distance between two water wells? Begin your response "
            "with either 'Yes' or 'No' and explain your answer."
        ),
    )

    G.add_edge("get_wells", "get_qualifier", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_qualifier",
        prompt=(
            "Is the spacing limit dependent upon well characteristics "
            "such as depth or production capability?"
            "If so, include that metric in the response (example: gallons per minute). "
        ),
    )

    G.add_edge("get_qualifier", "final")

    G.add_node(
        "final",
        prompt=(
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly four keys. The keys are "spacing", "units", '
            '"qualifier", and "explanation". The value of the "spacing" key '
            'should be the numerical value specifying the required distance '
            'betweeen wells, focus on spacing between wells only and ignore '
            'spacing requirements for other types of infrastructure. '
            'The "units" key should describe the units associated with the spacing. '
            'The value of the "qualifier" key should be the well characteristic '
            'metric that determines the spacing, if applicable. The value of the '
            '"explanation" key should be a string containing a short explanation '
            'for your choice.'
        ),
    )

    return G

def setup_graph_time(**kwargs):
    """Setup Graph to get drilling window restrictions.

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
            'In {DISTRICT_NAME}, how long after permit approval '
            'must drilling must commence? '
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
            'and "explanation". The value of the "time_period" key should '
            'be the numerical value associated with the drilling window the '
            '"units" key should describe the units associated with '
            'the drilling window. The value of the "explanation" key should '
            'be a string containing a short explanation for your choice.'
        ),
    )

    return G

def setup_graph_metering_device(**kwargs):
    """Setup Graph to get metering device requirements. 

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
            'Does the following text mention a requirement for a metering device? '
            'Metering devices are typically utilized to monitor water usage '
            'and might be required by the conservation district. '
            'Begin your response with either "Yes" or "No" and explain your '
            'answer.'
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            'Is a metering device that monitors water usage required in {DISTRICT_NAME}? '
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
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly two keys. The keys are "device" and '
            'and "explanation". The value of the "device" key should be '
            'a boolean value with the the value "True" if a device is '
            'required and "False" if a device is not required. The value '
            'of the "explanation" key should be a string containing a '
            'short explanation for your choice.'
        ),
    )

    return G

def setup_graph_drought(**kwargs):
    """Setup Graph to get district drought management plan.

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
            'Does the following text mention a drought management or contingency plan? '
            'Drought management plans are plans set forth by conservation district '
            'that specify actions or policies that could be implemented in the event '
            'of a drought such as imposing additional restrictions on well users. '
            'Begin your response with either "Yes" or "No" and explain your '
            'answer.'
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            'Does {DISTRICT_NAME} have a drought management or contingency plan?'
        ),
    )

    G.add_edge("init", "get_plan", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_plan",
        prompt=(
            "Summarize the drought management plan mentioned in the text? "
        ),
    )

    G.add_edge("get_plan", "final")

    G.add_node(
        "final",
        prompt=(
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly two keys. The keys are "drought_plan" and '
            'and "explanation". The value of the "drought_plan" key should '
            'be a boolean value with the the value "True" if a drought '
            'management plan is in place and "False" if not. The value '
            'of the "explanation" key should be a string containing a '
            'short explanation for your choice.'
        ),
    )

    return G

def setup_graph_contingency(**kwargs):
    """Setup Graph to get permit holder drought management plan
    requirements. 

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
            "Does the following text mention a requirement for well owners "
            "to develop a drought management or contingency plan? "
            "Drought contingency plans might specify actions "
            "that well owners and users should take in the event of a drought. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            'Does {DISTRICT_NAME} require well owners, users, or applicants '
            'to develop a drought management or contingency plan?'
        ),
    )

    G.add_edge("init", "get_plan", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_plan",
        prompt=(
            "Summarize the drought management plan requirements mentioned in the text?"
        ),
    )

    G.add_edge("get_plan", "final")
    G.add_edge("init", "final", condition=llm_response_starts_with_no)

    G.add_node(
        "final",
        prompt=(
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly two keys. The keys are "drought_plan" '
            'and "explanation". The value of the "drought_plan" key should '
            'be a boolean value with the the value "True" if a well owner '
            'is required to develop a drought management plan. The value '
            'of the "explanation" key should be a string containing a short '
            'explanation for your choice.'
        ),
    )

    return G

def setup_graph_plugging_reqs(**kwargs):
    """Setup Graph to get water well plugging requirements.

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
            'Does {DISTRICT_NAME} implement plugging requirements specific '
            'to water wells?'
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
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly two keys. The keys are "plugging_requirements" and '
            '"explanation". The value of the "plugging_requirements" key '
            'should be a boolean value with the the value "True" if the '
            'conservation district implements plugging requirements. '
            'The value of the "explanation" key should be a string containing a '
            'short explanation for your choice.'
        ),
    )

    return G

def setup_graph_external_transfer(**kwargs):
    """Setup Graph to get external transfer restrictions.

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
            'Does the following text mention restrictions or costs related to '
            'the external transport or export of water? External transport '
            'refers to cases in which well owners sell or transport water to '
            'a location outside of the district boundaries. '
            'Begin your response with either "Yes" or "No" and explain your '
            'answer.'
            '\n\n"""\n{text}\n"""'
            ),
        db_query=(
            'Does {DISTRICT_NAME} implement restrictions or costs related to '
            'the external transport or export of water? External transport refers to '
            'cases in which well owners sell or transport water to '
            'a location outside of the district boundaries. '
        )
    )

    G.add_edge("init", "get_restrictions", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_restrictions",
        prompt=(
            'What are the restrictions the text mentions?'
        ),
    )

    G.add_edge("get_restrictions", "get_permit")

    G.add_node(
        "get_permit",
        prompt=(
            'Is there a permit application fee for the external transfer of water? '
        ),
    )

    G.add_edge("get_permit", "get_cost")

    G.add_node(
        "get_cost",
        prompt=(
            'Is there a cost a associated with the external transfer of water? "'
            'Focus on costs that are specific to the transfer itself '
            '(dollars per gallon or acre-foot figures) rather than '
            'permit costs. Begin your answer with either "Yes" or "No" '
            'and explain your answer.'
        ),
    )

    G.add_edge("get_cost", "get_cost_amount", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_cost_amount",
        prompt=(
            'What is the dollar amount associated with the external transfer of water? '
            'Include the units associated with the cost (example: dollars per gallon). '
            'If the text does not mention a specific cost, then respond "Permit Specific" '
            'or with the rate structure mentioned in the text.'
        ),
    )

    G.add_edge("get_cost", "final", condition=llm_response_starts_with_no)
    G.add_edge("get_cost_amount", "final")

    G.add_node(
        "final",
        prompt=(
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include at least four keys. The keys are "transfer_restrictions", "requirements", '
            '"cost", and "explanation". The value of the "transfer_restrictions" key '
            'should be either "True" or "False" based on whether or not the conservation '
            'district restricts water transfer. The value "restrictions" should list the '
            'water transfer restrictions if applicable. The value of the "cost" key should '
            'either be "True" or "False" based on whether or not there are costs '
            'associated with the external transfer of water. If there is a specific cost mentioned '
            'the "cost_amount" key should reflect the dollar amount associated with the transfer '
            'of water. The value of the "explanation" key should be a string containing a short '
            'explanation for your choice.'
        ),
    )

    return G

def setup_graph_production_reporting(**kwargs):
    """Setup Graph to get production reporting requirements.

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
            'Does the following text mention a requirement for production reporting? '
            'Production reporting typically refers to the amount of water extracted '
            'from a well and may be subject to specific regulations. '
            'Begin your response with either "Yes" or "No" and explain your '
            'answer.'
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            'Does {DISTRICT_NAME} require production reporting for water wells? '
        ),

    )

    G.add_edge("init", "get_reporting", condition=llm_response_starts_with_yes)
    G.add_edge("init", "final", condition=llm_response_starts_with_no)


    G.add_node(
        "get_reporting",
        prompt=(
            'What are the requirements mentioned in the text? '
        ),
    )

    G.add_edge("get_reporting", "final")

    G.add_node(
        "final",
        prompt=(
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly two keys. The keys are "production_reporting" and '
            'and "explanation". The value of the "production_reporting" key '
            'should be a boolean value with the the value "True" if production '
            'reporting is required. The value of the "explanation" should be '
            'a string containing a short explanation for your choice.'
        ),
    )

    return G

def setup_graph_production_cost(**kwargs):
    """Setup Graph to get production cost.

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
            'Does the following text mention costs related to the production or '
            'extraction of water from wells? Such a cost would specify a cost per '
            'unit volume of water extracted. Focus on costs that are specific to '
            'the production or extraction itself (dollars per gallon or acre-foot '
            'figures) rather than permit costs. Begin your response with '
            'either "Yes" or "No" and explain your answer.'
            '\n\n"""\n{text}\n"""'
            ),
        db_query=(
            'Does {DISTRICT_NAME} charge well operators or owners to '
            'produce or extract water from a groundwater well?'
            'This is likely a dollar amount per gallon or acre-foot fee.'
        )
    )

    G.add_edge("init", "get_type", condition=llm_response_starts_with_yes)

    G.add_node(
        "get_type",
        prompt=(
            'Does the production cost apply broadly to all well users or '
            'does it vary by permit type, user, or other factors?'
        ),
    )

    G.add_edge("get_type", "get_cost_amount")

    G.add_node(
        "get_cost_amount",
        prompt=(
            'What is the dollar amount associated with the external transfer of water? '
            'Include the units associated with the cost (example: dollars per gallon). '
            'If the text does not mention a specific cost, then respond "Permit Specific" '
            'or with the rate structure mentioned in the text.'
        ),
    )

    G.add_edge("get_cost_amount", "final")

    G.add_node(
        "final",
        prompt=(
            'Respond based on our entire conversation so far. Return your '
            'answer in JSON format (not markdown). Your JSON file must '
            'include exactly four keys. The keys are "cost", "production_cost", '
            '"cost_units", and "explanation". The value of the "cost" key should '
            'be "True" if there are costs associated with the production '
            'of water. The value of the "production_cost" key should reflect the '
            'dollar amount associated with the production of water or "Permit Specific" '
            'if the cost varies. The value of the "cost_units" key should be a string '
            'containing the units associated with the cost (example: dollars per gallon). '
            'The value of the "explanation" key should be a string containing a short '
            'explanation for your choice.'
        ),
    )

    return G

def setup_graph_setback_features(**kwargs):
    """Setup Graph to get setback restrictions.

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
            "Does the following text mention restrictions related to how "
            "close a groundwater well can be located relative to property "
            "lines, buildings, septic systems, or 'other sources of "
            "contamination'?"
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            'Does {DISTRICT_NAME} restrict how close a groundwater well can be '
            'located relative to property lines, buildings, septic systems, or '
            '"other sources of contamination"? '
        ),

    )

    G.add_edge("init", "get_restrictions", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_restrictions",
        prompt=(
            "What are the restrictions mentioned in the text? "
        ),
    )

    G.add_edge("get_restrictions", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "setback_restrictions" and '
            'and "explanation". The value of the "setback_restrictions" key '
            'should be a boolean value with the value "True" if setback '
            'restrictions are imposed. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "choice."
        ),
    )

    return G

def setup_graph_redrilling(**kwargs):
    """Setup Graph to get redrilling restrictions.

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
            "Does the following text mention restrictions related to redrilling "
            "water wells? Redrilling refers to the process of deepening or widening "
            "an existing well to improve access to groundwater. Begin your response "
            "with either 'Yes' or 'No' and explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
        db_query=(
            'Does {DISTRICT_NAME} implement restrictions related to redrilling '
            'or deepening water wells?'
        )
    )

    G.add_edge("init", "get_redrilling", condition=llm_response_starts_with_yes)


    G.add_node(
        "get_redrilling",
        prompt=(
            "What are the redrilling restrictions mentioned in the text? "
        ),
    )

    G.add_edge("get_redrilling", "final")

    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "redrilling_restrictions" and '
            'and "explanation". The value of the "redrilling_restrictions" key '
            'should be a boolean value with the the value "True" if the conservation '
            'district implements redrilling restrictions. The value of the '
            '"explanation" key should be a string containing a short explanation '
            'for your choice.'
        ),
    )

    return G
