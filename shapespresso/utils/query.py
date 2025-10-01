import inspect
import sys

from loguru import logger
from urllib.error import URLError

from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError, QueryBadFormed


def endpoint_sparql_query(
        query: str,
        endpoint_url: str = 'http://localhost:1234/api/endpoint/sparql',
        mode: str = 'select'
) -> list[dict] | bool:
    """
    Execute a SPARQL query against a remote SPARQL endpoint;
    Support SELECT and ASK queries

    Args:
        query (str): SPARQL query to execute
        endpoint_url (str, optional): URL of the SPARQL endpoint;
            defaults to a local endpoint at 'http://localhost:1234/api/endpoint/sparql'
        mode (str, optional): query type — either 'select' (default) or 'ask';
            determines the format of the return value

    Returns:
        list[dict]: if mode is 'select', a list of query result rows, each as a dict {variable: value}
        bool: if mode is 'ask', the boolean result of the ASK query
        list: an empty list if the query fails due to endpoint or formatting errors
    """
    if 'wikidata' in endpoint_url:
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    else:
        sparql = SPARQLWrapper(endpoint_url)

    query = inspect.cleandoc(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        query_results = sparql.query().convert()

        if mode == 'ask':
            return query_results['boolean']
        else:
            results = list()
            for result in query_results["results"]["bindings"]:
                results.append({key: value['value'] for key, value in result.items()})
            return results
    except (EndPointInternalError, QueryBadFormed, URLError):
        logger.error(f"Error query: \n{query}")
        return list()
