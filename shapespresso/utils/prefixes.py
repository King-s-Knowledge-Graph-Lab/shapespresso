import re

from rdflib import Graph
from rdflib.namespace import NamespaceManager, Namespace


class NamespaceRegistry:
    DEFAULTS = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "yago": "http://yago-knowledge.org/resource/",
        "owl": "http://www.w3.org/2002/07/owl#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "geo": "http://www.opengis.net/ont/geosparql#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "schema": "http://schema.org/",
        "wd": "http://www.wikidata.org/entity/",
        "wdt": "http://www.wikidata.org/prop/direct/",
        "wikibase": "http://wikiba.se/ontology#",
        "bd": "http://www.bigdata.com/rdf#",
    }

    def __init__(self, custom_namespaces: dict[str, str] = None):
        """
        Initialize the namespace registry

        Args:
            custom_namespaces (dict[str, str], optional): namespaces to override or extend defaults
        """
        self.namespaces = self.DEFAULTS.copy()
        if custom_namespaces:
            self.namespaces.update(custom_namespaces)

    def as_dict(self) -> dict[str, str]:
        """
        Returns:
            dict[str, str]: namespace dictionary
        """
        return self.namespaces

    def create_namespace_manager(self) -> NamespaceManager:
        """
        Create an RDFLib NamespaceManager with the stored namespaces

        Returns:
            NamespaceManager: RDFLib-compatible namespace manager
        """
        g = Graph()
        namespace_manager = NamespaceManager(g)
        for prefix, url in self.namespaces.items():
            namespace_manager.bind(prefix, Namespace(url), override=True)
        return namespace_manager


def extract_prefix_declarations(text: str) -> dict[str, str]:
    """
    Extract all PREFIX declarations from text

    Args:
        text: input text

    Returns:
        dict: prefix -> namespace URI
    """
    pattern = r'PREFIX\s+([a-zA-Z_][\w\-]*)\s*:\s*<([^>]*)>'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return {prefix: uri for prefix, uri in matches}


def add_prefixes(text: str, prefixes: dict[str, str] = None) -> str:
    """
    Add missing PREFIX declarations to a SPARQL query or RDF-like text

    Args:
        text (str): text (e.g., query)
        prefixes (dict[str, str], optional): predefined prefix-to-URI mappings

    Returns:
        input text with any missing or conflicting PREFIX declarations added
    """
    # prefixes extracted from text
    namespaces = NamespaceRegistry(prefixes)
    prefix_pattern = r'\b([a-zA-Z_][\w\-]*)\:'
    prefix_matches = re.findall(prefix_pattern, text)
    text_prefixes = {key: value for key, value in namespaces.as_dict().items() if key in set(prefix_matches)}
    # prefixes defined in prefix declarations
    decl_prefixes = extract_prefix_declarations(text)
    # find emergent prefixes
    emergent_prefixes = {key: value for key, value in text_prefixes.items() if decl_prefixes.get(key) != value}
    # construct declarations
    declarations = [f"PREFIX {key}: <{value}>" for key, value in emergent_prefixes.items()]
    # add prefixes
    text = '\n'.join(declarations) + '\n' + text
    return text


def prefix_substitute(url: str) -> str:
    """
    Substitute a full URI with a prefixed form

    Args:
        url (str): the full URI

    Returns:
        url (str): the shortened prefixed URI or the original one if enclosed in angle brackets
    """
    namespaces = NamespaceRegistry()
    if url.startswith("http"):
        namespaces = sorted(namespaces.as_dict().items(), key=lambda x: len(x[1]), reverse=True)
        for key, value in namespaces:
            if value in url:
                url = url.replace(value, f"{key}:")
                return url
        return f"<{url}>"
    else:
        return url
