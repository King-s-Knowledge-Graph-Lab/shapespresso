import re

from loguru import logger
from typing import Any, Optional, Union

from jsonasobj import as_json
from ShExJSG import ShExJ
from pyjsg.jsglib import loads as jsg_loads

from rdflib import Graph
from rdflib.namespace import NamespaceManager

from pydantic import BaseModel
from openai.lib._pydantic import to_strict_json_schema

from pyshex.shex_evaluator import ShExEvaluator

from shapespresso.utils import NamespaceRegistry, extract_prefix_declarations
from shapespresso.parser.ShExC import ShExC


def position_start_line(shexc_text: str) -> int:
    """
    Find the line index where the ShExC start shape or first shape appears

    Args:
        shexc_text (str): ShExC text

    Returns:
        int: the line index
    """
    for idx, line in enumerate(shexc_text.split("\n")):
        if line.startswith("start") or line.startswith("<"):
            return idx
    return 0


def locate_comment(shexc_text: list[str], case: str) -> str | int:
    """
    Locate the relevant line following a comment in a ShExC text

    Args:
        shexc_text (list[str]): lines following or around the comment line
        case (str): extraction mode, either 'general' or 'constraint'

    Returns:
        str | int: the next non-empty line in document, or 0 if no suitable line is found
    """
    if case == "general":
        for line in shexc_text[1:]:
            if line:
                return line
    else:  # case == "constraint"
        line = shexc_text[0]
        if line[:line.index('#')].strip():
            return line[:line.index('#')].rstrip()
        else:
            for line in shexc_text[1:]:
                if line:
                    return line
    return 0


def remove_lines(text: str, removed_lines: list[int]) -> str:
    """
    Remove specific lines from text

    Args:
        text (str): input text
        removed_lines (list[int]): list of line numbers to be removed

    Returns:
        str: modified text with the specified lines removed
    """
    lines = text.splitlines()
    modified_lines = [line for i, line in enumerate(lines, start=1) if i not in removed_lines]
    return '\n'.join(modified_lines)


def shexc_to_shexj(
        shexc_text: str
) -> tuple[str, Optional[str], Optional[Union[NamespaceManager, Graph]], Optional[list[dict]]]:
    """
    Convert ShExC schema text to its equivalent ShExJ (JSON) representation

    Args:
        shexc_text (str): ShExC text

    Returns:
        shexj_text (str): ShExJ text
        base (Optional[str]): base URI declared in the schema (if any)
        namespaces (Optional[Union[NamespaceManager, Graph]]): parsed namespace declarations
        comments (Optional[list[dict]]): a list of extracted comment blocks with metadata
    """
    # shexj_text = as_json(parse(shexc_text))  # potential issues on recursion and choice
    # shexj_text = as_json(ShExEvaluator(schema=shexc_text)._schema)  # parse without error tracking

    # parse with error tracking
    shex_evaluator = ShExEvaluator(schema=shexc_text)
    schema = shex_evaluator._schema
    errors = shex_evaluator.errors
    if not schema and errors:
        error_lines = list(set([error["line"] for error in errors]))
        shexc_text = remove_lines(shexc_text, removed_lines=error_lines)
        shex_evaluator = ShExEvaluator(schema=shexc_text)
        if shex_evaluator._schema:
            shexj_text = as_json(shex_evaluator._schema)
        else:
            logger.error("Unable to parse the file!")
            return "", None, None, None
    elif shex_evaluator._schema:
        shexj_text = as_json(shex_evaluator._schema)
    else:
        logger.error("Unable to parse the file!")
        return "", None, None, None

    # load base, namespaces, and comments
    base = base_uri_parser_helper(shexc_text)
    namespaces = namespaces_parser_helper(shexc_text)
    comments = comment_parser_helper(shexc_text)
    return shexj_text, base, namespaces, comments


def shexj_to_shexc(
        shexj_text: str,
        base: Optional[str] = None,
        namespaces: Optional[Union[NamespaceManager, Graph]] = None,
        comments: Optional[list[dict]] = None
) -> str:
    """
    Convert ShExJ (JSON-based) schema text to ShExC (compact syntax) representation

    Args:
        shexj_text (str): ShExJ text
        base (Optional[str]): base URI to include in the ShExC output (if provided)
        namespaces (Optional[Union[NamespaceManager, Graph]]): namespace mappings to include as PREFIX declarations;
            if not provided, default namespaces will be used
        comments (Optional[list[dict]]): a list of comment metadata to insert into the schema

    Returns:
        shexc_text (str): ShExC text
    """
    if not namespaces:
        namespaces = NamespaceRegistry().create_namespace_manager()
    shex_json: ShExJ.Schema = jsg_loads(shexj_text, ShExJ)  # <class 'ShExJSG.ShExJ.Schema'>
    shexc_text = str(ShExC(shex_json, base, namespaces))
    shexc_text = insert_comments(shexc_text, comments)
    return shexc_text


def base_uri_parser_helper(shexc_text: str) -> Optional[str]:
    """
    Extract base URI from a ShExC schema if declared

    Args:
        shexc_text (str): ShExC text

    Returns:
        Optional[str]: the extracted base URI if found, otherwise None
    """
    base_pattern = r'^[Bb][Aa][Ss][Ee]\s+<(.+)>$'
    for line in shexc_text.split("\n"):
        match = re.match(base_pattern, line)
        if match:
            return match.group(1)
    return None


def namespaces_parser_helper(inputs: str | dict) -> Optional[Union[NamespaceManager, Graph]]:
    """
    Parse namespace prefix declarations

    Args:
        inputs (str, dict): a string containing namespace prefix declarations or a prefix-to-URI dictionary

    Returns:
        NamespaceManager with the parsed prefixes bound, otherwise None
    """
    g = Graph()
    if type(inputs) is str:
        prefixes = extract_prefix_declarations(inputs)
        for prefix, uri in prefixes.items():
            g.bind(prefix, uri)
    elif type(inputs) is dict:
        for prefix, uri in inputs.items():
            g.bind(prefix, uri)
    else:
        logger.error("TypeError: Incorrect inputs type for namespaces parser.")
    return NamespaceManager(g)


def comment_parser_helper(shexc_text: str) -> list[dict]:
    """
    Extract comment metadata from a ShExC text

    This helper function recognizes two types of comments:
    - general comments: lines starting with '#' before the start of the schema
    - constraint comments:
        - case 1: single-line comments starting with '#' after the schema start line
        - case 2: inline comments occurring after constraints on the same line

    Args:
        shexc_text (str): ShExC text

    Returns:
        list[dict]: a list of extracted comment blocks with metadata
    """
    comments = list()
    start_line_num = position_start_line(shexc_text)
    shexc_lines = shexc_text.split("\n")
    for idx, line in enumerate(shexc_lines):
        # general comments
        if idx < start_line_num:
            if line.strip().startswith("#"):
                comments.append({
                    "comment": line,
                    "type": "general",
                    "location": locate_comment(shexc_lines[idx:], "general")
                })
        # constraint comments
        else:
            if line.strip().startswith("#"):
                comments.append({
                    "comment": line,
                    "type": "constraint",
                    "location": locate_comment(shexc_lines[idx:], "constraint")
                })
            elif "#" in line:
                comments.append({
                    "comment": line[line.index("#"):],
                    "type": "constraint",
                    "location": locate_comment(shexc_lines[idx:], "constraint")
                })
    return comments


def insert_comments(shexc_text: str, comments: Optional[list[dict]]) -> str:
    """
    Insert comment strings back into a ShExC text at specified locations

    Args:
        shexc_text (str): ShExC text
        comments (Optional[list[dict]]): a list of extracted comment dictionaries with metadata

    Returns:
        str: ShExC text with comments inserted at their respective locations
    """
    shexc_lines = shexc_text.split("\n")
    if not comments:
        return shexc_text
    for comment in comments[::-1]:  # reverse the list during insertion since 'location' is the next line
        if comment["location"] == 0:
            shexc_lines.insert(0, comment["comment"])
            continue
        for idx, line in enumerate(shexc_lines):
            if line == comment["location"] or line.rstrip(' ;') == comment["location"]:
                if comment["type"] == "general":
                    shexc_lines.insert(idx, comment["comment"])
                else:
                    shexc_lines[idx] = line.rstrip() + '  ' + comment["comment"].lstrip()
                break
    return '\n'.join(shexc_lines)


def format_openai_json_schema(pydantic_model: type[BaseModel]) -> dict[str, Any]:
    """
    Format Pydantic model into `json_schema` type required by OpenAI API
    See source: https://github.com/openai/openai-python/blob/main/src/openai/lib/_parsing/_completions.py#L257

    Args:
        pydantic_model (BaseModel): Pydantic model

    Returns:
        (dict):
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": to_strict_json_schema(pydantic_model),
            "name": pydantic_model.__name__,
            "strict": True,
        },
    }
