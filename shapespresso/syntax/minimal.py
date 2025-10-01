from enum import Enum

from pydantic import BaseModel, AnyUrl


class ValueShape(BaseModel):
    name: str
    extra: AnyUrl
    predicate: AnyUrl
    values: list[AnyUrl]


class ValuesConstraint(BaseModel):
    values: list[AnyUrl]


class NodeKind(str, Enum):
    IRI = "iri"
    BNODE = "bnode"
    NONLITERAL = "nonliteral"
    LITERAL = "literal"


class NodeKindConstraint(BaseModel):
    node_kind: NodeKind


class NodeConstraintType(str, Enum):
    VALUE_SHAPE = "value_shape"
    NODE_KIND = "node_kind"
    VALUES_CONSTRAINT = "values_constraint"


class EXTRA(str, Enum):
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    INSTANCE_OF = "http://www.wikidata.org/prop/direct/P31"
    SUBCLASS_OF = "http://www.wikidata.org/prop/direct/P279"


class NodeConstraint(BaseModel):
    type: NodeConstraintType
    name: str = None
    extra: EXTRA = None
    predicate: EXTRA = None
    values: list[str] = None
    node_kind: NodeKind = None


class Cardinality(BaseModel):
    min: int
    max: int
