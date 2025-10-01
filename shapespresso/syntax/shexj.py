from typing import Literal

from pydantic import AnyUrl, BaseModel, model_validator


class IriStem(BaseModel):
    type: Literal["IriStem"]
    stem: AnyUrl


class NodeConstraint(BaseModel):
    type: Literal["NodeConstraint"]
    nodeKind: Literal["iri", "bnode", "nonliteral", "literal"] = None
    datatype: AnyUrl = None
    values: list[AnyUrl | IriStem] = None

    @model_validator(mode='after')
    def node_constraint_types_validator(self):
        types = [self.nodeKind, self.datatype, self.values]
        if sum(constraint is not None for constraint in types) != 1:
            raise ValueError("Exactly one of 'nodeKind', 'datatype', or 'values' must be set.")
        return self


class TripleConstraint(BaseModel):
    type: Literal["TripleConstraint"]
    inverse: bool = None
    predicate: AnyUrl
    valueExpr: NodeConstraint | str
    min: int = None
    max: int = None


class ShapeExpression(BaseModel):
    type: Literal["EachOf", "OneOf"]
    expressions: list[TripleConstraint]


class Shape(BaseModel):
    type: Literal["Shape"]
    id: str
    closed: bool = None
    extra: list[AnyUrl] = None
    expression: ShapeExpression | TripleConstraint


class Schema(BaseModel):
    type: Literal["Schema"]
    start: str
    shapes: list[Shape]
