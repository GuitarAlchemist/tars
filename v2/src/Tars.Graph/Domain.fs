namespace Tars.Graph

open System

type GraphNode =
    | Concept of name: string
    | Agent of id: string
    | File of path: string
    | Task of id: string
    | Belief of id: string * content: string

type GraphEdge =
    | RelatesTo of weight: float
    | CreatedBy
    | DependsOn
    | Solves
    | HasBelief
    | IsA
