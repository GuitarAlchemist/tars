namespace Tars.Core

open System

[<Measure>]
type token

[<Measure>]
type usd

[<Measure>]
type ms

[<Measure>]
type bytes

[<Measure>]
type cycles

[<Measure>]
type attention

[<Measure>]
type nodes

[<Measure>]
type joules

[<Measure>]
type requests

module Units =
    let toTokens (x: int) : int<token> = x * 1<token>
    let toUsd (x: decimal) : decimal<usd> = x * 1m<usd>
    let toMs (x: float) : float<ms> = x * 1.0<ms>
    let toBytes (x: int64) : int64<bytes> = x * 1L<bytes>
    let toCycles (x: int64) : int64<cycles> = x * 1L<cycles>
    let toAttention (x: float) : float<attention> = x * 1.0<attention>
    let toNodes (x: int) : int<nodes> = x * 1<nodes>
    let toJoules (x: float) : float<joules> = x * 1.0<joules>
    let toRequests (x: int) : int<requests> = x * 1<requests>
