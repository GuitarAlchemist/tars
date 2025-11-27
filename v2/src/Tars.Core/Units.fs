namespace Tars.Core

open System

[<Measure>]
type token

[<Measure>]
type usd

[<Measure>]
type ms

module Units =
    let toTokens (x: int) : int<token> = x * 1<token>
    let toUsd (x: decimal) : decimal<usd> = x * 1m<usd>
    let toMs (x: float) : float<ms> = x * 1.0<ms>
