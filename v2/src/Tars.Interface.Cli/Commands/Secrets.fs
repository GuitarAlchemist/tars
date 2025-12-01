module Tars.Interface.Cli.Commands.Secrets

open System
open Tars.Security

type SecretView =
    { Name: string
      Status: string
      Value: string }

let private mask (v: string) =
    if String.IsNullOrEmpty v then "****"
    elif v.Length <= 4 then "****"
    else v.Substring(0, 2) + "****" + v.Substring(v.Length - 2)

let private fetch name =
    match CredentialVault.getSecret name with
    | Ok v -> { Name = name; Status = "Set"; Value = mask v }
    | Error _ -> { Name = name; Status = "Not Set"; Value = "n/a" }

/// List selected TARS-related secrets/envs in masked form.
/// NOTE: Does not print full values.
let run () =
    let keys =
        [ "OLLAMA_BASE_URL"
          "DEFAULT_OLLAMA_MODEL"
          "OPENWEBUI_EMAIL"
          "OPENWEBUI_PASSWORD"
          "TARS_API_KEY"
          "TARS_API_URL" ]

    let rows = keys |> List.map fetch
    printfn "TARS Secrets (masked):"
    rows |> List.iter (fun r -> printfn "  %-22s %-8s %s" r.Name r.Status r.Value)
    0
