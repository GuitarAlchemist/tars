DESCRIBE {
    name: "Real FLUX Learning Script"
    version: "1.0"
    author: "TARS"
}

CONFIG {
    enable_learning: true
    target_language: "fsharp"
}

PATTERN railway_oriented {
    input: any
    output: Result<'T, 'E>
}

FSHARP {
    let processWithRailway input =
        input |> validate |> Result.bind process
}