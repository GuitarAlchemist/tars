META {
    name: "FLUX Test Script"
    version: "1.0.0"
    description: "First test of the revolutionary FLUX language"
}

REASONING {
    This is a test of the FLUX language system.
    FLUX = Functional Language Universal eXecution
    We're demonstrating multi-modal execution capabilities.
}

LANG("FSHARP") {
    printfn "🔥 Hello from F# in FLUX!"
    let x = 42
    printfn "The answer is %d" x
}

LANG("PYTHON") {
    print("🔥 Hello from Python in FLUX!")
    x = 42
    print(f"The answer is {x}")
}

DIAGNOSTIC {
    test: "Verify FLUX execution"
    validate: "Multi-language support"
}

(* This is a comment in FLUX - Revolutionary! *)
