
flux:
  name: "FunctionalRefactoring"
  description: "Convert imperative code to functional patterns"
  
  analyze:
    - detect_nesting: "Find deeply nested if statements"
    - identify_side_effects: "Locate printfn and mutable operations"
    - find_error_patterns: "Detect missing error handling"
  
  transform:
    - extract_types: "Create discriminated unions for data"
    - separate_concerns: "Split logging from business logic"
    - apply_functional_patterns: "Use map, filter, choose"
    - add_error_handling: "Wrap in Result or try-catch"
  
  validate:
    - compile_check: "Ensure code compiles"
    - behavior_preservation: "Verify same functionality"
    - performance_check: "Confirm no regression"
