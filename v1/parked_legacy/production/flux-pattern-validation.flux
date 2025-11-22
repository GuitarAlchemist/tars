PATTERN validation_chain {
    let validateNotEmpty str = 
        if String.IsNullOrWhiteSpace(str) then Error "Empty" else Ok str
    let validateLength max str =
        if str.Length > max then Error "Too long" else Ok str
}