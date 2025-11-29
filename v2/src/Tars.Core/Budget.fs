namespace Tars.Core

open System

/// Represents the cost of an operation
/// This type forms a Commutative Monoid under addition, with Zero as the identity.
type Cost =
    { Tokens: int<token>
      Money: decimal<usd>
      Duration: float<ms>
      CallCount: int<requests>
      Ram: int64<bytes>
      Vram: int64<bytes>
      Disk: int64<bytes>
      Network: int64<bytes>
      Cpu: int64<cycles>
      Attention: float<attention>
      Nodes: int<nodes>
      Energy: float<joules>
      Custom: Map<string, float> }

    static member Zero =
        { Tokens = 0<token>
          Money = 0m<usd>
          Duration = 0.0<ms>
          CallCount = 0<requests>
          Ram = 0L<bytes>
          Vram = 0L<bytes>
          Disk = 0L<bytes>
          Network = 0L<bytes>
          Cpu = 0L<cycles>
          Attention = 0.0<attention>
          Nodes = 0<nodes>
          Energy = 0.0<joules>
          Custom = Map.empty }

    static member (+)(a: Cost, b: Cost) =
        let mergeMaps m1 m2 =
            (m1, m2)
            ||> Map.fold (fun acc k v ->
                let existing = acc |> Map.tryFind k |> Option.defaultValue 0.0
                acc |> Map.add k (existing + v))

        { Tokens = a.Tokens + b.Tokens
          Money = a.Money + b.Money
          Duration = a.Duration + b.Duration
          CallCount = a.CallCount + b.CallCount
          Ram = a.Ram + b.Ram
          Vram = a.Vram + b.Vram
          Disk = a.Disk + b.Disk
          Network = a.Network + b.Network
          Cpu = a.Cpu + b.Cpu
          Attention = a.Attention + b.Attention
          Nodes = a.Nodes + b.Nodes
          Energy = a.Energy + b.Energy
          Custom = mergeMaps a.Custom b.Custom }

/// Represents the budget limits for a workflow
type Budget =
    { MaxTokens: int<token> option
      MaxMoney: decimal<usd> option
      MaxDuration: float<ms> option
      MaxCalls: int<requests> option
      MaxRam: int64<bytes> option
      MaxVram: int64<bytes> option
      MaxDisk: int64<bytes> option
      MaxNetwork: int64<bytes> option
      MaxCpu: int64<cycles> option
      MaxAttention: float<attention> option
      MaxNodes: int<nodes> option
      MaxEnergy: float<joules> option
      MaxCustom: Map<string, float> }

    static member Infinite =
        { MaxTokens = None
          MaxMoney = None
          MaxDuration = None
          MaxCalls = None
          MaxRam = None
          MaxVram = None
          MaxDisk = None
          MaxNetwork = None
          MaxCpu = None
          MaxAttention = None
          MaxNodes = None
          MaxEnergy = None
          MaxCustom = Map.empty }

/// Manages resource consumption against a budget
type BudgetGovernor(budget: Budget) =
    let mutable consumed = Cost.Zero
    let lockObj = obj ()

    member this.Consumed = consumed

    /// Returns the remaining budget for each dimension.
    /// Returns None if the dimension is unbounded.
    member this.Remaining: Budget =
        lock lockObj (fun () ->
            let remainingCustom =
                budget.MaxCustom
                |> Map.map (fun k max ->
                    let used = consumed.Custom |> Map.tryFind k |> Option.defaultValue 0.0
                    max - used)

            { MaxTokens = budget.MaxTokens |> Option.map (fun m -> m - consumed.Tokens)
              MaxMoney = budget.MaxMoney |> Option.map (fun m -> m - consumed.Money)
              MaxDuration = budget.MaxDuration |> Option.map (fun m -> m - consumed.Duration)
              MaxCalls = budget.MaxCalls |> Option.map (fun m -> m - consumed.CallCount)
              MaxRam = budget.MaxRam |> Option.map (fun m -> m - consumed.Ram)
              MaxVram = budget.MaxVram |> Option.map (fun m -> m - consumed.Vram)
              MaxDisk = budget.MaxDisk |> Option.map (fun m -> m - consumed.Disk)
              MaxNetwork = budget.MaxNetwork |> Option.map (fun m -> m - consumed.Network)
              MaxCpu = budget.MaxCpu |> Option.map (fun m -> m - consumed.Cpu)
              MaxAttention = budget.MaxAttention |> Option.map (fun m -> m - consumed.Attention)
              MaxNodes = budget.MaxNodes |> Option.map (fun m -> m - consumed.Nodes)
              MaxEnergy = budget.MaxEnergy |> Option.map (fun m -> m - consumed.Energy)
              MaxCustom = remainingCustom })

    /// Attempts to consume the specified cost.
    /// Returns Ok() if successful, or Error(reason) if the budget would be exceeded.
    member this.TryConsume(cost: Cost) : Result<unit, string> =
        lock lockObj (fun () ->
            let newTotal = consumed + cost

            let check (limit: 'a option) (current: 'a) (name: string) =
                match limit with
                | Some max when current > max ->
                    Error $"{name} budget exceeded. Max: {max}, Requested Total: {current}"
                | _ -> Ok()

            let r1 = check budget.MaxTokens newTotal.Tokens "Tokens"
            let r2 = check budget.MaxMoney newTotal.Money "Money"
            let r3 = check budget.MaxDuration newTotal.Duration "Duration"
            let r4 = check budget.MaxCalls newTotal.CallCount "Calls"
            let r5 = check budget.MaxRam newTotal.Ram "RAM"
            let r6 = check budget.MaxVram newTotal.Vram "VRAM"
            let r7 = check budget.MaxDisk newTotal.Disk "Disk"
            let r8 = check budget.MaxNetwork newTotal.Network "Network"
            let r9 = check budget.MaxCpu newTotal.Cpu "CPU"
            let r10 = check budget.MaxAttention newTotal.Attention "Attention"
            let r11 = check budget.MaxNodes newTotal.Nodes "Nodes"
            let r12 = check budget.MaxEnergy newTotal.Energy "Energy"

            // Check Custom
            let customErrors =
                budget.MaxCustom
                |> Map.fold
                    (fun errs k max ->
                        let current = newTotal.Custom |> Map.tryFind k |> Option.defaultValue 0.0

                        if current > max then
                            errs
                            @ [ $"Custom '{k}' budget exceeded. Max: {max}, Requested Total: {current}" ]
                        else
                            errs)
                    []

            match r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 with
            | Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok() when customErrors.IsEmpty ->
                consumed <- newTotal
                Ok()
            | Error e, _, _, _, _, _, _, _, _, _, _, _ -> Error e
            | _, Error e, _, _, _, _, _, _, _, _, _, _ -> Error e
            | _, _, Error e, _, _, _, _, _, _, _, _, _ -> Error e
            | _, _, _, Error e, _, _, _, _, _, _, _, _ -> Error e
            | _, _, _, _, Error e, _, _, _, _, _, _, _ -> Error e
            | _, _, _, _, _, Error e, _, _, _, _, _, _ -> Error e
            | _, _, _, _, _, _, Error e, _, _, _, _, _ -> Error e
            | _, _, _, _, _, _, _, Error e, _, _, _, _ -> Error e
            | _, _, _, _, _, _, _, _, Error e, _, _, _ -> Error e
            | _, _, _, _, _, _, _, _, _, Error e, _, _ -> Error e
            | _, _, _, _, _, _, _, _, _, _, Error e, _ -> Error e
            | _, _, _, _, _, _, _, _, _, _, _, Error e -> Error e
            | _ -> Error(String.concat "; " customErrors))

    /// Consumes the specified cost, updating the state even if the budget is exceeded.
    /// Returns Ok() if within budget, or Error(reason) if exceeded.
    member this.Consume(cost: Cost) : Result<unit, string> =
        lock lockObj (fun () ->
            let newTotal = consumed + cost
            consumed <- newTotal

            let check (limit: 'a option) (current: 'a) (name: string) =
                match limit with
                | Some max when current > max ->
                    Error $"{name} budget exceeded. Max: {max}, Requested Total: {current}"
                | _ -> Ok()

            let r1 = check budget.MaxTokens newTotal.Tokens "Tokens"
            let r2 = check budget.MaxMoney newTotal.Money "Money"
            let r3 = check budget.MaxDuration newTotal.Duration "Duration"
            let r4 = check budget.MaxCalls newTotal.CallCount "Calls"
            let r5 = check budget.MaxRam newTotal.Ram "RAM"
            let r6 = check budget.MaxVram newTotal.Vram "VRAM"
            let r7 = check budget.MaxDisk newTotal.Disk "Disk"
            let r8 = check budget.MaxNetwork newTotal.Network "Network"
            let r9 = check budget.MaxCpu newTotal.Cpu "CPU"
            let r10 = check budget.MaxAttention newTotal.Attention "Attention"
            let r11 = check budget.MaxNodes newTotal.Nodes "Nodes"
            let r12 = check budget.MaxEnergy newTotal.Energy "Energy"

            let customErrors =
                budget.MaxCustom
                |> Map.fold
                    (fun errs k max ->
                        let current = newTotal.Custom |> Map.tryFind k |> Option.defaultValue 0.0

                        if current > max then
                            errs
                            @ [ $"Custom '{k}' budget exceeded. Max: {max}, Requested Total: {current}" ]
                        else
                            errs)
                    []

            match r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 with
            | Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok() when customErrors.IsEmpty -> Ok()
            | Error e, _, _, _, _, _, _, _, _, _, _, _ -> Error e
            | _, Error e, _, _, _, _, _, _, _, _, _, _ -> Error e
            | _, _, Error e, _, _, _, _, _, _, _, _, _ -> Error e
            | _, _, _, Error e, _, _, _, _, _, _, _, _ -> Error e
            | _, _, _, _, Error e, _, _, _, _, _, _, _ -> Error e
            | _, _, _, _, _, Error e, _, _, _, _, _, _ -> Error e
            | _, _, _, _, _, _, Error e, _, _, _, _, _ -> Error e
            | _, _, _, _, _, _, _, Error e, _, _, _, _ -> Error e
            | _, _, _, _, _, _, _, _, Error e, _, _, _ -> Error e
            | _, _, _, _, _, _, _, _, _, Error e, _, _ -> Error e
            | _, _, _, _, _, _, _, _, _, _, Error e, _ -> Error e
            | _, _, _, _, _, _, _, _, _, _, _, Error e -> Error e
            | _ -> Error(String.concat "; " customErrors))

    /// Checks if the specified cost can be afforded without consuming it.
    member this.CanAfford(cost: Cost) : bool =
        lock lockObj (fun () ->
            let newTotal = consumed + cost

            let check (limit: 'a option) (current: 'a) =
                match limit with
                | Some max when current > max -> false
                | _ -> true

            let r1 = check budget.MaxTokens newTotal.Tokens
            let r2 = check budget.MaxMoney newTotal.Money
            let r3 = check budget.MaxDuration newTotal.Duration
            let r4 = check budget.MaxCalls newTotal.CallCount
            let r5 = check budget.MaxRam newTotal.Ram
            let r6 = check budget.MaxVram newTotal.Vram
            let r7 = check budget.MaxDisk newTotal.Disk
            let r8 = check budget.MaxNetwork newTotal.Network
            let r9 = check budget.MaxCpu newTotal.Cpu
            let r10 = check budget.MaxAttention newTotal.Attention
            let r11 = check budget.MaxNodes newTotal.Nodes
            let r12 = check budget.MaxEnergy newTotal.Energy

            let customOk =
                budget.MaxCustom
                |> Map.forall (fun k max ->
                    let current = newTotal.Custom |> Map.tryFind k |> Option.defaultValue 0.0
                    current <= max)

            r1
            && r2
            && r3
            && r4
            && r5
            && r6
            && r7
            && r8
            && r9
            && r10
            && r11
            && r12
            && customOk)

    /// Helper to consume just tokens
    member this.TryConsumeTokens(tokens: int) =
        this.TryConsume
            { Cost.Zero with
                Tokens = tokens * 1<token> }

    /// Helper to consume just a call
    member this.TryConsumeCall() =
        this.TryConsume
            { Cost.Zero with
                CallCount = 1<requests> }

    /// Allocates a portion of the remaining budget to a new child governor.
    /// The allocated amount is immediately consumed from this governor.
    member this.Allocate(childBudget: Budget) : Result<BudgetGovernor, string> =
        lock lockObj (fun () ->
            // 1. Validate: Child cannot be unbounded if parent is bounded
            let validate (pLimit: 'a option) (cLimit: 'a option) name =
                match pLimit, cLimit with
                | Some _, None -> Error $"{name} must be bounded because parent is bounded"
                | _ -> Ok()

            let v1 = validate budget.MaxTokens childBudget.MaxTokens "Tokens"
            let v2 = validate budget.MaxMoney childBudget.MaxMoney "Money"
            let v3 = validate budget.MaxDuration childBudget.MaxDuration "Duration"
            let v4 = validate budget.MaxCalls childBudget.MaxCalls "Calls"
            let v5 = validate budget.MaxRam childBudget.MaxRam "RAM"
            let v6 = validate budget.MaxVram childBudget.MaxVram "VRAM"
            let v7 = validate budget.MaxDisk childBudget.MaxDisk "Disk"
            let v8 = validate budget.MaxNetwork childBudget.MaxNetwork "Network"
            let v9 = validate budget.MaxCpu childBudget.MaxCpu "CPU"
            let v10 = validate budget.MaxAttention childBudget.MaxAttention "Attention"
            let v11 = validate budget.MaxNodes childBudget.MaxNodes "Nodes"
            let v12 = validate budget.MaxEnergy childBudget.MaxEnergy "Energy"

            // Custom validation
            let customErrors =
                budget.MaxCustom
                |> Map.fold
                    (fun errs k _ ->
                        if not (childBudget.MaxCustom.ContainsKey k) then
                            errs @ [ $"Custom '{k}' must be bounded because parent is bounded" ]
                        else
                            errs)
                    []

            match v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12 with
            | Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok(), Ok() when customErrors.IsEmpty ->
                // 2. Calculate cost of allocation
                let cost =
                    { Tokens = childBudget.MaxTokens |> Option.defaultValue 0<token>
                      Money = childBudget.MaxMoney |> Option.defaultValue 0m<usd>
                      Duration = childBudget.MaxDuration |> Option.defaultValue 0.0<ms>
                      CallCount = childBudget.MaxCalls |> Option.defaultValue 0<requests>
                      Ram = childBudget.MaxRam |> Option.defaultValue 0L<bytes>
                      Vram = childBudget.MaxVram |> Option.defaultValue 0L<bytes>
                      Disk = childBudget.MaxDisk |> Option.defaultValue 0L<bytes>
                      Network = childBudget.MaxNetwork |> Option.defaultValue 0L<bytes>
                      Cpu = childBudget.MaxCpu |> Option.defaultValue 0L<cycles>
                      Attention = childBudget.MaxAttention |> Option.defaultValue 0.0<attention>
                      Nodes = childBudget.MaxNodes |> Option.defaultValue 0<nodes>
                      Energy = childBudget.MaxEnergy |> Option.defaultValue 0.0<joules>
                      Custom = childBudget.MaxCustom }

                // 3. Try to consume from parent
                match this.TryConsume(cost) with
                | Ok() -> Ok(new BudgetGovernor(childBudget))
                | Error e -> Error $"Allocation failed: {e}"
            | Error e, _, _, _, _, _, _, _, _, _, _, _ -> Error e
            | _, Error e, _, _, _, _, _, _, _, _, _, _ -> Error e
            | _, _, Error e, _, _, _, _, _, _, _, _, _ -> Error e
            | _, _, _, Error e, _, _, _, _, _, _, _, _ -> Error e
            | _, _, _, _, Error e, _, _, _, _, _, _, _ -> Error e
            | _, _, _, _, _, Error e, _, _, _, _, _, _ -> Error e
            | _, _, _, _, _, _, Error e, _, _, _, _, _ -> Error e
            | _, _, _, _, _, _, _, Error e, _, _, _, _ -> Error e
            | _, _, _, _, _, _, _, _, Error e, _, _, _ -> Error e
            | _, _, _, _, _, _, _, _, _, Error e, _, _ -> Error e
            | _, _, _, _, _, _, _, _, _, _, Error e, _ -> Error e
            | _, _, _, _, _, _, _, _, _, _, _, Error e -> Error e
            | _ -> Error(String.concat "; " customErrors))

    /// Checks if the remaining budget for a specific dimension is below a certain percentage.
    /// Returns true if remaining < (Total * percentage).
    member this.IsCritical(percentage: float) : bool =
        lock lockObj (fun () ->
            let check (limit: 'a option) (current: 'a) (converter: 'a -> float) =
                match limit with
                | Some max ->
                    let total = converter max
                    let used = converter current
                    let remaining = total - used

                    if total > 0.0 then
                        (remaining / total) < percentage
                    else
                        true
                | None -> false

            let c1 = check budget.MaxTokens consumed.Tokens (fun x -> float x)
            let c2 = check budget.MaxMoney consumed.Money (fun x -> float (decimal x))
            // We can add more dimensions if needed, but Tokens/Money are the main ones.
            c1 || c2)
