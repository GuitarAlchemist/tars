#r "TarsEngine.FSharp.Core/bin/Release/net9.0/TarsEngine.FSharp.Core.dll"
#r "TarsEngine.FSharp.SelfImprovement/bin/Release/net9.0/TarsEngine.FSharp.SelfImprovement.dll"

open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.SpecKitWorkspace

let features = discoverFeatures None
for feature in features do
    printfn "Feature %s" feature.Id
    for task in feature.Tasks do
        printfn "  Task %s status=%s priority=%A" (defaultArg task.TaskId "(no id)") task.Status task.Priority
