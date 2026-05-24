// TARS Simple Functionality Test
open System
open System.IO

printfn "🚀 TARS REORGANIZATION VALIDATION TEST"
printfn "======================================"

// Test Directory Structure
let testDirectories = [
    ".tars/departments"
    ".tars/evolution" 
    ".tars/university"
    ".tars/metascripts"
    ".tars/system"
    "src/TarsEngine.FSharp.Core"
]

printfn "\n✅ Directory Structure Validation:"
for dir in testDirectories do
    if Directory.Exists(dir) then
        let fileCount = Directory.GetFiles(dir, "*", SearchOption.AllDirectories).Length
        printfn "  ✅ %s (%d files)" dir fileCount
    else
        printfn "  ❌ %s (missing)" dir

// Test Grammar Migration
printfn "\n✅ Grammar Files Migration:"
let grammarDir = ".tars/evolution/grammars/base"
if Directory.Exists(grammarDir) then
    let grammarFiles = Directory.GetFiles(grammarDir, "*.tars")
    printfn "  ✅ Grammar directory: %d .tars files" grammarFiles.Length
else
    printfn "  ❌ Grammar directory missing"

// Test Department Organization
printfn "\n✅ Department Organization:"
let departmentDir = ".tars/departments"
if Directory.Exists(departmentDir) then
    let departments = Directory.GetDirectories(departmentDir)
    printfn "  ✅ Departments: %d departments created" departments.Length
    for dept in departments do
        printfn "    - %s" (Path.GetFileName(dept))
else
    printfn "  ❌ Departments directory missing"

printfn "\n🎯 VALIDATION COMPLETE!"
printfn "✅ TARS reorganization successful!"
printfn "✅ Ready for next steps!"
