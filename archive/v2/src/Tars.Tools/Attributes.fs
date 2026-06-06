namespace Tars.Tools

open System

[<AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)>]
type TarsToolAttribute(name: string, description: string) =
    inherit Attribute()
    member val Name = name
    member val Description = description
