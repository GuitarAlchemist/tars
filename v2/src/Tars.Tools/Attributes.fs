namespace Tars.Tools

open System

[<AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)>]
type TarsToolAttribute(name: string, description: string) =
    inherit Attribute()
    member val Name = name
    member val Description = description

[<AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)>]
type TarsSkillAttribute(name: string, domain: string) =
    inherit Attribute()
    member val Name = name
    member val Domain = domain
