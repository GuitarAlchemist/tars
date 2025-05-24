namespace TarsEngine.FSharp.Core.CodeGen.Testing.Generators

open System
open Microsoft.Extensions.Logging

/// <summary>
/// Generator for test values.
/// </summary>
type TestValueGenerator(logger: ILogger<TestValueGenerator>) =
    
    let random = Random()
    
    /// <summary>
    /// Generates a random integer.
    /// </summary>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <returns>A random integer.</returns>
    member _.GenerateInt(min: int, max: int) =
        random.Next(min, max)
    
    /// <summary>
    /// Generates a random double.
    /// </summary>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <returns>A random double.</returns>
    member _.GenerateDouble(min: double, max: double) =
        min + (random.NextDouble() * (max - min))
    
    /// <summary>
    /// Generates a random boolean.
    /// </summary>
    /// <returns>A random boolean.</returns>
    member _.GenerateBool() =
        random.Next(2) = 0
    
    /// <summary>
    /// Generates a random string.
    /// </summary>
    /// <param name="length">The length of the string.</param>
    /// <returns>A random string.</returns>
    member _.GenerateString(length: int) =
        let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        let buffer = Array.zeroCreate<char> length
        
        for i in 0 .. length - 1 do
            buffer.[i] <- chars.[random.Next(chars.Length)]
        
        new String(buffer)
    
    /// <summary>
    /// Generates a random date time.
    /// </summary>
    /// <returns>A random date time.</returns>
    member _.GenerateDateTime() =
        let start = DateTime(1970, 1, 1)
        let range = (DateTime.Today - start).Days
        start.AddDays(random.Next(range))
    
    /// <summary>
    /// Generates a random GUID.
    /// </summary>
    /// <returns>A random GUID.</returns>
    member _.GenerateGuid() =
        Guid.NewGuid()
    
    /// <summary>
    /// Generates a random enum value.
    /// </summary>
    /// <param name="enumType">The enum type.</param>
    /// <returns>A random enum value.</returns>
    member _.GenerateEnum(enumType: Type) =
        let values = Enum.GetValues(enumType)
        values.GetValue(random.Next(values.Length))
    
    /// <summary>
    /// Generates a random array.
    /// </summary>
    /// <param name="elementType">The element type.</param>
    /// <param name="length">The length of the array.</param>
    /// <returns>A random array.</returns>
    member this.GenerateArray(elementType: Type, length: int) =
        let array = Array.CreateInstance(elementType, length)
        
        for i in 0 .. length - 1 do
            array.SetValue(this.GenerateValue(elementType), i)
        
        array
    
    /// <summary>
    /// Generates a random list.
    /// </summary>
    /// <param name="elementType">The element type.</param>
    /// <param name="length">The length of the list.</param>
    /// <returns>A random list.</returns>
    member this.GenerateList(elementType: Type, length: int) =
        let listType = typedefof<System.Collections.Generic.List<_>>.MakeGenericType(elementType)
        let list = Activator.CreateInstance(listType)
        let addMethod = listType.GetMethod("Add")
        
        for _ in 0 .. length - 1 do
            let value = this.GenerateValue(elementType)
            addMethod.Invoke(list, [|value|]) |> ignore
        
        list
    
    /// <summary>
    /// Generates a random dictionary.
    /// </summary>
    /// <param name="keyType">The key type.</param>
    /// <param name="valueType">The value type.</param>
    /// <param name="count">The number of entries.</param>
    /// <returns>A random dictionary.</returns>
    member this.GenerateDictionary(keyType: Type, valueType: Type, count: int) =
        let dictionaryType = typedefof<System.Collections.Generic.Dictionary<_,_>>.MakeGenericType(keyType, valueType)
        let dictionary = Activator.CreateInstance(dictionaryType)
        let addMethod = dictionaryType.GetMethod("Add")
        
        for _ in 0 .. count - 1 do
            let key = this.GenerateValue(keyType)
            let value = this.GenerateValue(valueType)
            
            try
                addMethod.Invoke(dictionary, [|key; value|]) |> ignore
            with
            | _ -> () // Ignore duplicate keys
        
        dictionary
    
    /// <summary>
    /// Generates a random value for a type.
    /// </summary>
    /// <param name="type">The type to generate a value for.</param>
    /// <returns>A random value.</returns>
    member this.GenerateValue(type': Type) =
        try
            if type' = typeof<int> then
                this.GenerateInt(-100, 100) :> obj
            elif type' = typeof<double> then
                this.GenerateDouble(-100.0, 100.0) :> obj
            elif type' = typeof<float> then
                float32(this.GenerateDouble(-100.0, 100.0)) :> obj
            elif type' = typeof<decimal> then
                decimal(this.GenerateDouble(-100.0, 100.0)) :> obj
            elif type' = typeof<bool> then
                this.GenerateBool() :> obj
            elif type' = typeof<string> then
                this.GenerateString(10) :> obj
            elif type' = typeof<DateTime> then
                this.GenerateDateTime() :> obj
            elif type' = typeof<Guid> then
                this.GenerateGuid() :> obj
            elif type'.IsEnum then
                this.GenerateEnum(type')
            elif type'.IsArray then
                this.GenerateArray(type'.GetElementType(), 3)
            elif type'.IsGenericType && type'.GetGenericTypeDefinition() = typedefof<System.Collections.Generic.List<_>> then
                let elementType = type'.GetGenericArguments().[0]
                this.GenerateList(elementType, 3)
            elif type'.IsGenericType && type'.GetGenericTypeDefinition() = typedefof<System.Collections.Generic.Dictionary<_,_>> then
                let keyType = type'.GetGenericArguments().[0]
                let valueType = type'.GetGenericArguments().[1]
                this.GenerateDictionary(keyType, valueType, 3)
            elif type'.IsClass && type' <> typeof<string> then
                null // For now, return null for complex types
            else
                Activator.CreateInstance(type') // Default constructor for other types
        with
        | ex ->
            logger.LogError(ex, "Error generating value for type: {Type}", type'.FullName)
            null
    
    /// <summary>
    /// Generates a C# code string for a value.
    /// </summary>
    /// <param name="value">The value to generate code for.</param>
    /// <param name="type">The type of the value.</param>
    /// <returns>The C# code string.</returns>
    member this.GenerateCSharpValueCode(value: obj, type': Type) =
        try
            if value = null then
                "null"
            elif type' = typeof<int> then
                value.ToString()
            elif type' = typeof<double> then
                $"{value}d"
            elif type' = typeof<float> then
                $"{value}f"
            elif type' = typeof<decimal> then
                $"{value}m"
            elif type' = typeof<bool> then
                value.ToString().ToLowerInvariant()
            elif type' = typeof<string> then
                $"\"{value}\""
            elif type' = typeof<DateTime> then
                let dt = value :?> DateTime
                $"new DateTime({dt.Year}, {dt.Month}, {dt.Day}, {dt.Hour}, {dt.Minute}, {dt.Second})"
            elif type' = typeof<Guid> then
                $"new Guid(\"{value}\")"
            elif type'.IsEnum then
                $"{type'.FullName}.{value}"
            elif type'.IsArray then
                let array = value :?> Array
                let elementType = type'.GetElementType()
                let elements = 
                    [|
                        for i in 0 .. array.Length - 1 do
                            let element = array.GetValue(i)
                            yield this.GenerateCSharpValueCode(element, elementType)
                    |]
                $"new {elementType.FullName}[] {{ {String.Join(", ", elements)} }}"
            elif type'.IsGenericType && type'.GetGenericTypeDefinition() = typedefof<System.Collections.Generic.List<_>> then
                let list = value
                let elementType = type'.GetGenericArguments().[0]
                let elements = ResizeArray<string>()
                
                let enumerator = (list :?> System.Collections.IEnumerable).GetEnumerator()
                while enumerator.MoveNext() do
                    let element = enumerator.Current
                    elements.Add(this.GenerateCSharpValueCode(element, elementType))
                
                $"new List<{elementType.FullName}>() {{ {String.Join(", ", elements)} }}"
            elif type'.IsGenericType && type'.GetGenericTypeDefinition() = typedefof<System.Collections.Generic.Dictionary<_,_>> then
                let dictionary = value
                let keyType = type'.GetGenericArguments().[0]
                let valueType = type'.GetGenericArguments().[1]
                let entries = ResizeArray<string>()
                
                let enumerator = (dictionary :?> System.Collections.IEnumerable).GetEnumerator()
                while enumerator.MoveNext() do
                    let entry = enumerator.Current
                    let key = entry.GetType().GetProperty("Key").GetValue(entry)
                    let value = entry.GetType().GetProperty("Value").GetValue(entry)
                    
                    let keyCode = this.GenerateCSharpValueCode(key, keyType)
                    let valueCode = this.GenerateCSharpValueCode(value, valueType)
                    
                    entries.Add($"{{ {keyCode}, {valueCode} }}")
                
                $"new Dictionary<{keyType.FullName}, {valueType.FullName}>() {{ {String.Join(", ", entries)} }}"
            else
                "null" // Default for other types
        with
        | ex ->
            logger.LogError(ex, "Error generating C# code for value: {Value}", value)
            "null"
    
    /// <summary>
    /// Generates an F# code string for a value.
    /// </summary>
    /// <param name="value">The value to generate code for.</param>
    /// <param name="type">The type of the value.</param>
    /// <returns>The F# code string.</returns>
    member this.GenerateFSharpValueCode(value: obj, type': Type) =
        try
            if value = null then
                "null"
            elif type' = typeof<int> then
                value.ToString()
            elif type' = typeof<double> then
                value.ToString()
            elif type' = typeof<float> then
                $"{value}f"
            elif type' = typeof<decimal> then
                $"{value}m"
            elif type' = typeof<bool> then
                value.ToString().ToLowerInvariant()
            elif type' = typeof<string> then
                $"\"{value}\""
            elif type' = typeof<DateTime> then
                let dt = value :?> DateTime
                $"DateTime({dt.Year}, {dt.Month}, {dt.Day}, {dt.Hour}, {dt.Minute}, {dt.Second})"
            elif type' = typeof<Guid> then
                $"Guid(\"{value}\")"
            elif type'.IsEnum then
                $"{type'.FullName}.{value}"
            elif type'.IsArray then
                let array = value :?> Array
                let elementType = type'.GetElementType()
                let elements = 
                    [|
                        for i in 0 .. array.Length - 1 do
                            let element = array.GetValue(i)
                            yield this.GenerateFSharpValueCode(element, elementType)
                    |]
                $"[| {String.Join("; ", elements)} |]"
            elif type'.IsGenericType && type'.GetGenericTypeDefinition() = typedefof<System.Collections.Generic.List<_>> then
                let list = value
                let elementType = type'.GetGenericArguments().[0]
                let elements = ResizeArray<string>()
                
                let enumerator = (list :?> System.Collections.IEnumerable).GetEnumerator()
                while enumerator.MoveNext() do
                    let element = enumerator.Current
                    elements.Add(this.GenerateFSharpValueCode(element, elementType))
                
                $"[ {String.Join("; ", elements)} ]"
            elif type'.IsGenericType && type'.GetGenericTypeDefinition() = typedefof<System.Collections.Generic.Dictionary<_,_>> then
                let dictionary = value
                let keyType = type'.GetGenericArguments().[0]
                let valueType = type'.GetGenericArguments().[1]
                let entries = ResizeArray<string>()
                
                let enumerator = (dictionary :?> System.Collections.IEnumerable).GetEnumerator()
                while enumerator.MoveNext() do
                    let entry = enumerator.Current
                    let key = entry.GetType().GetProperty("Key").GetValue(entry)
                    let value = entry.GetType().GetProperty("Value").GetValue(entry)
                    
                    let keyCode = this.GenerateFSharpValueCode(key, keyType)
                    let valueCode = this.GenerateFSharpValueCode(value, valueType)
                    
                    entries.Add($"{keyCode}, {valueCode}")
                
                $"dict [ {String.Join("; ", entries)} ]"
            else
                "null" // Default for other types
        with
        | ex ->
            logger.LogError(ex, "Error generating F# code for value: {Value}", value)
            "null"
    
    /// <summary>
    /// Generates a code string for a value.
    /// </summary>
    /// <param name="value">The value to generate code for.</param>
    /// <param name="type">The type of the value.</param>
    /// <param name="language">The language to generate code for.</param>
    /// <returns>The code string.</returns>
    member this.GenerateValueCode(value: obj, type': Type, language: string) =
        match language.ToLowerInvariant() with
        | "csharp" -> this.GenerateCSharpValueCode(value, type')
        | "fsharp" -> this.GenerateFSharpValueCode(value, type')
        | _ -> this.GenerateCSharpValueCode(value, type') // Default to C#
