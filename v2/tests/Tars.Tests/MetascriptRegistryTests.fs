namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Metascript.V1
open Tars.Metascript

/// Tests for FileMetascriptRegistry CRUD operations
type MetascriptRegistryTests() =

    let createTestDir () =
        let dir = Path.Combine(Path.GetTempPath(), $"tars-test-{Guid.NewGuid():N}")
        Directory.CreateDirectory(dir) |> ignore
        dir

    let cleanupDir dir =
        if Directory.Exists(dir) then
            Directory.Delete(dir, true)

    let createTestMetascript name content =
        { Name = name
          Blocks =
            [ { Type = MetascriptBlockType.Text
                Content = content
                LineNumber = 1
                ColumnNumber = 1
                Parameters = []
                Id = Guid.NewGuid().ToString("N")
                ParentId = None
                Metadata = Map.empty } ]
          FilePath = None
          Variables = Map.empty
          Metadata = Map.empty }

    [<Fact>]
    member _.``Register saves metascript as .tars file``() =
        task {
            let dir = createTestDir ()

            try
                let registry = FileMetascriptRegistry(dir) :> IMetascriptRegistry
                let metascript = createTestMetascript "test-script" "Hello, Test!"

                do! registry.Register(metascript)

                let filePath = Path.Combine(dir, "test-script.tars")
                Assert.True(File.Exists(filePath), $"File should exist at {filePath}")

                let content = File.ReadAllText(filePath)
                Assert.Contains("Hello, Test!", content)
            finally
                cleanupDir dir
        }

    [<Fact>]
    member _.``Get retrieves registered metascript``() =
        task {
            let dir = createTestDir ()

            try
                let registry = FileMetascriptRegistry(dir) :> IMetascriptRegistry
                let metascript = createTestMetascript "retrieve-test" "Content to retrieve"

                do! registry.Register(metascript)
                let! retrieved = registry.Get("retrieve-test")

                Assert.True(retrieved.IsSome, "Should retrieve registered metascript")
                Assert.Equal("retrieve-test", retrieved.Value.Name)

                let textBlock =
                    retrieved.Value.Blocks |> List.find (fun b -> b.Type = MetascriptBlockType.Text)

                Assert.Contains("Content to retrieve", textBlock.Content)
            finally
                cleanupDir dir
        }

    [<Fact>]
    member _.``Get returns None for non-existent metascript``() =
        task {
            let dir = createTestDir ()

            try
                let registry = FileMetascriptRegistry(dir) :> IMetascriptRegistry

                let! result = registry.Get("does-not-exist")

                Assert.True(result.IsNone, "Should return None for non-existent metascript")
            finally
                cleanupDir dir
        }

    [<Fact>]
    member _.``List returns all registered metascripts``() =
        task {
            let dir = createTestDir ()

            try
                let registry = FileMetascriptRegistry(dir) :> IMetascriptRegistry

                do! registry.Register(createTestMetascript "script-a" "A")
                do! registry.Register(createTestMetascript "script-b" "B")
                do! registry.Register(createTestMetascript "script-c" "C")

                let! all = registry.List()

                Assert.Equal(3, all.Length)
                let names = all |> List.map (fun m -> m.Name)
                Assert.Contains("script-a", names)
                Assert.Contains("script-b", names)
                Assert.Contains("script-c", names)
            finally
                cleanupDir dir
        }

    [<Fact>]
    member _.``Delete removes metascript``() =
        task {
            let dir = createTestDir ()

            try
                let registry = FileMetascriptRegistry(dir) :> IMetascriptRegistry
                let metascript = createTestMetascript "to-delete" "Delete me"

                do! registry.Register(metascript)
                let! beforeDelete = registry.Get("to-delete")
                Assert.True(beforeDelete.IsSome, "Should exist before delete")

                let! deleted = registry.Delete("to-delete")
                Assert.True(deleted, "Delete should return true")

                let! afterDelete = registry.Get("to-delete")
                Assert.True(afterDelete.IsNone, "Should not exist after delete")
            finally
                cleanupDir dir
        }

    [<Fact>]
    member _.``Register overwrites existing metascript``() =
        task {
            let dir = createTestDir ()

            try
                let registry = FileMetascriptRegistry(dir) :> IMetascriptRegistry

                do! registry.Register(createTestMetascript "overwrite" "Original content")
                do! registry.Register(createTestMetascript "overwrite" "Updated content")

                let! retrieved = registry.Get("overwrite")

                Assert.True(retrieved.IsSome)

                let textBlock =
                    retrieved.Value.Blocks |> List.find (fun b -> b.Type = MetascriptBlockType.Text)

                Assert.Contains("Updated content", textBlock.Content)
            finally
                cleanupDir dir
        }

    [<Fact>]
    member _.``Registry creates directory if not exists``() =
        task {
            let baseDir = createTestDir ()
            let nestedDir = Path.Combine(baseDir, "nested", "scripts")

            try
                Assert.False(Directory.Exists(nestedDir), "Nested dir should not exist initially")

                let registry = FileMetascriptRegistry(nestedDir) :> IMetascriptRegistry
                do! registry.Register(createTestMetascript "nested-script" "Nested content")

                Assert.True(Directory.Exists(nestedDir), "Directory should be created")

                let! retrieved = registry.Get("nested-script")
                Assert.True(retrieved.IsSome)
            finally
                cleanupDir baseDir
        }

    [<Fact>]
    member _.``Name lookup is case-insensitive``() =
        task {
            let dir = createTestDir ()

            try
                let registry = FileMetascriptRegistry(dir) :> IMetascriptRegistry

                do! registry.Register(createTestMetascript "CamelCase" "Content")

                let! lower = registry.Get("camelcase")
                let! upper = registry.Get("CAMELCASE")
                let! mixed = registry.Get("CamelCase")

                Assert.True(lower.IsSome, "Should find with lowercase")
                Assert.True(upper.IsSome, "Should find with uppercase")
                Assert.True(mixed.IsSome, "Should find with original case")
            finally
                cleanupDir dir
        }
