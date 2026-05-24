namespace Tars.Cortex

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Data.Sqlite
open Tars.Core

/// <summary>
/// SQLite-backed vector store for persistent storage.
/// Stores vectors, metadata, and supports efficient similarity search.
/// </summary>
type SqliteVectorStore(dbPath: string) =
    let connectionString = $"Data Source={dbPath}"
    /// Cap rows scanned per query to keep demo-friendly; tune or replace with ANN for scale
    let maxScanRows = 5000

    /// Initialize the database schema
    let initializeDb () =
        use conn = new SqliteConnection(connectionString)
        conn.Open()
        use cmd = conn.CreateCommand()
        cmd.CommandText <- """
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY
            );
            
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT NOT NULL,
                collection TEXT NOT NULL,
                vector BLOB NOT NULL,
                metadata TEXT NOT NULL,
                checksum TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used TEXT NOT NULL,
                PRIMARY KEY (collection, id),
                FOREIGN KEY (collection) REFERENCES collections(name)
            );
            
            CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection);
        """
        cmd.ExecuteNonQuery() |> ignore
    
    /// Convert float32 array to bytes
    let vectorToBytes (v: float32[]) : byte[] =
        let bytes = Array.zeroCreate<byte>(v.Length * 4)
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length)
        bytes
    
    /// Convert bytes to float32 array
    let bytesToVector (bytes: byte[]) : float32[] =
        let v = Array.zeroCreate<float32>(bytes.Length / 4)
        Buffer.BlockCopy(bytes, 0, v, 0, bytes.Length)
        v

    let computeChecksum (vector: float32[]) (payload: Map<string, string>) =
        use sha = System.Security.Cryptography.SHA256.Create()
        let vecBytes = vectorToBytes vector
        let metaBytes = System.Text.Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload))
        let combined = Array.append vecBytes metaBytes
        sha.ComputeHash(combined) |> Convert.ToHexString |> fun s -> s.ToLowerInvariant()
    
    do
        // Ensure directory exists
        let dir = Path.GetDirectoryName(dbPath)
        if not (String.IsNullOrEmpty(dir)) && not (Directory.Exists(dir)) then
            Directory.CreateDirectory(dir) |> ignore
        initializeDb()
    
    interface IVectorStore with
        member _.SaveAsync(collection: string, id: string, vector: float32[], payload: Map<string, string>) =
            task {
                use conn = new SqliteConnection(connectionString)
                do! conn.OpenAsync()
                
                // Ensure collection exists
                use cmdColl = conn.CreateCommand()
                cmdColl.CommandText <- "INSERT OR IGNORE INTO collections (name) VALUES (@name)"
                cmdColl.Parameters.AddWithValue("@name", collection) |> ignore
                do! cmdColl.ExecuteNonQueryAsync() |> Async.AwaitTask |> Async.Ignore
                
                // Check if entry exists for versioning
                use cmdCheck = conn.CreateCommand()
                cmdCheck.CommandText <- "SELECT version FROM vectors WHERE collection = @coll AND id = @id"
                cmdCheck.Parameters.AddWithValue("@coll", collection) |> ignore
                cmdCheck.Parameters.AddWithValue("@id", id) |> ignore
                let! existing = cmdCheck.ExecuteScalarAsync()
                let version = if isNull existing then 1 else (existing :?> int64 |> int) + 1
                
                // Upsert vector
                use cmd = conn.CreateCommand()
                cmd.CommandText <- """
                    INSERT OR REPLACE INTO vectors 
                    (id, collection, vector, metadata, checksum, version, created_at, last_used)
                    VALUES (@id, @coll, @vec, @meta, @chk, @ver, @created, @used)
                """
                cmd.Parameters.AddWithValue("@id", id) |> ignore
                cmd.Parameters.AddWithValue("@coll", collection) |> ignore
                cmd.Parameters.AddWithValue("@vec", vectorToBytes vector) |> ignore
                cmd.Parameters.AddWithValue("@meta", JsonSerializer.Serialize(payload)) |> ignore
                cmd.Parameters.AddWithValue("@chk", computeChecksum vector payload) |> ignore
                cmd.Parameters.AddWithValue("@ver", version) |> ignore
                cmd.Parameters.AddWithValue("@created", DateTime.UtcNow.ToString("o")) |> ignore
                cmd.Parameters.AddWithValue("@used", DateTime.UtcNow.ToString("o")) |> ignore
                do! cmd.ExecuteNonQueryAsync() |> Async.AwaitTask |> Async.Ignore
            } :> Task
        
        member _.SearchAsync(collection: string, queryVector: float32[], limit: int) =
            task {
                use conn = new SqliteConnection(connectionString)
                do! conn.OpenAsync()
                
                use cmd = conn.CreateCommand()
                let scanLimit = Math.Max(limit, maxScanRows)
                cmd.CommandText <- "SELECT id, vector, metadata FROM vectors WHERE collection = @coll LIMIT @lim"
                cmd.Parameters.AddWithValue("@coll", collection) |> ignore
                cmd.Parameters.AddWithValue("@lim", scanLimit) |> ignore
                
                use! reader = cmd.ExecuteReaderAsync()
                let results = ResizeArray<string * float32 * Map<string, string>>()
                
                while reader.Read() do
                    let id = reader.GetString(0)
                    let vecBytes = reader.GetValue(1) :?> byte[]
                    let vec = bytesToVector vecBytes
                    let metaJson = reader.GetString(2)
                    let meta = JsonSerializer.Deserialize<Map<string, string>>(metaJson)
                    
                    let sim = Similarity.cosineSimilarity queryVector vec
                    let dist = Similarity.similarityToDistance sim
                    results.Add((id, dist, meta))
                
                return results 
                       |> Seq.sortBy (fun (_, d, _) -> d) 
                       |> Seq.truncate limit 
                       |> Seq.toList
            }
    
    /// Get vector count in a collection
    member _.GetCountAsync(collection: string) =
        task {
            use conn = new SqliteConnection(connectionString)
            do! conn.OpenAsync()
            use cmd = conn.CreateCommand()
            cmd.CommandText <- "SELECT COUNT(*) FROM vectors WHERE collection = @coll"
            cmd.Parameters.AddWithValue("@coll", collection) |> ignore
            let! result = cmd.ExecuteScalarAsync()
            return result :?> int64 |> int
        }
    
    /// Get all collection names
    member _.GetCollectionsAsync() =
        task {
            use conn = new SqliteConnection(connectionString)
            do! conn.OpenAsync()
            use cmd = conn.CreateCommand()
            cmd.CommandText <- "SELECT name FROM collections"
            use! reader = cmd.ExecuteReaderAsync()
            let names = ResizeArray<string>()
            while reader.Read() do
                names.Add(reader.GetString(0))
            return names |> Seq.toList
        }
