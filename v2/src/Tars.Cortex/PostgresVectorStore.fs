namespace Tars.Cortex

open System
open System.Text.Json
open System.Threading.Tasks
open Npgsql
open Tars.Core

/// <summary>
/// Postgres-backed vector store using pgvector extension.
/// </summary>
type PostgresVectorStore(connectionString: string, ?dimension: int) =

    // Default to nomic-embed-text dims (768) unless caller overrides
    let vecDim = defaultArg dimension 768

    /// Initialize the database schema and extension
    let initializeDb () =
        use conn = new NpgsqlConnection(connectionString)
        conn.Open()
        use cmd = conn.CreateCommand()
        cmd.CommandText <- PostgresVectorStore.SchemaSql vecDim
        cmd.ExecuteNonQuery() |> ignore

    let computeChecksum (vector: float32[]) (payload: Map<string, string>) =
        use sha = System.Security.Cryptography.SHA256.Create()
        let bytes = Array.zeroCreate<byte> (vector.Length * 4)
        Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length)

        let metaBytes =
            System.Text.Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload))

        let combined = Array.append bytes metaBytes

        sha.ComputeHash(combined)
        |> Convert.ToHexString
        |> fun s -> s.ToLowerInvariant()

    // Convert float32[] to pgvector string format "[1.0,2.0,...]"
    let vectorToString (v: float32[]) = "[" + String.Join(",", v) + "]"

    do
        // Best effort init, might fail if db doesn't exist yet (handled by docker-compose)
        try
            initializeDb ()
        with _ ->
            ()

    /// The schema every construction applies. Creates what is missing and nothing else.
    ///
    /// It used to `DROP TABLE IF EXISTS vectors` first, "to ensure correct dimension
    /// (dev only)" - unconditionally, from the constructor, inside a `try` that
    /// swallowed every failure. `tars evolve` and the UI both construct a store at
    /// startup, so each of them wiped every persisted embedding, every time (#284).
    ///
    /// A table created for another dimension is re-typed only while it is **empty**,
    /// where there is nothing to lose. That is the table an upgrade leaves behind when
    /// a model's dimension was wrong: `mxbai` was mapped to 512, every vector it made
    /// was refused, and the column stayed `vector(512)` with no rows in it.
    ///
    /// One that holds vectors is left as it is. Writing a vector of the wrong size into
    /// it fails loudly - pgvector refuses it - which is the right outcome: which
    /// dimension should win is a migration decision, not something to settle by
    /// deleting data.
    ///
    /// DDL cannot take SQL parameters for a type definition, so the dimension is
    /// interpolated; it is an int, never caller text.
    static member SchemaSql(dimension: int) : string =
        $"""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT NOT NULL,
                collection TEXT NOT NULL,
                vector vector({dimension}),
                metadata JSONB NOT NULL,
                checksum TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL,
                last_used TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (collection, id),
                FOREIGN KEY (collection) REFERENCES collections(name) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection);

            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM vectors)
                   AND (SELECT format_type(atttypid, atttypmod)
                        FROM pg_attribute
                        WHERE attrelid = 'vectors'::regclass AND attname = 'vector')
                       <> 'vector({dimension})'
                THEN
                    ALTER TABLE vectors ALTER COLUMN vector TYPE vector({dimension});
                END IF;
            END $$;
        """

    interface IVectorStore with
        member _.SaveAsync(collection: string, id: string, vector: float32[], payload: Map<string, string>) =
            task {
                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()

                // Ensure collection
                use cmdColl = conn.CreateCommand()
                cmdColl.CommandText <- "INSERT INTO collections (name) VALUES (@name) ON CONFLICT (name) DO NOTHING"
                cmdColl.Parameters.AddWithValue("@name", collection) |> ignore
                do! cmdColl.ExecuteNonQueryAsync() |> Async.AwaitTask |> Async.Ignore

                // Check version
                use cmdCheck = conn.CreateCommand()
                cmdCheck.CommandText <- "SELECT version FROM vectors WHERE collection = @coll AND id = @id"
                cmdCheck.Parameters.AddWithValue("@coll", collection) |> ignore
                cmdCheck.Parameters.AddWithValue("@id", id) |> ignore
                let! existing = cmdCheck.ExecuteScalarAsync()
                let version = if isNull existing then 1 else (existing :?> int) + 1

                // Upsert
                use cmd = conn.CreateCommand()

                cmd.CommandText <-
                    """
                    INSERT INTO vectors 
                    (id, collection, vector, metadata, checksum, version, created_at, last_used)
                    VALUES (@id, @coll, @vec::vector, @meta::jsonb, @chk, @ver, @created, @used)
                    ON CONFLICT (collection, id) DO UPDATE SET
                    vector = EXCLUDED.vector,
                    metadata = EXCLUDED.metadata,
                    checksum = EXCLUDED.checksum,
                    version = EXCLUDED.version,
                    last_used = EXCLUDED.last_used
                """

                cmd.Parameters.AddWithValue("@id", id) |> ignore
                cmd.Parameters.AddWithValue("@coll", collection) |> ignore

                if vector.Length <> vecDim then
                    return
                        raise (
                            ArgumentException(
                                $"Vector dimension {vector.Length} does not match configured pgvector dimension {vecDim}"
                            )
                        )

                cmd.Parameters.AddWithValue("@vec", vectorToString vector) |> ignore

                cmd.Parameters.AddWithValue("@meta", JsonSerializer.Serialize(payload))
                |> ignore

                cmd.Parameters.AddWithValue("@chk", computeChecksum vector payload) |> ignore
                cmd.Parameters.AddWithValue("@ver", version) |> ignore
                cmd.Parameters.AddWithValue("@created", DateTime.UtcNow) |> ignore
                cmd.Parameters.AddWithValue("@used", DateTime.UtcNow) |> ignore

                do! cmd.ExecuteNonQueryAsync() |> Async.AwaitTask |> Async.Ignore
            }
            :> Task

        member _.SearchAsync(collection: string, queryVector: float32[], limit: int) =
            task {
                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()

                use cmd = conn.CreateCommand()
                // pgvector's <=> is cosine distance, the VectorMatch convention; closest first.
                cmd.CommandText <-
                    """
                    SELECT id, vector, metadata, (vector <=> @vec::vector) as distance
                    FROM vectors 
                    WHERE collection = @coll
                    ORDER BY distance ASC
                    LIMIT @lim
                """

                cmd.Parameters.AddWithValue("@coll", collection) |> ignore
                cmd.Parameters.AddWithValue("@vec", vectorToString queryVector) |> ignore
                cmd.Parameters.AddWithValue("@lim", limit) |> ignore

                use! reader = cmd.ExecuteReaderAsync()
                let results = ResizeArray<VectorMatch>()

                while reader.Read() do
                    let id = reader.GetString(0)
                    let distance = reader.GetDouble(3) |> float32
                    let metaJson = reader.GetString(2)
                    let meta = JsonSerializer.Deserialize<Map<string, string>>(metaJson)
                    results.Add({ Id = id; Distance = distance; Payload = meta })

                return results |> Seq.toList
            }

    member _.GetCountAsync(collection: string) =
        task {
            use conn = new NpgsqlConnection(connectionString)
            do! conn.OpenAsync()
            use cmd = conn.CreateCommand()
            cmd.CommandText <- "SELECT COUNT(*) FROM vectors WHERE collection = @coll"
            cmd.Parameters.AddWithValue("@coll", collection) |> ignore
            let! result = cmd.ExecuteScalarAsync()
            return result :?> int64 |> int
        }

    member _.GetCollectionsAsync() =
        task {
            use conn = new NpgsqlConnection(connectionString)
            do! conn.OpenAsync()
            use cmd = conn.CreateCommand()
            cmd.CommandText <- "SELECT name FROM collections"
            use! reader = cmd.ExecuteReaderAsync()
            let names = ResizeArray<string>()

            while reader.Read() do
                names.Add(reader.GetString(0))

            return names |> Seq.toList
        }
