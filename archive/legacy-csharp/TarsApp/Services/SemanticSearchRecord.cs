using Microsoft.Extensions.VectorData;

namespace TarsApp.Services;

public class SemanticSearchRecord
{
    [VectorStoreRecordKey]
    public required string Key { get; set; }

    [VectorStoreRecordData]
    public required string FileName { get; set; }

    [VectorStoreRecordData]
    public int PageNumber { get; set; }

    [VectorStoreRecordData]
    public required string Text { get; set; }

    [VectorStoreRecordVector(4096, DistanceFunction.CosineSimilarity)] // 4096 is the vector size for llama3
    public ReadOnlyMemory<float> Vector { get; set; }
}
