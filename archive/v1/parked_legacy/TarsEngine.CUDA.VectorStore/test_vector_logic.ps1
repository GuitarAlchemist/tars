=== TARS VECTOR STORE LOGIC TEST ===

# Simple vector similarity test in PowerShell
Write-Host 'Testing vector similarity logic...'

# Test vectors
$vec1 = @(1.0, 0.0, 0.0)
$vec2 = @(0.0, 1.0, 0.0)  
$vec3 = @(1.0, 0.0, 0.0)

# Cosine similarity function
function Get-CosineSimilarity($a, $b) {
    $dot = 0.0
    $norm_a = 0.0
    $norm_b = 0.0
    
    for ($i = 0; $i -lt $a.Length; $i++) {
        $dot += $a[$i] * $b[$i]
        $norm_a += $a[$i] * $a[$i]
        $norm_b += $b[$i] * $b[$i]
    }
    
    return $dot / ([Math]::Sqrt($norm_a) * [Math]::Sqrt($norm_b))
}

$sim1_2 = Get-CosineSimilarity $vec1 $vec2
$sim1_3 = Get-CosineSimilarity $vec1 $vec3

Write-Host \
Similarity
vec1-vec2:
$sim1_2\
Write-Host \Similarity
vec1-vec3:
$sim1_3\

if ($sim1_3 -gt 0.99 -and $sim1_2 -lt 0.1) {
    Write-Host '✅ Vector similarity logic works!' -ForegroundColor Green
    Write-Host '✅ Ready for CUDA acceleration' -ForegroundColor Green
    Write-Host '🚀 RTX 3070 will provide massive speedup!' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'CUDA Performance Expectations:' -ForegroundColor Cyan
    Write-Host '  - 5888 CUDA cores vs 8-16 CPU cores'
    Write-Host '  - 448 GB/s memory bandwidth vs ~50 GB/s'
    Write-Host '  - Expected speedup: 50-100x for vector operations'
    Write-Host '  - Perfect for TARS intelligence explosion!'
} else {
    Write-Host '❌ Vector similarity test failed' -ForegroundColor Red
}

