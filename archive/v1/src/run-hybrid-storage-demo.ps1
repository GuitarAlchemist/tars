# TARS Hybrid Consciousness Storage Demo
# Demonstrates optimal storage strategy: Volatile + Cached + Persistent + Long-term

Write-Host "💾🧠 TARS HYBRID CONSCIOUSNESS STORAGE DEMO" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Storage tier definitions
$StorageTiers = @{
    "Volatile" = @{
        description = "In-memory only, ultra-fast access"
        color = "Red"
        icon = "⚡"
        maxItems = 50
        persistence = $false
    }
    "Cached" = @{
        description = "In-memory with async disk writes"
        color = "Yellow"
        icon = "🔄"
        maxItems = 200
        persistence = $true
        persistInterval = 5
    }
    "Persistent" = @{
        description = "Immediate disk write, critical data"
        color = "Green"
        icon = "💾"
        maxItems = $null
        persistence = $true
        persistInterval = 0
    }
    "LongTerm" = @{
        description = "Compressed archival storage"
        color = "Blue"
        icon = "📚"
        maxItems = $null
        persistence = $true
        compressed = $true
    }
}

# Data type storage assignments
$DataTypeStorage = @{
    "CurrentThoughts" = "Volatile"
    "AttentionFocus" = "Volatile"
    "EmotionalState" = "Volatile"
    "WorkingMemory" = "Cached"
    "AgentContributions" = "Cached"
    "ConversationContext" = "Cached"
    "SelfAwareness" = "Persistent"
    "ConsciousnessLevel" = "Persistent"
    "PersonalityTraits" = "Persistent"
    "LongTermMemory" = "LongTerm"
}

# Initialize storage system
function Initialize-HybridStorage {
    Write-Host "🔧 Initializing Hybrid Consciousness Storage..." -ForegroundColor Cyan
    
    # Create storage directories
    $storageDir = ".tars\storage"
    @("volatile", "cached", "persistent", "longterm") | ForEach-Object {
        $dir = "$storageDir\$_"
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    # Initialize storage containers
    $global:storageContainers = @{
        "Volatile" = @{}
        "Cached" = @{}
        "Persistent" = @{}
        "LongTerm" = @{}
    }
    
    # Initialize statistics
    $global:storageStats = @{
        "Volatile" = @{ count = 0; reads = 0; writes = 0; lastAccess = (Get-Date) }
        "Cached" = @{ count = 0; reads = 0; writes = 0; lastAccess = (Get-Date) }
        "Persistent" = @{ count = 0; reads = 0; writes = 0; lastAccess = (Get-Date) }
        "LongTerm" = @{ count = 0; reads = 0; writes = 0; lastAccess = (Get-Date) }
    }
    
    Write-Host "  ✅ Initialized 4-tier hybrid storage system" -ForegroundColor Green
    Write-Host ""
}

# Store data in appropriate tier
function Store-ConsciousnessData {
    param(
        [string]$DataType,
        [string]$Id,
        [object]$Content,
        [double]$Importance = 0.5
    )
    
    $tier = $DataTypeStorage[$DataType]
    if (-not $tier) {
        $tier = "Cached" # Default tier
    }
    
    $tierInfo = $StorageTiers[$tier]
    $entry = @{
        id = $Id
        dataType = $DataType
        content = $Content
        importance = $Importance
        timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        tier = $tier
        accessCount = 1
        lastAccessed = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    }
    
    # Store in appropriate container
    $global:storageContainers[$tier][$Id] = $entry
    $global:storageStats[$tier].count++
    $global:storageStats[$tier].writes++
    $global:storageStats[$tier].lastAccess = Get-Date
    
    Write-Host "    $($tierInfo.icon) Stored in $tier tier: $DataType" -ForegroundColor $tierInfo.color
    
    # Handle tier-specific logic
    switch ($tier) {
        "Volatile" {
            # Check capacity limits
            if ($tierInfo.maxItems -and $global:storageContainers[$tier].Count -gt $tierInfo.maxItems) {
                # Remove least important items
                $itemsToRemove = $global:storageContainers[$tier].Values | 
                    Sort-Object { $_.importance * $_.accessCount } | 
                    Select-Object -First ($global:storageContainers[$tier].Count - $tierInfo.maxItems)
                
                foreach ($item in $itemsToRemove) {
                    $global:storageContainers[$tier].Remove($item.id)
                    Write-Host "      🗑️ Evicted from volatile: $($item.dataType)" -ForegroundColor Gray
                }
            }
        }
        "Cached" {
            # Simulate async persistence
            Write-Host "      ⏳ Queued for async persistence (${$tierInfo.persistInterval}s)" -ForegroundColor Gray
        }
        "Persistent" {
            # Simulate immediate disk write
            Write-Host "      💾 Written to disk immediately" -ForegroundColor Green
        }
        "LongTerm" {
            # Simulate compression
            Write-Host "      🗜️ Compressed and archived" -ForegroundColor Blue
        }
    }
}

# Retrieve data with tier fallback
function Get-ConsciousnessData {
    param([string]$Id)
    
    # Try each tier in order of speed
    $tiers = @("Volatile", "Cached", "Persistent", "LongTerm")
    
    foreach ($tier in $tiers) {
        if ($global:storageContainers[$tier].ContainsKey($Id)) {
            $entry = $global:storageContainers[$tier][$Id]
            $tierInfo = $StorageTiers[$tier]
            
            # Update access statistics
            $entry.accessCount++
            $entry.lastAccessed = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            $global:storageStats[$tier].reads++
            $global:storageStats[$tier].lastAccess = Get-Date
            
            Write-Host "    $($tierInfo.icon) Retrieved from $tier tier: $($entry.dataType)" -ForegroundColor $tierInfo.color
            
            # Promote frequently accessed items to faster tiers
            if ($tier -eq "Persistent" -and $entry.accessCount -gt 3) {
                $global:storageContainers["Cached"][$Id] = $entry
                Write-Host "      ⬆️ Promoted to Cached tier" -ForegroundColor Yellow
            }
            elseif ($tier -eq "Cached" -and $entry.accessCount -gt 5) {
                $global:storageContainers["Volatile"][$Id] = $entry
                Write-Host "      ⬆️ Promoted to Volatile tier" -ForegroundColor Red
            }
            
            return $entry
        }
    }
    
    Write-Host "    ❌ Data not found: $Id" -ForegroundColor Red
    return $null
}

# Demonstrate storage performance
function Demo-StoragePerformance {
    param([string]$Scenario)
    
    Write-Host ""
    Write-Host "🚀 STORAGE PERFORMANCE DEMO: $Scenario" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    # Simulate different data types being stored
    $dataTypes = @(
        @{ type = "CurrentThoughts"; content = "Processing user input about storage"; importance = 0.3 }
        @{ type = "WorkingMemory"; content = "User asked about volatile vs persistent state"; importance = 0.8 }
        @{ type = "AgentContributions"; content = "Memory agent contributed storage analysis"; importance = 0.7 }
        @{ type = "SelfAwareness"; content = "Understanding storage trade-offs"; importance = 0.9 }
        @{ type = "EmotionalState"; content = "Excited about hybrid architecture"; importance = 0.4 }
        @{ type = "PersonalityTraits"; content = "Analytical and thorough"; importance = 1.0 }
        @{ type = "LongTermMemory"; content = "Storage architecture principles learned"; importance = 0.95 }
    )
    
    Write-Host "📝 Storing consciousness data across tiers..." -ForegroundColor Yellow
    foreach ($data in $dataTypes) {
        $id = [System.Guid]::NewGuid().ToString().Substring(0, 8)
        Store-ConsciousnessData -DataType $data.type -Id $id -Content $data.content -Importance $data.importance
        Start-Sleep -Milliseconds 100 # Simulate processing time
    }
    
    Write-Host ""
    Write-Host "🔍 Retrieving data to demonstrate tier performance..." -ForegroundColor Yellow
    
    # Retrieve some data multiple times to show promotion
    $sampleIds = $global:storageContainers.Values | ForEach-Object { $_.Keys } | Select-Object -First 3
    foreach ($id in $sampleIds) {
        for ($i = 1; $i -le 3; $i++) {
            Write-Host "  Access #$i for ${id}:" -ForegroundColor Gray
            Get-ConsciousnessData -Id $id | Out-Null
            Start-Sleep -Milliseconds 50
        }
    }
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMilliseconds
    
    Write-Host ""
    Write-Host "⏱️ Performance: ${duration}ms total" -ForegroundColor Green
    Write-Host ""
}

# Show storage statistics
function Show-StorageStatistics {
    Write-Host ""
    Write-Host "📊 HYBRID STORAGE STATISTICS" -ForegroundColor Cyan
    Write-Host "============================" -ForegroundColor Cyan
    Write-Host ""
    
    $totalItems = 0
    $totalReads = 0
    $totalWrites = 0
    
    foreach ($tier in $StorageTiers.Keys) {
        $stats = $global:storageStats[$tier]
        $tierInfo = $StorageTiers[$tier]
        
        Write-Host "$($tierInfo.icon) $tier Tier:" -ForegroundColor $tierInfo.color
        Write-Host "  Items: $($stats.count)" -ForegroundColor White
        Write-Host "  Reads: $($stats.reads)" -ForegroundColor White
        Write-Host "  Writes: $($stats.writes)" -ForegroundColor White
        Write-Host "  Description: $($tierInfo.description)" -ForegroundColor Gray
        Write-Host ""
        
        $totalItems += $stats.count
        $totalReads += $stats.reads
        $totalWrites += $stats.writes
    }
    
    Write-Host "📈 Total Statistics:" -ForegroundColor Yellow
    Write-Host "  Total Items: $totalItems" -ForegroundColor White
    Write-Host "  Total Reads: $totalReads" -ForegroundColor White
    Write-Host "  Total Writes: $totalWrites" -ForegroundColor White
    Write-Host "  Memory Usage: $([GC]::GetTotalMemory($false) / 1MB) MB" -ForegroundColor White
    Write-Host ""
}

# Show tier architecture
function Show-TierArchitecture {
    Write-Host ""
    Write-Host "🏗️ HYBRID STORAGE ARCHITECTURE" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Data Flow: User Input → Tier Assignment → Storage → Retrieval" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($tier in @("Volatile", "Cached", "Persistent", "LongTerm")) {
        $tierInfo = $StorageTiers[$tier]
        Write-Host "$($tierInfo.icon) $tier Tier" -ForegroundColor $tierInfo.color
        Write-Host "  $($tierInfo.description)" -ForegroundColor White
        
        # Show data types in this tier
        $dataTypesInTier = $DataTypeStorage.GetEnumerator() | Where-Object { $_.Value -eq $tier } | ForEach-Object { $_.Key }
        if ($dataTypesInTier) {
            Write-Host "  Data Types: $($dataTypesInTier -join ', ')" -ForegroundColor Gray
        }
        Write-Host ""
    }
    
    Write-Host "🔄 Automatic Promotion:" -ForegroundColor Yellow
    Write-Host "  • Frequently accessed items move to faster tiers" -ForegroundColor White
    Write-Host "  • LRU eviction from volatile tier when capacity exceeded" -ForegroundColor White
    Write-Host "  • Async persistence for cached tier" -ForegroundColor White
    Write-Host "  • Immediate persistence for critical data" -ForegroundColor White
    Write-Host ""
}

# Main demo
function Start-HybridStorageDemo {
    Write-Host "💾🧠 TARS Hybrid Storage Demo Started" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 This demo shows optimal storage strategy:" -ForegroundColor Yellow
    Write-Host "  • Volatile: Ultra-fast temporary data (thoughts, attention)" -ForegroundColor Red
    Write-Host "  • Cached: Fast access + async persistence (working memory)" -ForegroundColor Yellow
    Write-Host "  • Persistent: Immediate disk write (critical data)" -ForegroundColor Green
    Write-Host "  • Long-term: Compressed archival (significant memories)" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Commands: 'architecture', 'performance', 'stats', 'help', 'exit'" -ForegroundColor Gray
    Write-Host ""
    
    $isRunning = $true
    while ($isRunning) {
        Write-Host ""
        $userInput = Read-Host "Command"
        
        switch ($userInput.ToLower().Trim()) {
            "exit" {
                $isRunning = $false
                Write-Host ""
                Write-Host "💾🧠 Hybrid storage demo completed! Optimal strategy demonstrated." -ForegroundColor Green
                break
            }
            "architecture" {
                Show-TierArchitecture
            }
            "performance" {
                Demo-StoragePerformance -Scenario "Balanced"
            }
            "stats" {
                Show-StorageStatistics
            }
            "help" {
                Write-Host ""
                Write-Host "💾🧠 Hybrid Storage Demo Commands:" -ForegroundColor Cyan
                Write-Host "• 'architecture' - Show tier architecture and data flow" -ForegroundColor White
                Write-Host "• 'performance' - Demonstrate storage performance" -ForegroundColor White
                Write-Host "• 'stats' - Show detailed storage statistics" -ForegroundColor White
                Write-Host "• 'exit' - End the demo" -ForegroundColor White
            }
            default {
                Write-Host "Unknown command. Type 'help' for available commands." -ForegroundColor Red
            }
        }
    }
}

# Initialize and start
Initialize-HybridStorage
Show-TierArchitecture
Demo-StoragePerformance -Scenario "Initial Load"
Start-HybridStorageDemo
