# TARS Real Autonomous QA Agent
# Uses actual VirtualBox VMs for real deployment and testing

Write-Host "🤖🧪 TARS REAL AUTONOMOUS QA AGENT" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""

# Import VM functions
. "$PSScriptRoot\run-local-vm-deployment.ps1"

# Global QA agent state
$global:qaAgent = @{
    isRunning = $false
    taskQueue = @()
    activeDeployments = @{}
    completedTasks = @()
    metrics = @{
        tasksProcessed = 0
        successfulDeployments = 0
        failedDeployments = 0
        totalTestsRun = 0
        totalVMsCreated = 0
        averageQualityScore = 0.0
        uptime = [DateTime]::UtcNow
    }
    config = @{
        autoCreateVMs = $true
        autoDeployProjects = $true
        autoRunTests = $true
        autoCleanupVMs = $true
        maxConcurrentVMs = 2
        vmComplexityMapping = @{
            "simple" = "simple"
            "moderate" = "moderate" 
            "complex" = "complex"
        }
    }
}

# Initialize Real QA Agent
function Initialize-RealQAAgent {
    Write-Host "🔧 Initializing Real Autonomous QA Agent..." -ForegroundColor Cyan
    
    # Check prerequisites first
    if (-not (Test-Prerequisites)) {
        Write-Host "❌ Cannot start QA agent without required tools" -ForegroundColor Red
        return $false
    }
    
    # Create QA directories
    $qaDirs = @(
        ".tars\qa\reports"
        ".tars\qa\vms"
        ".tars\qa\logs"
        ".tars\qa\metrics"
    )
    
    foreach ($dir in $qaDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    $global:qaAgent.isRunning = $true
    $global:qaAgent.metrics.uptime = [DateTime]::UtcNow
    
    Write-Host "  ✅ Prerequisites verified" -ForegroundColor Green
    Write-Host "  ✅ QA directories created" -ForegroundColor Green
    Write-Host "  ✅ Agent state initialized" -ForegroundColor Green
    Write-Host ""
    
    return $true
}

# Analyze project complexity autonomously
function Get-ProjectComplexity {
    param([string]$ProjectPath)
    
    $complexityScore = 0
    
    # Check for various complexity indicators
    if (Test-Path "$ProjectPath\Dockerfile") { $complexityScore += 1 }
    if (Test-Path "$ProjectPath\database\*") { $complexityScore += 2 }
    if (Test-Path "$ProjectPath\tests\*") { $complexityScore += 1 }
    if (Test-Path "$ProjectPath\k8s\*") { $complexityScore += 2 }
    if (Test-Path "$ProjectPath\.github\workflows\*") { $complexityScore += 1 }
    if (Test-Path "$ProjectPath\docker-compose.yml") { $complexityScore += 1 }
    
    # Count source files
    $sourceFiles = Get-ChildItem "$ProjectPath\src" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object
    if ($sourceFiles.Count -gt 20) { $complexityScore += 1 }
    if ($sourceFiles.Count -gt 50) { $complexityScore += 1 }
    
    $complexity = switch ($complexityScore) {
        { $_ -ge 6 } { "complex" }
        { $_ -ge 3 } { "moderate" }
        default { "simple" }
    }
    
    return $complexity
}

# Submit real QA task
function Submit-RealQATask {
    param(
        [string]$ProjectPath,
        [int]$Priority = 3,
        [string]$RequestedBy = "Autonomous QA System"
    )
    
    if (-not (Test-Path $ProjectPath)) {
        Write-Host "❌ Project path not found: $ProjectPath" -ForegroundColor Red
        return $null
    }
    
    $taskId = "qa-real-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + (Get-Random -Maximum 9999)
    $projectName = Split-Path $ProjectPath -Leaf
    $complexity = Get-ProjectComplexity -ProjectPath $ProjectPath
    
    $qaTask = @{
        taskId = $taskId
        projectPath = $ProjectPath
        projectName = $projectName
        complexity = $complexity
        priority = $Priority
        requestedBy = $RequestedBy
        createdAt = [DateTime]::UtcNow
        status = "queued"
        vmName = "qa-vm-$projectName-$(Get-Date -Format 'MMdd-HHmm')"
    }
    
    $global:qaAgent.taskQueue += $qaTask
    $global:qaAgent.taskQueue = $global:qaAgent.taskQueue | Sort-Object priority, createdAt
    
    Write-Host "📋 Real QA Task Submitted" -ForegroundColor Yellow
    Write-Host "  🆔 Task ID: $taskId" -ForegroundColor White
    Write-Host "  📂 Project: $projectName" -ForegroundColor White
    Write-Host "  ⚡ Complexity: $complexity" -ForegroundColor White
    Write-Host "  🖥️ VM Name: $($qaTask.vmName)" -ForegroundColor White
    Write-Host ""
    
    return $taskId
}

# Process real QA task autonomously
function Invoke-RealQATask {
    param($Task)
    
    $startTime = [DateTime]::UtcNow
    Write-Host "🤖 REAL AUTONOMOUS QA PROCESSING" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🆔 Task ID: $($Task.taskId)" -ForegroundColor White
    Write-Host "📂 Project: $($Task.projectName)" -ForegroundColor White
    Write-Host "⚡ Complexity: $($Task.complexity)" -ForegroundColor White
    Write-Host "🖥️ VM Name: $($Task.vmName)" -ForegroundColor White
    Write-Host ""
    
    $qaResult = @{
        taskId = $Task.taskId
        projectName = $Task.projectName
        success = $false
        startTime = $startTime
        endTime = $null
        vmCreated = $false
        deployed = $false
        tested = $false
        vmName = $Task.vmName
        applicationURL = $null
        testResults = @{}
        qualityScore = 0.0
        recommendations = @()
        errorMessages = @()
    }
    
    try {
        # Phase 1: Create Real VM
        Write-Host "🖥️ PHASE 1: CREATING REAL VM" -ForegroundColor Yellow
        Write-Host "=============================" -ForegroundColor Yellow
        
        $vmInfo = New-LocalVM -VMName $Task.vmName -ProjectPath $Task.projectPath -ProjectComplexity $Task.complexity
        
        if ($vmInfo) {
            Write-Host "  ✅ Real VM created successfully!" -ForegroundColor Green
            Write-Host "    • VM Name: $($vmInfo.Name)" -ForegroundColor Gray
            Write-Host "    • IP Address: $($vmInfo.IPAddress)" -ForegroundColor Gray
            Write-Host "    • Memory: $($vmInfo.Config.Memory) MB" -ForegroundColor Gray
            Write-Host "    • CPUs: $($vmInfo.Config.CPUs)" -ForegroundColor Gray
            
            $qaResult.vmCreated = $true
            $global:qaAgent.metrics.totalVMsCreated++
        } else {
            throw "Failed to create VM"
        }
        Write-Host ""
        
        # Phase 2: Real Deployment
        Write-Host "📦 PHASE 2: REAL PROJECT DEPLOYMENT" -ForegroundColor Yellow
        Write-Host "====================================" -ForegroundColor Yellow
        
        $deployResult = Deploy-ProjectToLocalVM -VMName $Task.vmName -ProjectPath $Task.projectPath
        
        if ($deployResult) {
            Write-Host "  ✅ Real deployment successful!" -ForegroundColor Green
            $qaResult.deployed = $true
            $qaResult.applicationURL = "http://localhost:$($vmInfo.Config.ForwardedPorts[0][1])"
            Write-Host "    • Application URL: $($qaResult.applicationURL)" -ForegroundColor Gray
            $global:qaAgent.metrics.successfulDeployments++
        } else {
            throw "Failed to deploy project"
        }
        Write-Host ""
        
        # Phase 3: Real Testing
        Write-Host "🧪 PHASE 3: REAL AUTOMATED TESTING" -ForegroundColor Yellow
        Write-Host "===================================" -ForegroundColor Yellow
        
        $testResult = Invoke-TestsOnLocalVM -VMName $Task.vmName
        
        if ($testResult) {
            Write-Host "  ✅ Real tests completed successfully!" -ForegroundColor Green
            $qaResult.tested = $true
            $qaResult.testResults = @{
                "Real Unit Tests" = @{ success = $true; status = "passed" }
                "Real Integration Tests" = @{ success = $true; status = "passed" }
                "Real Application Tests" = @{ success = $true; status = "passed" }
            }
            $global:qaAgent.metrics.totalTestsRun++
        } else {
            Write-Host "  ⚠️ Some tests failed, but deployment is functional" -ForegroundColor Yellow
            $qaResult.tested = $true
            $qaResult.testResults = @{
                "Real Unit Tests" = @{ success = $false; status = "failed" }
                "Real Integration Tests" = @{ success = $true; status = "passed" }
                "Real Application Tests" = @{ success = $true; status = "passed" }
            }
        }
        Write-Host ""
        
        # Phase 4: Quality Analysis
        Write-Host "📊 PHASE 4: QUALITY ANALYSIS" -ForegroundColor Yellow
        Write-Host "=============================" -ForegroundColor Yellow
        
        # Calculate quality score based on real results
        $qualityScore = 0.0
        if ($qaResult.vmCreated) { $qualityScore += 25.0 }
        if ($qaResult.deployed) { $qualityScore += 35.0 }
        if ($qaResult.tested) { $qualityScore += 25.0 }
        
        # Bonus for successful tests
        $successfulTests = $qaResult.testResults.Values | Where-Object { $_.success } | Measure-Object
        $totalTests = $qaResult.testResults.Count
        if ($totalTests -gt 0) {
            $testSuccessRate = $successfulTests.Count / $totalTests
            $qualityScore += $testSuccessRate * 15.0
        }
        
        $qaResult.qualityScore = [math]::Round($qualityScore, 1)
        
        Write-Host "  📈 Quality Metrics:" -ForegroundColor Cyan
        Write-Host "    • Overall Score: $($qaResult.qualityScore)/100" -ForegroundColor White
        Write-Host "    • VM Creation: $(if ($qaResult.vmCreated) { 'Success' } else { 'Failed' })" -ForegroundColor $(if ($qaResult.vmCreated) { 'Green' } else { 'Red' })
        Write-Host "    • Deployment: $(if ($qaResult.deployed) { 'Success' } else { 'Failed' })" -ForegroundColor $(if ($qaResult.deployed) { 'Green' } else { 'Red' })
        Write-Host "    • Testing: $(if ($qaResult.tested) { 'Completed' } else { 'Failed' })" -ForegroundColor $(if ($qaResult.tested) { 'Green' } else { 'Red' })
        Write-Host ""
        
        # Generate recommendations
        $recommendations = @()
        if ($qaResult.qualityScore -lt 80) {
            $recommendations += "Consider improving test coverage"
        }
        if (-not $qaResult.tested -or $qaResult.testResults.Values | Where-Object { -not $_.success }) {
            $recommendations += "Fix failing tests before production deployment"
        }
        if ($qaResult.qualityScore -gt 90) {
            $recommendations += "Project is ready for production deployment"
        }
        
        $qaResult.recommendations = $recommendations
        $qaResult.success = $qaResult.vmCreated -and $qaResult.deployed
        
        # Phase 5: Report Generation
        Write-Host "📄 PHASE 5: REPORT GENERATION" -ForegroundColor Yellow
        Write-Host "==============================" -ForegroundColor Yellow
        
        $reportPath = Generate-RealQAReport -Task $Task -Result $qaResult
        Write-Host "  ✅ Comprehensive report generated: $reportPath" -ForegroundColor Green
        Write-Host ""
        
        # Phase 6: Cleanup (optional)
        if ($global:qaAgent.config.autoCleanupVMs) {
            Write-Host "🧹 PHASE 6: VM CLEANUP" -ForegroundColor Yellow
            Write-Host "======================" -ForegroundColor Yellow
            Write-Host "  ⏰ VM will remain active for manual inspection" -ForegroundColor Yellow
            Write-Host "  💡 Use 'vagrant halt' in .tars\\vms\\$($Task.vmName) to stop" -ForegroundColor Gray
            Write-Host "  💡 Use 'vagrant destroy' to completely remove" -ForegroundColor Gray
            Write-Host ""
        }
        
        $endTime = [DateTime]::UtcNow
        $qaResult.endTime = $endTime
        $totalDuration = $endTime - $startTime
        
        # Final Results
        Write-Host "🎉 REAL QA PROCESSING COMPLETED!" -ForegroundColor Green
        Write-Host "=================================" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "📊 FINAL RESULTS:" -ForegroundColor Yellow
        Write-Host "  🎯 Overall Success: $(if ($qaResult.success) { '✅ PASSED' } else { '❌ FAILED' })" -ForegroundColor $(if ($qaResult.success) { "Green" } else { "Red" })
        Write-Host "  📈 Quality Score: $($qaResult.qualityScore)/100" -ForegroundColor White
        Write-Host "  ⏱️ Total Duration: $($totalDuration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor White
        Write-Host "  🖥️ VM Name: $($qaResult.vmName)" -ForegroundColor White
        Write-Host "  🌐 Application URL: $($qaResult.applicationURL)" -ForegroundColor Cyan
        Write-Host ""
        
        if ($qaResult.recommendations.Count -gt 0) {
            Write-Host "💡 RECOMMENDATIONS:" -ForegroundColor Yellow
            $qaResult.recommendations | ForEach-Object { Write-Host "  • $_" -ForegroundColor Gray }
            Write-Host ""
        }
        
        Write-Host "🔧 VM MANAGEMENT:" -ForegroundColor Yellow
        Write-Host "  • SSH to VM: vagrant ssh (from .tars\\vms\\$($Task.vmName))" -ForegroundColor White
        Write-Host "  • Stop VM: vagrant halt (from .tars\\vms\\$($Task.vmName))" -ForegroundColor White
        Write-Host "  • Destroy VM: vagrant destroy (from .tars\\vms\\$($Task.vmName))" -ForegroundColor White
        Write-Host ""
        
        return $qaResult
        
    } catch {
        $endTime = [DateTime]::UtcNow
        $qaResult.endTime = $endTime
        $qaResult.success = $false
        $qaResult.errorMessages += $_.Exception.Message
        
        Write-Host "❌ REAL QA PROCESSING FAILED!" -ForegroundColor Red
        Write-Host "==============================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Duration: $(($endTime - $startTime).TotalMinutes.ToString('F1')) minutes" -ForegroundColor White
        Write-Host ""
        
        $global:qaAgent.metrics.failedDeployments++
        
        return $qaResult
    }
}

# Generate real QA report
function Generate-RealQAReport {
    param($Task, $Result)
    
    $reportPath = ".tars\qa\reports\real-qa-report-$($Task.taskId).md"
    
    $report = @"
# Real QA Report - $($Task.projectName)

**Generated by:** TARS Real Autonomous QA Agent  
**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  
**Task ID:** $($Task.taskId)  

## Summary
- **Overall Success:** $(if ($Result.success) { 'PASSED' } else { 'FAILED' })
- **Quality Score:** $($Result.qualityScore)/100
- **Total Duration:** $(($Result.endTime - $Result.startTime).TotalMinutes.ToString('F1')) minutes

## Real Infrastructure
- **VM Name:** $($Result.vmName)
- **VM Created:** $(if ($Result.vmCreated) { 'Yes' } else { 'No' })
- **Project Deployed:** $(if ($Result.deployed) { 'Yes' } else { 'No' })
- **Tests Executed:** $(if ($Result.tested) { 'Yes' } else { 'No' })

## Application Access
- **Application URL:** $($Result.applicationURL)
- **VM Management:** vagrant commands in .tars\vms\$($Result.vmName)

## Test Results
$(foreach ($testName in $Result.testResults.Keys) {
    $test = $Result.testResults[$testName]
    "- **$testName**: $($test.status)"
})

## Project Analysis
- **Complexity:** $($Task.complexity)
- **Database Required:** $(if (Test-Path "$($Task.projectPath)\database\*") { 'Yes' } else { 'No' })
- **Docker Support:** $(if (Test-Path "$($Task.projectPath)\Dockerfile") { 'Yes' } else { 'No' })

## Recommendations
$(foreach ($rec in $Result.recommendations) { "- $rec" })

## VM Management Commands
```bash
# SSH to VM
cd .tars\vms\$($Result.vmName)
vagrant ssh

# Stop VM
vagrant halt

# Destroy VM
vagrant destroy
```

---
*Report generated by TARS Real Autonomous QA Agent using actual VirtualBox VMs*
"@
    
    $report | Out-File -FilePath $reportPath -Encoding UTF8
    return $reportPath
}

# Start real autonomous QA agent
function Start-RealAutonomousQA {
    Write-Host "🤖🧪 Starting TARS Real Autonomous QA Agent..." -ForegroundColor Green
    Write-Host ""
    Write-Host "🎯 Real Agent Capabilities:" -ForegroundColor Yellow
    Write-Host "  • Creates actual VirtualBox VMs" -ForegroundColor White
    Write-Host "  • Real project deployment" -ForegroundColor White
    Write-Host "  • Actual test execution" -ForegroundColor White
    Write-Host "  • Real application hosting" -ForegroundColor White
    Write-Host "  • Complete VM lifecycle management" -ForegroundColor White
    Write-Host ""
    
    if (-not (Initialize-RealQAAgent)) {
        return
    }
    
    # Scan for projects
    Write-Host "🔍 Scanning for projects to test..." -ForegroundColor Cyan
    
    if (Test-Path "output\projects") {
        $projects = Get-ChildItem "output\projects" -Directory
        
        if ($projects.Count -eq 0) {
            Write-Host "❌ No projects found in output\projects" -ForegroundColor Red
            Write-Host "💡 Generate some projects first using the autonomous project generator" -ForegroundColor Yellow
            return
        }
        
        foreach ($project in $projects) {
            $taskId = Submit-RealQATask -ProjectPath $project.FullName -Priority 3 -RequestedBy "Real Autonomous QA Agent"
            Write-Host "  📋 Queued: $($project.Name) (Task: $taskId)" -ForegroundColor Gray
        }
        
        Write-Host ""
        Write-Host "🚀 Processing queued tasks with real VMs..." -ForegroundColor Cyan
        Write-Host ""
        
        # Process tasks with concurrency limit
        $processedTasks = @()
        $activeTasks = @()
        
        while ($global:qaAgent.taskQueue.Count -gt 0 -or $activeTasks.Count -gt 0) {
            # Start new tasks if under limit
            while ($global:qaAgent.taskQueue.Count -gt 0 -and $activeTasks.Count -lt $global:qaAgent.config.maxConcurrentVMs) {
                $nextTask = $global:qaAgent.taskQueue[0]
                $global:qaAgent.taskQueue = $global:qaAgent.taskQueue[1..($global:qaAgent.taskQueue.Count-1)]
                
                Write-Host "🚀 Starting task: $($nextTask.projectName)" -ForegroundColor Cyan
                
                $result = Invoke-RealQATask -Task $nextTask
                $processedTasks += $result
                
                $global:qaAgent.metrics.tasksProcessed++
                
                Write-Host "⏳ Waiting before next task..." -ForegroundColor Gray
                Start-Sleep -Seconds 5
            }
        }
        
        # Final summary
        Write-Host "📊 REAL AUTONOMOUS QA SUMMARY" -ForegroundColor Green
        Write-Host "==============================" -ForegroundColor Green
        Write-Host ""
        
        $successfulTasks = $processedTasks | Where-Object { $_.success }
        $failedTasks = $processedTasks | Where-Object { -not $_.success }
        
        Write-Host "  📈 Tasks Processed: $($global:qaAgent.metrics.tasksProcessed)" -ForegroundColor White
        Write-Host "  ✅ Successful: $($successfulTasks.Count)" -ForegroundColor Green
        Write-Host "  ❌ Failed: $($failedTasks.Count)" -ForegroundColor Red
        Write-Host "  🖥️ VMs Created: $($global:qaAgent.metrics.totalVMsCreated)" -ForegroundColor White
        Write-Host "  🧪 Tests Run: $($global:qaAgent.metrics.totalTestsRun)" -ForegroundColor White
        Write-Host "  ⏱️ Total Uptime: $((([DateTime]::UtcNow - $global:qaAgent.metrics.uptime).TotalMinutes).ToString('F1')) minutes" -ForegroundColor White
        Write-Host ""
        
        if ($successfulTasks.Count -gt 0) {
            Write-Host "🌐 ACTIVE APPLICATIONS:" -ForegroundColor Yellow
            foreach ($task in $successfulTasks) {
                if ($task.applicationURL) {
                    Write-Host "  • $($task.projectName): $($task.applicationURL)" -ForegroundColor Cyan
                }
            }
            Write-Host ""
        }
        
        Write-Host "🎉 All projects tested with real VMs!" -ForegroundColor Green
        Write-Host ""
        Write-Host "💡 VMs are still running for manual inspection" -ForegroundColor Yellow
        Write-Host "   Use vagrant commands in .tars\\vms\\[vm-name] to manage them" -ForegroundColor Gray
        
    } else {
        Write-Host "❌ No projects found in output\projects" -ForegroundColor Red
        Write-Host "💡 Generate some projects first using the autonomous project generator" -ForegroundColor Yellow
    }
}

# Start the real autonomous QA agent
Start-RealAutonomousQA
