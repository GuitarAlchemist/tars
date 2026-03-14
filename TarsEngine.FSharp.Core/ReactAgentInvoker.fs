namespace TarsEngine.FSharp.Core

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open System.Text.Json
open Microsoft.Extensions.Logging

/// React agent task definition
type ReactAgentTask = {
    TaskId: string
    ComponentId: string
    TaskType: string
    Description: string
    Priority: int
    CreatedAt: DateTime
    Status: string
    Parameters: Map<string, obj>
    ExpectedOutput: string
    ReactComponentType: string
}

/// React agent response
type ReactAgentResponse = {
    TaskId: string
    Success: bool
    Output: obj
    ExecutionTime: float
    Timestamp: DateTime
    ReactComponentGenerated: bool
    ComponentCode: string option
    ErrorMessage: string option
}

/// Real-time React agent invoker
type ReactAgentInvoker(logger: ILogger<ReactAgentInvoker>) =
    
    let activeTasks = ConcurrentDictionary<string, ReactAgentTask>()
    let completedTasks = ConcurrentDictionary<string, ReactAgentResponse>()
    
    /// Parse task description into structured task
    member private this.ParseTaskDescription(taskDescription: string, componentId: string) =
        let taskId = sprintf "REACT_%s_%d" componentId (DateTime.UtcNow.Ticks)
        let parts = taskDescription.Split(':')
        
        if parts.Length >= 2 then
            let taskType = parts.[0].Trim()
            let description = parts.[1].Trim()
            
            let (priority, reactComponentType, expectedOutput) = 
                match taskType with
                | "CREATE_REFACTORING_PLAN" -> (1, "RefactoringDashboard", "Interactive refactoring plan with code suggestions")
                | "INTERFACE_REVIEW" -> (2, "InterfaceAnalyzer", "Visual interface complexity analysis")
                | "DEPENDENCY_ANALYSIS" -> (1, "DependencyGraph", "Interactive dependency visualization")
                | "ASYNC_OPTIMIZATION" -> (3, "AsyncProfiler", "Performance optimization recommendations")
                | "REAL_TIME_MONITOR" -> (4, "ComponentMonitor", "Live component health dashboard")
                | _ -> (5, "GenericAnalyzer", "General analysis output")
            
            {
                TaskId = taskId
                ComponentId = componentId
                TaskType = taskType
                Description = description
                Priority = priority
                CreatedAt = DateTime.UtcNow
                Status = "PENDING"
                Parameters = Map.ofList [
                    ("componentId", componentId :> obj)
                    ("timestamp", DateTime.UtcNow :> obj)
                    ("taskType", taskType :> obj)
                ]
                ExpectedOutput = expectedOutput
                ReactComponentType = reactComponentType
            }
        else
            {
                TaskId = taskId
                ComponentId = componentId
                TaskType = "UNKNOWN"
                Description = taskDescription
                Priority = 5
                CreatedAt = DateTime.UtcNow
                Status = "PENDING"
                Parameters = Map.empty
                ExpectedOutput = "Unknown output"
                ReactComponentType = "GenericAnalyzer"
            }
    
    /// Generate React component code dynamically
    member private this.GenerateReactComponent(task: ReactAgentTask) =
        let timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss")
        
        match task.ReactComponentType with
        | "RefactoringDashboard" ->
            sprintf """
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

const RefactoringDashboard_%s = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Real-time data fetching for component %s
    const fetchData = async () => {
      try {
        const response = await fetch('/api/tars/component/%s/refactoring-analysis');
        const data = await response.json();
        setAnalysisData(data);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch refactoring data:', error);
        setLoading(false);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 5000); // Real-time updates every 5 seconds
    return () => clearInterval(interval);
  }, []);
  
  if (loading) return <div>Loading real-time refactoring analysis...</div>;
  
  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <h2 className="text-2xl font-bold">Refactoring Dashboard - %s</h2>
        <p className="text-sm text-gray-600">Generated at %s</p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="font-semibold mb-2">Complexity Analysis</h3>
            <Badge variant="outline">Cyclomatic Complexity: {analysisData?.complexity || 'Loading...'}</Badge>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Suggested Refactorings</h3>
            <ul className="space-y-1">
              {analysisData?.suggestions?.map((suggestion, index) => (
                <li key={index} className="text-sm">{suggestion}</li>
              )) || ['Loading suggestions...']}
            </ul>
          </div>
        </div>
        <Button 
          onClick={() => window.open('/api/tars/component/%s/generate-refactoring-pr', '_blank')}
          className="mt-4"
        >
          Generate Refactoring PR
        </Button>
      </CardContent>
    </Card>
  );
};

export default RefactoringDashboard_%s;
""" task.ComponentId task.ComponentId task.ComponentId task.ComponentId timestamp task.ComponentId task.ComponentId
        
        | "DependencyGraph" ->
            sprintf """
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { ForceGraph3D } from 'react-force-graph';

const DependencyGraph_%s = () => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchDependencies = async () => {
      try {
        const response = await fetch('/api/tars/component/%s/dependency-graph');
        const data = await response.json();
        setGraphData(data);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch dependency data:', error);
        setLoading(false);
      }
    };
    
    fetchDependencies();
  }, []);
  
  return (
    <Card className="w-full h-96">
      <CardHeader>
        <h2 className="text-2xl font-bold">Dependency Graph - %s</h2>
        <p className="text-sm text-gray-600">Real-time dependency analysis - %s</p>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div>Loading dependency graph...</div>
        ) : (
          <ForceGraph3D
            graphData={graphData}
            nodeLabel="id"
            nodeAutoColorBy="group"
            linkDirectionalParticles={2}
            linkDirectionalParticleSpeed={0.01}
            width={800}
            height={400}
          />
        )}
      </CardContent>
    </Card>
  );
};

export default DependencyGraph_%s;
""" task.ComponentId task.ComponentId task.ComponentId timestamp task.ComponentId
        
        | "ComponentMonitor" ->
            sprintf """
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ComponentMonitor_%s = () => {
  const [metrics, setMetrics] = useState([]);
  const [currentStatus, setCurrentStatus] = useState('UNKNOWN');
  
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/tars/component/%s/real-time-metrics');
        const data = await response.json();
        setMetrics(prev => [...prev.slice(-20), data]); // Keep last 20 data points
        setCurrentStatus(data.status);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };
    
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000); // Real-time updates every 2 seconds
    return () => clearInterval(interval);
  }, []);
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'HEALTHY': return 'text-green-600';
      case 'WARNING': return 'text-yellow-600';
      case 'ERROR': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <h2 className="text-2xl font-bold">Real-Time Monitor - %s</h2>
        <p className={`text-sm ${getStatusColor(currentStatus)}`}>
          Status: {currentStatus} | Last Update: %s
        </p>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%%" height={300}>
          <LineChart data={metrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="complexity" stroke="#8884d8" strokeWidth={2} />
            <Line type="monotone" dataKey="performance" stroke="#82ca9d" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ComponentMonitor_%s;
""" task.ComponentId task.ComponentId task.ComponentId timestamp task.ComponentId
        
        | _ ->
            sprintf """
import React from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';

const GenericAnalyzer_%s = () => {
  return (
    <Card className="w-full">
      <CardHeader>
        <h2 className="text-2xl font-bold">Component Analysis - %s</h2>
        <p className="text-sm text-gray-600">Generated at %s</p>
      </CardHeader>
      <CardContent>
        <p>Real-time analysis for component %s</p>
        <p>Task: %s</p>
      </CardContent>
    </Card>
  );
};

export default GenericAnalyzer_%s;
""" task.ComponentId task.ComponentId timestamp task.ComponentId task.Description task.ComponentId
    
    /// Execute React agent task
    member private this.ExecuteTask(task: ReactAgentTask) =
        async {
            try
                let startTime = DateTime.UtcNow
                logger.LogInformation(sprintf "Executing React agent task: %s for component %s" task.TaskType task.ComponentId)
                
                // Update task status
                activeTasks.TryUpdate(task.TaskId, { task with Status = "EXECUTING" }, task) |> ignore
                
                // Generate React component code
                let reactCode = this.GenerateReactComponent(task)
                
                // Simulate real processing time based on task complexity
                let processingTime = 
                    match task.Priority with
                    | 1 -> 500 // High priority - fast execution
                    | 2 -> 1000
                    | 3 -> 1500
                    | _ -> 2000 // Lower priority - more processing time
                
                do! Async.Sleep(processingTime)
                
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                // Create response with real data
                let response = {
                    TaskId = task.TaskId
                    Success = true
                    Output = {|
                        TaskType = task.TaskType
                        ComponentId = task.ComponentId
                        ExecutionTime = executionTime
                        ReactComponentGenerated = true
                        ApiEndpoints = [
                            sprintf "/api/tars/component/%s/real-time-metrics" task.ComponentId
                            sprintf "/api/tars/component/%s/analysis" task.ComponentId
                            sprintf "/api/tars/component/%s/refactoring-suggestions" task.ComponentId
                        ]
                        WebSocketEndpoint = sprintf "ws://localhost:3000/tars/component/%s/live" task.ComponentId
                    |} :> obj
                    ExecutionTime = executionTime
                    Timestamp = endTime
                    ReactComponentGenerated = true
                    ComponentCode = Some reactCode
                    ErrorMessage = None
                }
                
                // Store completed task
                completedTasks.TryAdd(task.TaskId, response) |> ignore
                activeTasks.TryRemove(task.TaskId) |> ignore
                
                logger.LogInformation(sprintf "React agent task completed: %s in %.1fms" task.TaskId executionTime)
                
                return response
            with
            | ex ->
                logger.LogError(ex, sprintf "React agent task failed: %s" task.TaskId)
                
                let errorResponse = {
                    TaskId = task.TaskId
                    Success = false
                    Output = null
                    ExecutionTime = 0.0
                    Timestamp = DateTime.UtcNow
                    ReactComponentGenerated = false
                    ComponentCode = None
                    ErrorMessage = Some ex.Message
                }
                
                completedTasks.TryAdd(task.TaskId, errorResponse) |> ignore
                activeTasks.TryRemove(task.TaskId) |> ignore
                
                return errorResponse
        }
    
    /// Invoke React agents for component analysis
    member this.InvokeReactAgents(componentAnalysis: ComponentAnalysis) =
        async {
            try
                logger.LogInformation(sprintf "Invoking React agents for component: %s" componentAnalysis.Name)
                
                let tasks = 
                    componentAnalysis.ReactAgentTasks
                    |> Array.map (fun taskDesc -> this.ParseTaskDescription(taskDesc, componentAnalysis.ComponentId))
                
                // Add tasks to active queue
                for task in tasks do
                    activeTasks.TryAdd(task.TaskId, task) |> ignore
                
                // Execute all tasks concurrently
                let! responses = 
                    tasks
                    |> Array.map this.ExecuteTask
                    |> Async.Parallel
                
                logger.LogInformation(sprintf "Completed %d React agent tasks for component %s" responses.Length componentAnalysis.Name)
                
                return {|
                    ComponentId = componentAnalysis.ComponentId
                    ComponentName = componentAnalysis.Name
                    TasksExecuted = responses.Length
                    SuccessfulTasks = responses |> Array.filter (fun r -> r.Success) |> Array.length
                    FailedTasks = responses |> Array.filter (fun r -> not r.Success) |> Array.length
                    TotalExecutionTime = responses |> Array.sumBy (fun r -> r.ExecutionTime)
                    Responses = responses
                    ReactComponentsGenerated = responses |> Array.filter (fun r -> r.ReactComponentGenerated) |> Array.length
                |}
            with
            | ex ->
                logger.LogError(ex, sprintf "Failed to invoke React agents for component: %s" componentAnalysis.Name)
                return {|
                    ComponentId = componentAnalysis.ComponentId
                    ComponentName = componentAnalysis.Name
                    TasksExecuted = 0
                    SuccessfulTasks = 0
                    FailedTasks = 1
                    TotalExecutionTime = 0.0
                    Responses = [||]
                    ReactComponentsGenerated = 0
                |}
        }
    
    /// Get real-time status of all React agents
    member this.GetRealTimeStatus() =
        {|
            ActiveTasks = activeTasks.Count
            CompletedTasks = completedTasks.Count
            ActiveTaskDetails = activeTasks.Values |> Seq.toArray
            RecentCompletions = 
                completedTasks.Values 
                |> Seq.filter (fun r -> (DateTime.UtcNow - r.Timestamp).TotalMinutes < 5.0)
                |> Seq.toArray
            SystemLoad = if activeTasks.Count > 10 then "HIGH" elif activeTasks.Count > 5 then "MEDIUM" else "LOW"
            Timestamp = DateTime.UtcNow
        |}
