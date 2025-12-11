C:\Users\spare\source\repos\tars\v2\src\Tars.Cortex\Tars.Cortex.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved. [C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Tars.Interface.Cli.fsproj]
C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Tars.Metascript.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved. [C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Tars.Interface.Cli.fsproj]
C:\Users\spare\source\repos\tars\v2\src\Tars.Evolution\Tars.Evolution.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved. [C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Tars.Interface.Cli.fsproj]
C:\Users\spare\source\repos\tars\v2\src\Tars.Graph\Tars.Graph.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved. [C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Tars.Interface.Cli.fsproj]
C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Tars.Interface.Cli.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved.
C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Tars.Interface.Cli.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved.
C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Tars.Metascript.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved.
C:\Users\spare\source\repos\tars\v2\src\Tars.Evolution\Tars.Evolution.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved.
C:\Users\spare\source\repos\tars\v2\src\Tars.Graph\Tars.Graph.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved.
C:\Users\spare\source\repos\tars\v2\src\Tars.Cortex\Tars.Cortex.fsproj : warning NU1608: Detected package version outside of dependency constraint: FSharp.Compiler.Service 43.10.100 requires FSharp.Core (= 10.0.100) but version FSharp.Core 10.0.101 was resolved.
[15:22:21 INF] Loading workflow from demo_workflow.json...
[15:22:21 INF] Workflow 'JokeGenerator' loaded successfully.
[15:22:21 INF] Starting execution...
[15:22:25 ERR] Workflow execution failed
System.Net.Http.HttpRequestException: No connection could be made because the target machine actively refused it. (localhost:8000)
 ---> System.Net.Sockets.SocketException (10061): No connection could be made because the target machine actively refused it.
   at System.Net.Sockets.Socket.AwaitableSocketAsyncEventArgs.ThrowException(SocketError error, CancellationToken cancellationToken)
   at System.Net.Sockets.Socket.AwaitableSocketAsyncEventArgs.System.Threading.Tasks.Sources.IValueTaskSource.GetResult(Int16 token)
   at System.Net.Http.HttpConnectionPool.ConnectToTcpHostAsync(String host, Int32 port, HttpRequestMessage initialRequest, Boolean async, CancellationToken cancellationToken)
   --- End of inner exception stack trace ---
   at System.Net.Http.HttpConnectionPool.ConnectToTcpHostAsync(String host, Int32 port, HttpRequestMessage initialRequest, Boolean async, CancellationToken cancellationToken)
   at System.Net.Http.HttpConnectionPool.ConnectAsync(HttpRequestMessage request, Boolean async, CancellationToken cancellationToken)
   at System.Net.Http.HttpConnectionPool.CreateHttp11ConnectionAsync(HttpRequestMessage request, Boolean async, CancellationToken cancellationToken)
   at System.Net.Http.HttpConnectionPool.InjectNewHttp11ConnectionAsync(QueueItem queueItem)
   at System.Threading.Tasks.TaskCompletionSourceWithCancellation`1.WaitWithCancellationAsync(CancellationToken cancellationToken)
   at System.Net.Http.HttpConnectionPool.SendWithVersionDetectionAndRetryAsync(HttpRequestMessage request, Boolean async, Boolean doRequestAuth, CancellationToken cancellationToken)
   at System.Net.Http.RedirectHandler.SendAsync(HttpRequestMessage request, Boolean async, CancellationToken cancellationToken)
   at System.Net.Http.SocketsHttpHandler.<SendAsync>g__CreateHandlerAndSendAsync|115_0(HttpRequestMessage request, CancellationToken cancellationToken)
   at System.Net.Http.HttpClient.<SendAsync>g__Core|83_0(HttpRequestMessage request, HttpCompletionOption completionOption, CancellationTokenSource cts, Boolean disposeCts, CancellationTokenSource pendingRequestsCts, CancellationToken originalCancellationToken)
   at Tars.Llm.OpenAiCompatibleClient.sendChatAsync@114-1.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Llm\OpenAiCompatibleClient.fs:line 150
   at Tars.Llm.LlmService.Tars-Llm-LlmService-ILlmService-CompleteAsync@132.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Llm\LlmService.fs:line 138
   at Tars.Metascript.Engine.executeStep@1109.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1168
   at Tars.Metascript.Engine.startTask@1665.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1668
   at Tars.Metascript.Engine.schedulerLoop@1637-1.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1689
   at Tars.Metascript.Engine.run@1708.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1751
   at Tars.Interface.Cli.Commands.RunCommand.run@27-27.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Commands\RunCommand.fs:line 93
