[13:43:11 INF] Starting Evolutive Grammar (Macro) Demo...
[13:43:11 INF] Registering macro 'demo_greeting'...
[13:43:11 INF] Registry contains: ["demo_greeting"]
[13:43:11 INF] Executing main workflow...
[13:43:15 ERR] Execution failed
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
   at Tars.Llm.OpenAiCompatibleClient.sendChatAsync@106-1.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Llm\OpenAiCompatibleClient.fs:line 115
   at Tars.Llm.LlmService.Tars-Llm-LlmService-ILlmService-CompleteAsync@115.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Llm\LlmService.fs:line 120
   at Tars.Metascript.Engine.executeStep@1110.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1169
   at Tars.Metascript.Engine.run@1622.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1675
   at Tars.Metascript.Engine.executeStep@1110.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1602
   at Tars.Metascript.Engine.executeStep@1110.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1611
   at Tars.Metascript.Engine.run@1622.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\Engine.fs:line 1675
   at Tars.Interface.Cli.Commands.MacroDemo.run@16-1.MoveNext() in C:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\Commands\MacroDemo.fs:line 120
