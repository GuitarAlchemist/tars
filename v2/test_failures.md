  Determining projects to restore...
  All projects are up-to-date for restore.
  Tars.Security -> C:\Users\spare\source\repos\tars\v2\src\Tars.Security\bin\Debug\net10.0\Tars.Security.dll
  Tars.Core -> C:\Users\spare\source\repos\tars\v2\src\Tars.Core\bin\Debug\net10.0\Tars.Core.dll
  Tars.Sandbox -> C:\Users\spare\source\repos\tars\v2\src\Tars.Sandbox\bin\Debug\net10.0\Tars.Sandbox.dll
  Tars.Llm -> C:\Users\spare\source\repos\tars\v2\src\Tars.Llm\bin\Debug\net10.0\Tars.Llm.dll
  Tars.Connectors -> C:\Users\spare\source\repos\tars\v2\src\Tars.Connectors\bin\Debug\net10.0\Tars.Connectors.dll
  Tars.Kernel -> C:\Users\spare\source\repos\tars\v2\src\Tars.Kernel\bin\Debug\net10.0\Tars.Kernel.dll
  Tars.Cortex -> C:\Users\spare\source\repos\tars\v2\src\Tars.Cortex\bin\Debug\net10.0\Tars.Cortex.dll
  Tars.Metascript -> C:\Users\spare\source\repos\tars\v2\src\Tars.Metascript\bin\Debug\net10.0\Tars.Metascript.dll
  Tars.Tools -> C:\Users\spare\source\repos\tars\v2\src\Tars.Tools\bin\Debug\net10.0\Tars.Tools.dll
  Tars.Graph -> C:\Users\spare\source\repos\tars\v2\src\Tars.Graph\bin\Debug\net10.0\Tars.Graph.dll
  Tars.Evolution -> C:\Users\spare\source\repos\tars\v2\src\Tars.Evolution\bin\Debug\net10.0\Tars.Evolution.dll
  Tars.Tests -> C:\Users\spare\source\repos\tars\v2\tests\Tars.Tests\bin\Debug\net10.0\Tars.Tests.dll
Test run for C:\Users\spare\source\repos\tars\v2\tests\Tars.Tests\bin\Debug\net10.0\Tars.Tests.dll (.NETCoreApp,Version=v10.0)
VSTest version 18.0.1 (x64)

Starting test execution, please wait...
A total of 1 test files matched the specified pattern.
[xUnit.net 00:00:00.38]     Tars.Tests.ToolTests.runCommand executes in sandbox [FAIL]
[xUnit.net 00:00:00.38]     Tars.Tests.SandboxTests.Can run python script in sandbox [FAIL]
  Failed Tars.Tests.ToolTests.runCommand executes in sandbox [43 ms]
  Error Message:
   Assert.Equal() Failure: Strings differ
           Γåô (pos 0)
Expected: "hello from sandbox"
Actual:   "Sandbox Error: Docker Error: Docker API r"┬╖┬╖┬╖
           Γåæ (pos 0)
  Stack Trace:
     at <StartupCode$Tars-Tests>.$ToolTests.runCommand executes in sandbox@12.MoveNext() in C:\Users\spare\source\repos\tars\v2\tests\Tars.Tests\ToolTests.fs:line 18
--- End of stack trace from previous location ---
  Failed Tars.Tests.SandboxTests.Can run python script in sandbox [221 ms]
  Error Message:
   Failed to run container: Docker Error: Docker API responded with status code=NotFound, response={"message":"No such image: tars-sandbox:latest"}

  Stack Trace:
     at <StartupCode$Tars-Tests>.$SandboxTests.Can run python script in sandbox@12.MoveNext() in C:\Users\spare\source\repos\tars\v2\tests\Tars.Tests\SandboxTests.fs:line 28
--- End of stack trace from previous location ---
[xUnit.net 00:00:00.39]     Tars.Tests.SandboxTests.Sandbox has no internet access [FAIL]
  Failed Tars.Tests.SandboxTests.Sandbox has no internet access [9 ms]
  Error Message:
   Failed to run container: Docker Error: Docker API responded with status code=NotFound, response={"message":"No such image: tars-sandbox:latest"}

  Stack Trace:
     at <StartupCode$Tars-Tests>.$SandboxTests.Sandbox has no internet access@33.MoveNext() in C:\Users\spare\source\repos\tars\v2\tests\Tars.Tests\SandboxTests.fs:line 54
--- End of stack trace from previous location ---
[xUnit.net 00:00:00.39]     Tars.Tests.ToolTests.runCommand runs in isolated OS [FAIL]
  Failed Tars.Tests.ToolTests.runCommand runs in isolated OS [17 ms]
  Error Message:
   Assert.Contains() Failure: Sub-string not found
String:    "Sandbox Error: Docker Error: Docker API r"┬╖┬╖┬╖
Not found: "PRETTY_NAME="Debian"
  Stack Trace:
     at <StartupCode$Tars-Tests>.$ToolTests.runCommand runs in isolated OS@23.MoveNext() in C:\Users\spare\source\repos\tars\v2\tests\Tars.Tests\ToolTests.fs:line 28
--- End of stack trace from previous location ---

Failed!  - Failed:     4, Passed:   363, Skipped:     0, Total:   367, Duration: 5 s - Tars.Tests.dll (net10.0)
