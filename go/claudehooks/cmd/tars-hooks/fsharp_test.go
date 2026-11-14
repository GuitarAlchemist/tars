package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fsharpRepo is a temporary tree, in forward slashes, holding src/App/{A,B}.fsproj and
// src/App/sub/deep/.
func fsharpRepo(t *testing.T) string {
	t.Helper()
	root := filepath.ToSlash(t.TempDir())
	deep := filepath.Join(root, "src", "App", "sub", "deep")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"B.fsproj", "A.fsproj", "Lib.fs"} {
		if err := os.WriteFile(filepath.Join(root, "src", "App", f), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// dotnetSystem: dotnet on PATH, `dotnet build` answering build, the tool lists answering lists.
func dotnetSystem(calls *[]string, env map[string]string, build func() (string, bool, bool), lists map[string]string) *system {
	sys := fakeSystem("", time.Now())
	sys.getenv = func(name string) string { return env[name] }
	sys.lookPath = func(name string) bool { return name == "dotnet" }
	sys.output = func(dir, name string, args ...string) (string, bool) {
		return lists[name+" "+strings.Join(args, " ")], true
	}
	sys.runTimeout = func(limit time.Duration, name string, args ...string) (string, bool, bool) {
		if name == "dotnet" && len(args) > 0 && args[0] == "build" && build != nil {
			return build()
		}
		return "", true, false
	}
	return recording(sys, calls)
}

func TestFsharpCheckBuildsTheNearestProject(t *testing.T) {
	root := fsharpRepo(t)
	app := root + "/src/App"
	for _, c := range []struct {
		desc, stdin string
		env         map[string]string
		want        string // the build command, "" for none
	}{
		{"an edited .fs builds its directory's first project", editPayload("Edit", app+"/Lib.fs"), nil, "dotnet build --no-restore -nologo -clp:NoSummary -v:q " + app + "/A.fsproj"},
		{"the walk goes up from a nested directory", editPayload("Write", app+"/sub/deep/X.fsi"), nil, "dotnet build --no-restore -nologo -clp:NoSummary -v:q " + app + "/A.fsproj"},
		{"MultiEdit counts", editPayload("MultiEdit", app+"/Lib.fs"), nil, "dotnet build --no-restore -nologo -clp:NoSummary -v:q " + app + "/A.fsproj"},
		{"an edited .fsproj is built itself", editPayload("Edit", `C:\x\B.fsproj`), nil, `dotnet build --no-restore -nologo -clp:NoSummary -v:q C:\x\B.fsproj`},
		{"CLAUDE_FILE_PATH wins over stdin", editPayload("Edit", "/nowhere/a.cs"), map[string]string{"CLAUDE_FILE_PATH": app + "/Lib.fs"}, "dotnet build --no-restore -nologo -clp:NoSummary -v:q " + app + "/A.fsproj"},
		{"no tool name: a direct invocation", "", map[string]string{"CLAUDE_FILE_PATH": app + "/Lib.fs"}, "dotnet build --no-restore -nologo -clp:NoSummary -v:q " + app + "/A.fsproj"},
		{"CLAUDE_TOOL_NAME wins over stdin", editPayload("Edit", app+"/Lib.fs"), map[string]string{"CLAUDE_TOOL_NAME": "Read"}, ""},
		{"a backslash path names no directory", editPayload("Edit", strings.ReplaceAll(app, "/", `\`)+`\Lib.fs`), nil, ""},
		{"no project above", editPayload("Edit", root+"/Lone.fs"), nil, ""},
		{".fsx is not built", editPayload("Edit", app+"/run.fsx"), nil, ""},
		{"v1/ is skipped", editPayload("Edit", root+"/v1/src/App/Lib.fs"), nil, ""},
		{"parked_legacy/ is skipped", editPayload("Edit", root+"/parked_legacy/Lib.fs"), nil, ""},
		{"autonomous_backups/ is skipped", editPayload("Edit", root+"/autonomous_backups/Lib.fs"), nil, ""},
		{".tars/ is skipped", editPayload("Edit", root+"/.tars/Lib.fs"), nil, ""},
		{"Read is ignored", editPayload("Read", app+"/Lib.fs"), nil, ""},
		{"CLAUDE_DISABLE_FSHARP_CHECK", editPayload("Edit", app+"/Lib.fs"), map[string]string{"CLAUDE_DISABLE_FSHARP_CHECK": "1"}, ""},
		{"not JSON", "garbage", nil, ""},
	} {
		var calls []string
		code, out, errs := runHook(t, dotnetSystem(&calls, c.env, nil, nil), "fsharp-check", c.stdin)
		var builds []string
		for _, call := range calls {
			if strings.HasPrefix(call, "dotnet build") {
				builds = append(builds, call)
			}
		}
		got := strings.Join(builds, "; ")
		if code != 0 || out != "" || errs != "" || got != c.want {
			t.Errorf("%s: exit %d, out %q, stderr %q, built %q, want %q", c.desc, code, out, errs, got, c.want)
		}
	}
}

func TestFsharpCheckReportsFailuresOnly(t *testing.T) {
	root := fsharpRepo(t)
	in := editPayload("Edit", root+"/src/App/Lib.fs")
	var buildOut strings.Builder
	buildOut.WriteString("  Determining projects to restore...\r\n")
	for i := 0; i < 20; i++ {
		buildOut.WriteString("C:/r/Lib.fs(3,5): error FS0039: The value 'x' is not defined [A.fsproj]\r\n")
	}
	failing := func() (string, bool, bool) {
		return "Lib.fs(1,1): warning FS0064: less generic\n" + buildOut.String(), false, false
	}

	var calls []string
	code, _, errs := runHook(t, dotnetSystem(&calls, nil, failing, nil), "fsharp-check", in)
	lines := strings.Split(strings.TrimRight(errs, "\n"), "\n")
	if code != 0 || lines[0] != "[fsharp-check] dotnet build failed for A.fsproj:" || len(lines) != 16 ||
		lines[1] != "Lib.fs(1,1): warning FS0064: less generic" || strings.Contains(errs, "Determining") || strings.Contains(errs, "\r") {
		t.Errorf("exit %d, stderr:\n%s", code, errs)
	}
	for desc, build := range map[string]func() (string, bool, bool){
		"a green build": func() (string, bool, bool) { return "error FS0001 in an old log line", true, false },
		"a hung build":  func() (string, bool, bool) { return "error FS0001: partial", false, true },
	} {
		if code, out, errs := runHook(t, dotnetSystem(&calls, nil, build, nil), "fsharp-check", in); code != 0 || out != "" || errs != "" {
			t.Errorf("%s: exit %d, out %q, stderr %q", desc, code, out, errs)
		}
	}
	// Without dotnet nothing runs.
	calls = nil
	sys := dotnetSystem(&calls, nil, failing, nil)
	sys.lookPath = func(string) bool { return false }
	if code, _, errs := runHook(t, sys, "fsharp-check", in); code != 0 || errs != "" || len(calls) != 0 {
		t.Errorf("without dotnet: exit %d, stderr %q, calls %v", code, errs, calls)
	}
}

func TestFsharpFormat(t *testing.T) {
	const header = "Package Id      Version      Commands\n-------------------------------------\n"
	global := map[string]string{"dotnet tool list -g": header + "Fantomas        6.3.0        fantomas\n"}
	local := map[string]string{"dotnet tool list -g": header + "dotnet-ef       8.0.0        dotnet-ef\n", "dotnet tool list": header + "fantomas        6.3.0        fantomas\n"}
	none := map[string]string{"dotnet tool list -g": header + "fantomas-tool   1.0.0        x\n", "dotnet tool list": header}
	file := "/r/src/App/Lib.fs"
	for _, c := range []struct {
		desc, stdin string
		env         map[string]string
		lists       map[string]string
		want        string
	}{
		{"a global Fantomas formats the file", editPayload("Edit", file), nil, global, "lookPath dotnet; dotnet tool list -g; dotnet fantomas " + file},
		{"a local Fantomas formats the file", editPayload("Write", file), nil, local, "lookPath dotnet; dotnet tool list -g; dotnet tool list; dotnet fantomas " + file},
		{"no Fantomas: nothing", editPayload("Edit", file), nil, none, "lookPath dotnet; dotnet tool list -g; dotnet tool list"},
		{".fsi formats", editPayload("Edit", "/r/Lib.fsi"), nil, global, "lookPath dotnet; dotnet tool list -g; dotnet fantomas /r/Lib.fsi"},
		{".fsproj is not formatted", editPayload("Edit", "/r/A.fsproj"), nil, global, ""},
		{".fsx is not formatted", editPayload("Edit", "/r/a.fsx"), nil, global, ""},
		{"v1/ is skipped", editPayload("Edit", "/r/v1/Lib.fs"), nil, global, ""},
		{`v1\ is skipped too`, editPayload("Edit", `C:\r\v1\Lib.fs`), nil, global, ""},
		{"v2/ is formatted", editPayload("Edit", "/r/v2/Lib.fs"), nil, global, "lookPath dotnet; dotnet tool list -g; dotnet fantomas /r/v2/Lib.fs"},
		{"Bash is ignored", editPayload("Bash", file), nil, global, ""},
		{"CLAUDE_DISABLE_FSHARP_FORMAT", editPayload("Edit", file), map[string]string{"CLAUDE_DISABLE_FSHARP_FORMAT": "1"}, global, ""},
	} {
		var calls []string
		code, out, errs := runHook(t, dotnetSystem(&calls, c.env, nil, c.lists), "fsharp-format", c.stdin)
		if got := strings.Join(calls, "; "); code != 0 || out != "" || errs != "" || got != c.want {
			t.Errorf("%s: exit %d, out %q, stderr %q, ran %q, want %q", c.desc, code, out, errs, got, c.want)
		}
	}
}
