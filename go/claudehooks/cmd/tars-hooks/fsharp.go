package main

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// The F# hooks (PostToolUse on Write|Edit) were fsharp-check.sh and fsharp-format.sh: a fast
// local `dotnet build` of the edited file's project, and an opt-in Fantomas pass. Both report,
// never block: they exit 0 whatever happens, and stay silent on success.

// editTarget is the tool and the file of the edit: CLAUDE_TOOL_NAME and CLAUDE_FILE_PATH, or the
// stdin payload's tool_name and tool_input.file_path when CLAUDE_FILE_PATH is unset.
func editTarget(e *env) (tool, file string) {
	tool, file = e.sys.getenv("CLAUDE_TOOL_NAME"), e.sys.getenv("CLAUDE_FILE_PATH")
	if file == "" {
		p, _ := e.payload()
		file = field(p, "tool_input", "file_path")
		if tool == "" {
			tool = field(p, "tool_name")
		}
	}
	return tool, file
}

// isEditTool: with a tool name, only Write, Edit and MultiEdit; without one (a direct
// invocation), anything.
func isEditTool(tool string) bool {
	return tool == "" || tool == "Write" || tool == "Edit" || tool == "MultiEdit"
}

// inSkippedTree: the legacy and generated trees CLAUDE.md names. v2/ is the active codebase and
// is not skipped. The scripts' `case */v1/*` saw forward slashes only; a Windows path is
// matched here too, so fsharp-format no longer rewrites a legacy file edited through one.
func inSkippedTree(file string) bool {
	f := strings.ReplaceAll(file, `\`, "/")
	for _, dir := range []string{"/v1/", "/parked_legacy/", "/autonomous_backups/", "/.tars/"} {
		if strings.Contains(f, dir) {
			return true
		}
	}
	return false
}

// fsharpCheck runs `dotnet build --no-restore -nologo -clp:NoSummary -v:q <project>` for the
// project holding an edited .fs, .fsi or .fsproj (30 s cap) and prints its error and warning
// lines on stderr when it fails. CLAUDE_DISABLE_FSHARP_CHECK turns it off.
func fsharpCheck(e *env) int {
	if e.sys.getenv("CLAUDE_DISABLE_FSHARP_CHECK") != "" {
		return 0
	}
	tool, file := editTarget(e)
	if !isEditTool(tool) || file == "" {
		return 0
	}
	if !strings.HasSuffix(file, ".fs") && !strings.HasSuffix(file, ".fsi") && !strings.HasSuffix(file, ".fsproj") {
		return 0 // .fsx scripts are evaluated, not built
	}
	if inSkippedTree(file) {
		return 0
	}
	proj := projectOf(file)
	if proj == "" || !e.sys.lookPath("dotnet") {
		return 0
	}
	out, ok, timedOut := e.sys.runTimeout(30*time.Second, "dotnet", "build", "--no-restore", "-nologo", "-clp:NoSummary", "-v:q", proj)
	if timedOut || ok {
		return 0 // a hung build is not reported: the goal is fast feedback
	}
	fmt.Fprintf(e.stderr, "[fsharp-check] dotnet build failed for %s:\n", filepath.Base(proj))
	n := 0
	for _, l := range strings.Split(out, "\n") {
		if n < 15 && diagnostic.MatchString(l) {
			fmt.Fprintln(e.stderr, strings.TrimRight(l, "\r"))
			n++
		}
	}
	return 0
}

var diagnostic = regexp.MustCompile(`(error|warning) [A-Z]+[0-9]+`)

// projectOf is the .fsproj to build: the file itself, or the first *.fsproj of the nearest
// directory above it that has one. The walk splits on forward slashes, as the script's dirname
// did: a Windows path with backslashes names no directory, so the hook stays silent there and
// does not start a 30 s build after every edit on this machine (one heavy build at a time).
func projectOf(file string) string {
	if strings.HasSuffix(file, ".fsproj") {
		return file
	}
	dir := path.Dir(file)
	for dir != "" && dir != "." && dir != "/" {
		if p := firstProject(dir); p != "" {
			return p
		}
		parent := path.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

// firstProject is `ls "$dir"/*.fsproj | head -1`.
func firstProject(dir string) string {
	entries, err := os.ReadDir(dir + "/")
	if err != nil {
		return ""
	}
	for _, en := range entries { // sorted by name
		if !en.IsDir() && !strings.HasPrefix(en.Name(), ".") && strings.HasSuffix(en.Name(), ".fsproj") {
			return dir + "/" + en.Name()
		}
	}
	return ""
}

// fsharpFormat runs `dotnet fantomas <file>` (15 s cap, output discarded) on an edited .fs or
// .fsi when Fantomas is installed as a global or local dotnet tool; a no-op otherwise.
// CLAUDE_DISABLE_FSHARP_FORMAT turns it off.
func fsharpFormat(e *env) int {
	if e.sys.getenv("CLAUDE_DISABLE_FSHARP_FORMAT") != "" {
		return 0
	}
	tool, file := editTarget(e)
	if !isEditTool(tool) || file == "" {
		return 0
	}
	if !strings.HasSuffix(file, ".fs") && !strings.HasSuffix(file, ".fsi") {
		return 0
	}
	if inSkippedTree(file) || !e.sys.lookPath("dotnet") {
		return 0
	}
	if global, _ := e.sys.output("", "dotnet", "tool", "list", "-g"); !fantomasListed.MatchString(global) {
		if local, _ := e.sys.output("", "dotnet", "tool", "list"); !fantomasListed.MatchString(local) {
			return 0
		}
	}
	_, _, _ = e.sys.runTimeout(15*time.Second, "dotnet", "fantomas", file) // the script ignored the outcome
	return 0
}

// grep -qi '^fantomas[[:space:]]' over the lines of `dotnet tool list`.
var fantomasListed = regexp.MustCompile(`(?im)^fantomas[ \t]`)
