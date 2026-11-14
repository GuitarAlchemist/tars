package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// system is the outside world a hook reads: the clock, the environment and the processes it
// runs. Files are touched directly; tests give each hook its own temporary repository.
type system struct {
	now    func() time.Time
	getenv func(string) string
	// output runs a program and returns its stdout, trimmed, and whether it exited 0 (the
	// scripts' `& git … 2>$null`).
	output func(dir, name string, args ...string) (string, bool)
	// lookPath says whether a program is on PATH (`command -v`).
	lookPath func(name string) bool
	// runTimeout runs a program in the working directory with stdout and stderr together
	// (`timeout N cmd 2>&1`): ok when it exited 0, timedOut when the limit killed it.
	runTimeout func(limit time.Duration, name string, args ...string) (out string, ok, timedOut bool)
	// executable is this program's own path.
	executable func() (string, error)
	// toplevel is the repository holding the working directory (`git rev-parse --show-toplevel`).
	toplevel func() (string, bool)
}

func realSystem() *system {
	return &system{
		now:    time.Now,
		getenv: os.Getenv,
		output: func(dir, name string, args ...string) (string, bool) {
			cmd := exec.Command(name, args...)
			cmd.Dir = dir
			out, err := cmd.Output()
			return strings.TrimSpace(string(out)), err == nil
		},
		lookPath: func(name string) bool {
			_, err := exec.LookPath(name)
			return err == nil
		},
		runTimeout: func(limit time.Duration, name string, args ...string) (string, bool, bool) {
			ctx, cancel := context.WithTimeout(context.Background(), limit)
			defer cancel()
			cmd := exec.CommandContext(ctx, name, args...)
			// dotnet leaves build-server processes holding the output pipes: stop waiting for
			// them shortly after the program itself has exited or been killed.
			cmd.WaitDelay = 2 * time.Second
			out, err := cmd.CombinedOutput()
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return string(out), false, true
			}
			return string(out), err == nil, false
		},
		executable: os.Executable,
		toplevel: func() (string, bool) {
			dir, err := os.Getwd()
			if err != nil {
				return "", false
			}
			return findTopLevel(dir)
		},
	}
}

// findTopLevel is the nearest ancestor of dir (itself included) holding a `.git` directory or
// worktree file, with forward slashes as git prints it. It replaces a `git rev-parse` process,
// which cost 1 to 2.5 s on a loaded Windows machine, on every prompt and every edit.
func findTopLevel(dir string) (string, bool) {
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return filepath.ToSlash(dir), true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// repoRoot is the repository holding the hook's working directory, "" outside one (every
// digest hook then exits 0, as the scripts did).
func (e *env) repoRoot() string {
	root, ok := e.sys.toplevel()
	if !ok {
		return ""
	}
	return root
}

// git is a git query's stdout in root, "" when it prints nothing (the scripts kept whatever
// `& git … 2>$null` printed).
func (e *env) git(root string, args ...string) string {
	out, _ := e.sys.output("", "git", append([]string{"-C", root}, args...)...)
	return out
}

// payload reads stdin as a JSON object; nil when it is empty or not an object.
func (e *env) payload() (map[string]any, []byte) {
	raw, _ := io.ReadAll(e.stdin)
	var m map[string]any
	if json.Unmarshal(bytes.TrimSpace(raw), &m) != nil {
		return nil, raw
	}
	return m, raw
}

// field is a nested string member, "" when absent or not a string.
func field(m map[string]any, path ...string) string {
	var v any = m
	for _, k := range path {
		obj, ok := v.(map[string]any)
		if !ok {
			return ""
		}
		v = obj[k]
	}
	s, _ := v.(string)
	return s
}

// age is the time since path was modified; ok false when it does not exist.
func (e *env) age(path string) (time.Duration, bool) {
	st, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	return e.sys.now().Sub(st.ModTime()), true
}

// psInt is PowerShell's [int] cast of a double: rounded to the nearest, halves to even.
func psInt(f float64) int64 {
	return int64(math.RoundToEven(f))
}

// writeContext prints what the scripts printed with
// `@{ additionalContext = $ctx } | ConvertTo-Json -Compress`.
func writeContext(w io.Writer, ctx string) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)                                    // ConvertTo-Json writes < > & as they are
	_ = enc.Encode(map[string]string{"additionalContext": ctx}) // stdout, as fmt.Print
}

// writeText writes a file as `Set-Content -Encoding UTF8` did in PowerShell 7: UTF-8 without a
// byte-order mark, a line break appended.
func writeText(path, text string) error {
	return os.WriteFile(path, []byte(text+"\n"), 0o644)
}

// bom is the byte-order mark Windows PowerShell's Set-Content wrote at the head of a UTF-8 file.
var bom = string(rune(0xFEFF))

// counter reads a whole-number file (`^\d+$` once trimmed), 0 otherwise.
func counter(path string) int {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(strings.TrimPrefix(string(raw), bom))
	if s == "" || strings.Trim(s, "0123456789") != "" {
		return 0
	}
	n := 0
	for _, c := range s {
		n = n*10 + int(c-'0')
		if n > 1<<30 {
			return 0
		}
	}
	return n
}

// envCount is an environment variable matching `^\d+$`, def otherwise (the scripts' override
// pattern for thresholds).
func envCount(e *env, name string, def int) int {
	v := e.sys.getenv(name)
	if v == "" || strings.Trim(v, "0123456789") != "" || len(v) > 9 {
		return def
	}
	n := 0
	for _, c := range v {
		n = n*10 + int(c-'0')
	}
	return n
}
