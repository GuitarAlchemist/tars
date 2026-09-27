namespace Tars.Tools

open Tars.Core
open Tars.Core.ToolMetadata

/// What each tool expects and what it does, for the graph editor.
///
/// A sidecar keyed by name, so a tool is described without touching where it is
/// defined, and a tool nobody has described is simply not offered yet. The count is
/// deliberately small to begin with: the editor shows "N of M tools described", and
/// a short honest catalog beats a long guessed one.
///
/// **Only tools whose input is a JSON object belong here.** Some TARS tools take the
/// raw string instead — `hash_text` hashes whatever it is handed — and the executor
/// serializes a node's arguments to JSON before calling the tool, so describing one
/// of those as an object would make it hash `{"text":"..."}` rather than the text.
/// Those need a convention for "this tool takes the bare string", which is not in
/// this slice; until then, leaving them undescribed is the honest answer.
module ToolDescriptors =

    /// The first tools described, one per approval tier.
    ///
    /// Each schema was read off the tool's own definition rather than guessed:
    /// `StandardTools.readFile`, `GitTools.writeCode`, `SystemTools.runShell`.
    let builtIn : ToolDescriptor list =
        [
          // -------------------------------------------------- reads, changes nothing
          { Name = "read_file"
            InputSchema =
                objectSchema
                    [ "path", "string", "Path to the file, relative to the working directory or absolute." ]
                    [ "path" ]
            Required = [ "path" ]
            Approval = Approval.observes "reads a file from disk and returns its text" }

          // `path` is required although the tool takes it optionally, because the
          // optional case does not work: `ToolHelpers.parseStringArg` falls back to
          // the raw input when the property is absent, so `{}` makes `list_dir` look
          // for a directory literally named "{}". Advertising a default that does not
          // exist is how an editor produces a call that always fails.
          { Name = "list_dir"
            InputSchema =
                objectSchema
                    [ "path", "string", "Directory to list, relative to the working directory or absolute." ]
                    [ "path" ]
            Required = [ "path" ]
            Approval = Approval.observes "lists the entries of a directory" }

          // Directory only: `WorkflowTools.countLines` calls `Directory.GetFiles` on
          // whatever it is given, so a file path errors rather than being counted.
          { Name = "count_lines"
            InputSchema =
                objectSchema
                    [ "path", "string", "Directory to count, searched recursively."
                      "pattern", "string", "Glob for which files to include. Defaults to *.fs." ]
                    [ "path" ]
            Required = [ "path" ]
            Approval = Approval.observes "counts lines in the files under a directory without changing them" }

          // ------------------------------------------------ changes this machine
          { Name = "write_code"
            InputSchema =
                objectSchema
                    [ "path", "string", "File to write, relative or absolute. Directories are created as needed."
                      "content", "string", "The complete new contents of the file." ]
                    [ "path"; "content" ]
            Required = [ "path"; "content" ]
            Approval =
                Approval.mutates "writes a file to disk, replacing it if it already exists, and creates directories" }

          { Name = "run_shell"
            InputSchema =
                objectSchema
                    [ "command", "string", "The shell command to run."
                      "timeout", "integer", "Seconds to wait before giving up. Defaults to 30." ]
                    [ "command" ]
            Required = [ "command" ]
            Approval =
                Approval.mutates
                    "runs an arbitrary shell command on this machine, which can do anything the user can" }

          // -------------------------------------------- leaves the machine or costs
          { Name = "fetch_webpage"
            InputSchema =
                objectSchema
                    [ "url", "string", "The page to fetch."
                      "max_length", "integer", "Cut the extracted text at this many characters. Defaults to 10000." ]
                    [ "url" ]
            Required = [ "url" ]
            Approval = Approval.escapes "sends a request to an external site, which tells that site what was asked for" }
        ]

    /// Register the built-in descriptions. Safe to call more than once.
    let registerBuiltIn () = describeAll builtIn
