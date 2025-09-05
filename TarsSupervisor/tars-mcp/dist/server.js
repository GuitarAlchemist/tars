// server.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { spawn } from "node:child_process";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
const PROJECT = process.env.TARS_SUPERVISOR_DIR || process.cwd(); // folder with TarsSupervisor.Cli
const DOTNET = ["dotnet", "run", "--"]; // run inside PROJECT
function run(cmd) {
    return new Promise((resolve) => {
        const p = spawn(cmd[0], cmd.slice(1), { cwd: PROJECT, shell: false });
        let out = "", err = "";
        p.stdout.on("data", (d) => (out += d.toString()));
        p.stderr.on("data", (d) => (err += d.toString()));
        p.on("close", (code) => resolve({ code: code ?? -1, out, err }));
    });
}
function latestVersionDir(base) {
    const root = join(base, "output", "versions");
    if (!existsSync(root))
        return null;
    const dirs = readdirSync(root, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
        .sort()
        .reverse();
    return dirs[0] ? join(root, dirs[0]) : null;
}
const server = new McpServer({
    name: "tars-mcp",
    version: "0.1.0",
    description: "Expose TARS Supervisor commands as MCP tools",
});
// ---------------- Tools ----------------
server.tool("tars.plan", "Generate PLAN.md and next_steps.trsx via the supervisor", { type: "object", properties: {}, additionalProperties: false }, async () => {
    const { code, out, err } = await run([...DOTNET, "plan"]);
    if (code !== 0)
        throw new Error(err || out || `plan exited ${code}`);
    const ver = latestVersionDir(PROJECT);
    let plan = "";
    if (ver && existsSync(join(ver, "PLAN.md"))) {
        plan = readFileSync(join(ver, "PLAN.md"), "utf8");
    }
    return { content: [{ type: "text", text: plan || out || "OK" }] };
});
server.tool("tars.iterate", "Run full iteration (plan -> validate -> metrics)", { type: "object", properties: {}, additionalProperties: false }, async () => {
    const { code, out, err } = await run([...DOTNET, "iterate"]);
    if (code !== 0)
        throw new Error(err || out || `iterate exited ${code}`);
    return { content: [{ type: "text", text: out }] };
});
server.tool("tars.report", "Summarize runs from metrics.jsonl", { type: "object", properties: {}, additionalProperties: false }, async () => {
    const { code, out, err } = await run([...DOTNET, "report"]);
    if (code !== 0)
        throw new Error(err || out || `report exited ${code}`);
    return { content: [{ type: "text", text: out }] };
});
server.tool("tars.rollback", "Point CURRENT to last SUCCESS version", { type: "object", properties: {}, additionalProperties: false }, async () => {
    const { code, out, err } = await run([...DOTNET, "rollback"]);
    if (code !== 0)
        throw new Error(err || out || `rollback exited ${code}`);
    return { content: [{ type: "text", text: out }] };
});
// ----- Resources (registered BEFORE connect) -----
server.resource("Latest PLAN.md", "tars://latest/plan", async () => {
    const root = join(PROJECT, "output", "versions");
    const dirs = existsSync(root)
        ? readdirSync(root, { withFileTypes: true })
            .filter(d => d.isDirectory())
            .map(d => d.name)
            .sort()
            .reverse()
        : [];
    if (!dirs[0])
        throw new Error("No versions yet");
    const uri = "tars://latest/plan";
    const text = readFileSync(join(root, dirs[0], "PLAN.md"), "utf8");
    return { contents: [{ uri, text, mimeType: "text/markdown" }] };
});
server.resource("Latest next_steps.trsx", "tars://latest/next_steps", async () => {
    const root = join(PROJECT, "output", "versions");
    const dirs = existsSync(root)
        ? readdirSync(root, { withFileTypes: true })
            .filter(d => d.isDirectory())
            .map(d => d.name)
            .sort()
            .reverse()
        : [];
    if (!dirs[0])
        throw new Error("No versions yet");
    const uri = "tars://latest/next_steps";
    const text = readFileSync(join(root, dirs[0], "next_steps.trsx"), "utf8");
    return { contents: [{ uri, text, mimeType: "text/plain" }] };
});
// ---------------- Transport (connect AFTER registration) ----------------
const transport = new StdioServerTransport();
await server.connect(transport);
