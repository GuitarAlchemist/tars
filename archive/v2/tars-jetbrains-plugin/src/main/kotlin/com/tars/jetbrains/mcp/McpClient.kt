package com.tars.jetbrains.mcp

import com.google.gson.Gson
import com.google.gson.JsonObject
import java.io.*
import java.util.concurrent.atomic.AtomicInteger

/**
 * MCP (Model Context Protocol) client for communicating with TARS server.
 */
class McpClient(private val command: List<String>) {
    private var process: Process? = null
    private var reader: BufferedReader? = null
    private var writer: BufferedWriter? = null
    private val requestId = AtomicInteger(0)
    private val gson = Gson()
    
    var isConnected: Boolean = false
        private set
    
    /**
     * Connect to the TARS MCP server.
     */
    fun connect(): Boolean {
        return try {
            val processBuilder = ProcessBuilder(command)
            processBuilder.redirectErrorStream(false)
            process = processBuilder.start()
            
            reader = BufferedReader(InputStreamReader(process!!.inputStream))
            writer = BufferedWriter(OutputStreamWriter(process!!.outputStream))
            
            // Send initialize request
            val initRequest = mapOf(
                "jsonrpc" to "2.0",
                "method" to "initialize",
                "id" to requestId.incrementAndGet(),
                "params" to mapOf(
                    "protocolVersion" to "2024-11-05",
                    "capabilities" to emptyMap<String, Any>(),
                    "clientInfo" to mapOf(
                        "name" to "tars-jetbrains",
                        "version" to "1.0.0"
                    )
                )
            )
            
            sendRequest(initRequest)
            val response = readResponse()
            
            if (response != null && !response.has("error")) {
                // Send initialized notification
                sendRequest(mapOf(
                    "jsonrpc" to "2.0",
                    "method" to "notifications/initialized",
                    "params" to emptyMap<String, Any>()
                ))
                isConnected = true
            }
            
            isConnected
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    /**
     * Disconnect from the server.
     */
    fun disconnect() {
        isConnected = false
        try {
            writer?.close()
            reader?.close()
            process?.destroy()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    /**
     * List available tools.
     */
    fun listTools(): List<ToolInfo> {
        if (!isConnected) return emptyList()
        
        val request = mapOf(
            "jsonrpc" to "2.0",
            "method" to "tools/list",
            "id" to requestId.incrementAndGet(),
            "params" to emptyMap<String, Any>()
        )
        
        sendRequest(request)
        val response = readResponse()
        
        if (response?.has("result") == true) {
            val result = response.getAsJsonObject("result")
            val tools = result.getAsJsonArray("tools")
            return tools.map { tool ->
                val obj = tool.asJsonObject
                ToolInfo(
                    name = obj.get("name").asString,
                    description = obj.get("description")?.asString ?: ""
                )
            }
        }
        return emptyList()
    }
    
    /**
     * Call a tool with arguments.
     */
    fun callTool(name: String, arguments: Map<String, Any>): String {
        if (!isConnected) return "Error: Not connected"
        
        val request = mapOf(
            "jsonrpc" to "2.0",
            "method" to "tools/call",
            "id" to requestId.incrementAndGet(),
            "params" to mapOf(
                "name" to name,
                "arguments" to arguments
            )
        )
        
        sendRequest(request)
        val response = readResponse()
        
        if (response?.has("result") == true) {
            val result = response.getAsJsonObject("result")
            val content = result.getAsJsonArray("content")
            if (content.size() > 0) {
                val first = content[0].asJsonObject
                return first.get("text")?.asString ?: "No response"
            }
        } else if (response?.has("error") == true) {
            val error = response.getAsJsonObject("error")
            return "Error: ${error.get("message")?.asString}"
        }
        
        return "No response"
    }
    
    private fun sendRequest(request: Map<String, Any>) {
        val json = gson.toJson(request)
        writer?.write(json)
        writer?.newLine()
        writer?.flush()
    }
    
    private fun readResponse(): JsonObject? {
        return try {
            val line = reader?.readLine()
            if (line != null) {
                gson.fromJson(line, JsonObject::class.java)
            } else null
        } catch (e: Exception) {
            null
        }
    }
}

data class ToolInfo(
    val name: String,
    val description: String
)
