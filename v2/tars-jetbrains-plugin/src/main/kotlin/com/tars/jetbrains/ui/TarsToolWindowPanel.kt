package com.tars.jetbrains.ui

import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.components.JBTextArea
import com.intellij.util.ui.JBUI
import com.tars.jetbrains.mcp.McpClient
import com.tars.jetbrains.settings.TarsSettings
import java.awt.BorderLayout
import java.awt.Dimension
import javax.swing.*

/**
 * Main panel for the TARS tool window.
 */
class TarsToolWindowPanel(private val project: Project) : JPanel(BorderLayout()) {
    
    private val settings = TarsSettings.getInstance()
    private var client: McpClient? = null
    
    private val outputArea = JBTextArea().apply {
        isEditable = false
        lineWrap = true
        wrapStyleWord = true
    }
    
    private val inputField = JTextField()
    private val sendButton = JButton("Send")
    private val connectButton = JButton("Connect")
    private val toolsCombo = JComboBox<String>()
    
    init {
        border = JBUI.Borders.empty(8)
        
        // Header with connection controls
        val headerPanel = JPanel(BorderLayout()).apply {
            add(connectButton, BorderLayout.WEST)
            add(JLabel("  Tool: "), BorderLayout.CENTER)
            add(toolsCombo, BorderLayout.EAST)
        }
        
        // Output area
        val scrollPane = JBScrollPane(outputArea).apply {
            preferredSize = Dimension(400, 300)
        }
        
        // Input area
        val inputPanel = JPanel(BorderLayout()).apply {
            add(inputField, BorderLayout.CENTER)
            add(sendButton, BorderLayout.EAST)
        }
        
        add(headerPanel, BorderLayout.NORTH)
        add(scrollPane, BorderLayout.CENTER)
        add(inputPanel, BorderLayout.SOUTH)
        
        // Event handlers
        connectButton.addActionListener { toggleConnection() }
        sendButton.addActionListener { sendMessage() }
        inputField.addActionListener { sendMessage() }
        
        appendOutput("Welcome to TARS AI Assistant!\n")
        appendOutput("Click 'Connect' to connect to TARS MCP server.\n")
    }
    
    private fun toggleConnection() {
        if (client?.isConnected == true) {
            client?.disconnect()
            client = null
            connectButton.text = "Connect"
            toolsCombo.removeAllItems()
            appendOutput("\n[Disconnected from TARS]\n")
        } else {
            connect()
        }
    }
    
    private fun connect() {
        val serverPath = settings.serverPath
        if (serverPath.isBlank()) {
            Messages.showWarningDialog(
                project,
                "Please configure the TARS server path in Settings > Tools > TARS",
                "TARS Configuration"
            )
            return
        }
        
        appendOutput("\nConnecting to TARS...\n")
        
        // Run connection in background
        Thread {
            try {
                val command = serverPath.split(" ")
                client = McpClient(command)
                
                if (client!!.connect()) {
                    SwingUtilities.invokeLater {
                        connectButton.text = "Disconnect"
                        appendOutput("[Connected to TARS]\n")
                        
                        // Load tools
                        val tools = client!!.listTools()
                        toolsCombo.removeAllItems()
                        toolsCombo.addItem("think")  // Default
                        tools.forEach { tool ->
                            if (tool.name != "think") {
                                toolsCombo.addItem(tool.name)
                            }
                        }
                        appendOutput("Loaded ${tools.size} tools\n")
                    }
                } else {
                    SwingUtilities.invokeLater {
                        appendOutput("[Failed to connect]\n")
                    }
                }
            } catch (e: Exception) {
                SwingUtilities.invokeLater {
                    appendOutput("[Error: ${e.message}]\n")
                }
            }
        }.start()
    }
    
    private fun sendMessage() {
        val message = inputField.text.trim()
        if (message.isEmpty()) return
        
        inputField.text = ""
        appendOutput("\nYou: $message\n")
        
        if (client?.isConnected != true) {
            appendOutput("TARS: [Not connected. Click 'Connect' first.]\n")
            return
        }
        
        // Run tool call in background
        Thread {
            try {
                val toolName = toolsCombo.selectedItem as? String ?: "think"
                val result = client!!.callTool(toolName, mapOf("problem" to message))
                
                SwingUtilities.invokeLater {
                    appendOutput("TARS: $result\n")
                }
            } catch (e: Exception) {
                SwingUtilities.invokeLater {
                    appendOutput("TARS: [Error: ${e.message}]\n")
                }
            }
        }.start()
    }
    
    private fun appendOutput(text: String) {
        outputArea.append(text)
        outputArea.caretPosition = outputArea.document.length
    }
}
