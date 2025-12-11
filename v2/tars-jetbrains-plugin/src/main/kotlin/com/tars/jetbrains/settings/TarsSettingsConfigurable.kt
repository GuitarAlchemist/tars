package com.tars.jetbrains.settings

import com.intellij.openapi.options.Configurable
import com.intellij.openapi.ui.DialogPanel
import com.intellij.ui.dsl.builder.bindText
import com.intellij.ui.dsl.builder.panel
import com.intellij.ui.dsl.builder.rows
import javax.swing.JComponent

/**
 * Settings configurable for TARS plugin.
 */
class TarsSettingsConfigurable : Configurable {
    
    private val settings = TarsSettings.getInstance()
    private var panel: DialogPanel? = null
    private var serverPath: String = settings.serverPath
    
    override fun getDisplayName(): String = "TARS"
    
    override fun createComponent(): JComponent {
        panel = panel {
            group("Connection") {
                row("Server Command:") {
                    textField()
                        .bindText(::serverPath)
                        .comment("Command to start TARS MCP server")
                }.resizableRow()
            }
            
            group("About") {
                row {
                    label("TARS AI Assistant v1.0.0")
                }
                row {
                    browserLink("GitHub", "https://github.com/GuitarAlchemist/tars")
                }
            }
        }
        return panel!!
    }
    
    override fun isModified(): Boolean {
        return serverPath != settings.serverPath
    }
    
    override fun apply() {
        settings.serverPath = serverPath
    }
    
    override fun reset() {
        serverPath = settings.serverPath
        panel?.reset()
    }
    
    override fun disposeUIResources() {
        panel = null
    }
}
