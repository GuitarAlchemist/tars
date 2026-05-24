package com.tars.jetbrains.settings

import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage

/**
 * Persistent settings for TARS plugin.
 */
@State(
    name = "TarsSettings",
    storages = [Storage("tars.xml")]
)
class TarsSettings : PersistentStateComponent<TarsSettings.State> {
    
    data class State(
        var serverPath: String = "dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- mcp server"
    )
    
    private var state = State()
    
    var serverPath: String
        get() = state.serverPath
        set(value) { state.serverPath = value }
    
    override fun getState(): State = state
    
    override fun loadState(state: State) {
        this.state = state
    }
    
    companion object {
        fun getInstance(): TarsSettings {
            return ApplicationManager.getApplication().getService(TarsSettings::class.java)
        }
    }
}
