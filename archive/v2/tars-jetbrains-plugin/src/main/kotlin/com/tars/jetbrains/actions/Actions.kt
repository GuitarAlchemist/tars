package com.tars.jetbrains.actions

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.wm.ToolWindowManager

/**
 * Action to connect to TARS MCP server.
 */
class ConnectAction : AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        // Open the TARS tool window
        val toolWindow = ToolWindowManager.getInstance(project).getToolWindow("TARS")
        toolWindow?.show()
    }
}

/**
 * Action to ask TARS a question.
 */
class AskQuestionAction : AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        val question = Messages.showInputDialog(
            project,
            "What would you like to ask TARS?",
            "Ask TARS",
            null
        )
        
        if (!question.isNullOrBlank()) {
            // Open tool window and send the question
            val toolWindow = ToolWindowManager.getInstance(project).getToolWindow("TARS")
            toolWindow?.show()
            
            Messages.showInfoMessage(
                project,
                "Question sent to TARS tool window. Check the TARS panel for the response.",
                "TARS"
            )
        }
    }
}

/**
 * Action to explain selected code.
 */
class ExplainSelectionAction : AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val editor = e.getData(com.intellij.openapi.actionSystem.CommonDataKeys.EDITOR) ?: return
        
        val selectedText = editor.selectionModel.selectedText
        
        if (selectedText.isNullOrBlank()) {
            Messages.showWarningDialog(
                project,
                "Please select some code first.",
                "No Selection"
            )
            return
        }
        
        // Open tool window
        val toolWindow = ToolWindowManager.getInstance(project).getToolWindow("TARS")
        toolWindow?.show()
        
        Messages.showInfoMessage(
            project,
            "Code selection sent to TARS. Check the TARS panel for explanation.",
            "TARS"
        )
    }
    
    override fun update(e: AnActionEvent) {
        val editor = e.getData(com.intellij.openapi.actionSystem.CommonDataKeys.EDITOR)
        e.presentation.isEnabled = editor?.selectionModel?.hasSelection() == true
    }
}
