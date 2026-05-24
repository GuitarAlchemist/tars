window.blazorDiagnostics = {
    checkConnection: function() {
        console.log("Blazor connection check initiated");
        return "Connection working";
    },
    
    logError: function(error) {
        console.error("Blazor error:", error);
    }
};

// Monitor SignalR connection
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        if (window.Blazor) {
            console.log("Blazor initialized successfully");
        } else {
            console.error("Blazor failed to initialize after timeout");
        }
    }, 3000);
});