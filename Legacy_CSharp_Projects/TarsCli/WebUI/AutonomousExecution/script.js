document.addEventListener('DOMContentLoaded', function() {
    // Navigation
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    const contentSections = document.querySelectorAll('.content-section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the target section
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            // Update active nav link
            navLinks.forEach(navLink => navLink.classList.remove('active'));
            this.classList.add('active');
            
            // Show target section, hide others
            contentSections.forEach(section => section.classList.remove('active'));
            targetSection.classList.add('active');
            
            // Update URL hash
            window.location.hash = targetId;
        });
    });
    
    // Handle initial page load with hash
    if (window.location.hash) {
        const targetId = window.location.hash.substring(1);
        const targetLink = document.querySelector(`.navbar-nav .nav-link[href="#${targetId}"]`);
        const targetSection = document.getElementById(targetId);
        
        if (targetLink && targetSection) {
            navLinks.forEach(navLink => navLink.classList.remove('active'));
            targetLink.classList.add('active');
            
            contentSections.forEach(section => section.classList.remove('active'));
            targetSection.classList.add('active');
        }
    }
    
    // Initialize charts
    initializeCharts();
    
    // Initialize event handlers
    initializeEventHandlers();
    
    // Fetch data
    fetchData();
});

function initializeCharts() {
    // Execution Performance Chart
    const executionPerformanceCtx = document.getElementById('executionPerformanceChart');
    if (executionPerformanceCtx) {
        new Chart(executionPerformanceCtx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
                datasets: [
                    {
                        label: 'Success Rate',
                        data: [75, 78, 80, 79, 85, 87],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Execution Count',
                        data: [12, 15, 18, 20, 22, 24],
                        borderColor: '#0d6efd',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Success Rate (%)'
                        },
                        max: 100
                    },
                    y1: {
                        beginAtZero: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Execution Count'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }
}

function initializeEventHandlers() {
    // Execute buttons
    const executeButtons = document.querySelectorAll('.btn-primary:contains("Execute")');
    executeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const improvementId = this.closest('tr').querySelector('td:first-child').textContent;
            executeImprovement(improvementId);
        });
    });
    
    // View buttons
    const viewButtons = document.querySelectorAll('.btn-outline-primary:contains("View")');
    viewButtons.forEach(button => {
        button.addEventListener('click', function() {
            const id = this.closest('tr').querySelector('td:first-child').textContent;
            if (id.startsWith('IMP-')) {
                viewImprovement(id);
            } else if (id.startsWith('CTX-')) {
                viewExecution(id);
            }
        });
    });
    
    // Monitor buttons
    const monitorButtons = document.querySelectorAll('.btn-outline-secondary:contains("Monitor")');
    monitorButtons.forEach(button => {
        button.addEventListener('click', function() {
            const contextId = this.closest('tr').querySelector('td:first-child').textContent;
            monitorExecution(contextId);
        });
    });
    
    // Rollback buttons
    const rollbackButtons = document.querySelectorAll('.btn-outline-danger:contains("Rollback")');
    rollbackButtons.forEach(button => {
        button.addEventListener('click', function() {
            const contextId = this.closest('tr').querySelector('td:first-child').textContent;
            rollbackExecution(contextId);
        });
    });
    
    // Approve buttons
    const approveButtons = document.querySelectorAll('.btn-outline-success:contains("Approve")');
    approveButtons.forEach(button => {
        button.addEventListener('click', function() {
            const contextId = this.closest('tr').querySelector('td:first-child').textContent;
            approveExecution(contextId);
        });
    });
    
    // Reject buttons
    const rejectButtons = document.querySelectorAll('.btn-outline-danger:contains("Reject")');
    rejectButtons.forEach(button => {
        button.addEventListener('click', function() {
            const contextId = this.closest('tr').querySelector('td:first-child').textContent;
            rejectExecution(contextId);
        });
    });
}

// API Functions
async function fetchData() {
    try {
        // Fetch improvements
        const improvements = await fetchImprovements();
        updateImprovementsTable(improvements);
        
        // Fetch executions
        const executions = await fetchExecutions();
        updateExecutionsTable(executions);
        
        // Fetch dashboard data
        const dashboardData = await fetchDashboardData();
        updateDashboard(dashboardData);
    } catch (error) {
        console.error('Error fetching data:', error);
        showNotification('Error fetching data', 'error');
    }
}

async function fetchImprovements() {
    // In a real implementation, this would make an API call
    // For now, return mock data
    return [
        {
            id: 'IMP-456',
            name: 'Implement Caching for API Responses',
            category: 'Performance',
            status: 'Pending',
            priority: 'High',
            impact: 'High',
            effort: 'Medium',
            risk: 'Low',
            created: '2023-06-15'
        },
        // More improvements...
    ];
}

async function fetchExecutions() {
    // In a real implementation, this would make an API call
    // For now, return mock data
    return [
        {
            id: 'CTX-123',
            improvement: 'Refactor Authentication Service',
            status: 'Completed',
            started: '2023-06-15 10:30',
            completed: '2023-06-15 10:45',
            duration: '15m',
            mode: 'Real',
            environment: 'Development'
        },
        // More executions...
    ];
}

async function fetchDashboardData() {
    // In a real implementation, this would make an API call
    // For now, return mock data
    return {
        totalExecutions: 24,
        totalExecutionsChange: '+3 today',
        successRate: '87%',
        successRateChange: '+2% this week',
        pendingReviews: 5,
        pendingReviewsChange: '+2 since yesterday',
        rollbacks: 3,
        rollbacksChange: 'No change',
        recentExecutions: [
            {
                id: 'CTX-123',
                improvement: 'Refactor Authentication Service',
                status: 'Completed',
                started: '2 hours ago'
            },
            // More executions...
        ],
        pendingImprovements: [
            {
                id: 'IMP-456',
                name: 'Implement Caching for API Responses',
                category: 'Performance',
                priority: 'High',
                impact: 'High',
                effort: 'Medium',
                risk: 'Low'
            },
            // More improvements...
        ]
    };
}

function updateImprovementsTable(improvements) {
    // In a real implementation, this would update the table with the fetched data
    console.log('Updating improvements table with', improvements.length, 'improvements');
}

function updateExecutionsTable(executions) {
    // In a real implementation, this would update the table with the fetched data
    console.log('Updating executions table with', executions.length, 'executions');
}

function updateDashboard(data) {
    // In a real implementation, this would update the dashboard with the fetched data
    console.log('Updating dashboard with', data);
}

// Action Functions
async function executeImprovement(improvementId) {
    try {
        console.log('Executing improvement:', improvementId);
        showNotification(`Executing improvement: ${improvementId}`, 'info');
        
        // In a real implementation, this would make an API call
        // For now, simulate a delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        showNotification(`Improvement ${improvementId} execution started`, 'success');
    } catch (error) {
        console.error('Error executing improvement:', error);
        showNotification(`Error executing improvement: ${error.message}`, 'error');
    }
}

function viewImprovement(improvementId) {
    console.log('Viewing improvement:', improvementId);
    // In a real implementation, this would navigate to the improvement details page
    showNotification(`Viewing improvement: ${improvementId}`, 'info');
}

function viewExecution(contextId) {
    console.log('Viewing execution:', contextId);
    // In a real implementation, this would navigate to the execution details page
    showNotification(`Viewing execution: ${contextId}`, 'info');
}

async function monitorExecution(contextId) {
    try {
        console.log('Monitoring execution:', contextId);
        showNotification(`Monitoring execution: ${contextId}`, 'info');
        
        // In a real implementation, this would make an API call
        // For now, simulate a delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Show a modal with execution progress
        // For now, just show a notification
        showNotification(`Execution ${contextId} progress: 75%`, 'info');
    } catch (error) {
        console.error('Error monitoring execution:', error);
        showNotification(`Error monitoring execution: ${error.message}`, 'error');
    }
}

async function rollbackExecution(contextId) {
    try {
        console.log('Rolling back execution:', contextId);
        
        // Confirm rollback
        if (!confirm(`Are you sure you want to roll back execution ${contextId}?`)) {
            return;
        }
        
        showNotification(`Rolling back execution: ${contextId}`, 'info');
        
        // In a real implementation, this would make an API call
        // For now, simulate a delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        showNotification(`Execution ${contextId} rolled back successfully`, 'success');
    } catch (error) {
        console.error('Error rolling back execution:', error);
        showNotification(`Error rolling back execution: ${error.message}`, 'error');
    }
}

async function approveExecution(contextId) {
    try {
        console.log('Approving execution:', contextId);
        
        // Prompt for comment
        const comment = prompt(`Enter a comment for approving execution ${contextId}:`);
        
        showNotification(`Approving execution: ${contextId}`, 'info');
        
        // In a real implementation, this would make an API call
        // For now, simulate a delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        showNotification(`Execution ${contextId} approved successfully`, 'success');
    } catch (error) {
        console.error('Error approving execution:', error);
        showNotification(`Error approving execution: ${error.message}`, 'error');
    }
}

async function rejectExecution(contextId) {
    try {
        console.log('Rejecting execution:', contextId);
        
        // Prompt for comment
        const comment = prompt(`Enter a comment for rejecting execution ${contextId}:`);
        
        showNotification(`Rejecting execution: ${contextId}`, 'info');
        
        // In a real implementation, this would make an API call
        // For now, simulate a delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        showNotification(`Execution ${contextId} rejected successfully`, 'success');
    } catch (error) {
        console.error('Error rejecting execution:', error);
        showNotification(`Error rejecting execution: ${error.message}`, 'error');
    }
}

// Utility Functions
function showNotification(message, type = 'info') {
    // In a real implementation, this would show a toast notification
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // For now, just log to console
    // In a real implementation, we would use a toast library
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'info' ? 'primary' : type === 'success' ? 'success' : type === 'error' ? 'danger' : 'primary'} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add to document
    document.body.appendChild(toast);
    
    // Initialize and show toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove after shown
    toast.addEventListener('hidden.bs.toast', function() {
        document.body.removeChild(toast);
    });
}

// jQuery-like utility for button text matching
HTMLCollection.prototype.forEach = Array.prototype.forEach;
NodeList.prototype.forEach = Array.prototype.forEach;

if (!Element.prototype.matches) {
    Element.prototype.matches = Element.prototype.msMatchesSelector || Element.prototype.webkitMatchesSelector;
}

if (!Element.prototype.closest) {
    Element.prototype.closest = function(s) {
        var el = this;
        do {
            if (el.matches(s)) return el;
            el = el.parentElement || el.parentNode;
        } while (el !== null && el.nodeType === 1);
        return null;
    };
}

NodeList.prototype.filter = Array.prototype.filter;
HTMLCollection.prototype.filter = Array.prototype.filter;

if (!NodeList.prototype.filter) {
    NodeList.prototype.filter = Array.prototype.filter;
}

Object.defineProperty(Element.prototype, 'textContent', {
    get: function() {
        return this.innerText || this.innerHTML.replace(/<[^>]*>/g, '');
    }
});

HTMLElement.prototype.contains = function(text) {
    return this.textContent.includes(text);
};
