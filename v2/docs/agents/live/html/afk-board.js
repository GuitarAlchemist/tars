async function fetchBoardData() {
    try {
        const response = await fetch('./afk-runs.json', { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to fetch board data:', error);
        return null;
    }
}

function escapeHTML(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function getGithubUrl(ref) {
    if (!ref) return '#';
    const escapedRef = escapeHTML(ref);
    // Remove # and prepend with base url placeholder, assuming it's an issue or PR number
    // For a real implementation, this might need to parse repo details, but for now we just make a generic link if it has a #
    if (escapedRef.startsWith('#')) {
        // Just keeping it simple for now, links to `#` or a generic issue search if we don't have the full URL
        return `https://github.com/issues?q=${encodeURIComponent(escapedRef)}`;
    }
    return escapedRef;
}

function createCardHTML(run) {
    const risk = escapeHTML(run.risk || 'low');
    const riskClass = `risk-${risk.toLowerCase()}`;
    const agent = escapeHTML(run.agent);

    const tags = [];
    if (run.agent) tags.push(`<span class="tag">Agent: ${agent}</span>`);

    let markersHTML = '';
    const markers = [];

    if (run.is_stale) markers.push('<div class="marker marker-warning">⚠️ Stale</div>');
    if (run.is_blocked) markers.push('<div class="marker marker-error">🚫 Blocked</div>');
    if (run.is_duplicate) markers.push('<div class="marker marker-warning">📋 Duplicate</div>');
    if (run.ci_failing) markers.push('<div class="marker marker-error">❌ CI Failing</div>');
    if (run.needs_human_review) markers.push('<div class="marker marker-warning">👀 Needs Human Review</div>');

    if (markers.length > 0) {
        markersHTML = `<div class="marker-container">${markers.join('')}</div>`;
    }

    let linksHTML = '';
    const links = [];

    const issue = escapeHTML(run.issue);
    const pr = escapeHTML(run.pr);

    if (run.issue) links.push(`<a href="${getGithubUrl(run.issue)}" target="_blank" rel="noopener noreferrer">Issue ${issue}</a>`);
    if (run.pr) links.push(`<a href="${getGithubUrl(run.pr)}" target="_blank" rel="noopener noreferrer">PR ${pr}</a>`);

    if (run.evidence && run.evidence.length > 0) {
        const evidenceLinks = run.evidence.map(e => `<span>${escapeHTML(e)}</span>`).join(' &middot; ');
        links.push(`<span class="evidence-links">Evidence: ${evidenceLinks}</span>`);
    }

    if (links.length > 0) {
        linksHTML = `<div class="card-links">${links.join(' | ')}</div>`;
    }

    const lastSignal = run.last_signal ? escapeHTML(new Date(run.last_signal).toLocaleString()) : 'Unknown';
    const nextAction = escapeHTML(run.next_action || 'None specified');
    const title = escapeHTML(run.title || 'Untitled Work');

    return `
        <div class="card ${riskClass}">
            <div class="card-header">
                <h3 class="card-title">
                    <a href="${getGithubUrl(run.issue)}" target="_blank" rel="noopener noreferrer">${issue || ''} ${title}</a>
                </h3>
            </div>
            <div class="card-meta">
                ${tags.join('')}
                <span class="tag ${riskClass}">Risk: ${risk}</span>
            </div>
            ${markersHTML}
            <div class="card-body">
                <div><strong>Last signal:</strong> ${lastSignal}</div>
                <div><strong>Next action:</strong> ${nextAction}</div>
            </div>
            ${linksHTML}
        </div>
    `;
}

function renderBoard(data) {
    if (!data) return;

    // Clear existing cards in all columns
    const columns = document.querySelectorAll('.column .cards');
    columns.forEach(col => {
        col.innerHTML = '';
    });

    const summaryCounts = {
        total: 0,
        blocked: 0,
        needs_review: 0,
        ci_failing: 0,
        stale: 0
    };

    data.forEach(run => {
        summaryCounts.total++;
        if (run.is_blocked) summaryCounts.blocked++;
        if (run.needs_human_review) summaryCounts.needs_review++;
        if (run.ci_failing) summaryCounts.ci_failing++;
        if (run.is_stale) summaryCounts.stale++;

        const state = (run.state || 'queued').toLowerCase();
        // Find column by data-state attribute
        const column = document.querySelector(`.column[data-state="${state}"] .cards`);

        if (column) {
            column.insertAdjacentHTML('beforeend', createCardHTML(run));
        } else {
            console.warn(`Unknown state '${state}' for run ${run.issue}`);
            // Fallback to queued
            const queuedCol = document.querySelector(`.column[data-state="queued"] .cards`);
            if (queuedCol) {
                queuedCol.insertAdjacentHTML('beforeend', createCardHTML(run));
            }
        }
    });

    // Handle empty columns
    columns.forEach(col => {
        if (col.children.length === 0) {
            col.innerHTML = '<div class="empty-state">No work in this state</div>';
        }
    });

    // Update Summary
    const summaryContainer = document.getElementById('summary-counts');
    if (summaryContainer) {
        summaryContainer.innerHTML = `
            <div class="summary-item">Total: ${summaryCounts.total}</div>
            <div class="summary-item">Needs Review: ${summaryCounts.needs_review}</div>
            <div class="summary-item">Blocked: ${summaryCounts.blocked}</div>
            <div class="summary-item">CI Failing: ${summaryCounts.ci_failing}</div>
            <div class="summary-item">Stale: ${summaryCounts.stale}</div>
        `;
    }

    // Update timestamp
    const timeContainer = document.getElementById('last-updated-time');
    if (timeContainer) {
        timeContainer.textContent = new Date().toLocaleTimeString();
    }
}

async function refreshBoard() {
    const boardData = await fetchBoardData();
    renderBoard(boardData);
}

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    refreshBoard();
    // Refresh every 60 seconds
    setInterval(refreshBoard, 60_000);
});
