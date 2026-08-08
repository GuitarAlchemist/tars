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

function formatRef(ref) {
    if (!ref) return '';
    const val = String(ref).trim();
    if (val.startsWith('#') || isNaN(val)) {
        return val;
    }
    return '#' + val;
}

function getGithubUrl(ref, type) {
    if (!ref) return '#';
    const cleanRef = String(ref).replace('#', '').trim();
    if (!isNaN(cleanRef)) {
        // Assume default repo matches typical GuitarAlchemist repo
        return `https://github.com/GuitarAlchemist/tars/pull/${cleanRef}`;
    }
    return ref;
}

function createCardHTML(run) {
    const risk = escapeHTML(run.risk || 'low');
    const riskClass = `risk-${risk.toLowerCase()}`;
    const agent = escapeHTML(run.agent || 'Unknown');

    const tags = [];
    if (agent) tags.push(`<span class="tag">Agent: ${agent}</span>`);

    let markersHTML = '';
    const markers = [];

    // Check various representations of markers
    if (run.is_stale || run.state === 'stale') {
        markers.push('<div class="marker marker-warning">⚠️ Stale</div>');
    }
    if (run.is_blocked || run.state === 'blocked') {
        markers.push('<div class="marker marker-error">🚫 Blocked</div>');
    }
    if (run.is_duplicate || run.state === 'duplicate' || run.state === 'duplicate-agent-pr') {
        markers.push('<div class="marker marker-warning">📋 Duplicate</div>');
    }
    if (run.ci_failing || run.state === 'ci-failing') {
        markers.push('<div class="marker marker-error">❌ CI Failing</div>');
    }
    if (run.needs_human_review || run.state === 'needs-human-review' || run.state === 'human-review') {
        markers.push('<div class="marker marker-warning">👀 Needs Human Review</div>');
    }
    if (run.state === 'ci-green') {
        markers.push('<div class="marker marker-success">✅ CI Green</div>');
    }

    if (markers.length > 0) {
        markersHTML = `<div class="marker-container">${markers.join('')}</div>`;
    }

    let linksHTML = '';
    const links = [];

    const issue = formatRef(run.issue);
    const pr = formatRef(run.pr);

    if (run.issue) {
        const issueUrl = `https://github.com/GuitarAlchemist/tars/issues/${String(run.issue).replace('#', '')}`;
        links.push(`<a href="${issueUrl}" target="_blank" rel="noopener noreferrer" aria-label="Issue ${issue}">${issue}</a>`);
    }
    if (run.pr) {
        links.push(`<a href="${getGithubUrl(run.pr)}" target="_blank" rel="noopener noreferrer" aria-label="PR ${pr}">PR ${pr}</a>`);
    }

    if (run.evidence && run.evidence.length > 0) {
        const evidenceLinks = run.evidence.map(e => {
            if (e && typeof e === 'object') {
                const url = e.url || '#';
                const typeName = e.type || 'finding';
                const text = e.finding ? `${typeName}: ${e.finding}` : typeName;
                if (url !== '#') {
                    return `<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(text)}</a>`;
                }
                return `<span>${escapeHTML(text)}</span>`;
            }
            return `<span>${escapeHTML(e)}</span>`;
        }).join(' &middot; ');
        links.push(`<span class="evidence-links">Evidence: ${evidenceLinks}</span>`);
    }

    if (links.length > 0) {
        linksHTML = `<div class="card-links">${links.join(' | ')}</div>`;
    }

    // Support both 'last_signal' and 'last_signal_at'
    const signalTime = run.last_signal || run.last_signal_at;
    const lastSignal = signalTime ? escapeHTML(new Date(signalTime).toLocaleString()) : 'Unknown';
    const nextAction = escapeHTML(run.next_action || 'None specified');
    // Support both 'title' and 'summary'
    const title = escapeHTML(run.title || run.summary || 'Untitled Work');

    return `
        <div class="card ${riskClass}" role="listitem" tabindex="0">
            <div class="card-header">
                <h3 class="card-title">
                    ${title}
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
    const columns = document.querySelectorAll('.column');
    columns.forEach(col => {
        const cardsContainer = col.querySelector('.cards');
        if (cardsContainer) {
            cardsContainer.innerHTML = '';
        }
    });

    const summaryCounts = {
        total: 0,
        blocked: 0,
        needs_review: 0,
        ci_failing: 0,
        stale: 0,
        done: 0
    };

    data.forEach(run => {
        summaryCounts.total++;
        if (run.is_blocked || run.state === 'blocked') summaryCounts.blocked++;
        if (run.needs_human_review || run.state === 'needs-human-review' || run.state === 'human-review') summaryCounts.needs_review++;
        if (run.ci_failing || run.state === 'ci-failing') summaryCounts.ci_failing++;
        if (run.is_stale || run.state === 'stale') summaryCounts.stale++;
        if (run.state === 'done') summaryCounts.done++;

        // Map state strings to columns cleanly
        let state = (run.state || 'queued').toLowerCase();

        // Canonical state mapping to align vocabulary & DOM attributes
        if (state === 'pr-open') state = 'pr-opened';
        if (state === 'human-review') state = 'needs-human-review';

        const columnContainer = document.querySelector(`.column[data-state="${state}"] .cards`);

        if (columnContainer) {
            columnContainer.insertAdjacentHTML('beforeend', createCardHTML(run));
        } else {
            console.warn(`Unknown state '${state}' mapped for run. Falling back to queued column.`);
            const queuedCol = document.querySelector(`.column[data-state="queued"] .cards`);
            if (queuedCol) {
                queuedCol.insertAdjacentHTML('beforeend', createCardHTML(run));
            }
        }
    });

    // Handle empty columns & update counters in headings
    columns.forEach(col => {
        const state = col.getAttribute('data-state');
        const cardsContainer = col.querySelector('.cards');
        const header = col.querySelector('h2');

        if (cardsContainer) {
            const cardCount = cardsContainer.querySelectorAll('.card').length;
            if (header) {
                header.setAttribute('data-count', cardCount);
            }
            if (cardCount === 0) {
                cardsContainer.innerHTML = '<div class="empty-state">No work in this state</div>';
            }
        }
    });

    // Update Summary Section with accessible count cards
    const summaryContainer = document.getElementById('summary-counts');
    if (summaryContainer) {
        summaryContainer.innerHTML = `
            <div class="summary-item" tabindex="0">Total Runs: ${summaryCounts.total}</div>
            <div class="summary-item" tabindex="0">Needs Review: ${summaryCounts.needs_review}</div>
            <div class="summary-item" tabindex="0">Blocked: ${summaryCounts.blocked}</div>
            <div class="summary-item" tabindex="0">CI Failing: ${summaryCounts.ci_failing}</div>
            <div class="summary-item" tabindex="0">Stale: ${summaryCounts.stale}</div>
            <div class="summary-item" tabindex="0">Done: ${summaryCounts.done}</div>
        `;
    }

    // Update timestamp
    const timeContainer = document.getElementById('last-updated-time');
    if (timeContainer) {
        timeContainer.textContent = new Date().toLocaleString();
    }
}

async function refreshBoard() {
    const boardData = await fetchBoardData();
    if (boardData) {
        renderBoard(boardData);
    }
}

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    refreshBoard();
    // Refresh every 60 seconds
    setInterval(refreshBoard, 60_000);
});
