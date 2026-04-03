/**
 * ROSE Plugin Module for Radical Edge Explorer
 *
 * Handles workflow submission results and live state updates via SSE.
 */

export const name = 'rose';

export const notificationConfig = {
    topic:   'workflow_state',
    idField: 'wf_id',
};

export function onNotification(data, page, api) {
    const wfId = data.data?.wf_id || '';
    if (!wfId) return;

    const entryId = `rose-task-${api.edgeName}-${wfId}`;
    const entry   = document.getElementById(entryId);
    if (!entry) return;

    if (data.topic === 'task_event') {
        const logEl = entry.querySelector('.rose-task-log');
        if (!logEl) return;
        const d      = data.data;
        const color  = d.ok ? 'var(--green, #4caf50)' : 'var(--red, #f44336)';
        const icon   = d.ok ? '✓' : '✗';
        const excerpt = api.escHtml(d.excerpt || '');
        logEl.insertAdjacentHTML('beforeend',
            `<div style="color:${color};font-size:0.85em;">[task.${d.task_id}] ${icon} ${excerpt}</div>`
        );
        return;
    }

    if (data.topic !== 'workflow_state') return;

    const state   = (data.data?.state || '?').toUpperCase();
    const stateEl = entry.querySelector('.rose-task-state');
    if (!stateEl) return;

    const isOk      = state === 'COMPLETED';
    const isRunning = state === 'RUNNING' || state === 'INITIALIZING';
    const isFailed  = state === 'FAILED' || state === 'CANCELED';

    stateEl.className = `rose-task-state badge ${
        isOk      ? 'badge-green'  :
        isRunning ? 'badge-blue'   :
        isFailed  ? 'badge-red'    : 'badge-orange'}`;
    stateEl.textContent = state;

    const logEl = entry.querySelector('.rose-task-log');
    if (!logEl) return;

    const ts    = new Date().toLocaleTimeString();
    const stats = data.data?.stats;
    const error = data.data?.error;

    let logHtml = `<span style="color:var(--muted);font-size:0.9em;">[${ts}] <b>${state}</b></span>`;
    if (stats?.iteration !== undefined) {
        const metric = stats.metric_value != null ? ` metric=${stats.metric_value}` : '';
        logHtml += `<br><span style="font-size:0.85em;">iteration ${stats.iteration}${metric}</span>`;
    }
    if (error) {
        logHtml += `<pre class="err">${api.escHtml(error)}</pre>`;
    }

    // Write state info into a stable child so task event lines below survive.
    // On first write, clear the "Waiting…" placeholder.
    let si = logEl.querySelector('.rose-state-info');
    if (!si) {
        logEl.innerHTML = '';
        si = document.createElement('div');
        si.className = 'rose-state-info';
        logEl.appendChild(si);
    }
    si.innerHTML = logHtml;
}
