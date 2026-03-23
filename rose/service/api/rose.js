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
    if (data.topic !== 'workflow_state') return;

    const wfId = data.data?.wf_id || '';
    const state = (data.data?.state || '?').toUpperCase();
    if (!wfId) return;

    const entryId  = `rose-task-${api.edgeName}-${wfId}`;
    const entry    = document.getElementById(entryId);
    if (!entry) return;

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
        const metric = stats.metric_value !== undefined ? ` metric=${stats.metric_value}` : '';
        logHtml += `<br><span style="font-size:0.85em;">iteration ${stats.iteration}${metric}</span>`;
    }
    if (error) {
        logHtml += `<pre class="err">${api.escHtml(error)}</pre>`;
    }
    logEl.innerHTML = logHtml;
}
