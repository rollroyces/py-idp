/* py-idp-web — single-page UI logic (v2: Apple/PDF-reader design).
 *
 * Talks to /api/* for our extras and to the upstream /extract, /templates,
 * /healthz endpoints for the real work. No frameworks; vanilla DOM APIs.
 */

(() => {
  'use strict';

  // ───────────────────────────── State ──────────────────────────────────
  const state = {
    apiKey: localStorage.getItem('idp_api_key') || '',
    // apiBase lets the static (GitHub-Pages) build talk to a remote backend.
    // Empty = same-origin (works with `./start.sh`). Set via the sidebar field,
    // ?api_base=... on load, or localStorage 'idp_api_base'.
    apiBase: localStorage.getItem('idp_api_base') || '',
    reviewer: localStorage.getItem('idp_reviewer') || '',
    lastResult: null,
    lastRunId: null,
    lastFile: null,                // last File object picked
    batch: { queue: [], running: 0, concurrency: 2 },
  };

  // ───────────────────────────── Helpers ────────────────────────────────
  // Resolve a relative API path (/extract, /api/status, /templates/...)
  // against state.apiBase. When apiBase is empty we return the path unchanged
  // so same-origin calls (the local ./start.sh mode) keep working.
  function apiUrl(path) {
    if (path == null) return '';
    if (/^https?:\/\//i.test(path)) return path;          // already absolute
    const base = (state.apiBase || '').replace(/\/+$/, ''); // strip trailing /
    const p = path.startsWith('/') ? path : '/' + path;
    return base ? base + p : p;
  }

  function authHeaders(extra) {
    const h = { ...(extra || {}) };
    if (state.apiKey) h['X-API-Key'] = state.apiKey;
    return h;
  }

  const $  = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

  function basename(p) {
    if (!p) return '';
    const ix = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
    return ix >= 0 ? p.slice(ix + 1) : p;
  }

  // ───────────────────────────── Error envelopes ─────────────────────────
  // Errors from the API come back as `{"error": {"code", "message", "type",
  // "request_id", "details"}}` (see idp.errors.error_envelope). We carry the
  // whole envelope forward so toasts can show the code AND we can render a
  // dedicated detail panel with copy-to-clipboard for bug reports.
  function makeApiError(body, status, statusText) {
    const env = body?.error;
    if (env && typeof env === 'object') {
      const err = new Error(env.message || statusText || `HTTP ${status}`);
      err.code = env.code || 'IDP-INT-001';
      err.type = env.type || null;
      err.requestId = env.request_id || null;
      err.details = env.details || null;
      err.status = status;
      err.isIdpError = true;
      return err;
    }
    // Fallback for FastAPI's default {"detail": "..."} shape (e.g. 404 on
    // unmounted routes, 401/403 before our exception handlers fire).
    const detail = body?.detail || statusText || `HTTP ${status}`;
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    err.code = status === 400 ? 'IDP-VAL-002'
            : status === 401 ? 'IDP-AUTH-001'
            : status === 403 ? 'IDP-AUTH-002'
            : status === 404 ? 'IDP-NF-001'
            : status === 413 ? 'IDP-PAYLOAD-001'
            : status === 429 ? 'IDP-RATE-001'
            : status >= 500 ? 'IDP-INT-001'
            : 'IDP-CLIENT-001';
    err.type = null;
    err.requestId = null;
    err.details = null;
    err.status = status;
    err.isIdpError = false;
    return err;
  }

  function toast(msg, kind = 'info', ms = 3000, opts = {}) {
    // opts: { code?, requestId?, sticky?: bool }
    const el = $('#toast');
    const code = opts.code ? `<span class="toast-code">${escapeHtml(opts.code)}</span>` : '';
    el.innerHTML = `${escapeHtml(msg)}${code}`;
    el.className = `toast ${kind}`;
    if (opts.requestId) el.dataset.requestId = opts.requestId;
    else delete el.dataset.requestId;
    const ttl = opts.sticky ? 0 : ms;
    if (ttl > 0) setTimeout(() => el.classList.add('hidden'), ttl);
  }

  // Catalog of well-known codes so the detail panel can show a recovery hint.
  // Kept tiny on purpose — full descriptions live in idp.errors docstrings.
  const ERROR_HINTS = {
    'IDP-RATE-001':     'Slow down — wait 60s then retry, or raise IDP_RATE_LIMIT_PER_MINUTE on the server.',
    'IDP-PARSE-001':    'The document couldn\'t be parsed. Check that the file extension and contents match (PDF / image / text).',
    'IDP-SCHEMA-001':   'The LLM output didn\'t match the requested schema. Re-run with strict=False to see what came back.',
    'IDP-BACKEND-001':  'The selected LLM backend is unreachable. Verify the API key or that the local model (Ollama / HF-VLM) is running.',
    'IDP-STORE-001':    'A storage backend failed (disk full, permission denied, migration error). Check server logs.',
    'IDP-CONF-001':     'Server is misconfigured (missing or invalid env var). See server startup logs.',
    'IDP-TMPL-404':     'No template with that name is registered. Check the templates/ directory.',
    'IDP-TMPL-001':     'A template file is malformed (bad YAML, missing field). Fix the .md file and reload.',
    'IDP-VAL-001':      'The HTTP request body or query string failed validation. See the details panel for the field list.',
    'IDP-VAL-002':      'Missing or malformed field in the request body. See the message for which one.',
    'IDP-AUTH-001':     'Missing X-API-Key header. Set it in the sidebar or via ?api_key=…',
    'IDP-AUTH-002':     'Invalid X-API-Key. Check the server\'s IDP_API_KEY setting.',
    'IDP-PAYLOAD-001':  'Upload too large. Lower IDP_MAX_UPLOAD_BYTES on the server, or send a smaller file.',
    'IDP-NF-001':       'That route does not exist on this server.',
    'IDP-INT-001':      'Unhandled server error. Check server logs and the request_id when reporting.',
    'IDP-CLIENT-001':   'Client-side error (non-IDPError). Check the network tab.',
  };

  function renderErrorPanel(err, ctxLabel = 'Error') {
    const root = $('#extract-result');
    if (!root) return;
    const code = err.code || 'IDP-INT-001';
    const hint = ERROR_HINTS[code] || 'No recovery hint for this code. Check server logs and report the code + request_id.';
    const detailsJson = err.details ? JSON.stringify(err.details, null, 2) : null;
    root.innerHTML = `
      <div class="error-card" role="alert">
        <div class="error-card-head">
          <span class="error-badge">${escapeHtml(code)}</span>
          <span class="error-title">${escapeHtml(ctxLabel)}</span>
        </div>
        <div class="error-message">${escapeHtml(err.message || 'Unknown error')}</div>
        ${err.type ? `<div class="error-meta"><span class="k">type</span><code>${escapeHtml(err.type)}</code></div>` : ''}
        ${err.requestId ? `<div class="error-meta"><span class="k">request_id</span><code>${escapeHtml(err.requestId)}</code><button class="btn-ghost xs" data-copy="${escapeHtml(err.requestId)}" title="Copy">⧉</button></div>` : ''}
        <div class="error-hint"><strong>What to try:</strong> ${escapeHtml(hint)}</div>
        ${detailsJson ? `
          <details class="error-details">
            <summary>Details</summary>
            <pre class="json-block">${escapeHtml(detailsJson)}</pre>
          </details>
        ` : ''}
        <div class="error-actions">
          <button class="btn-ghost small" data-copy='${escapeHtml(JSON.stringify({code, message: err.message, type: err.type, request_id: err.requestId, details: err.details}))}'>Copy report</button>
        </div>
      </div>`;
    // Wire copy-to-clipboard
    $$('#extract-result [data-copy]').forEach(b =>
      b.addEventListener('click', () => {
        const v = b.dataset.copy;
        navigator.clipboard?.writeText(v).then(
          () => toast('Copied', 'success', 1500),
          () => toast('Copy failed', 'error', 2000)
        );
      })
    );
  }

  function fmtVal(v) {
    if (v === null || v === undefined) return '∅';
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }

  function fmtTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts * 1000);
    return d.toLocaleString();
  }

  function fmtBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  async function api(path, opts = {}) {
    const headers = authHeaders(opts.headers);
    const res = await fetch(apiUrl(path), { ...opts, headers });
    let body;
    try { body = await res.json(); } catch { body = null; }
    if (!res.ok) {
      throw makeApiError(body, res.status, res.statusText);
    }
    return body;
  }

  async function pollStatus() {
    try {
      const s = await api('/api/status');
      $('#status-line').textContent =
        `v${s.version} · ${s.default_backend} · runs ${s.history_count} · corrections ${s.corrections_count}`;
      $('#status-pill').dataset.state = 'on';
      $('#status-text').textContent = 'online';
      return s;
    } catch (e) {
      $('#status-pill').dataset.state = 'off';
      $('#status-text').textContent = 'offline';
      $('#status-line').textContent = e.message || 'cannot reach server';
      return null;
    }
  }

  // ───────────────────────────── Sidebar nav ────────────────────────────
  function setupNav() {
    $$('.nav-item').forEach(btn => {
      btn.addEventListener('click', () => switchView(btn.dataset.tab));
    });
    // Keyboard shortcuts: E B H T A
    document.addEventListener('keydown', e => {
      if (e.target.matches('input, textarea, select')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const map = { e: 'extract', b: 'batch', h: 'history', t: 'templates', a: 'about' };
      if (map[e.key.toLowerCase()]) switchView(map[e.key.toLowerCase()]);
    });
  }

  function switchView(name) {
    $$('.nav-item').forEach(n => n.classList.toggle('is-active', n.dataset.tab === name));
    $$('.view').forEach(v => v.classList.toggle('is-active', v.dataset.view === name));
    const title = { extract: 'Extract', batch: 'Batch', review: 'HITL Review',
                    history: 'History', templates: 'Templates', about: 'About' }[name] || name;
    $('#page-title').textContent = title;
    $('#breadcrumb').textContent =
      name === 'extract' ? 'Document Intelligence'
      : name === 'about' ? 'About'
      : `Document Intelligence · ${title}`;

    if (name === 'history')    refreshHistory();
    if (name === 'review')     { refreshCorrections(); renderReviewFields(); updateReviewBadge(); }
    if (name === 'templates')  refreshTemplates();
  }

  // ───────────────────────────── Extract ────────────────────────────────
  function setupExtract() {
    const dz = $('#drop-single');
    const fileInput = $('#file-single');
    const runBtn = $('#btn-run-extract');
    const clearBtn = $('#btn-clear-extract');

    const onPick = (f) => {
      state.lastFile = f;
      $('#file-single-name').textContent = f ? `${f.name} · ${fmtBytes(f.size)}` : 'No file selected';
      runBtn.disabled = !f;
      if (f) renderPreview(f);
      else  $('#preview-frame').innerHTML = $('#preview-empty-tpl').innerHTML || '';
    };

    ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('is-drag'); }));
    ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('is-drag'); }));
    dz.addEventListener('drop', e => { const f = e.dataTransfer?.files?.[0]; if (f) onPick(f); });
    dz.addEventListener('click', e => { if (e.target.closest('label')) return; fileInput.click(); });
    fileInput.addEventListener('change', e => onPick(e.target.files?.[0]));

    clearBtn.addEventListener('click', () => {
      onPick(null);
      fileInput.value = '';
      $('#extract-result').innerHTML =
        `<div class="result-empty">
           <svg viewBox="0 0 120 120" class="empty-svg"><circle cx="60" cy="60" r="40" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M44 60 L54 70 L78 46" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
           <h2>Results will appear here</h2>
           <p>Upload a document, then press <strong>Extract</strong>.</p>
         </div>`;
      state.lastResult = null;
      state.lastRunId = null;
    });

    runBtn.addEventListener('click', async () => {
      if (!state.lastFile) return;
      runBtn.disabled = true;
      $('#extract-progress').classList.remove('hidden');

      const fd = new FormData();
      fd.append('file', state.lastFile);
      const schema = $('#schema-picker').value;
      const backend = $('#backend-picker').value;
      if (schema)  fd.append('schema_name', schema);
      if (backend) fd.append('backend', backend);

      try {
        const body = await api('/extract', { method: 'POST', body: fd });
        state.lastResult = body;
        const hist = await api('/api/history?limit=20');
        const targetBase = basename(state.lastFile.name);
        const mine = (hist.runs || []).find(r =>
          basename(r.filename) === targetBase &&
          (r.schema_name || '') === (body.schema_name || '')
        );
        if (mine) state.lastRunId = mine.id;

        renderResult(body);
        renderReviewFields();
        updateReviewBadge();
        toast(`Extracted ${Object.keys(body.extraction || {}).length} fields`, 'success');
      } catch (e) {
        const code = e.code || 'IDP-INT-001';
        renderErrorPanel(e, `Extract failed (${code})`);
        toast(`${e.message} · ${code}`, 'error', 8000, { code, requestId: e.requestId, sticky: true });
      } finally {
        runBtn.disabled = !state.lastFile;
        $('#extract-progress').classList.add('hidden');
      }
    });
  }

  // ───────────────────────────── Preview ─────────────────────────────────
  function renderPreview(file) {
    const frame = $('#preview-frame');
    const name = file.name;
    const size = fmtBytes(file.size);
    const mime = file.type || '';
    const isText = /\.(txt|md|csv|json|log)$/i.test(name) || mime.startsWith('text/');
    const isImage = /^image\//.test(mime) || /\.(png|jpe?g|gif|webp|bmp)$/i.test(name);
    const isPdf   = /\.pdf$/i.test(name) || mime === 'application/pdf';

    let bodyHtml = '';
    if (isImage) {
      const url = URL.createObjectURL(file);
      bodyHtml = `<div class="preview-doc-body image-content"><img src="${url}" alt="${escapeHtml(name)}" /></div>`;
    } else if (isText) {
      const reader = new FileReader();
      reader.onload = () => {
        frame.querySelector('.preview-doc-body').textContent = reader.result;
      };
      reader.readAsText(file);
      bodyHtml = `<div class="preview-doc-body text-content">Loading…</div>`;
    } else if (isPdf) {
      bodyHtml = `<div class="preview-doc-body pdf-content">
        <svg viewBox="0 0 80 100" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="6" y="6" width="68" height="88" rx="4"/>
          <path d="M52 6 L74 28" />
          <path d="M52 6 L52 28 L74 28" />
          <line x1="14" y1="40" x2="62" y2="40" />
          <line x1="14" y1="50" x2="62" y2="50" />
          <line x1="14" y1="60" x2="48" y2="60" />
        </svg>
        <div><strong>${escapeHtml(name)}</strong></div>
        <div class="subtle">PDF · preview not rendered inline.<br>Open the file in your viewer to inspect contents.</div>
      </div>`;
    } else {
      bodyHtml = `<div class="preview-doc-body text-content">Binary file — preview unavailable.</div>`;
    }

    frame.innerHTML = `
      <div class="preview-doc">
        <div class="preview-doc-bar">
          <span class="filename">${escapeHtml(name)}</span>
          <span class="meta"><span>${size}</span><span>${escapeHtml(mime || 'unknown')}</span></span>
        </div>
        ${bodyHtml}
      </div>
    `;
  }

  // ───────────────────────────── Confidence rings ───────────────────────
  function ringMarkup(name, value) {
    const pct = Math.round(value * 100);
    const R = 14;
    const C = 2 * Math.PI * R;
    const offset = C * (1 - value);
    const cls = value < 0.6 ? 'low' : value < 0.85 ? 'mid' : '';
    return `
      <div class="confidence-row${value < 0.6 ? ' is-low' : ''}">
        <div>
          <div class="field-name">${escapeHtml(name)}</div>
          <div class="subtle" style="font-size:11px; font-family:var(--font-mono); margin-top:2px;">confidence</div>
        </div>
        <div class="confidence-ring ${cls}">
          <svg width="40" height="40" viewBox="0 0 40 40">
            <circle class="track" cx="20" cy="20" r="${R}"/>
            <circle class="fill" cx="20" cy="20" r="${R}"
                    stroke-dasharray="${C}" stroke-dashoffset="${C}"
                    data-target="${offset}"/>
          </svg>
          <div class="pct">${pct}%</div>
        </div>
      </div>`;
  }

  function animateRings(root) {
    requestAnimationFrame(() => {
      $$('.confidence-ring .fill', root).forEach(c => {
        const target = parseFloat(c.dataset.target);
        c.style.transition = 'none';
        c.setAttribute('stroke-dashoffset', c.getAttribute('stroke-dasharray'));
        requestAnimationFrame(() => {
          c.style.transition = 'stroke-dashoffset 0.9s cubic-bezier(0.16, 1, 0.3, 1)';
          c.setAttribute('stroke-dashoffset', target);
        });
      });
    });
  }

  // ───────────────────────────── Result panel ───────────────────────────
  function renderResult(body) {
    const root = $('#extract-result');
    const conf = body.confidence || {};
    const confEntries = Object.entries(conf).sort((a, b) => a[1] - b[1]);
    const extraction = body.extraction || {};
    const lowCount = confEntries.filter(([, v]) => v < 0.6).length;

    const metaChips = [
      `<span class="chip">schema <strong>${escapeHtml(body.schema_name || '—')}</strong></span>`,
      `<span class="chip">backend <strong>${escapeHtml(body.backend_name || '—')}</strong></span>`,
      body.classification ? `<span class="chip">doc <strong>${escapeHtml(body.classification)}</strong></span>` : '',
      body.mode             ? `<span class="chip">mode <strong>${escapeHtml(body.mode)}</strong></span>` : '',
      body.template_used   ? `<span class="chip">template <strong>${escapeHtml(body.template_used)}</strong></span>` : '',
      (body.validation && body.validation.valid !== undefined)
        ? `<span class="chip ${body.validation.valid ? 'success' : 'warn'}">validation <strong>${body.validation.valid ? 'PASS' : 'FAIL'}</strong></span>`
        : '',
    ].filter(Boolean).join('');

    const banner = lowCount > 0
      ? `<div class="banner warn">⚠ ${lowCount} low-confidence field${lowCount > 1 ? 's' : ''} flagged — switch to <strong>HITL Review</strong></div>`
      : `<div class="banner success">✓ All fields above 0.6 confidence</div>`;

    const rings = confEntries.length
      ? `<div class="confidence-list">${confEntries.map(([k, v]) => ringMarkup(k, v)).join('')}</div>`
      : `<p class="subtle" style="margin-top: 12px;">No per-field confidence returned.</p>`;

    root.innerHTML = `
      <div class="result-meta">${metaChips}</div>
      ${banner}
      <div class="section-title">Extracted fields</div>
      <pre class="json-block">${escapeHtml(JSON.stringify(extraction, null, 2))}</pre>
      <div class="section-title">Per-field confidence</div>
      ${rings}
      ${body.validation ? `
        <div class="section-title">Validation</div>
        <pre class="json-block">${escapeHtml(JSON.stringify(body.validation, null, 2))}</pre>
      ` : ''}
    `;
    animateRings(root);
  }

  // ───────────────────────────── Review / HITL ──────────────────────────
  function updateReviewBadge() {
    const body = state.lastResult;
    if (!body) { $('#nav-review-badge').hidden = true; return; }
    const low = Object.values(body.confidence || {}).filter(v => v < 0.6).length;
    const badge = $('#nav-review-badge');
    if (low > 0) { badge.hidden = false; badge.textContent = String(low); }
    else         { badge.hidden = true; }
  }

  function renderReviewFields() {
    const root = $('#review-fields');
    const body = state.lastResult;
    if (!body) {
      root.className = 'review-empty';
      root.innerHTML = '<p class="subtle">Run an extraction first — low-confidence fields will appear here.</p>';
      return;
    }
    const conf = body.confidence || {};
    const low = Object.entries(conf).filter(([, v]) => v < 0.6).sort((a, b) => a[1] - b[1]);
    if (low.length === 0) {
      root.className = 'review-empty';
      root.innerHTML = '<p class="subtle">✓ No fields below 0.6 confidence — nothing to review.</p>';
      return;
    }
    root.className = '';
    const ext = body.extraction || {};
    const rows = low.map(([k, v]) => {
      const raw = ext[k];
      const display = (raw === null || typeof raw === 'object') ? '' : String(raw);
      return `
        <div class="field-row is-low" data-field="${escapeHtml(k)}">
          <div class="field-name">${escapeHtml(k)}<span class="confidence-pill low">${(v * 100).toFixed(0)}%</span></div>
          <input type="text" placeholder="(empty)" value="${escapeHtml(display)}" data-original="${escapeHtml(display)}" />
          <button class="btn-ghost small" data-action="reset" data-field="${escapeHtml(k)}">Reset</button>
        </div>`;
    }).join('');

    root.innerHTML = `
      <p class="subtle" style="margin-bottom: 14px;">${low.length} field${low.length > 1 ? 's' : ''} below 0.6 confidence.</p>
      <div class="review-actions">
        <input type="text" placeholder="Reviewer name" value="${escapeHtml(state.reviewer)}" id="reviewer-name" style="max-width:240px;" />
        <button class="btn-primary" id="btn-save-corrections">Save corrections</button>
      </div>
      <div>${rows}</div>
      <textarea id="correction-note" rows="2" placeholder="Note (optional)" style="margin-top: 14px;"></textarea>
    `;

    $$('#review-fields [data-action="reset"]').forEach(b =>
      b.addEventListener('click', e => {
        const k = e.currentTarget.dataset.field;
        const inp = $(`#review-fields [data-field="${k}"] input`);
        if (inp) inp.value = inp.dataset.original;
      })
    );
    $('#btn-save-corrections')?.addEventListener('click', submitCorrection);
    $('#reviewer-name')?.addEventListener('change', e => {
      state.reviewer = e.target.value || '';
      localStorage.setItem('idp_reviewer', state.reviewer);
      const top = $('#reviewer-name-top');
      if (top) top.value = state.reviewer;
    });
  }

  async function submitCorrection() {
    if (!state.lastResult || !state.lastRunId) {
      toast('No run to correct yet — extract something first.', 'warn');
      return;
    }
    const before = { ...(state.lastResult.extraction || {}) };
    const after = { ...before };
    $$('#review-fields .field-row').forEach(row => {
      const k = row.dataset.field;
      const inp = $('input', row);
      if (inp) after[k] = inp.value;
    });
    const note = $('#correction-note')?.value || '';
    try {
      await api('/api/correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_id: state.lastRunId,
          reviewer: state.reviewer || 'anonymous',
          before, after, note,
        }),
      });
      toast('Correction saved', 'success');
      refreshCorrections();
    } catch (e) {
      toast(`Save failed: ${e.message}`, 'error');
    }
  }

  async function refreshCorrections() {
    const list = $('#corrections-list');
    try {
      const data = await api('/api/corrections?limit=50');
      const items = data.corrections || [];
      if (items.length === 0) {
        list.className = 'corrections-empty';
        list.innerHTML = '<p class="subtle">No corrections saved yet.</p>';
        return;
      }
      list.className = '';
      list.innerHTML = items.map(c => {
        const changed = Object.keys(c.after || {}).filter(k => JSON.stringify(c.before?.[k]) !== JSON.stringify(c.after?.[k]));
        const diffRows = changed.length === 0
          ? '<p class="subtle small" style="margin-top:8px;">No field changes (note-only)</p>'
          : changed.map(k => `
              <div class="diff-row">
                <code>${escapeHtml(k)}</code>
                <span class="diff-before">${escapeHtml(fmtVal(c.before?.[k]))}</span>
                <span class="diff-arrow">→</span>
                <span class="diff-after">${escapeHtml(fmtVal(c.after?.[k]))}</span>
              </div>
            `).join('');
        return `
          <div class="correction-item">
            <div class="head">
              <span class="who">${escapeHtml(c.reviewer || 'anonymous')}</span>
              <span class="when">${fmtTime(c.ts)}</span>
            </div>
            ${c.notes ? `<div class="note">"${escapeHtml(c.notes)}"</div>` : ''}
            <div style="margin-top: 10px;">${diffRows}</div>
          </div>`;
      }).join('');
    } catch (e) {
      list.className = 'corrections-empty';
      list.innerHTML = `<p class="subtle">Failed to load: ${escapeHtml(e.message)}</p>`;
    }
  }

  // ───────────────────────────── Batch (with drag-reorder) ──────────────
  function setupBatch() {
    const dz = $('#drop-batch');
    const fileInput = $('#file-batch');
    const runBtn = $('#btn-run-batch');
    const clearBtn = $('#btn-clear-batch');
    const queue = $('#batch-queue');
    const concurrencySelect = $('#batch-concurrency');
    concurrencySelect.value = '2';

    const render = () => {
      $('#file-batch-count').textContent = files.length
        ? `${files.length} file${files.length > 1 ? 's' : ''} queued`
        : 'No files queued';
      runBtn.disabled = files.length === 0;
      if (files.length === 0) {
        queue.className = 'queue';
        queue.innerHTML = `<div class="queue-empty">
          <svg viewBox="0 0 120 120" class="empty-svg" style="width:64px;height:64px;"><rect x="22" y="18" width="76" height="92" rx="6" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="40" y="38" width="40" height="56" rx="3" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
          <p style="margin-top:12px;">No files queued yet.</p>
        </div>`;
        return;
      }
      queue.className = 'queue';
      queue.innerHTML = files.map((f, i) => `
        <div class="queue-item" draggable="true" data-i="${i}">
          <span class="queue-handle" title="Drag to reorder">⋮⋮</span>
          <span class="filename">${escapeHtml(f.name)}</span>
          <span class="size">${fmtBytes(f.size)}</span>
          <span class="queue-badge ${f._state || 'queued'}">${f._state || 'queued'}</span>
          ${f._errorCode ? `<span class="queue-error-code" title="${escapeHtml(f._error || '')}">${escapeHtml(f._errorCode)}</span>` : ''}
          <button class="queue-remove" data-rm="${i}" title="Remove">✕</button>
        </div>
      `).join('');
      $$('#batch-queue [data-rm]').forEach(b =>
        b.addEventListener('click', e => {
          e.stopPropagation();
          files.splice(+e.currentTarget.dataset.rm, 1);
          render();
        })
      );
      setupDragReorder();
    };

    function setupDragReorder() {
      let dragSrc = null;
      $$('#batch-queue .queue-item').forEach(item => {
        item.addEventListener('dragstart', e => {
          dragSrc = +item.dataset.i;
          item.classList.add('is-dragging');
          e.dataTransfer.effectAllowed = 'move';
          e.dataTransfer.setData('text/plain', String(dragSrc));
        });
        item.addEventListener('dragend', () => {
          item.classList.remove('is-dragging');
          $$('.queue-item.is-dragover').forEach(el => el.classList.remove('is-dragover'));
          dragSrc = null;
        });
        item.addEventListener('dragover', e => {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'move';
          if (dragSrc !== null && +item.dataset.i !== dragSrc) {
            item.classList.add('is-dragover');
          }
        });
        item.addEventListener('dragleave', () => item.classList.remove('is-dragover'));
        item.addEventListener('drop', e => {
          e.preventDefault();
          const src = +e.dataTransfer.getData('text/plain');
          const dst = +item.dataset.i;
          if (src !== dst && !isNaN(src) && !isNaN(dst)) {
            const [moved] = files.splice(src, 1);
            files.splice(dst, 0, moved);
            render();
          }
        });
      });
    }

    let files = [];
    ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('is-drag'); }));
    ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('is-drag'); }));
    dz.addEventListener('drop', e => { files.push(...(e.dataTransfer?.files || [])); render(); });
    dz.addEventListener('click', e => { if (e.target.closest('label')) return; fileInput.click(); });
    fileInput.addEventListener('change', e => { files.push(...e.target.files); render(); });

    clearBtn.addEventListener('click', () => { files = []; render(); });

    runBtn.addEventListener('click', async () => {
      if (files.length === 0) return;
      runBtn.disabled = true;
      const concurrency = parseInt($('#batch-concurrency').value, 10) || 2;
      const schema = $('#batch-schema').value;
      const backend = $('#backend-picker').value;

      files.forEach(f => f._state = 'running');
      render();

      const tasks = files.map(async (file) => {
        while (state.batch.running >= concurrency) {
          await new Promise(r => setTimeout(r, 50));
        }
        state.batch.running++;
        try {
          const fd = new FormData();
          fd.append('file', file);
          if (schema)  fd.append('schema_name', schema);
          if (backend) fd.append('backend', backend);
          const body = await api('/extract', { method: 'POST', body: fd });
          file._state = 'done';
          file._result = body;
        } catch (e) {
          file._state = 'error';
          file._error = e.message;
          file._errorCode = e.code || 'IDP-INT-001';
          file._requestId = e.requestId || null;
        } finally {
          state.batch.running--;
          render();
        }
      });
      await Promise.all(tasks);
      const ok = files.filter(f => f._state === 'done').length;
      const err = files.filter(f => f._state === 'error').length;
      toast(`Batch done — ${ok} ok, ${err} error${err === 1 ? '' : 's'}`, err === 0 ? 'success' : 'warn');
      runBtn.disabled = files.length === 0;
      pollStatus();
    });

    render();
  }

  // ───────────────────────────── History ────────────────────────────────
  async function refreshHistory() {
    const list = $('#history-list');
    try {
      const data = await api('/api/history?limit=50');
      const runs = data.runs || [];
      if (runs.length === 0) {
        list.className = 'history-empty';
        list.innerHTML = '<p class="subtle">No runs yet.</p>';
        return;
      }
      list.className = 'history-list';
      list.innerHTML = runs.map(r => {
        const result = r.result || {};
        const fields = result.extraction ? Object.keys(result.extraction).length : 0;
        const conf = result.confidence || {};
        const low = Object.values(conf).filter(v => v < 0.6).length;
        const fname = basename(r.filename) || '(no name)';
        return `
          <details class="history-item">
            <summary>
              <span class="run-title">${escapeHtml(fname)}</span>
              <span class="run-meta">
                <span>${escapeHtml(r.schema_name || '')}</span>
                <span>${fmtTime(r.ts)}</span>
                ${fields ? `<span><strong>${fields}</strong> fields</span>` : ''}
                ${low    ? `<span class="warn"><strong>${low}</strong> low-conf</span>` : ''}
              </span>
            </summary>
            <div class="run-body">
              ${result.extraction ? `<div class="section-title">Extraction</div><pre class="json-block">${escapeHtml(JSON.stringify(result.extraction, null, 2))}</pre>` : ''}
              ${result.confidence ? `<div class="section-title">Confidence</div><pre class="json-block">${escapeHtml(JSON.stringify(result.confidence, null, 2))}</pre>` : ''}
              ${result.validation ? `<div class="section-title">Validation</div><pre class="json-block">${escapeHtml(JSON.stringify(result.validation, null, 2))}</pre>` : ''}
            </div>
          </details>`;
      }).join('');
    } catch (e) {
      list.className = 'history-empty';
      list.innerHTML = `<p class="subtle">Failed to load: ${escapeHtml(e.message)}</p>`;
    }
  }

  // ───────────────────────────── Templates ──────────────────────────────
  async function refreshTemplates() {
    const list = $('#templates-list');
    try {
      const headers = authHeaders();
      const res = await fetch(apiUrl('/templates'), { headers });
      if (res.status === 401 || res.status === 403) {
        list.className = 'templates-loading';
        list.innerHTML = `<p class="subtle">Templates require an API key. Pass <code>?api_key=…</code> in the URL or set localStorage.</p>`;
        return;
      }
      const templates = await res.json();
      if (!Array.isArray(templates) || templates.length === 0) {
        list.className = 'templates-loading';
        list.innerHTML = '<p class="subtle">No templates registered. Add <code>.md</code> files to the templates/ directory.</p>';
        return;
      }
      list.className = 'templates-list';
      const details = await Promise.all(templates.map(async t => {
        try {
          const r = await fetch(apiUrl(`/templates/${encodeURIComponent(t.name)}`), { headers });
          if (!r.ok) return { t, body: null };
          return { t, body: await r.json() };
        } catch { return { t, body: null }; }
      }));
      list.innerHTML = details.map(({ t, body }) => `
        <details class="template-item">
          <summary>
            <span class="t-name">${escapeHtml(t.name)}</span>
            <span class="t-meta">
              <span>schema <code>${escapeHtml(t.schema || '?')}</code></span>
              <span>v${t.version}</span>
              ${(t.mime_types || []).map(m => `<span class="chip">${escapeHtml(m)}</span>`).join('')}
            </span>
          </summary>
          ${body?.body ? `<div class="t-body"><pre>${escapeHtml(body.body)}</pre></div>` : '<div class="t-body"><p class="subtle">No body.</p></div>'}
        </details>
      `).join('');
    } catch (e) {
      list.className = 'templates-loading';
      list.innerHTML = `<p class="subtle">Failed to load: ${escapeHtml(e.message)}</p>`;
    }
  }

  // ───────────────────────────── Misc ───────────────────────────────────
  function setupMisc() {
    $('#btn-clear-history')?.addEventListener('click', async () => {
      if (!confirm('Clear all run history? Corrections are kept.')) return;
      try {
        await api('/api/history/clear', { method: 'POST' });
        toast('History cleared', 'success');
        refreshHistory();
        pollStatus();
      } catch (e) {
        toast(`Clear failed: ${e.message}`, 'error');
      }
    });
    $('#btn-refresh-corrections')?.addEventListener('click', refreshCorrections);

    // Top reviewer field
    const top = $('#reviewer-name-top');
    if (top) {
      top.value = state.reviewer;
      top.addEventListener('change', e => {
        state.reviewer = e.target.value || '';
        localStorage.setItem('idp_reviewer', state.reviewer);
        const low = $('#reviewer-name');
        if (low) low.value = state.reviewer;
      });
    }

    // Backend-URL field (sidebar). Empty = same-origin (local ./start.sh mode).
    // When non-empty, every fetch goes through apiUrl() → prefix with state.apiBase.
    const baseInput = $('#api-base-url');
    if (baseInput) {
      baseInput.value = state.apiBase;
      baseInput.addEventListener('change', e => {
        let v = (e.target.value || '').trim();
        // Allow-without-trailing-slash only. Strip if user pasted one.
        v = v.replace(/\/+$/, '');
        e.target.value = v;
        state.apiBase = v;
        if (v) localStorage.setItem('idp_api_base', v);
        else   localStorage.removeItem('idp_api_base');
        // Refresh status pill so the user sees the effect immediately.
        pollStatus();
      });
    }

    // Static-demo banner: respect persisted dismissal and hide once a backend
    // is configured (the static-only message is no longer true).
    const banner = document.querySelector('.static-demo-banner');
    if (banner) {
      if (state.apiBase || localStorage.getItem('idp_banner_disabled') === '1') {
        banner.remove();
      } else {
        banner.querySelector('.static-demo-dismiss')
          ?.addEventListener('click', () => {
            banner.remove();
            localStorage.setItem('idp_banner_disabled', '1');
          });
      }
    }
  }

  // ───────────────────────────── Util ───────────────────────────────────
  function escapeHtml(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ───────────────────────────── Boot ───────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    setupNav();
    setupExtract();
    setupBatch();
    setupMisc();

    const params = new URLSearchParams(location.search);
    const k = params.get('api_key');
    if (k) { state.apiKey = k; localStorage.setItem('idp_api_key', k); }
    const b = params.get('api_base');
    if (b) {
      state.apiBase = b.replace(/\/+$/, '');
      localStorage.setItem('idp_api_base', state.apiBase);
    }

    pollStatus();
    setInterval(pollStatus, 15000);
  });
})();