/**
 * app.js — SBI MF FAQ Assistant Frontend Logic
 *
 * Responsibilities:
 *   - Server health polling (status badge in header)
 *   - Quick-start chip → populate input and submit
 *   - Input validation and character counter
 *   - POST /chat → render user bubble + typing indicator → render bot bubble
 *   - Citation card, confidence tier badge, PII notice rendering
 *   - Welcome screen hide/show
 *   - Error toast notifications
 */

'use strict';

/* ── DOM Refs ─────────────────────────────────────────────────────────────── */

const queryInput    = document.getElementById('queryInput');
const sendBtn       = document.getElementById('sendBtn');
const chatFeed      = document.getElementById('chatFeed');
const welcomeScreen = document.getElementById('welcomeScreen');
const charCounter   = document.getElementById('charCounter');
const statusDot     = document.getElementById('statusDot');
const statusLabel   = document.getElementById('statusLabel');
const toast         = createToast();

/* ── State ────────────────────────────────────────────────────────────────── */

let isLoading = false;

/* ── Helpers ──────────────────────────────────────────────────────────────── */

function now() {
  return new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Format answer text: strip "Source:" and "Last updated" lines (shown in citation card) */
function formatAnswer(text) {
  const lines = text.split('\n');
  const body  = [];

  for (const line of lines) {
    const trimmed = line.trim();
    // Source and date lines are rendered in the citation card below the bubble
    if (!trimmed.startsWith('Source:') && !trimmed.startsWith('Last updated from sources:')) {
      body.push(escapeHtml(trimmed));
    }
  }

  return body.filter(Boolean).join('<br>');
}

/** Extract "Source: <url>" from raw answer text */
function extractSource(text) {
  const m = text.match(/Source:\s*(https?:\/\/\S+)/i);
  return m ? m[1].trim() : '';
}

/** Extract "Last updated from sources: <date>" from raw answer text */
function extractDate(text) {
  const m = text.match(/Last updated from sources:\s*(\S+)/i);
  return m ? m[1].trim() : '';
}

/* ── Toast ────────────────────────────────────────────────────────────────── */

function createToast() {
  const el = document.createElement('div');
  el.className = 'toast';
  el.setAttribute('role', 'alert');
  document.body.appendChild(el);
  return el;
}

let toastTimer = null;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4000);
}

/* ── Status Badge ─────────────────────────────────────────────────────────── */

async function checkHealth() {
  const ctrl = new AbortController();
  const tid  = setTimeout(() => ctrl.abort(), 5000);
  try {
    const res  = await fetch('/health', { signal: ctrl.signal });
    clearTimeout(tid);
    const data = await res.json();

    statusDot.className = 'status-dot';

    if (data.engine_ready && data.ollama_online) {
      statusDot.classList.add('status-dot--ok');
      const model = data.ollama_model ? ` · ${data.ollama_model}` : '';
      statusLabel.textContent = `Ready${model}`;
    } else if (data.engine_ready && !data.ollama_online) {
      statusDot.classList.add('status-dot--degraded');
      statusLabel.textContent = 'Ollama offline';
    } else {
      statusDot.classList.add('status-dot--degraded');
      statusLabel.textContent = 'Loading models…';
    }
  } catch (e) {
    clearTimeout(tid);
    statusDot.className = 'status-dot status-dot--error';
    statusLabel.textContent = 'Server unreachable';
  }
}

// Poll health every 15 s; immediate first check
checkHealth();
setInterval(checkHealth, 15_000);

/* ── Welcome Screen ───────────────────────────────────────────────────────── */

function hideWelcome() {
  if (welcomeScreen && welcomeScreen.parentNode) {
    welcomeScreen.style.animation = 'fadeUp 0.25s ease reverse both';
    setTimeout(() => welcomeScreen.remove(), 250);
  }
}

/* ── Quick-Start Chips ────────────────────────────────────────────────────── */

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const query = chip.dataset.query;
    if (query) {
      queryInput.value = query;
      updateCounter();
      updateSendBtn();
      queryInput.focus();
      submitQuery();
    }
  });
});

/* ── Character Counter ────────────────────────────────────────────────────── */

function updateCounter() {
  const len = queryInput.value.length;
  charCounter.textContent = `${len} / 500`;
  charCounter.className = 'input-wrap__counter';
  if (len > 450) charCounter.classList.add('warn');
  if (len >= 500) charCounter.classList.add('over');
}

function updateSendBtn() {
  const val = queryInput.value.trim();
  sendBtn.disabled = val.length < 2 || val.length > 500 || isLoading;
}

queryInput.addEventListener('input', () => {
  updateCounter();
  updateSendBtn();
});

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !sendBtn.disabled) {
    e.preventDefault();
    submitQuery();
  }
});

sendBtn.addEventListener('click', () => {
  if (!sendBtn.disabled) submitQuery();
});

/* ── Render: User Bubble ──────────────────────────────────────────────────── */

function appendUserBubble(text) {
  const msg = document.createElement('div');
  msg.className = 'msg msg--user';
  msg.innerHTML = `
    <div class="msg__bubble">${escapeHtml(text)}</div>
    <span class="msg__time">${now()}</span>
  `;
  chatFeed.appendChild(msg);
  scrollToBottom();
}

/* ── Render: Typing Indicator ─────────────────────────────────────────────── */

function appendTypingIndicator() {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg--bot';
  wrap.id = 'typingIndicator';
  wrap.innerHTML = `
    <div class="typing-indicator" aria-label="Assistant is thinking">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatFeed.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

/* ── Render: Tier Badge ───────────────────────────────────────────────────── */

function tierBadge(tier) {
  if (!tier || tier === 'ADVISORY') return ''; // advisory is shown as refusal, no badge needed in body
  const icons = {
    HIGH:     '✦',
    MODERATE: '◈',
    LOW:      '◇',
  };
  const icon = icons[tier] || '';
  return `<span class="tier-badge tier-badge--${tier}" title="Retrieval confidence: ${tier}">${icon} ${tier}</span>`;
}

/* ── Render: Bot Bubble ───────────────────────────────────────────────────── */

function appendBotBubble(data) {
  const {
    answer,
    confidence_tier,
    source_url,
    publication_date,
    is_advisory,
    pii_redacted,
    latency_ms,
  } = data;

  // Extract source/date from answer text if API fields are empty
  const srcUrl  = source_url  || extractSource(answer);
  const srcDate = publication_date || extractDate(answer);

  // Format body text (strip source/date lines — shown in card below)
  const bodyHtml = formatAnswer(answer);

  // Citation card (only for non-advisory answers with a source)
  let citationHtml = '';
  if (!is_advisory && srcUrl) {
    citationHtml = `
      <div class="citation-card">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
          <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
        </svg>
        <a class="citation-card__link" href="${escapeHtml(srcUrl)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(srcUrl)}">${escapeHtml(srcUrl)}</a>
        ${srcDate ? `<span class="citation-card__date">Updated: ${escapeHtml(srcDate)}</span>` : ''}
        ${tierBadge(confidence_tier)}
      </div>
    `;
  }

  // PII notice
  let piiHtml = '';
  if (pii_redacted && pii_redacted.length > 0) {
    piiHtml = `
      <div class="pii-notice">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
        Sensitive info redacted from your query: ${pii_redacted.map(escapeHtml).join(', ')}
      </div>
    `;
  }

  // Latency note (shown as part of timestamp)
  const latencyNote = latency_ms ? ` · ${(latency_ms / 1000).toFixed(1)}s` : '';

  const msg = document.createElement('div');
  msg.className = 'msg msg--bot';
  msg.innerHTML = `
    <div class="msg__bubble">
      ${bodyHtml}
      ${citationHtml}
      ${piiHtml}
    </div>
    <span class="msg__time">${now()}${latencyNote}</span>
  `;
  chatFeed.appendChild(msg);
  scrollToBottom();
}

/* ── Scroll ───────────────────────────────────────────────────────────────── */

function scrollToBottom() {
  requestAnimationFrame(() => {
    document.querySelector('.main').scrollTop = 99999;
  });
}

/* ── Core Submit ──────────────────────────────────────────────────────────── */

async function submitQuery() {
  const query = queryInput.value.trim();
  if (!query || isLoading) return;

  // Hide welcome banner on first message
  hideWelcome();

  // Lock UI
  isLoading = true;
  queryInput.value = '';
  updateCounter();
  updateSendBtn();
  queryInput.disabled = true;

  // Render user bubble + typing indicator
  appendUserBubble(query);
  appendTypingIndicator();

  // Set up a manual timeout (120 s) using AbortController for broad browser compat
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), 120_000);

  try {
    const res = await fetch('/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query }),
      signal:  controller.signal,
    });
    clearTimeout(timeoutId);

    removeTypingIndicator();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    appendBotBubble(data);

  } catch (err) {
    clearTimeout(timeoutId);
    removeTypingIndicator();
    // DOMException with name 'AbortError' is thrown when the controller aborts
    if (err.name === 'AbortError' || err.name === 'TimeoutError') {
      showToast('Request timed out. Ollama may be slow — please try again.');
      appendBotBubble({
        answer: 'The request timed out. The local LLM (Ollama) may be under heavy load. Please try again in a moment.',
        confidence_tier: '',
        source_url: '',
        publication_date: '',
        is_advisory: false,
        pii_redacted: [],
        latency_ms: 0,
      });
    } else {
      showToast(`Error: ${err.message}`);
      appendBotBubble({
        answer: `An error occurred: ${err.message}. Please check that the API server and Ollama are running.`,
        confidence_tier: '',
        source_url: '',
        publication_date: '',
        is_advisory: false,
        pii_redacted: [],
        latency_ms: 0,
      });
    }
  } finally {
    isLoading = false;
    queryInput.disabled = false;
    updateSendBtn();
    queryInput.focus();
  }
}
