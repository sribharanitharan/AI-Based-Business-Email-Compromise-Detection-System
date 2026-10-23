const API = window.location.origin;

// ── Page navigation ──────────────────────────────────────────────────────────
function showPage(name, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  if (el) el.classList.add('active');
  document.getElementById('pageTitle').textContent = {
    analyze:  'Analyze Email',
    thread:   'Thread Analysis',
    dashboard:'Dashboard',
    stream:   'Live Mailbox Stream',
    models:   'Model Comparison',
    history:  'History',
    review:   'Analyst Review Queue',
    threat:   'Threat Intelligence',
    about:    'About'
  }[name] || name;
  if (name === 'dashboard') loadDashboard();
  if (name === 'stream')    initStreamPage();
  if (name === 'models')    loadModelComparison();
  if (name === 'history')   loadHistory();
  if (name === 'review')    loadReviewQueue();
  if (name === 'threat')    loadThreatIntel();
  if (name === 'thread')    initThreadPage();
}

// ── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('topbarTime').textContent =
    new Date().toLocaleTimeString('en-IN', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const r = await fetch(API + '/api/health', { signal: AbortSignal.timeout(2000) });
    const d = await r.json();
    if (d.models_loaded) {
      dot.className = 'status-dot online';
      text.textContent = 'Models Ready';
    } else {
      dot.className = 'status-dot';
      text.textContent = 'Run train.py';
    }
  } catch {
    dot.className = 'status-dot offline';
    text.textContent = 'Server Offline';
  }
}
checkHealth();
setInterval(checkHealth, 10000);

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3500);
}

// ── Sample data ───────────────────────────────────────────────────────────────
const SAMPLES = {
  bec: {
    subject:  'URGENT: Wire Transfer Required Immediately',
    body:     'Hi,\n\nI need you to process an urgent wire transfer of $85,000 to our new vendor account. This is extremely time sensitive and must be done TODAY. Do not discuss this with anyone until the transfer is complete. Keep this confidential.\n\nRouting: 021000021\nAccount: 3748291002\n\nACT NOW!\n\nCEO',
    from_addr:'ceo@comp4ny.com',
    to_addr:  'finance@company.com',
    reply_to: 'attacker@gmail.com',
    urls:     'http://bit.ly/wire-confirm\nhttp://203.0.113.5/login'
  },
  payroll: {
    subject:  'Important: Change of direct deposit bank account details',
    body:     'Hello HR,\n\nI need to update my direct deposit bank account routing details starting with this upcoming pay period. Please update my bank to:\nBank: Chase Federal\nRouting: 021000021\nAccount: 9948271049\n\nPlease confirm as soon as this is updated.\n\nThanks,\nJohn Smith (VP Operations)',
    from_addr:'john.smith@comp4ny-hr.co',
    to_addr:  'hr-payroll@company.com',
    reply_to: 'hacker-mailbox@gmail.com',
    urls:     ''
  },
  gift: {
    subject:  'Quick favor needed from store',
    body:     'Are you at your desk right now? I am currently stuck in a confidential board meeting and cannot talk. I need you to quickly purchase 5 Apple/iTunes Gift Cards ($100 each) for an emergency client incentive. Scratch the backs and email me pictures of the codes ASAP. Do not tell anyone yet.',
    from_addr:'executive.director@company-vip.xyz',
    to_addr:  'staff@company.com',
    reply_to: 'vip-exec@outlook.com',
    urls:     ''
  },
  legit: {
    subject:  'Q3 Sales Report – Please Review',
    body:     'Hi team,\n\nPlease find the Q3 sales report attached for your review. Key highlights include a 12% increase in new customers and strong retention numbers across all regions.\n\nPlease share your feedback by end of week.\n\nBest regards,\nAlice',
    from_addr:'alice@company.com',
    to_addr:  'team@company.com',
    reply_to: '',
    urls:     'https://drive.google.com/file/d/abc123'
  }
};

function loadSample(type) {
  const s = SAMPLES[type];
  if (!s) return;
  document.getElementById('f-subject').value = s.subject;
  document.getElementById('f-body').value    = s.body;
  document.getElementById('f-from').value    = s.from_addr;
  document.getElementById('f-to').value      = s.to_addr;
  document.getElementById('f-replyto').value = s.reply_to;
  document.getElementById('f-urls').value    = s.urls;
  showToast(`Loaded sample email`);
}

function clearForm() {
  ['f-subject','f-body','f-from','f-to','f-replyto','f-urls']
    .forEach(id => { document.getElementById(id).value = ''; });
  document.getElementById('resultEmpty').classList.remove('hidden');
  document.getElementById('resultContent').classList.add('hidden');
  _currentEmailId = null;
}

// ── File upload (.eml) ────────────────────────────────────────────────────────
async function uploadEmlFile(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  const formData = new FormData();
  formData.append("file", file);
  showToast(`📁 Uploading & parsing ${file.name}...`);
  try {
    const res = await fetch(API + '/api/analyze/eml', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Failed to analyze .eml file');
    const data = await res.json();
    document.getElementById('f-subject').value = data.subject_preview || '';
    document.getElementById('f-from').value = data.sender || '';
    renderResult(data);
    showToast(`✅ Analyzed ${file.name}: [${data.risk_level}] Risk`);
  } catch (err) {
    showToast('❌ Error parsing .eml: ' + err.message);
  }
}

// ── Real-Time Streaming Traffic Simulator ────────────────────────────────────
let _trafficTimer = null;
async function toggleLiveTraffic() {
  const btn = document.getElementById('liveTrafficBtn');
  if (_trafficTimer) {
    clearInterval(_trafficTimer);
    _trafficTimer = null;
    if (btn) btn.innerHTML = '⚡ Live Traffic Stream';
    showToast('⏹️ Live traffic simulation stopped');
    return;
  }

  showToast('▶️ Starting Real-time Email Stream...');
  if (btn) btn.innerHTML = '⏹️ Stop Stream';

  async function pump() {
    try {
      const res = await fetch(API + '/api/simulate/traffic?count=1', { method: 'POST' });
      const data = await res.json();
      if (data.results && data.results.length > 0) {
        const item = data.results[0];
        showToast(`📬 Incoming Email: "${item.subject_preview}" -> [${item.risk_level}]`);
        const curPage = document.querySelector('.page.active');
        if (curPage && curPage.id === 'page-dashboard') loadDashboard();
        if (curPage && curPage.id === 'page-history') loadHistory();
        if (curPage && curPage.id === 'page-review') loadReviewQueue();
      }
    } catch (e) {
      console.error(e);
    }
  }
  pump();
  _trafficTimer = setInterval(pump, 4000);
}

// ── Analyze ────────────────────────────────────────────────────────────────────
let _currentEmailId = null;

async function analyzeEmail() {
  const btn  = document.getElementById('analyzeBtn');
  const body = document.getElementById('f-body').value.trim();
  if (!body) { showToast('⚠️ Please enter email body'); return; }

  btn.classList.add('loading');
  btn.innerHTML = '<span class="pulse">⏳</span> Analyzing...';

  const urlsRaw = document.getElementById('f-urls').value;
  const urls    = urlsRaw.split('\n').map(u => u.trim()).filter(Boolean);

  const payload = {
    subject:   document.getElementById('f-subject').value,
    body,
    from_addr: document.getElementById('f-from').value,
    to_addr:   document.getElementById('f-to').value,
    reply_to:  document.getElementById('f-replyto').value,
    urls
  };

  try {
    const res  = await fetch(API + '/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    _currentEmailId = data.email_id;
    renderResult(data);
  } catch(e) {
    showToast('❌ Error: ' + (e.message || 'Could not connect to server'));
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Email';
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
const RISK_COLORS = {
  CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e'
};

function renderResult(d) {
  document.getElementById('resultEmpty').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');

  // Score ring
  const pct       = Math.round(d.risk_score * 100);
  const circumf   = 326.7;
  const offset    = circumf - (circumf * d.risk_score);
  const fill      = document.getElementById('ringFill');
  const color     = RISK_COLORS[d.risk_level] || '#3b82f6';
  fill.style.stroke           = color;
  fill.style.strokeDashoffset = offset;
  fill.style.filter           = `drop-shadow(0 0 8px ${color}66)`;
  document.getElementById('ringScore').textContent = pct + '%';
  document.getElementById('ringScore').style.color = color;

  // Badge
  const badge = document.getElementById('riskBadge');
  badge.textContent = d.risk_level;
  badge.className   = 'risk-badge ' + d.risk_level;
  document.getElementById('becType').textContent =
    d.bec_type !== 'N/A' ? '⚠️ ' + d.bec_type : '✅ No BEC pattern detected';

  // Action
  const actionBox = document.getElementById('actionBox');
  actionBox.querySelector('#actionText').textContent = d.recommended_action;
  const iconMap = { CRITICAL:'🚫', HIGH:'⚠️', MEDIUM:'🔍', LOW:'✅' };
  actionBox.querySelector('.action-icon').textContent = iconMap[d.risk_level] || '⚡';

  // Model scores
  const grid = document.getElementById('modelScoresGrid');
  const modelLabels = {
    lr:'Logistic Reg.', xgb:'XGBoost',
    gb:'Grad. Boosting', svm:'SVM',
    lightgbm:'LightGBM', catboost:'CatBoost'
  };
  grid.innerHTML = Object.entries(d.model_scores).map(([k, v]) => `
    <div class="ms-item">
      <div class="ms-name">${modelLabels[k] || k}</div>
      <div class="ms-val">${(v * 100).toFixed(1)}%</div>
      <div class="ms-bar"><div class="ms-bar-fill" style="width:${v*100}%"></div></div>
    </div>
  `).join('');

  // SHAP waterfall
  const shapSection = document.getElementById('shapSection');
  const shapList    = document.getElementById('shapList');
  if (d.top_shap_features && d.top_shap_features.length > 0) {
    shapSection.style.display = '';
    const maxImpact = Math.max(...d.top_shap_features.map(f => Math.abs(f.impact)));
    shapList.innerHTML = d.top_shap_features.map(f => {
      const pct   = maxImpact > 0 ? (Math.abs(f.impact) / maxImpact * 100).toFixed(0) : 0;
      const isPos = f.impact >= 0;
      return `
        <div class="shap-item">
          <div class="shap-feat">${f.feature.replace(/_/g,' ')}</div>
          <div class="shap-bar-wrap">
            <div class="shap-bar ${isPos ? 'shap-pos' : 'shap-neg'}" style="width:${pct}%"></div>
          </div>
          <div class="shap-val ${isPos ? 'shap-pos-txt' : 'shap-neg-txt'}">${isPos?'+':''}${f.impact.toFixed(4)}</div>
        </div>
      `;
    }).join('');
  } else {
    shapSection.style.display = 'none';
  }

  // Flags
  const flagsList = document.getElementById('flagsList');
  if (!d.flags || d.flags.length === 0) {
    flagsList.innerHTML = '<div class="no-flags">✅ No threat indicators detected</div>';
  } else {
    const iconMap2 = {
      header:'📨', linguistic:'🗣️', social_eng:'🎭',
      url:'🔗', domain:'🌐', content:'📝', thread:'🧵', threat_intel:'🛡️'
    };
    flagsList.innerHTML = d.flags.map(f => `
      <div class="flag-item">
        <span class="flag-icon">${iconMap2[f.type] || '🚩'}</span>
        <span>${f.message}</span>
      </div>
    `).join('');
  }

  // Meta
  document.getElementById('resultMeta').textContent =
    `ID: ${d.email_id}  •  Confidence: ${(d.confidence*100).toFixed(1)}%  •  ${d.timestamp}`;
}

// ── PDF Report download ────────────────────────────────────────────────────────
async function downloadReport() {
  if (!_currentEmailId) { showToast('⚠️ No email analyzed yet'); return; }
  showToast('📄 Generating PDF report...');
  window.open(API + `/api/report/${_currentEmailId}`, '_blank');
}

// ── Analyst Feedback ──────────────────────────────────────────────────────────
async function submitFeedback(label) {
  if (!_currentEmailId) { showToast('⚠️ No email analyzed yet'); return; }
  try {
    const res = await fetch(API + '/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email_id:      _currentEmailId,
        analyst_label: label,
        analyst_notes: `Manually classified via dashboard as ${label}`
      })
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    showToast(`✅ ${d.message} (${label})`);
  } catch(e) {
    showToast('❌ Feedback error: ' + e.message);
  }
}

// ── CSV Export ────────────────────────────────────────────────────────────────
function exportCSV() {
  showToast('📥 Downloading CSV...');
  window.location.href = API + '/api/export/csv';
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
let riskChart = null, modelChart = null, becTypeChart = null;

async function loadDashboard() {
  try {
    const stats = await fetch(API + '/api/stats').then(r => r.json());

    document.getElementById('s-total').textContent     = stats.total_analyzed;
    document.getElementById('s-bec').textContent       = stats.bec_detected;
    document.getElementById('s-legit').textContent     = stats.legitimate;
    document.getElementById('s-rate').textContent      = stats.detection_rate + '%';
    document.getElementById('s-uncertain').textContent = stats.uncertain_count ?? '—';
    document.getElementById('s-feedback').textContent  = stats.feedback_count ?? '—';

    // AUC
    const auc = stats.ensemble_auc || 0;
    document.getElementById('aucValue').textContent = auc.toFixed(4);
    setTimeout(() => {
      document.getElementById('aucBar').style.width = (auc * 100) + '%';
    }, 100);

    buildRiskChart(stats.risk_breakdown || {});
    buildModelChart(stats.model_f1 || {});
    buildBecTypeChart(stats.bec_types || {});

  } catch(e) {
    showToast('Could not load dashboard — is the server running?');
  }
}

function buildRiskChart(rb) {
  const ctx = document.getElementById('riskChart').getContext('2d');
  if (riskChart) riskChart.destroy();
  riskChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Critical', 'High', 'Medium', 'Low / Safe'],
      datasets: [{
        data: [rb.CRITICAL||0, rb.HIGH||0, rb.MEDIUM||0, rb.LOW||0],
        backgroundColor: ['#ef4444cc','#f97316cc','#eab308cc','#22c55ecc'],
        borderColor:     ['#ef4444',  '#f97316',  '#eab308',  '#22c55e'],
        borderWidth: 1.5, borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94a3b8', font: {size: 11}, padding: 12, boxWidth: 12 }
        }
      }
    }
  });
}

function buildModelChart(mf) {
  const labelMap = {
    logistic_regression:'LR', xgboost:'XGBoost',
    gradient_boosting:'Grad.B', svm:'SVM', lightgbm:'LightGBM',
    catboost:'CatBoost', ensemble:'Ensemble'
  };
  const labels = Object.keys(mf).map(k => labelMap[k] || k);
  const values = Object.values(mf).map(v => (v*100).toFixed(1));
  const colors = ['#3b82f6','#6366f1','#06b6d4','#8b5cf6','#22c55e','#f59e0b','#ef4444'];

  const ctx = document.getElementById('modelChart').getContext('2d');
  if (modelChart) modelChart.destroy();
  modelChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'F1 Score (%)',
        data: values,
        backgroundColor: colors.map(c => c + '99'),
        borderColor: colors,
        borderWidth: 1.5, borderRadius: 6
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { color: '#64748b', font: {size: 10}, callback: v => v + '%' },
          grid:  { color: 'rgba(255,255,255,.04)' }
        },
        x: { ticks: { color: '#94a3b8', font: {size: 10} }, grid: { display: false } }
      }
    }
  });
}

function buildBecTypeChart(bec_types) {
  const labels = Object.keys(bec_types);
  const values = Object.values(bec_types);
  const colors = ['#ef4444','#f97316','#eab308','#06b6d4','#8b5cf6','#22c55e','#3b82f6','#f43f5e'];

  const ctx = document.getElementById('becTypeChart').getContext('2d');
  if (becTypeChart) becTypeChart.destroy();

  if (!labels.length) {
    // Draw empty state
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = '#475569';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('No BEC emails detected yet', ctx.canvas.width/2, ctx.canvas.height/2);
    return;
  }

  becTypeChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Count',
        data: values,
        backgroundColor: colors.slice(0, labels.length).map(c => c + 'cc'),
        borderColor:     colors.slice(0, labels.length),
        borderWidth: 1.5, borderRadius: 6
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#64748b', font: {size: 10} },
          grid:  { color: 'rgba(255,255,255,.04)' }
        },
        y: { ticks: { color: '#94a3b8', font: {size: 10} }, grid: { display: false } }
      }
    }
  });
}

// ── History ───────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const data = await fetch(API + '/api/history').then(r => r.json());
    const el   = document.getElementById('historyTable');

    if (!data.length) {
      el.innerHTML = '<div class="history-empty">No emails analyzed yet. Go to Analyze Email to get started.</div>';
      return;
    }

    el.innerHTML = `
      <table class="history-table">
        <thead>
          <tr>
            <th>ID</th><th>Time</th><th>Subject</th>
            <th>Sender</th><th>Score</th><th>Level</th><th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${data.map(e => `
            <tr>
              <td style="font-family:monospace;color:var(--text3)">${e.email_id}</td>
              <td style="color:var(--text3)">${e.timestamp.split(' ')[1]}</td>
              <td title="${e.subject_preview||''}">${(e.subject_preview||'—').substring(0,40)}</td>
              <td style="color:var(--text3)">${(e.sender||'—').substring(0,25)}</td>
              <td class="history-score" style="color:${RISK_COLORS[e.risk_level]||'#fff'}">
                ${(e.risk_score*100).toFixed(1)}%
              </td>
              <td><span class="h-badge ${e.risk_level}">${e.risk_level}</span></td>
              <td>
                <button class="btn btn-ghost btn-xs" onclick="window.open('${API}/api/report/${e.email_id}','_blank')">PDF</button>
                <button class="btn btn-ghost btn-xs" onclick="loadExplain('${e.email_id}')">SHAP</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch(e) {
    showToast('Could not load history');
  }
}

async function loadExplain(email_id) {
  try {
    const d = await fetch(API + `/api/explain/${email_id}`).then(r => r.json());
    if (!d.top_features || !d.top_features.length) {
      showToast('SHAP data not available for this email');
      return;
    }
    const lines = d.top_features.map(f =>
      `${f.feature.replace(/_/g,' ')}: ${f.impact > 0 ? '+':''} ${f.impact.toFixed(4)}`
    ).join('\n');
    alert(`SHAP Top Features for ${email_id}:\n\n${lines}\n\nPositive = BEC signal | Negative = Legitimate signal`);
  } catch(e) {
    showToast('Could not fetch SHAP explanation');
  }
}

async function clearHistory() {
  try {
    await fetch(API + '/api/history', { method: 'DELETE' });
    loadHistory();
    showToast('🗑 History cleared');
  } catch { showToast('Error clearing history'); }
}

// ── Review Queue ──────────────────────────────────────────────────────────────
async function loadReviewQueue() {
  try {
    const data = await fetch(API + '/api/history?limit=500').then(r => r.json());
    const uncertain = data.filter(e => e.risk_score >= 0.40 && e.risk_score <= 0.60);
    const el = document.getElementById('reviewQueue');

    if (!uncertain.length) {
      el.innerHTML = '<div class="history-empty">✅ No uncertain emails in queue. All predictions are confident.</div>';
      return;
    }

    el.innerHTML = `
      <div style="margin-bottom:12px;color:var(--text3);font-size:13px">
        ${uncertain.length} email(s) with risk score between 40%–60% need analyst review:
      </div>
      ${uncertain.map(e => `
        <div class="review-item card glass" style="margin-bottom:10px;padding:14px 18px">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
            <div>
              <span class="h-badge ${e.risk_level}" style="margin-right:8px">${e.risk_level}</span>
              <span style="color:var(--text2);font-weight:600">${e.subject_preview || '(no subject)'}</span>
            </div>
            <div style="color:var(--text3);font-size:12px">${e.timestamp} · ${e.sender}</div>
          </div>
          <div style="margin-top:8px;color:var(--accent);font-size:13px">Risk: ${(e.risk_score*100).toFixed(1)}% · ID: ${e.email_id}</div>
          <div class="btn-row" style="margin-top:10px;gap:6px">
            <button class="btn btn-ghost btn-xs" style="color:#ef4444;border-color:#ef4444" onclick="quickFeedback('${e.email_id}','BEC',this)">🚨 Confirm BEC</button>
            <button class="btn btn-ghost btn-xs" style="color:#22c55e;border-color:#22c55e" onclick="quickFeedback('${e.email_id}','LEGITIMATE',this)">✅ Mark Legit</button>
            <button class="btn btn-ghost btn-xs" onclick="window.open('${API}/api/report/${e.email_id}','_blank')">📄 Report</button>
          </div>
        </div>
      `).join('')}
    `;
  } catch(e) {
    showToast('Could not load review queue');
  }
}

async function quickFeedback(emailId, label, btn) {
  try {
    btn.disabled = true;
    btn.textContent = '⏳ Saving...';
    const res = await fetch(API + '/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email_id: emailId, analyst_label: label, analyst_notes: 'Via review queue' })
    });
    const d = await res.json();
    showToast(`✅ Feedback recorded: ${label}`);
    btn.closest('.review-item').style.opacity = '0.4';
  } catch(e) {
    showToast('❌ Error: ' + e.message);
  }
}

// ── Threat Intel ──────────────────────────────────────────────────────────────
async function loadThreatIntel() {
  try {
    const d  = await fetch(API + '/api/threat-intel').then(r => r.json());
    const el = document.getElementById('threatFeedList');

    const statusColor = {
      ACTIVE: '#22c55e', SIMULATED: '#f59e0b', OFFLINE: '#ef4444'
    };

    el.innerHTML = `
      <div style="margin-bottom:16px">
        <span style="color:var(--text3)">Total IOC Count:</span>
        <span style="font-weight:700;color:var(--accent);margin-left:8px">${d.total_ioc_count?.toLocaleString()}</span>
        <span style="color:var(--text3);margin-left:16px">Last Sync: ${d.last_sync}</span>
      </div>
      ${d.feeds.map(feed => `
        <div class="card" style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:14px 18px;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
              <span style="font-weight:600;color:var(--text1)">${feed.name}</span>
              <span style="margin-left:10px;font-size:11px;padding:2px 8px;border-radius:20px;background:${statusColor[feed.status]}22;color:${statusColor[feed.status]}">${feed.status}</span>
            </div>
            <span style="color:var(--text3);font-size:12px">${feed.entries?.toLocaleString()} entries</span>
          </div>
          <div style="color:var(--text3);font-size:12px;margin-top:4px">${feed.description}</div>
          <div style="color:var(--text4,#475569);font-size:11px;margin-top:2px">Updated: ${feed.last_updated}</div>
        </div>
      `).join('')}
      <div class="card" style="background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.2);border-radius:10px;padding:14px 18px;color:#94a3b8;font-size:13px;margin-top:4px">
        ℹ️ ${d.integration_note}
      </div>
    `;
  } catch(e) {
    showToast('Could not load threat intel');
  }
}

// ── Thread Analysis ───────────────────────────────────────────────────────────
let threadCount = 0;

function initThreadPage() {
  if (document.querySelectorAll('.thread-email-slot').length === 0) {
    addThreadEmail();
    addThreadEmail();
  }
}

function addThreadEmail() {
  threadCount++;
  const container = document.getElementById('threadEmails');
  const div = document.createElement('div');
  div.className = 'thread-email-slot card';
  div.style.cssText = 'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px;margin-bottom:12px';
  div.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <span style="font-weight:600;color:var(--text2)">Email ${threadCount} <span style="color:var(--text3);font-size:12px;font-weight:400">(oldest first)</span></span>
      <button class="btn btn-ghost btn-xs" style="color:#ef4444" onclick="this.closest('.thread-email-slot').remove()">✕ Remove</button>
    </div>
    <div class="form-row" style="gap:8px">
      <div class="form-group" style="flex:2">
        <input class="input" placeholder="Subject" style="margin-bottom:6px" />
      </div>
      <div class="form-group" style="flex:1">
        <input class="input" placeholder="From (sender@domain.com)" style="margin-bottom:6px" />
      </div>
    </div>
    <textarea class="textarea" rows="4" placeholder="Email body..."></textarea>
  `;
  container.appendChild(div);
}

function clearThread() {
  document.getElementById('threadEmails').innerHTML = '';
  document.getElementById('threadResult').style.display = 'none';
  threadCount = 0;
}

function loadThreadSample() {
  clearThread();
  const slots = [
    { subject: 'Q3 Invoice Payment – Follow Up', from: 'vendor@supplierco.com',
      body: 'Hi Sarah,\n\nJust following up on the Q3 invoice we sent last week. Please let us know if you need any additional documentation.\n\nBest,\nMike' },
    { subject: 'Re: Q3 Invoice Payment – Follow Up', from: 'sarah@company.com',
      body: 'Hi Mike,\n\nThanks for following up! The invoice is being processed. Should be done by end of week.\n\nBest,\nSarah' },
    { subject: 'Re: Q3 Invoice Payment – Follow Up', from: 'vendor@supp1ierco.com',
      body: 'URGENT - Sarah please note our bank account has changed. New account for wire transfer: Chase Bank Routing 021000021 Account 9988776655. Please redirect the $142,000 payment immediately. Do not share this with anyone. CEO has approved.\n\nMike' },
  ];
  slots.forEach(s => {
    addThreadEmail();
    const slot = document.querySelectorAll('.thread-email-slot');
    const last = slot[slot.length - 1];
    last.querySelectorAll('input')[0].value = s.subject;
    last.querySelectorAll('input')[1].value = s.from;
    last.querySelector('textarea').value = s.body;
  });
  showToast('🧵 Sample thread loaded (includes thread hijacking attack in email 3)');
}

async function analyzeThread() {
  const slots = document.querySelectorAll('.thread-email-slot');
  if (slots.length < 1) { showToast('⚠️ Add at least one email to the thread'); return; }

  const btn = document.getElementById('threadAnalyzeBtn');
  btn.innerHTML = '<span class="pulse">⏳</span> Analyzing...';
  btn.disabled = true;

  const emails = Array.from(slots).map(slot => ({
    subject:   slot.querySelectorAll('input')[0]?.value || '',
    from_addr: slot.querySelectorAll('input')[1]?.value || '',
    body:      slot.querySelector('textarea')?.value || '',
    to_addr:   '', reply_to: '', urls: []
  }));

  try {
    const res = await fetch(API + '/api/analyze/thread', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emails, thread_subject: emails[0]?.subject || '' })
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    renderThreadResult(d);
  } catch(e) {
    showToast('❌ Thread analysis error: ' + e.message);
  } finally {
    btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Thread';
    btn.disabled = false;
  }
}

function renderThreadResult(d) {
  const el = document.getElementById('threadResultContent');
  const overallColor = RISK_COLORS[d.email_results[d.email_results.length-1]?.risk_level] || '#3b82f6';

  el.innerHTML = `
    <div style="display:flex;gap:20px;align-items:center;padding:0 0 16px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:16px">
      <div>
        <div style="font-size:28px;font-weight:800;color:${overallColor}">${(d.thread_risk*100).toFixed(1)}%</div>
        <div style="color:var(--text3);font-size:12px">Thread Risk Score</div>
      </div>
      <div>
        <div style="font-size:18px;font-weight:700;color:var(--text2)">${d.thread_length} Email${d.thread_length!==1?'s':''}</div>
        <div style="color:var(--text3);font-size:12px">in Thread</div>
      </div>
      ${d.thread_flags?.length ? `
        <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:8px 14px">
          <div style="font-weight:600;color:#ef4444;font-size:13px">⚠️ ${d.thread_flags.length} Thread-Level Alert${d.thread_flags.length!==1?'s':''}</div>
        </div>
      ` : ''}
    </div>
    ${d.thread_flags?.length ? `
      <div style="margin-bottom:16px">
        <div class="flags-title" style="margin-bottom:8px">🧵 Thread-Level Threats</div>
        ${d.thread_flags.map(f => `<div class="flag-item"><span class="flag-icon">🧵</span><span>${f.message}</span></div>`).join('')}
      </div>
    ` : ''}
    <div>
      <div class="flags-title" style="margin-bottom:10px">Individual Email Results</div>
      ${d.email_results.map(r => `
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:8px;border-left:3px solid ${RISK_COLORS[r.risk_level]||'#475569'}">
          <span style="font-weight:700;color:var(--text3);font-size:12px">#${r.thread_position}</span>
          <span class="h-badge ${r.risk_level}">${r.risk_level}</span>
          <span style="color:var(--text2);flex:1">${r.subject_preview || '(no subject)'}</span>
          <span style="color:${RISK_COLORS[r.risk_level]};font-weight:700">${(r.risk_score*100).toFixed(1)}%</span>
          ${r.flags?.length ? `<span style="color:var(--text3);font-size:12px">${r.flags.length} flag${r.flags.length!==1?'s':''}</span>` : ''}
        </div>
      `).join('')}
    </div>
  `;
  document.getElementById('threadResult').style.display = '';
}

// ── Keyboard shortcut: Ctrl+Enter to analyze ──────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') analyzeEmail();
});

// ══════════════════════════════════════════════════════════════════════════════
// 📡 LIVE MAILBOX STREAM (Target: sribharani101@gmail.com)
// ══════════════════════════════════════════════════════════════════════════════

let _streamScanInterval = null;
let _streamScannedCount = 0;
let _lastStreamEmail = null;

function initStreamPage() {
  const mail = document.getElementById('cfg-email')?.value || 'sribharani101@gmail.com';
  document.getElementById('streamTargetMail').textContent = mail;
}

function toggleMailboxSettings() {
  const p = document.getElementById('mailboxSettingsPanel');
  p.style.display = (p.style.display === 'none' || !p.style.display) ? 'block' : 'none';
}

async function saveMailboxConfig() {
  const email = document.getElementById('cfg-email').value.trim() || 'sribharani101@gmail.com';
  const app_password = document.getElementById('cfg-app-pass').value.trim();
  const statusEl = document.getElementById('imapConnStatus');

  statusEl.textContent = '⏳ Testing connection...';
  try {
    const res = await fetch(API + '/api/mailbox/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, app_password })
    });
    const d = await res.json();
    document.getElementById('streamTargetMail').textContent = email;
    statusEl.textContent = d.status === 'connected' ? '🟢 Live Gmail IMAP Connected' : '🟡 Live Stream Active (' + email + ')';
    showToast(d.message);
  } catch (err) {
    statusEl.textContent = '⚠️ Config saved locally';
    showToast('Saved mailbox target: ' + email);
  }
}

function toggleStreamAutoScan() {
  const btn = document.getElementById('streamToggleBtn');
  const icon = document.getElementById('streamToggleIcon');
  const text = document.getElementById('streamToggleText');

  if (_streamScanInterval) {
    clearInterval(_streamScanInterval);
    _streamScanInterval = null;
    icon.textContent = '▶';
    text.textContent = 'Start Live Auto-Scan';
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-primary');
    showToast('⏸️ Live Mailbox scan paused');
    return;
  }

  icon.textContent = '⏸';
  text.textContent = 'Pause Auto-Scan';
  btn.classList.remove('btn-primary');
  btn.classList.add('btn-danger');
  showToast('▶️ Real-time scan started for ' + (document.getElementById('streamTargetMail').textContent || 'sribharani101@gmail.com'));

  fetchNextLiveEmail();
  _streamScanInterval = setInterval(fetchNextLiveEmail, 4000);
}

async function fetchNextLiveEmail() {
  try {
    const res = await fetch(API + '/api/mailbox/stream-next', { method: 'POST' });
    if (!res.ok) throw new Error('Mailbox fetch failed');
    const data = await res.json();
    if (data.email) {
      renderLiveStreamEmail(data.email);
    }
  } catch (err) {
    console.error('Stream error:', err);
  }
}

async function scan100MailboxFolder() {
  const btn = document.getElementById('scanFolderBtn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="pulse">⏳</span> Scanning 120+ Mails...';
  }
  showToast('📂 Loading 120 .eml files from data/raw/emails...');

  try {
    const res = await fetch(API + '/api/mailbox/scan-folder', { method: 'POST' });
    if (!res.ok) throw new Error('Folder scan failed');
    const data = await res.json();

    showToast(`⚡ Ingested ${data.total_scanned} emails! Streaming results dynamically...`);

    // Dynamically pump each email into the stream ticker one after another
    let idx = 0;
    const interval = setInterval(() => {
      if (idx >= data.results.length) {
        clearInterval(interval);
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '📂 Scan 120+ Mailbox Folder';
        }
        showToast(`✅ Finished! ${data.total_scanned} Scanned: ${data.bec_detected} BEC, ${data.legitimate} Legitimate`);
        return;
      }
      const item = data.results[idx];
      renderLiveStreamEmail(item);
      idx++;
    }, 120);

  } catch (err) {
    showToast('❌ Error scanning folder: ' + err.message);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '📂 Scan 120+ Mailbox Folder';
    }
  }
}

function renderLiveStreamEmail(item) {
  _lastStreamEmail = item;
  _streamScannedCount++;
  document.getElementById('streamCount').textContent = _streamScannedCount;

  // 1. Update Left Card (Current Focus)
  document.getElementById('activeEmailSubject').textContent = item.subject_preview || '(No Subject)';
  document.getElementById('activeEmailSender').textContent = `From: ${item.sender || 'Unknown'} · Scanned: ${item.timestamp}`;

  const badge = document.getElementById('activeEmailBadge');
  badge.className = `risk-badge ${item.risk_level.toLowerCase()}`;
  badge.textContent = `${item.risk_level} · ${(item.risk_score * 100).toFixed(0)}%`;

  document.getElementById('liveEmailBodyWrap').textContent = `Subject: ${item.subject_preview}\nFrom: ${item.sender}\nTo: ${item.to_addr || 'sribharani101@gmail.com'}\nStatus: Flagged as ${item.classification} (${item.bec_type || 'General'})\nRecommended Action: ${item.recommended_action}\nConfidence: ${(item.confidence * 100).toFixed(1)}%`;

  // Model Bars
  const barsContainer = document.getElementById('liveModelBars');
  const scores = item.model_scores || {};
  const modelLabels = {
    xgb: 'XGBoost',
    cb: 'CatBoost',
    gb: 'Gradient Boosting',
    lr: 'Logistic Regression',
    svm: 'Support Vector Machine',
    iso: 'Isolation Forest (Anomaly)'
  };

  let barsHtml = '';
  for (const [k, name] of Object.entries(modelLabels)) {
    const val = scores[k] !== undefined ? scores[k] : 0.5;
    const pct = (val * 100).toFixed(0);
    const color = val >= 0.75 ? '#ef4444' : (val >= 0.55 ? '#f97316' : (val >= 0.30 ? '#eab308' : '#10b981'));
    barsHtml += `
      <div style="font-size:11px; display:flex; justify-content:space-between; margin-bottom:2px;">
        <span style="color:var(--text2)">${name}</span>
        <span style="font-weight:700; color:${color}">${pct}%</span>
      </div>
      <div style="height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden; margin-bottom:6px;">
        <div style="width:${pct}%; height:100%; background:${color}; transition: width 0.5s ease;"></div>
      </div>
    `;
  }
  barsContainer.innerHTML = barsHtml;

  // Flags
  const flagsWrap = document.getElementById('liveFlagsSection');
  const flagsList = document.getElementById('liveFlagsList');
  if (item.flags && item.flags.length > 0) {
    flagsWrap.style.display = 'block';
    flagsList.innerHTML = item.flags.map(f => `
      <div class="flag-item" style="padding:6px 10px; margin-bottom:4px; font-size:12px;">
        <span class="flag-icon">${f.type === 'domain' ? '🌐' : (f.type === 'social_eng' ? '🎭' : '⚠️')}</span>
        <span>${f.message}</span>
      </div>
    `).join('');
  } else {
    flagsWrap.style.display = 'none';
  }

  // 2. Prepend to Right Column (Ticker Stream)
  const ticker = document.getElementById('streamTickerList');
  const emptyState = ticker.querySelector('.text-muted');
  if (emptyState) emptyState.remove();

  const borderColor = RISK_COLORS[item.risk_level] || '#38bdf8';
  const card = document.createElement('div');
  card.style.cssText = `
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-left: 4px solid ${borderColor};
    border-radius: 8px;
    padding: 12px 14px;
    animation: fadeIn 0.4s ease;
    cursor: pointer;
    transition: transform 0.2s, background 0.2s;
  `;
  card.onmouseover = () => { card.style.background = 'rgba(255, 255, 255, 0.06)'; };
  card.onmouseout  = () => { card.style.background = 'rgba(255, 255, 255, 0.03)'; };
  card.onclick = () => { renderLiveStreamEmail(item); };

  card.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px;">
      <span style="font-weight:700; color:var(--text1); font-size:13px; max-width:70%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
        ${item.subject_preview || '(No Subject)'}
      </span>
      <span class="h-badge ${item.risk_level}" style="font-size:10px; padding:2px 8px;">${item.risk_level} (${(item.risk_score*100).toFixed(0)}%)</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text3);">
      <span>From: ${item.sender || 'Unknown'}</span>
      <span>${item.timestamp.split(' ')[1] || item.timestamp}</span>
    </div>
    ${item.bec_type && item.bec_type !== 'N/A' ? `
      <div style="margin-top:6px; font-size:11px; color:#f87171; display:flex; align-items:center; gap:4px;">
        <span>🎯 Category:</span> <strong>${item.bec_type}</strong>
      </div>
    ` : ''}
  `;

  ticker.insertBefore(card, ticker.firstChild);
  if (ticker.children.length > 30) {
    ticker.removeChild(ticker.lastChild);
  }

  showToast(`📥 Analyzed email: "${item.subject_preview}" [${item.risk_level}]`);
}

function clearStreamFeed() {
  const ticker = document.getElementById('streamTickerList');
  ticker.innerHTML = `
    <div style="text-align:center; padding:40px 20px; color:var(--text-muted); font-size:13px;">
      <div style="font-size:32px; margin-bottom:8px;">🧹</div>
      Feed cleared. Waiting for next live emails!
    </div>
  `;
  _streamScannedCount = 0;
  document.getElementById('streamCount').textContent = '0';
  showToast('Stream feed cleared');
}


// ══════════════════════════════════════════════════════════════════════════════
// ⚖️ MODEL COMPARISON SECTION
// ══════════════════════════════════════════════════════════════════════════════

let _comparisonChart = null;

async function loadModelComparison() {
  try {
    const res = await fetch(API + '/api/models/comparison');
    const d = await res.json();

    // 1. Render Matrix Cards
    const matrixGrid = document.getElementById('modelMatrixGrid');
    matrixGrid.innerHTML = d.models.map(m => `
      <div class="card glass" style="border-top: 3px solid #38bdf8; padding:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <h4 style="margin:0; font-size:15px; font-weight:700; color:var(--text1);">${m.name}</h4>
          <span style="font-size:11px; padding:2px 8px; border-radius:12px; background:rgba(56,189,248,0.15); color:#38bdf8; font-weight:600;">${m.type}</span>
        </div>
        <div style="font-size:12px; color:var(--text3); margin-bottom:12px;">Architecture: <strong style="color:var(--text2)">${m.family}</strong></div>
        
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; background:rgba(0,0,0,0.2); padding:10px; border-radius:6px; margin-bottom:12px; font-size:12px;">
          <div>
            <div style="color:var(--text3); font-size:11px;">Ensemble Weight</div>
            <div style="font-weight:700; font-size:14px; color:#38bdf8;">${(m.weight * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div style="color:var(--text3); font-size:11px;">F1 Benchmark</div>
            <div style="font-weight:700; font-size:14px; color:#10b981;">${(m.f1_score * 100).toFixed(1)}%</div>
          </div>
        </div>

        <div style="font-size:11px; color:var(--text3); line-height:1.4;">
          <strong style="color:var(--text2)">Detection Focus:</strong> ${m.detection_focus}
        </div>
        <div style="margin-top:8px; font-size:11px; color:var(--text3);">
          ⚡ Latency: <strong style="color:#10b981">${m.latency}</strong>
        </div>
      </div>
    `).join('');

    // 2. Render Comparison Table
    const tbody = document.getElementById('modelComparisonTableBody');
    tbody.innerHTML = d.models.map(m => `
      <tr>
        <td><strong>${m.name}</strong></td>
        <td><span style="color:var(--text2)">${m.family}</span></td>
        <td><span style="color:#38bdf8; font-weight:700">${(m.weight * 100).toFixed(1)}%</span></td>
        <td><span style="color:#10b981; font-weight:700">${(m.f1_score * 100).toFixed(1)}%</span></td>
        <td style="font-size:12px; color:var(--text2)">${m.detection_focus}</td>
        <td><span style="color:var(--text3); font-size:11px">${m.latency}</span></td>
      </tr>
    `).join('');

    // 3. Render Chart
    renderModelComparisonChart(d.models);

  } catch (err) {
    console.error('Model comparison load failed:', err);
  }
}

function renderModelComparisonChart(modelsList) {
  const ctx = document.getElementById('liveModelComparisonChart')?.getContext('2d');
  if (!ctx) return;

  const email = _lastStreamEmail;
  const labels = modelsList.map(m => m.name);
  const scores = modelsList.map(m => {
    if (email && email.model_scores && email.model_scores[m.id] !== undefined) {
      return (email.model_scores[m.id] * 100).toFixed(1);
    }
    return (m.weight * 400).toFixed(1); // Baseline default
  });

  const colors = scores.map(s => s >= 75 ? '#ef4444' : (s >= 55 ? '#f97316' : (s >= 30 ? '#eab308' : '#10b981')));

  if (_comparisonChart) {
    _comparisonChart.destroy();
  }

  _comparisonChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: email ? `Model Risk Verdict for: "${email.subject_preview?.substring(0, 24)}..."` : 'Model Baseline Detection Sensitivity (%)',
        data: scores,
        backgroundColor: colors,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } }
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: '#64748b', callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        x: {
          ticks: { color: '#94a3b8' },
          grid: { display: false }
        }
      }
    }
  });

  // Calculate consensus
  if (email && email.model_scores) {
    const vals = Object.values(email.model_scores);
    const becVotes = vals.filter(v => v >= 0.5).length;
    const rate = Math.round((Math.max(becVotes, vals.length - becVotes) / vals.length) * 100);
    const consensusEl = document.getElementById('ensembleConsensusVal');
    const descEl = document.getElementById('ensembleConsensusDesc');
    if (consensusEl) {
      consensusEl.textContent = rate + '%';
      consensusEl.style.color = rate >= 80 ? '#10b981' : '#f59e0b';
    }
    if (descEl) {
      descEl.textContent = `${becVotes} of ${vals.length} models voted BEC (${rate}% agreement)`;
    }
  }
}
