const form = document.getElementById('stream-form');
const strategySelect = document.getElementById('strategy');
const beamWidthField = document.getElementById('beam-width-field');
const beamThrottleField = document.getElementById('beam-throttle-field');
const includeAttentionCheckbox = document.getElementById('includeAttention');
const connectionStatus = document.getElementById('connection-status');
const greedyLog = document.getElementById('greedy-log');
const beamTable = document.getElementById('beam-table');
const beamHistory = document.getElementById('beam-history');
const finalTranslation = document.getElementById('final-translation');
const attentionDump = document.getElementById('attention-dump');
const modePill = document.getElementById('mode-pill');
const attentionPill = document.getElementById('attention-pill');
const throttlePill = document.getElementById('throttle-pill');
const stopButton = document.getElementById('stop-button');
const apiBaseInput = document.getElementById('apiBase');
const beamChart = document.getElementById('beam-chart');
const beamChartEmpty = document.getElementById('beam-chart-empty');
const beamLegend = document.getElementById('beam-legend');

let eventSource = null;
let currentMode = 'greedy';
const beamRows = new Map();
const beamSeries = new Map();
let beamEventIndex = 0;
let currentSnapshotIndex = 0;
const MAX_SERIES_POINTS = 60;
const BEAM_COLORS = ['#2563eb', '#0ea5e9', '#16a34a', '#f97316', '#9333ea', '#db2777'];

function resetOutputs() {
  greedyLog.innerHTML = '';
  beamTable.innerHTML = '';
  beamHistory.innerHTML = '';
  finalTranslation.textContent = '';
  attentionDump.textContent = '';
  beamRows.clear();
  resetBeamVisuals();
}

function setStatus(text, state) {
  connectionStatus.textContent = text;
  connectionStatus.className = `status-badge ${state}`;
}

function setModePill(strategy) {
  modePill.textContent = strategy === 'beam' ? 'Beam search' : 'Greedy';
}

function setAttentionPill(enabled, captured) {
  if (captured) {
    attentionPill.textContent = 'Attention: captured';
  } else {
    attentionPill.textContent = enabled ? 'Attention: requested' : 'Attention: off';
  }
}

function setThrottlePill(value) {
  throttlePill.textContent = value > 0 ? `Throttle: ${value}ms` : 'Throttle: off';
}

function appendLog(container, message) {
  const entry = document.createElement('p');
  entry.className = 'log-entry';
  entry.textContent = message;
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
}

function renderBeamTable() {
  const sorted = Array.from(beamRows.values()).sort((a, b) => a.rank - b.rank);
  beamTable.innerHTML = '';
  sorted.forEach((beam) => {
    const row = document.createElement('tr');
    const rankCell = document.createElement('td');
    rankCell.textContent = `#${beam.rank}`;
    row.appendChild(rankCell);

    const scoreCell = document.createElement('td');
    scoreCell.textContent = beam.score.toFixed(3);
    row.appendChild(scoreCell);

    const tokensCell = document.createElement('td');
    tokensCell.textContent = beam.partialTranslation || beam.tokens.join(', ');
    row.appendChild(tokensCell);

    beamTable.appendChild(row);
  });
}

function resetBeamVisuals() {
  beamSeries.clear();
  beamEventIndex = 0;
  currentSnapshotIndex = 0;
  renderBeamChart();
  renderBeamLegend();
}

function renderBeamChart() {
  if (!beamChart) {
    return;
  }

  const rect = beamChart.getBoundingClientRect();
  const width = Math.max(200, Math.floor(rect.width));
  const height = Math.max(160, Math.floor(rect.height));
  beamChart.width = width;
  beamChart.height = height;
  const ctx = beamChart.getContext('2d');
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, width, height);

  const seriesEntries = Array.from(beamSeries.entries()).filter(([, points]) => points.length > 0);
  if (seriesEntries.length === 0) {
    if (beamChartEmpty) {
      beamChartEmpty.style.display = 'flex';
    }
    return;
  }
  if (beamChartEmpty) {
    beamChartEmpty.style.display = 'none';
  }

  const padding = 24;
  const xValues = seriesEntries.flatMap(([, points]) => points.map((point) => point.index));
  const yValues = seriesEntries.flatMap(([, points]) => points.map((point) => point.score));
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  let minY = Math.min(...yValues);
  let maxY = Math.max(...yValues);
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  } else {
    const pad = (maxY - minY) * 0.1;
    minY -= pad;
    maxY += pad;
  }
  const xRange = Math.max(1, maxX - minX);
  const yRange = Math.max(1e-6, maxY - minY);

  ctx.strokeStyle = '#cbd5f5';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, height - padding);
  ctx.stroke();

  seriesEntries.forEach(([rank, points], seriesIndex) => {
    const color = BEAM_COLORS[seriesIndex % BEAM_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, idx) => {
      const x =
        padding + ((point.index - minX) / xRange) * Math.max(1, width - padding * 2);
      const y =
        height - padding - ((point.score - minY) / yRange) * Math.max(1, height - padding * 2);
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });
}

function renderBeamLegend() {
  if (!beamLegend) {
    return;
  }
  beamLegend.innerHTML = '';
  const entries = Array.from(beamSeries.entries()).filter(([, points]) => points.length > 0);
  entries.forEach(([rank], idx) => {
    const item = document.createElement('span');
    item.className = 'legend-item';
    const dot = document.createElement('span');
    dot.className = 'legend-dot';
    dot.style.background = BEAM_COLORS[idx % BEAM_COLORS.length];
    item.appendChild(dot);
    item.appendChild(document.createTextNode(`Rank ${rank}`));
    beamLegend.appendChild(item);
  });
}

function handleTokenEvent(data) {
  appendLog(
    greedyLog,
    `step ${data.step + 1}: ${data.token} (id ${data.tokenId}) → ${data.partialTranslation}`,
  );
}

function handleBeamEvent(data) {
  beamRows.set(data.rank, data);
  if (data.rank === 1) {
    beamEventIndex += 1;
    currentSnapshotIndex = beamEventIndex;
  }
  const snapshotIndex = currentSnapshotIndex || beamEventIndex || 0;
  const points = beamSeries.get(data.rank) ?? [];
  points.push({ index: snapshotIndex, score: data.score });
  while (points.length > MAX_SERIES_POINTS) {
    points.shift();
  }
  beamSeries.set(data.rank, points);
  appendLog(
    beamHistory,
    `rank ${data.rank}: ${data.partialTranslation} (score ${data.score.toFixed(3)})`,
  );
  renderBeamTable();
  renderBeamChart();
  renderBeamLegend();
}

function handleCompletedEvent(data, includeAttention) {
  finalTranslation.textContent = data.translation || '(empty translation)';

  if (includeAttention && data.attention) {
    attentionDump.textContent = JSON.stringify(data.attention, null, 2);
    setAttentionPill(false, true);
  } else if (includeAttention) {
    attentionDump.textContent = 'Attention snapshots were requested but not returned by the server.';
  } else {
    attentionDump.textContent = '';
  }

  setStatus('Completed', 'offline');
  setFormEnabled(true);
}

function buildUrl(params) {
  const base = apiBaseInput.value.trim() || window.location.origin;
  const url = new URL('/transformer/translate/stream', base);
  const search = new URLSearchParams();
  search.set('text', params.text);
  if (params.maxLength) {
    search.set('maxLength', String(params.maxLength));
  }
  if (params.strategy && params.strategy !== 'greedy') {
    search.set('strategy', params.strategy);
  }
  if (params.strategy === 'beam') {
    search.set('beamWidth', String(params.beamWidth || 4));
    if (params.beamThrottleMs && params.beamThrottleMs > 0) {
      search.set('beamThrottleMs', String(params.beamThrottleMs));
    }
  }
  if (params.includeAttention) {
    search.set('includeAttention', 'true');
  }
  url.search = search.toString();
  return url.toString();
}

function setFormEnabled(enabled) {
  Array.from(form.elements).forEach((el) => {
    el.disabled = !enabled && el !== stopButton;
  });
  stopButton.disabled = enabled;
}

function stopStream(reason = 'Stopped by user') {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  setStatus(reason, 'offline');
  setFormEnabled(true);
}

strategySelect.addEventListener('change', () => {
  const isBeam = strategySelect.value === 'beam';
  beamWidthField.style.display = isBeam ? 'flex' : 'none';
  beamThrottleField.style.display = isBeam ? 'flex' : 'none';
  currentMode = isBeam ? 'beam' : 'greedy';
  setModePill(currentMode);
});

stopButton.addEventListener('click', () => {
  stopStream();
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!form.text.value.trim()) {
    form.text.focus();
    return;
  }

  if (eventSource) {
    stopStream('Restarting');
  }

  resetOutputs();
  currentMode = strategySelect.value;
  setModePill(currentMode);
  const includeAttention = includeAttentionCheckbox.checked;
  setAttentionPill(includeAttention, false);
  const throttleValue = Number(form.beamThrottleMs?.value) || 0;
  setThrottlePill(throttleValue);

  const payload = {
    text: form.text.value.trim(),
    maxLength: Number(form.maxLength.value) || undefined,
    strategy: strategySelect.value,
    beamWidth: Number(form.beamWidth?.value) || undefined,
    includeAttention,
    beamThrottleMs: throttleValue > 0 ? throttleValue : undefined,
  };

  const endpoint = buildUrl(payload);
  eventSource = new EventSource(endpoint);
  setStatus('Connecting…', 'offline');
  setFormEnabled(false);

  eventSource.onopen = () => {
    setStatus('Streaming', 'online');
  };

  eventSource.onerror = () => {
    appendLog(beamHistory, 'Stream error — check server logs.');
    stopStream('Error');
  };

  eventSource.onmessage = (evt) => {
    if (!evt.data) {
      return;
    }
    try {
      const data = JSON.parse(evt.data);
      if (!data?.type) {
        return;
      }
      if (data.type === 'token') {
        handleTokenEvent(data);
      } else if (data.type === 'beam') {
        handleBeamEvent(data);
      } else if (data.type === 'completed') {
        handleCompletedEvent(data, includeAttention);
      }
    } catch (err) {
      appendLog(beamHistory, `Failed to parse event: ${err}`);
    }
  };
});

setModePill(strategySelect.value);
beamWidthField.style.display = strategySelect.value === 'beam' ? 'flex' : 'none';
beamThrottleField.style.display = strategySelect.value === 'beam' ? 'flex' : 'none';
setAttentionPill(false, false);
setThrottlePill(0);
setStatus('Idle', 'offline');
setFormEnabled(true);
renderBeamChart();
renderBeamLegend();
window.addEventListener('resize', () => {
  renderBeamChart();
});
