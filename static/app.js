const form = document.getElementById('predictForm');
const hoursInput = document.getElementById('hoursInput');
const chartImg = document.getElementById('chartImg');
const coneImg = document.getElementById('coneImg');
const tableBody = document.querySelector('#resultsTable tbody');
const errorBox = document.getElementById('errorBox');
const downloadBtn = document.getElementById('downloadBtn');
const explainerEl = document.getElementById('gruExplainer');
const missionBriefEl = document.getElementById('missionBrief');
const missionRecEl = document.getElementById('missionRecommendation');

let lastPredictions = [];

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('d-none');
}

function clearError() {
  errorBox.textContent = '';
  errorBox.classList.add('d-none');
}

function renderTable(preds) {
  tableBody.innerHTML = '';
  preds.forEach((v, i) => {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = i + 1; // hour +h
    const td = document.createElement('td');
    td.textContent = Number(v).toFixed(2);
    tr.appendChild(th);
    tr.appendChild(td);
    tableBody.appendChild(tr);
  });
}

function enableDownload(preds) {
  downloadBtn.disabled = preds.length === 0;
  if (preds.length === 0) return;

  const rows = ['hour,prediction'];
  preds.forEach((v, i) => rows.push(`${i + 1},${Number(v).toFixed(2)}`));
  const csvContent = rows.join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  downloadBtn.onclick = () => {
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions_pm25.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
}

function renderExplainer(preds) {
  if (!explainerEl) return;
  if (!preds || preds.length === 0) {
    explainerEl.innerHTML = `
      <p class="mb-2">
        Imagina a <strong>Gru</strong> coordinando a los minions: nuestra
        <strong>GRU</strong> (Gated Recurrent Unit) "recuerda" las últimas horas
        y planifica las siguientes para anticipar la concentración de PM2.5.
      </p>
      <p class="text-muted mb-0">
        Tras realizar una predicción, aquí verás una explicación sencilla con
        flechas de tendencia (↗️/↘️/→), el promedio estimado y recomendaciones
        rápidas según el umbral de 15 µg/m³.
      </p>`;
    return;
  }

  const n = preds.length;
  const start = Number(preds[0]);
  const end = Number(preds[n - 1]);
  const diff = end - start;
  const avg = preds.reduce((a, b) => a + Number(b), 0) / n;
  const who24h = 15; // µg/m³

  let arrow = '→';
  let trendText = 'estable';
  if (diff > 1.0) { arrow = '↗️'; trendText = 'subiendo'; }
  else if (diff < -1.0) { arrow = '↘️'; trendText = 'bajando'; }

  let riskText = 'condiciones favorables';
  let advice = 'Mantén actividades normales; monitorea la tendencia.';
  if (avg >= 35) {
    riskText = 'calidad del aire mala';
    advice = 'Reduce actividad al aire libre, considera mascarilla y ventilación controlada.';
  } else if (avg >= who24h) {
    riskText = 'calidad del aire moderada';
    advice = 'Modera actividad prolongada al aire libre y ventila con precaución.';
  }

  explainerEl.innerHTML = `
    <p class="mb-2">
      Como <strong>Gru</strong> organizando al equipo, nuestra <strong>GRU</strong>
      mira las últimas horas y coordina las próximas <strong>${n}</strong>.
      Hoy la predicción termina <strong>${trendText}</strong> ${arrow} (de ${start.toFixed(1)} a ${end.toFixed(1)} µg/m³).
    </p>
    <p class="mb-2">
      Promedio estimado: <strong>${avg.toFixed(1)} µg/m³</strong>. Umbral recomendado (OMS, 24h):
      <strong>${who24h} µg/m³</strong>.
    </p>
    <p class="mb-0">
      Interpretación rápida: <em>${riskText}</em>. Recomendación: ${advice}
    </p>`;
}

function renderMissionBrief(preds) {
  if (!missionRecEl) return;
  if (!preds || preds.length === 0) {
    missionRecEl.className = 'alert alert-info mb-0';
    missionRecEl.textContent = 'Aún no hay predicción. Realiza una para obtener el plan de misión.';
    return;
  }

  // Usamos el valor máximo previsto para una decisión conservadora
  const maxPred = Math.max(...preds.map(Number));
  const who24h = 15; // µg/m³

  let level = 'bueno';
  let ship = 'Nave Ligera';
  let gear = 'sin filtros especiales';
  let icon = '🟢';
  let note = 'Condiciones favorables para exploración estándar.';

  if (maxPred >= 35) {
    level = 'malo';
    ship = 'Nave Hermética';
    gear = 'filtros HEPA + respiradores';
    icon = '🔴';
    note = 'Limita exposición exterior; operaciones desde cabina sellada.';
  } else if (maxPred >= who24h) {
    level = 'moderado';
    ship = 'Nave con Filtros';
    gear = 'filtros estándar y mascarillas';
    icon = '🟠';
    note = 'Evita actividad prolongada al aire libre; ventila con cautela.';
  }

  missionRecEl.className = 'alert alert-secondary mb-0';
  missionRecEl.innerHTML = `
    <strong>${icon} Calidad del aire: ${level}</strong><br/>
    Pico previsto: <strong>${maxPred.toFixed(1)} µg/m³</strong>.
    Recomendación: <strong>${ship}</strong> con <strong>${gear}</strong>.
    <br/>
    ${note}
  `;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearError();

  const hours = parseInt(hoursInput.value, 10);
  if (Number.isNaN(hours) || hours < 1 || hours > 24) {
    showError('Ingrese un número de horas entre 1 y 24.');
    return;
  }

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hours })
    });
    const data = await resp.json();
    if (!resp.ok) {
      showError(data.error || 'Error al procesar la predicción.');
      return;
    }

    lastPredictions = data.predictions || [];
    chartImg.src = `data:image/png;base64,${data.chart_base64}`;
    if (data.cone_chart_base64) {
      coneImg.src = `data:image/png;base64,${data.cone_chart_base64}`;
    }
    renderTable(lastPredictions);
    enableDownload(lastPredictions);
    renderExplainer(lastPredictions);
    renderMissionBrief(lastPredictions);
  } catch (err) {
    showError('No se pudo conectar con el servidor.');
  }
});