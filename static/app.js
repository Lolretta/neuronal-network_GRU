const form = document.getElementById('predictForm');
const hoursInput = document.getElementById('hoursInput');
const chartImg = document.getElementById('chartImg');
const tableBody = document.querySelector('#resultsTable tbody');
const errorBox = document.getElementById('errorBox');
const downloadBtn = document.getElementById('downloadBtn');

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
    renderTable(lastPredictions);
    enableDownload(lastPredictions);
  } catch (err) {
    showError('No se pudo conectar con el servidor.');
  }
});