function buildGrid(total) {
  const grid = document.getElementById('grid');
  grid.style.setProperty('--total', total);

  for (let i = 0; i < total; i++) {
    const card = document.createElement('div');
    card.className = 'cam-card';
    card.id = 'card' + i;
    card.innerHTML = `
      <div class="box">
        <img id="feed${i}" alt="Cámara ${i + 1} apagada" />
        <div class="off-overlay" id="off${i}">CÁMARA APAGADA</div>
      </div>
      <div class="row">
        <span class="cam-title">Cámara ${i + 1}</span>
        <small id="info${i}"></small>
      </div>
      <div class="row controls">
        <label class="toggle">
          <input type="checkbox" id="cam${i}" onchange="toggleCam(${i}, this.checked)">
          <span class="slider"></span>
          <span class="toggle-label">Activa</span>
        </label>
        <label class="toggle rec">
          <input type="checkbox" id="rec${i}" onchange="toggleRec(${i}, this.checked)">
          <span class="slider"></span>
          <span class="toggle-label">Grabar</span>
        </label>
      </div>
    `;
    grid.appendChild(card);
  }
}

function setFeed(i, on) {
  const img = document.getElementById('feed' + i);
  const off = document.getElementById('off' + i);
  const rec = document.getElementById('rec' + i);
  if (on) {
    img.src = `/feed/${i}?_=${Date.now()}`;
    off.style.display = 'none';
    rec.disabled = false;
  } else {
    img.removeAttribute('src');
    off.style.display = 'flex';
    // sin cámara prendida no hay nada que grabar
    rec.checked = false;
    rec.disabled = true;
  }
}

function toggleCam(i, on) {
  fetch(`/cam/${i}/${on ? 'enable' : 'disable'}`, { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (!d.ok) {
        document.getElementById('cam' + i).checked = !on;
        return;
      }
      setFeed(i, on);
      if (!on) {
        // apagar la cámara corta cualquier grabación en curso (ver disable_camera)
        fetch(`/record/${i}/stop`, { method: 'POST' });
      }
    });
}

function toggleRec(i, on) {
  fetch(`/record/${i}/${on ? 'start' : 'stop'}`, { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (!d.ok) {
        document.getElementById('rec' + i).checked = !on;
      }
    });
}

function pollStatus() {
  fetch('/status').then(r => r.json()).then(d => {
    d.cams.forEach(c => {
      const info = document.getElementById('info' + c.idx);
      if (info) info.textContent = `${c.res}` + (c.recording ? ' · grabando' : '');

      const camToggle = document.getElementById('cam' + c.idx);
      if (camToggle && camToggle.checked !== c.enabled) {
        camToggle.checked = c.enabled;
        setFeed(c.idx, c.enabled);
      }

      const recToggle = document.getElementById('rec' + c.idx);
      if (recToggle) {
        recToggle.disabled = !c.enabled;
        if (recToggle.checked !== c.recording) {
          recToggle.checked = c.recording;
        }
      }
    });
  }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  const total = window.TOTAL_CAMS || 0;
  buildGrid(total);
  // el servidor arranca todas las cámaras activas por defecto
  for (let i = 0; i < total; i++) {
    document.getElementById('cam' + i).checked = true;
    setFeed(i, true);
  }
  pollStatus();
  setInterval(pollStatus, 2000);
});
