/**
 * Scroll-driven foveated sampling playback.
 * Sequence: source + fixation → global + flat views → repeat.
 */

const DEFAULT_RUN = 'runs/seoul';

const state = {
  manifest: null,
  runBase: '',
  sourceImage: null,
  beats: [],
};

function getQueryRun() {
  const params = new URLSearchParams(window.location.search);
  return params.get('run') || params.get('manifest')?.replace(/\/manifest\.json$/, '') || null;
}

async function loadManifest(runBase) {
  const base = runBase.replace(/\/$/, '');
  const res = await fetch(`${base}/manifest.json`);
  if (!res.ok) throw new Error(`Could not load ${base}/manifest.json`);
  return { manifest: await res.json(), runBase: base };
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

function computeReveal(element) {
  const rect = element.getBoundingClientRect();
  const vh = window.innerHeight;
  const start = vh * 0.92;
  const end = vh * 0.28;
  const raw = (start - rect.top) / (start - end);
  return smoothstep(clamp(raw, 0, 1));
}

function updateScrollReveal() {
  for (const beat of state.beats) {
    beat.style.setProperty('--reveal', computeReveal(beat).toFixed(4));
  }
  const outro = document.getElementById('outro');
  if (outro) outro.style.setProperty('--reveal', computeReveal(outro).toFixed(4));
}

let scrollScheduled = false;
function onScroll() {
  if (scrollScheduled) return;
  scrollScheduled = true;
  requestAnimationFrame(() => {
    updateScrollReveal();
    scrollScheduled = false;
  });
}

function drawSourceCanvas(canvas, img, manifest, fixationIndex) {
  const [height, width] = manifest.image_size;
  const maxW = Math.min(720, window.innerWidth * 0.92);
  const scale = maxW / width;
  const cw = Math.round(width * scale);
  const ch = Math.round(height * scale);

  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, cw, ch);

  const fix = manifest.fixations.find((f) => f.index === fixationIndex)
    ?? manifest.fixations[fixationIndex - 1];
  if (!fix || fix.row == null || fix.col == null) return;

  const crossSize = Math.max(cw, ch) * 0.028;
  ctx.strokeStyle = 'rgba(196, 165, 116, 0.92)';
  ctx.lineWidth = Math.max(1.5, crossSize * 0.07);
  ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
  ctx.shadowBlur = 6;

  const x = fix.col * cw;
  const y = fix.row * ch;
  ctx.beginPath();
  ctx.moveTo(x - crossSize, y);
  ctx.lineTo(x + crossSize, y);
  ctx.moveTo(x, y - crossSize);
  ctx.lineTo(x, y + crossSize);
  ctx.stroke();
}

function createFixationSet(fix, runBase) {
  const section = document.createElement('section');
  section.className = 'beat fixation-set';
  section.dataset.fixationIndex = String(fix.index);

  const inner = document.createElement('div');
  inner.className = 'fixation-set-inner';

  const source = document.createElement('div');
  source.className = 'fixation-source';

  const label = document.createElement('p');
  label.className = 'beat-index';
  label.textContent = `Fixation ${fix.index}`;

  const frame = document.createElement('div');
  frame.className = 'source-frame';

  const canvas = document.createElement('canvas');
  frame.appendChild(canvas);
  state._canvases = state._canvases || [];
  state._canvases.push({ canvas, fixationIndex: fix.index });

  source.appendChild(label);
  source.appendChild(frame);

  const views = document.createElement('div');
  views.className = 'fixation-views';

  const pair = document.createElement('div');
  pair.className = 'views-pair';

  const [imgH, imgW] = state.manifest.image_size || [1, 1];
  pair.appendChild(createViewCell(
    'view-frame--global',
    'Foveated sampling',
    `${runBase}/${fix.views.global_cartesian}`,
    `${imgW} / ${imgH}`,
  ));
  pair.appendChild(createViewCell(
    'view-frame--flat',
    'Cortical magnification',
    `${runBase}/${fix.views.flat}`,
  ));
  views.appendChild(pair);

  inner.appendChild(source);
  inner.appendChild(views);
  section.appendChild(inner);
  return section;
}

function createViewCell(className, subtitle, imgSrc, aspectRatio = null) {
  const cell = document.createElement('div');
  cell.className = 'view-cell';

  const caption = document.createElement('p');
  caption.className = 'view-subtitle';
  caption.textContent = subtitle;

  const frame = document.createElement('div');
  frame.className = `view-frame ${className}`;
  if (aspectRatio) {
    frame.style.aspectRatio = aspectRatio;
  }

  const img = document.createElement('img');
  img.src = imgSrc;
  img.alt = '';
  img.loading = 'lazy';
  frame.appendChild(img);

  cell.appendChild(caption);
  cell.appendChild(frame);
  return cell;
}

function buildSequence(manifest, runBase) {
  const container = document.getElementById('sequence');
  container.innerHTML = '';
  state.beats = [];

  for (const fix of manifest.fixations) {
    const fixationSet = createFixationSet(fix, runBase);
    container.appendChild(fixationSet);
    state.beats.push(fixationSet);
  }
}

async function drawAllCanvases() {
  if (!state._canvases || !state.sourceImage) return;
  for (const { canvas, fixationIndex } of state._canvases) {
    drawSourceCanvas(canvas, state.sourceImage, state.manifest, fixationIndex);
  }
}

async function initRun(runBase) {
  const sequence = document.getElementById('sequence');
  sequence.innerHTML = '<p class="loading">Loading…</p>';

  const { manifest, runBase: base } = await loadManifest(runBase);
  state.manifest = manifest;
  state.runBase = base;
  state._canvases = [];

  state.sourceImage = await loadImage(`${base}/${manifest.source_image}`);

  buildSequence(manifest, base);
  await drawAllCanvases();

  updateScrollReveal();
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', () => {
    drawAllCanvases();
    updateScrollReveal();
  });
}

async function bootstrap() {
  const runBase = getQueryRun() || DEFAULT_RUN;
  try {
    await initRun(runBase);
  } catch (err) {
    console.error(err);
    document.getElementById('sequence').innerHTML =
      `<p class="error">${err.message}</p>`;
  }
}

bootstrap();
