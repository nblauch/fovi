/**
 * Scroll-driven foveated sampling playback.
 * Sequence: source + fixation → image-space + 3D + flat views → repeat.
 */

const DEFAULT_RUN = 'runs/seoul';
const DEFAULT_VIEW_ORDER = ['global_cartesian', 'manifold_3d', 'flat'];
const VIEW_DEFINITIONS = {
  global_cartesian: {
    className: 'view-frame--global',
    subtitle: 'Foveated sensor array',
    aspectRatio: (manifest) => {
      const [imgH, imgW] = manifest.image_size || [1, 1];
      return `${imgW} / ${imgH}`;
    },
  },
  manifold_3d: {
    className: 'view-frame--manifold-3d',
    subtitle: '3D sensor manifold',
  },
  flat: {
    className: 'view-frame--flat',
    subtitle: 'V1-like flat manifold',
  },
};

const state = {
  manifest: null,
  runBase: '',
  sourceImage: null,
  storyBeats: [],
  beats: [],
  plotlyViews: [],
  plotlyObserver: null,
  highlightTimer: null,
  highlightRaf: null,
  highlight: null,
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
  for (const beat of state.storyBeats) {
    beat.style.setProperty('--reveal', computeReveal(beat).toFixed(4));
  }
  for (const beat of state.beats) {
    beat.style.setProperty('--reveal', computeReveal(beat).toFixed(4));
  }
  const outro = document.getElementById('outro');
  if (outro) outro.style.setProperty('--reveal', computeReveal(outro).toFixed(4));
}

function collectStoryBeats() {
  state.storyBeats = Array.from(document.querySelectorAll('.story-reveal'));
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

function drawSourceCanvas(canvas, img, manifest, fixationIndex, maxWidth = 576) {
  const [height, width] = manifest.image_size;
  const maxW = Math.min(maxWidth, window.innerWidth * 0.92);
  const scale = maxW / width;
  const cw = Math.round(width * scale);
  const ch = Math.round(height * scale);

  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, cw, ch);

  if (fixationIndex == null) return;

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

function syncHighlightVideo(video, phaseSeconds) {
  if (!video || video.readyState < 1) return;
  const duration = Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : state.manifest?.params?.manifold_video_duration || 3;
  const targetTime = phaseSeconds % duration;
  if (Math.abs(video.currentTime - targetTime) > 0.16) {
    video.currentTime = targetTime;
  }
  video.play().catch(() => {});
}

function syncHighlightVideos() {
  const highlight = state.highlight;
  if (!highlight) return;

  const elapsed = (performance.now() - highlight.startedAt) / 1000;
  const nominalDuration = state.manifest?.params?.manifold_video_duration || 3;
  const phase = elapsed % nominalDuration;
  highlight.videos.forEach((video) => syncHighlightVideo(video, phase));
  state.highlightRaf = requestAnimationFrame(syncHighlightVideos);
}

function updateIntroHighlight(index) {
  const highlight = state.highlight;
  if (!highlight || !state.sourceImage || !state.manifest?.fixations?.length) return;

  const fix = state.manifest.fixations[index % state.manifest.fixations.length];
  highlight.label.textContent = `Fixation ${fix.index}`;
  drawSourceCanvas(
    highlight.canvas,
    state.sourceImage,
    state.manifest,
    fix.index,
    340,
  );
  highlight.globalImg.src = `${state.runBase}/${fix.views.global_cartesian}`;
  const flatView = highlight.intuitiveCheckbox?.checked && fix.views.flat_schwartz
    ? fix.views.flat_schwartz
    : fix.views.flat;
  highlight.flatImg.src = `${state.runBase}/${flatView}`;

  highlight.videos.forEach((video, videoIndex) => {
    const isActive = videoIndex === index % highlight.videos.length;
    video.classList.toggle('is-active', isActive);
  });
}

function buildIntroHighlight(manifest, runBase) {
  const root = document.getElementById('intro-highlight');
  if (!root || !manifest.fixations?.length) return;

  if (state.highlightTimer) {
    window.clearInterval(state.highlightTimer);
    state.highlightTimer = null;
  }
  if (state.highlightRaf) {
    cancelAnimationFrame(state.highlightRaf);
    state.highlightRaf = null;
  }

  root.innerHTML = '';
  const label = document.createElement('p');
  label.className = 'intro-highlight-label';

  const grid = document.createElement('div');
  grid.className = 'intro-highlight-grid';

  function appendPanelTitle(panel, text) {
    const title = document.createElement('p');
    title.className = 'view-subtitle intro-highlight-title';
    title.textContent = text;
    panel.appendChild(title);
  }

  const sourcePanel = document.createElement('figure');
  sourcePanel.className = 'intro-highlight-panel intro-highlight-source';
  appendPanelTitle(sourcePanel, 'Source image');
  const sourceCanvas = document.createElement('canvas');
  sourcePanel.appendChild(sourceCanvas);

  const globalPanel = document.createElement('figure');
  globalPanel.className = 'intro-highlight-panel';
  appendPanelTitle(globalPanel, 'Foveated sensor array');
  const globalImg = document.createElement('img');
  globalImg.alt = '';
  globalPanel.appendChild(globalImg);

  const manifoldPanel = document.createElement('figure');
  manifoldPanel.className = 'intro-highlight-panel';
  appendPanelTitle(manifoldPanel, '3D sensor manifold');
  const videoStack = document.createElement('div');
  videoStack.className = 'intro-highlight-video-stack';
  const videos = manifest.fixations.map((fix) => {
    const video = document.createElement('video');
    video.src = `${runBase}/${fix.views.manifold_3d}`;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.setAttribute('playsinline', '');
    video.setAttribute('aria-hidden', 'true');
    videoStack.appendChild(video);
    return video;
  });
  manifoldPanel.appendChild(videoStack);

  const flatPanel = document.createElement('figure');
  flatPanel.className = 'intro-highlight-panel';
  appendPanelTitle(flatPanel, 'V1-like flat manifold');
  const flatImg = document.createElement('img');
  flatImg.alt = '';
  const flatToggle = document.createElement('label');
  flatToggle.className = 'flat-toggle';
  flatToggle.innerHTML = `
    <input type="checkbox">
    <span>Make intuitive</span>
  `;
  const flatNote = document.createElement('p');
  flatNote.className = 'flat-note';
  flatNote.textContent = 'Reoriented for intuitive visualization.';
  const flatCheckbox = flatToggle.querySelector('input');
  flatCheckbox.addEventListener('change', () => {
    flatNote.classList.toggle('is-visible', flatCheckbox.checked);
    updateIntroHighlight(state.highlight.activeIndex);
  });
  flatPanel.appendChild(flatImg);
  flatPanel.appendChild(flatToggle);
  flatPanel.appendChild(flatNote);

  grid.appendChild(sourcePanel);
  grid.appendChild(globalPanel);
  grid.appendChild(manifoldPanel);
  grid.appendChild(flatPanel);
  root.appendChild(label);
  root.appendChild(grid);

  state.highlight = {
    label,
    canvas: sourceCanvas,
    globalImg,
    flatImg,
    intuitiveCheckbox: flatCheckbox,
    videos,
    startedAt: performance.now(),
    activeIndex: 0,
  };
  updateIntroHighlight(0);
  syncHighlightVideos();
  state.highlightTimer = window.setInterval(() => {
    state.highlight.activeIndex = (state.highlight.activeIndex + 1) % manifest.fixations.length;
    updateIntroHighlight(state.highlight.activeIndex);
  }, 500);
}

function createSourceImageSet() {
  const section = document.createElement('section');
  section.className = 'beat source-image-set';

  const inner = document.createElement('div');
  inner.className = 'fixation-set-inner';

  const label = document.createElement('p');
  label.className = 'beat-index';
  label.textContent = 'Source image';

  const frame = document.createElement('div');
  frame.className = 'source-frame';

  const canvas = document.createElement('canvas');
  frame.appendChild(canvas);
  state._canvases = state._canvases || [];
  state._canvases.push({ canvas, fixationIndex: null });

  inner.appendChild(label);
  inner.appendChild(frame);
  section.appendChild(inner);
  return section;
}

function getViewOrder(manifest) {
  return manifest.timing?.view_order || DEFAULT_VIEW_ORDER;
}

function isVideoSource(src) {
  const path = src.split(/[?#]/)[0].toLowerCase();
  return ['.mp4', '.webm', '.ogg'].some((ext) => path.endsWith(ext));
}

function isPlotlySource(src) {
  return src.split(/[?#]/)[0].toLowerCase().endsWith('.json');
}

function viewSourceForFixation(fix, viewName) {
  if (viewName === 'manifold_3d') {
    return fix.views?.manifold_3d_plotly || fix.views?.manifold_3d;
  }
  return fix.views?.[viewName];
}

function createFixationSet(fix, runBase) {
  const section = document.createElement('section');
  section.className = 'beat fixation-set';
  section.dataset.fixationIndex = String(fix.index);

  const inner = document.createElement('div');
  inner.className = 'fixation-set-inner';

  const label = document.createElement('p');
  label.className = 'beat-index';
  label.textContent = `Fixation ${fix.index}`;

  const views = document.createElement('div');
  views.className = 'fixation-views';

  const pair = document.createElement('div');
  pair.className = 'views-pair';

  for (const viewName of getViewOrder(state.manifest)) {
    const viewPath = viewSourceForFixation(fix, viewName);
    const viewDef = VIEW_DEFINITIONS[viewName];
    if (!viewPath || !viewDef) continue;

    const aspectRatio = typeof viewDef.aspectRatio === 'function'
      ? viewDef.aspectRatio(state.manifest)
      : viewDef.aspectRatio;
    pair.appendChild(createViewCell(
      viewDef.className,
      viewDef.subtitle,
      `${runBase}/${viewPath}`,
      aspectRatio,
      {
        intuitiveSrc: viewName === 'flat' && fix.views?.flat_schwartz
          ? `${runBase}/${fix.views.flat_schwartz}`
          : null,
      },
    ));
  }
  views.appendChild(pair);

  inner.appendChild(label);
  inner.appendChild(views);
  section.appendChild(inner);
  return section;
}

async function renderPlotlyManifold(container) {
  if (container.dataset.rendered === 'true') return;
  if (!window.Plotly) {
    container.textContent = 'Interactive plot unavailable';
    return;
  }

  container.dataset.rendered = 'true';
  container.textContent = '';

  const res = await fetch(container.dataset.src);
  if (!res.ok) throw new Error(`Could not load ${container.dataset.src}`);
  const data = await res.json();

  const trace = {
    type: 'scatter3d',
    mode: 'markers',
    x: data.x,
    y: data.y,
    z: data.z,
    marker: {
      color: data.color,
      size: data.marker_size || 2,
      opacity: 1,
    },
    hoverinfo: 'skip',
  };
  const axisRange = data.range || [-0.51, 0.51];
  const hiddenCenteredAxis = {
    visible: false,
    showgrid: false,
    zeroline: false,
    showticklabels: false,
    range: axisRange,
    autorange: false,
  };
  const layout = {
    autosize: true,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      bgcolor: 'rgba(0,0,0,0)',
      aspectmode: 'cube',
      camera: data.camera || { eye: { x: 1.65, y: -1.9, z: 0.75 } },
      xaxis: hiddenCenteredAxis,
      yaxis: hiddenCenteredAxis,
      zaxis: hiddenCenteredAxis,
    },
  };
  const config = {
    displayModeBar: false,
    responsive: true,
    scrollZoom: true,
  };
  await window.Plotly.newPlot(container, [trace], layout, config);
  requestAnimationFrame(() => window.Plotly.Plots.resize(container));
}

function observePlotlyViews() {
  if (!state.plotlyViews.length) return;
  if (!('IntersectionObserver' in window)) {
    state.plotlyViews.forEach((view) => renderPlotlyManifold(view).catch(console.error));
    return;
  }

  if (state.plotlyObserver) {
    state.plotlyObserver.disconnect();
  }
  state.plotlyObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      state.plotlyObserver.unobserve(entry.target);
      renderPlotlyManifold(entry.target).catch((err) => {
        console.error(err);
        entry.target.textContent = 'Interactive plot failed to load';
      });
    }
  }, { rootMargin: '400px 0px' });

  state.plotlyViews.forEach((view) => state.plotlyObserver.observe(view));
}

function createViewCell(className, subtitle, mediaSrc, aspectRatio = null, options = {}) {
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

  let hint = null;
  let flatToggle = null;
  let flatNote = null;
  if (isVideoSource(mediaSrc)) {
    const video = document.createElement('video');
    video.src = mediaSrc;
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.preload = 'metadata';
    video.setAttribute('playsinline', '');
    video.setAttribute('aria-label', subtitle);
    frame.appendChild(video);
  } else if (isPlotlySource(mediaSrc)) {
    const plot = document.createElement('div');
    plot.className = 'plotly-manifold';
    plot.dataset.src = mediaSrc;
    plot.setAttribute('role', 'img');
    plot.setAttribute('aria-label', subtitle);
    plot.textContent = 'Loading interactive plot…';
    state.plotlyViews.push(plot);
    frame.appendChild(plot);

    hint = document.createElement('div');
    hint.className = 'plotly-hint';
    hint.setAttribute('aria-hidden', 'true');
    hint.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" focusable="false">
        <path d="M8 6h8M8 18h8M6 8v8M18 8v8M7 7l-2 2M17 7l2 2M7 17l-2-2M17 17l2-2" />
      </svg>
      <span>Drag to rotate, scroll to zoom</span>
    `;
  } else {
    const img = document.createElement('img');
    img.src = mediaSrc;
    img.alt = '';
    img.loading = 'lazy';
    frame.appendChild(img);

    if (options.intuitiveSrc) {
      flatToggle = document.createElement('label');
      flatToggle.className = 'flat-toggle';
      flatToggle.innerHTML = `
        <input type="checkbox">
        <span>Make intuitive</span>
      `;
      const checkbox = flatToggle.querySelector('input');
      checkbox.addEventListener('change', () => {
        const isIntuitive = checkbox.checked;
        img.src = isIntuitive ? options.intuitiveSrc : mediaSrc;
        flatNote.hidden = !isIntuitive;
      });

      flatNote = document.createElement('p');
      flatNote.className = 'flat-note';
      flatNote.textContent = 'Reoriented for intuitive visualization.';
      flatNote.hidden = true;
    }
  }

  cell.appendChild(caption);
  if (flatToggle) {
    cell.appendChild(flatToggle);
  }
  if (flatNote) {
    cell.appendChild(flatNote);
  }
  if (hint) {
    cell.appendChild(hint);
  }
  cell.appendChild(frame);
  return cell;
}

function buildSequence(manifest, runBase) {
  const container = document.getElementById('sequence');
  container.innerHTML = '';
  state.beats = [];
  state.plotlyViews = [];

  const sourceImageSet = createSourceImageSet();
  container.appendChild(sourceImageSet);
  state.beats.push(sourceImageSet);

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

  buildIntroHighlight(manifest, base);
  buildSequence(manifest, base);
  await drawAllCanvases();
  observePlotlyViews();

  updateScrollReveal();
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', () => {
    if (state.highlight) {
      updateIntroHighlight(state.highlight.activeIndex);
    }
    drawAllCanvases();
    for (const view of state.plotlyViews) {
      if (view.dataset.rendered === 'true' && window.Plotly) {
        window.Plotly.Plots.resize(view);
      }
    }
    updateScrollReveal();
  });
}

async function bootstrap() {
  collectStoryBeats();
  const runBase = getQueryRun() || DEFAULT_RUN;
  try {
    await initRun(runBase);
  } catch (err) {
    console.error(err);
    updateScrollReveal();
    document.getElementById('sequence').innerHTML =
      `<p class="error">${err.message}</p>`;
  }
}

bootstrap();
