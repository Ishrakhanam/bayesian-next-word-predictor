const inputField = document.getElementById('textInput');
const predictionsContainer = document.getElementById('predictions');
const predictionCount = document.getElementById('predictionCount');
const statusLabel = document.getElementById('statusLabel');
const vocabSize = document.getElementById('vocabSize');
const tokenCount = document.getElementById('tokenCount');
const contextWindow = document.getElementById('contextWindow');
const retrainButton = document.getElementById('retrainButton');
const pageTitle = document.getElementById('pageTitle');
const navItems = document.querySelectorAll('.nav-item[data-page]');
const corpusCount = document.getElementById('corpusCount');
const corpusText = document.getElementById('corpusText');
const top1Accuracy = document.getElementById('top1Accuracy');
const top5Accuracy = document.getElementById('top5Accuracy');
const evaluatedExamples = document.getElementById('evaluatedExamples');
const evaluationDetails = document.getElementById('evaluationDetails');
let latestPredictionRequest = 0;

function setStatus(message, isReady = true) {
  statusLabel.textContent = message;
  statusLabel.style.color = isReady ? '#a9d6d2' : '#f1d9a8';
}

function renderPlaceholder() {
  predictionsContainer.innerHTML = `
    <div class="placeholder">
      Start typing a sentence to generate Bayesian next-word predictions.
    </div>
  `;
  predictionCount.textContent = '0 results';
}

function renderPredictions(items) {
  if (!items || !items.length) {
    renderPlaceholder();
    return;
  }

  predictionCount.textContent = `${items.length} results`;
  predictionsContainer.replaceChildren();

  items.forEach((item, index) => {
    const percent = Math.max(0, Math.min(100, Number(item.probability || 0) * 100));
    const predictionItem = document.createElement('div');
    predictionItem.className = 'prediction-item';
    const row = document.createElement('div');
    row.className = 'prediction-row';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'predict-button';
    button.textContent = `${index + 1}. ${item.word}`;
    const probability = document.createElement('span');
    probability.className = 'prediction-probability';
    probability.textContent = `${percent.toFixed(2)}%`;
    const track = document.createElement('div');
    track.className = 'progress-track';
    const fill = document.createElement('div');
    fill.className = 'progress-fill';
    fill.style.width = `${percent}%`;
    track.append(fill);
    row.append(button, probability);
    predictionItem.append(row, track);
    predictionsContainer.append(predictionItem);

    button.addEventListener('click', () => {
      const word = item.word;
      const currentText = inputField.value.trim();
      inputField.value = currentText ? `${currentText} ${word}` : word;
      fetchPredictions();
    });
  });
}

async function fetchModelInfo() {
  try {
    const response = await fetch('/api/model-info');
    if (!response.ok) {
      throw new Error(`Model info request failed: ${response.status}`);
    }
    const data = await response.json();
    vocabSize.textContent = data.vocabulary_size || 0;
    tokenCount.textContent = data.total_tokens || 0;
    contextWindow.textContent = data.context_window || 0;
  } catch (error) {
    console.error('Unable to load model info:', error);
    setStatus('Model info error', false);
  }
}

async function fetchPredictions() {
  const requestId = ++latestPredictionRequest;
  const text = inputField.value.trim();
  if (!text) {
    renderPlaceholder();
    return;
  }

  setStatus('Predicting…', false);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, top_k: 5 }),
    });
    if (!response.ok) {
      throw new Error(`Prediction request failed: ${response.status}`);
    }
    const result = await response.json();

    if (requestId !== latestPredictionRequest) {
      return;
    }
    if (result && result.predictions) {
      renderPredictions(result.predictions);
      setStatus('Ready', true);
    } else {
      renderPlaceholder();
      setStatus('No predictions', false);
    }
  } catch (error) {
    console.error('Prediction request failed:', error);
    if (requestId !== latestPredictionRequest) {
      return;
    }
    renderPlaceholder();
    setStatus('Offline', false);
  }
}

async function fetchCorpus() {
  corpusCount.textContent = 'Loading';
  corpusText.textContent = '';
  try {
    const response = await fetch('/api/corpus');
    if (!response.ok) {
      throw new Error(`Corpus request failed: ${response.status}`);
    }
    const data = await response.json();
    corpusCount.textContent = `${data.sentence_count} sentences`;
    corpusText.textContent = data.sentences.join('\n');
    setStatus('Ready', true);
  } catch (error) {
    console.error('Unable to load the training corpus:', error);
    corpusCount.textContent = 'Unavailable';
    corpusText.textContent = 'The local training corpus could not be loaded.';
    setStatus('Corpus error', false);
  }
}

async function fetchMetrics() {
  top1Accuracy.textContent = '--';
  top5Accuracy.textContent = '--';
  evaluatedExamples.textContent = '--';
  evaluationDetails.textContent = 'Evaluating the held-out corpus…';
  setStatus('Evaluating…', false);
  try {
    const response = await fetch('/api/metrics');
    if (!response.ok) {
      throw new Error(`Metrics request failed: ${response.status}`);
    }
    const data = await response.json();
    top1Accuracy.textContent = `${(data.top1_accuracy * 100).toFixed(2)}%`;
    top5Accuracy.textContent = `${(data.top5_accuracy * 100).toFixed(2)}%`;
    evaluatedExamples.textContent = data.evaluated_examples;
    evaluationDetails.textContent =
      `${data.training_sentences} training sentences · ${data.test_sentences} test sentences · ` +
      `${data.training_vocabulary_size} training vocabulary words`;
    setStatus('Ready', true);
  } catch (error) {
    console.error('Unable to load evaluation metrics:', error);
    evaluationDetails.textContent = 'The model evaluation could not be loaded.';
    setStatus('Metrics error', false);
  }
}

function showPage(name) {
  const titles = {
    predictor: 'Predictive language assistant',
    corpus: 'Training corpus',
    metrics: 'Model evaluation',
  };
  if (!titles[name]) {
    return;
  }

  document.querySelectorAll('.dashboard-view').forEach((view) => {
    const isActive = view.id === `${name}View`;
    view.hidden = !isActive;
  });
  navItems.forEach((item) => {
    const isActive = item.dataset.page === name;
    item.classList.toggle('active', isActive);
    if (isActive) {
      item.setAttribute('aria-current', 'page');
    } else {
      item.removeAttribute('aria-current');
    }
  });
  pageTitle.textContent = titles[name];

  if (name === 'corpus') {
    fetchCorpus();
  } else if (name === 'metrics') {
    fetchMetrics();
  }
}

async function retrainModel() {
  setStatus('Retraining…', false);
  try {
    const response = await fetch('/api/retrain', {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error(`Retraining request failed: ${response.status}`);
    }
    const data = await response.json();
    if (data && data.status === 'success') {
      await fetchModelInfo();
      setStatus('Ready', true);
      if (inputField.value.trim()) {
        fetchPredictions();
      }
      if (!document.getElementById('metricsView').hidden) {
        await fetchMetrics();
      }
    }
  } catch (error) {
    console.error('Retraining failed:', error);
    setStatus('Error', false);
  }
}

navItems.forEach((item) => {
  item.addEventListener('click', () => showPage(item.dataset.page));
});

inputField.addEventListener('input', (event) => {
  const value = event.target.value;
  if (!value.trim()) {
    renderPlaceholder();
    return;
  }
  fetchPredictions();
});

retrainButton.addEventListener('click', retrainModel);

renderPlaceholder();
fetchModelInfo();
inputField.focus();
