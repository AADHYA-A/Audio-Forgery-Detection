// static/app.js — Audio Forgery Detection Dashboard
document.addEventListener('DOMContentLoaded', () => {

    /* ─── Element refs ─────────────────────────────────── */
    const overlay         = document.getElementById('loadingOverlay');
    const fileInput       = document.getElementById('fileInput');
    const dropzone        = document.getElementById('dropzone');
    const dropFilename    = document.getElementById('dropFilename');
    const recordBtn       = document.getElementById('recordBtn');
    const recordStatus    = document.getElementById('recordStatus');
    const recordTimer     = document.getElementById('recordTimer');
    const timerDisplay    = document.getElementById('timerDisplay');
    const analyzeBtn      = document.getElementById('analyzeBtn');
    const analyzeBtnLabel = document.getElementById('analyzeBtnLabel');
    const resultsEmpty    = document.getElementById('resultsEmpty');
    const resultsGrid     = document.getElementById('resultsGrid');

    /* ─── State ─────────────────────────────────────────── */
    let selectedFile   = null;
    let activeTab      = 'upload';
    let activeModel    = 'both';
    let mediaRecorder  = null;
    let audioChunks    = [];
    let timerInterval  = null;
    let timerSeconds   = 0;
    let selectedSample = null;   // currently selected sample card element

    /* ─── Loading ────────────────────────────────────────── */
    const showLoading = () => overlay.classList.add('active');
    const hideLoading = () => overlay.classList.remove('active');

    /* ─── Tabs ───────────────────────────────────────────── */
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            activeTab = tab.dataset.tab;
            document.getElementById('tab-' + activeTab).classList.add('active');
            updateAnalyzeLabel();
        });
    });

    /* ─── Model cards ────────────────────────────────────── */
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            activeModel = card.dataset.model;
            updateAnalyzeLabel();
        });
    });

    function updateAnalyzeLabel() {
        const modelMap = { both: 'Both Models', cnn_lstm: 'CNN + LSTM', xgboost: 'GFCC + XGBoost' };
        analyzeBtnLabel.textContent = `Analyze — ${modelMap[activeModel]}`;
    }

    /* ─── Dropzone ───────────────────────────────────────── */
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            selectedFile = fileInput.files[0];
            dropFilename.textContent = `✓ ${selectedFile.name}`;
            resetResults();
        }
    });

    dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', e => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            selectedFile = file;
            dropFilename.textContent = `✓ ${file.name}`;
            resetResults();
        }
    });

    function resetResults() {
        resultsEmpty.style.display = 'flex';
        resultsGrid.style.display = 'none';
    }

    /* ─── Recording ──────────────────────────────────────── */
    recordBtn.addEventListener('click', async () => {
        if (recordBtn.classList.contains('recording')) {
            mediaRecorder.stop();
            recordBtn.classList.remove('recording');
            recordStatus.textContent = 'Processing…';
            clearInterval(timerInterval);
            recordTimer.style.display = 'none';
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunks, { type: 'audio/wav' });
                selectedFile = new File([blob], 'recording.wav', { type: 'audio/wav' });
                recordStatus.textContent = `Recording saved (${formatTime(timerSeconds)})`;
                stream.getTracks().forEach(t => t.stop());
                resetResults();
            };
            mediaRecorder.start();
            recordBtn.classList.add('recording');
            recordStatus.textContent = 'Recording… click to stop';
            timerSeconds = 0;
            timerDisplay.textContent = '0:00';
            recordTimer.style.display = 'block';
            timerInterval = setInterval(() => {
                timerSeconds++;
                timerDisplay.textContent = formatTime(timerSeconds);
            }, 1000);
        } catch (err) {
            recordStatus.textContent = 'Microphone access denied';
        }
    });

    function formatTime(s) {
        return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    }

    /* ═══════════════════════════════════════════════════════
       SAMPLE GALLERY — click to select, then analyze
       ═══════════════════════════════════════════════════════ */
    document.querySelectorAll('.sample-card').forEach(card => {
        card.addEventListener('click', (e) => {
            // Don't trigger selection when clicking on audio controls
            if (e.target.closest('audio')) return;

            // Deselect previous
            if (selectedSample) selectedSample.classList.remove('selected');

            // Select this card
            selectedSample = card;
            card.classList.add('selected');

            // Switch to dataset tab if not already
            if (activeTab !== 'dataset') {
                document.querySelectorAll('.tab').forEach(t => {
                    t.classList.remove('active');
                    if (t.dataset.tab === 'dataset') t.classList.add('active');
                });
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                document.getElementById('tab-dataset').classList.add('active');
                activeTab = 'dataset';
            }

            updateAnalyzeLabel();
        });
    });

    /* ─── Analyze ────────────────────────────────────────── */
    analyzeBtn.addEventListener('click', async () => {
        let file = null;
        let filename = null;

        if (activeTab === 'upload') {
            file = selectedFile;
            if (!file) { flashError('Please select or drop an audio file first.'); return; }
            filename = file.name;
        } else if (activeTab === 'record') {
            file = selectedFile;
            if (!file) { flashError('Please record audio first.'); return; }
            filename = 'recording.wav';
        } else if (activeTab === 'dataset') {
            if (!selectedSample) { flashError('Please click a sample card to select it.'); return; }
            const sampleFilename = selectedSample.dataset.filename;
            filename = sampleFilename;
            showLoading();
            try {
                const r = await fetch(`/dataset/${sampleFilename}`);
                if (!r.ok) throw new Error('File not found');
                const blob = await r.blob();
                file = new File([blob], sampleFilename, { type: 'audio/wav' });
            } catch (e) {
                hideLoading();
                flashError('Could not load dataset file: ' + e.message);
                return;
            }
        }

        showLoading();
        const form = new FormData();
        form.append('file', file);

        try {
            const r = await fetch('/predict', { method: 'POST', body: form });
            const data = await r.json();
            hideLoading();
            if (data.error) {
                flashError('Prediction error: ' + data.error);
                return;
            }
            showResults(data, filename);
        } catch (err) {
            hideLoading();
            flashError('Prediction failed: ' + err.message);
        }
    });

    /* ═══════════════════════════════════════════════════════
       SHOW RESULTS — with comparison view
       ═══════════════════════════════════════════════════════ */
    function showResults(data, filename) {
        resultsEmpty.style.display = 'none';
        resultsGrid.style.display = 'flex';

        const xgb = data.xgboost;
        const cnn = data.cnn_lstm;
        const audioUrl = data.audio_url || null;
        // Update top stat cards if metrics are present
        if (xgb && xgb.metrics) {
            const acc = (xgb.metrics.accuracy * 100).toFixed(1) + '%';
            const xgbAccEl = document.getElementById('xgbAccuracy');
            if (xgbAccEl) xgbAccEl.textContent = acc;
        }
        if (cnn && cnn.metrics) {
            const acc = (cnn.metrics.accuracy * 100).toFixed(1) + '%';
            const cnnAccEl = document.getElementById('cnnAccuracy');
            if (cnnAccEl) cnnAccEl.textContent = acc;
        }

        // Determine which models to show
        const showXgb = (activeModel === 'both' || activeModel === 'xgboost') && xgb && xgb.label !== null;
        const showCnn = (activeModel === 'both' || activeModel === 'cnn_lstm') && cnn && cnn.label !== null;
        const showBoth = showXgb && showCnn;

        // Audio playback
        const audioPlayer = document.getElementById('audioPlayer');
        const audioFilenameLabel = document.getElementById('audioFilenameLabel');
        if (audioUrl) {
            document.getElementById('resultAudio').src = audioUrl;
            audioFilenameLabel.textContent = filename || '';
            audioPlayer.style.display = 'block';
        } else {
            audioPlayer.style.display = 'none';
        }

        // Comparison header
        const compHeader = document.getElementById('comparisonHeader');
        const compSub = document.getElementById('comparisonSub');
        if (showBoth) {
            compHeader.style.display = 'block';
            compSub.textContent = `Comparing CNN‑LSTM vs GFCC+XGBoost on "${filename || 'audio'}"`;
        } else {
            compHeader.style.display = 'block';
            const modelName = showCnn ? 'CNN‑LSTM' : 'GFCC+XGBoost';
            compSub.textContent = `${modelName} prediction on "${filename || 'audio'}"`;
        }

        // Comparison grid
        const compGrid = document.getElementById('comparisonGrid');
        const cardCnn = document.getElementById('cardCnn');
        const cardXgb = document.getElementById('cardXgboost');

        if (showBoth) {
            compGrid.style.display = 'grid';
            compGrid.style.gridTemplateColumns = '1fr 1fr';
            cardCnn.style.display = 'block';
            cardXgb.style.display = 'block';
        } else if (showCnn) {
            compGrid.style.display = 'grid';
            compGrid.style.gridTemplateColumns = '1fr';
            cardCnn.style.display = 'block';
            cardXgb.style.display = 'none';
        } else if (showXgb) {
            compGrid.style.display = 'grid';
            compGrid.style.gridTemplateColumns = '1fr';
            cardCnn.style.display = 'none';
            cardXgb.style.display = 'block';
        } else {
            compGrid.style.display = 'none';
        }

        // Set individual model results
        if (showXgb) setModelResult('Xgboost', xgb, 'GFCC + XGBoost');
        if (showCnn) setModelResult('Cnn', cnn, 'CNN‑LSTM');

        // Consensus
        const consensusCard = document.getElementById('cardConsensus');
        const verdictCons   = document.getElementById('verdictConsensus');
        const descCons      = document.getElementById('descConsensus');
        consensusCard.className = 'result-card result-card-full';

        if (showBoth) {
            const xgbFake = xgb.label === 'fake';
            const cnnFake = cnn.label === 'fake';
            if (xgbFake === cnnFake) {
                const isForged = xgbFake;
                verdictCons.textContent = isForged ? '⚠ FORGED AUDIO' : '✓ GENUINE AUDIO';
                verdictCons.className = isForged ? 'result-verdict forged' : 'result-verdict genuine';
                descCons.textContent = isForged
                    ? 'Both models agree: this audio is synthetic/forged'
                    : 'Both models agree: this audio is authentic/genuine';
                consensusCard.classList.add(isForged ? 'forged' : 'genuine');
            } else {
                // Disagreement — trust higher confidence
                const xgbConf = xgb.confidence || 0;
                const cnnConf = cnn.confidence || 0;
                const trusted = xgbConf >= cnnConf ? xgb : cnn;
                const trustedName = xgbConf >= cnnConf ? 'XGBoost' : 'CNN‑LSTM';
                const isForged = trusted.label === 'fake';
                verdictCons.textContent = isForged ? '⚠ LIKELY FORGED' : '✓ LIKELY GENUINE';
                verdictCons.className = isForged ? 'result-verdict forged' : 'result-verdict genuine';
                descCons.textContent = `Models disagree — trusting ${trustedName} (higher confidence: ${((trusted.confidence || 0) * 100).toFixed(1)}%)`;
                consensusCard.classList.add(isForged ? 'forged' : 'genuine');
            }
        } else {
            const single = showXgb ? xgb : (showCnn ? cnn : null);
            if (single && single.label) {
                const isForged = single.label === 'fake';
                verdictCons.textContent = isForged ? '⚠ FORGED AUDIO' : '✓ GENUINE AUDIO';
                verdictCons.className = isForged ? 'result-verdict forged' : 'result-verdict genuine';
                descCons.textContent = isForged ? 'Model predicts this audio is synthetic/forged' : 'Model predicts this audio is authentic/genuine';
                consensusCard.classList.add(isForged ? 'forged' : 'genuine');
            } else {
                verdictCons.textContent = 'N/A';
                verdictCons.className = 'result-verdict na';
                descCons.textContent = 'No model prediction available';
            }
        }

        // Store metrics globally for the viz overlay
        window.latestXgbMetrics = data.xgboost && data.xgboost.metrics ? data.xgboost.metrics : null;
        window.latestCnnMetrics = data.cnn_lstm && data.cnn_lstm.metrics ? data.cnn_lstm.metrics : null;
        // Metrics
        showMetrics(data, showCnn, showXgb);
    }

    function setModelResult(key, value, modelLabel) {
        const verdict = document.getElementById('verdict' + key);
        const badge   = document.getElementById('badge' + key);
        const desc    = document.getElementById('desc' + key);
        const conf    = document.getElementById('conf' + key);
        const card    = document.getElementById('card' + key);

        if (!verdict || !badge || !desc || !conf || !card) {
            console.warn('setModelResult: missing DOM element for key', key);
            return;
        }

        card.className = 'result-card';

        if (!value || value.label === null) {
            verdict.textContent = 'N/A';
            verdict.className   = 'result-verdict na';
            badge.textContent   = 'Not loaded';
            badge.className     = 'result-badge';
            desc.textContent    = `${modelLabel} model not available`;
            conf.textContent    = '';
        } else if (value.label === 'fake') {
            verdict.textContent = '⚠ FORGED';
            verdict.className   = 'result-verdict forged';
            badge.textContent   = 'Forgery Detected';
            badge.className     = 'result-badge forged';
            desc.textContent    = `${modelLabel} classified this as synthetic/forged`;
            conf.textContent    = `Confidence: ${((value.confidence || 0) * 100).toFixed(1)}%`;
            card.classList.add('forged');
        } else {
            verdict.textContent = '✓ GENUINE';
            verdict.className   = 'result-verdict genuine';
            badge.textContent   = 'Authentic';
            badge.className     = 'result-badge genuine';
            desc.textContent    = `${modelLabel} classified this as authentic/genuine`;
            conf.textContent    = `Confidence: ${((value.confidence || 0) * 100).toFixed(1)}%`;
            card.classList.add('genuine');
        }
    }

    // ─── Metrics display ────────────────────────────────
    function renderCompareChart(xgbMetrics, cnnMetrics) {
        const container = document.getElementById('compareChart');
        if (!container) return;
        // Prepare metric values or placeholders
        const format = (val) => val !== null && val !== undefined ? (val * 100).toFixed(1) : '--';
        const accXgb = format(xgbMetrics?.accuracy);
        const precXgb = format(xgbMetrics?.precision);
        const recXgb = format(xgbMetrics?.recall);
        const f1Xgb = format(xgbMetrics?.f1_score);
        const accCnn = format(cnnMetrics?.accuracy);
        const precCnn = format(cnnMetrics?.precision);
        const recCnn = format(cnnMetrics?.recall);
        const f1Cnn = format(cnnMetrics?.f1_score);
        const metrics = [
            { label: 'Accuracy',  xgb: accXgb, cnn: accCnn },
            { label: 'Precision', xgb: precXgb, cnn: precCnn },
            { label: 'Recall',    xgb: recXgb, cnn: recCnn },
            { label: 'F1 Score',  xgb: f1Xgb, cnn: f1Cnn },
        ];
        container.innerHTML = metrics.map(m => `
            <div class="viz-cmp-group">
                <div class="viz-cmp-label">${m.label}</div>
                <div class="viz-cmp-bars">
                    <div class="viz-cmp-bar-wrap">
                        <div class="viz-cmp-bar xgb-bar" style="width:${m.xgb}%"></div>
                        <span class="viz-cmp-pct">${m.xgb}%</span>
                    </div>
                    <div class="viz-cmp-bar-wrap">
                        <div class="viz-cmp-bar cnn-bar" style="width:${m.cnn}%"></div>
                        <span class="viz-cmp-pct">${m.cnn}%</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    function showMetrics(data, showCnn, showXgb) {
        const section = document.getElementById('metricsSection');
        const cnnCard = document.getElementById('metricsCnn');
        const xgbCard = document.getElementById('metricsXgboost');

        const cnnMetrics = data.cnn_lstm && data.cnn_lstm.metrics;
        const xgbMetrics = data.xgboost && data.xgboost.metrics;

        const hasAny = (showCnn && cnnMetrics) || (showXgb && xgbMetrics);
        section.style.display = hasAny ? 'block' : 'none';

        if (showCnn && cnnMetrics) {
            cnnCard.style.display = 'block';
            renderMetricBars('metricsCnnBars', cnnMetrics);
            document.getElementById('confusionCnnImg').src = '/static/confusion_cnn_lstm.png?' + Date.now();
        } else {
            cnnCard.style.display = 'none';
        }

        if (showXgb && xgbMetrics) {
            xgbCard.style.display = 'block';
            renderMetricBars('metricsXgboostBars', xgbMetrics);
            document.getElementById('confusionXgboostImg').src = '/static/confusion_xgboost.png?' + Date.now();
        } else {
            xgbCard.style.display = 'none';
        }
    }

    function renderMetricBars(containerId, metrics) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        const keys = ['accuracy', 'precision', 'recall', 'f1_score'];
        const labels = { accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1_score: 'F1 Score' };

        keys.forEach(key => {
            const val = metrics[key];
            if (val === undefined || val === null) return;
            const pct = (val * 100).toFixed(1);

            const row = document.createElement('div');
            row.className = 'metric-bar-row';
            row.innerHTML = `
                <span class="metric-bar-label">${labels[key]}</span>
                <div class="metric-bar-track">
                    <div class="metric-bar-fill" style="width: ${pct}%"></div>
                </div>
                <span class="metric-bar-value">${pct}%</span>
            `;
            container.appendChild(row);
        });
    }

    /* ─── Flash error ────────────────────────────────────── */
    function flashError(msg) {
        const el = document.createElement('div');
        el.textContent = msg;
        Object.assign(el.style, {
            position: 'fixed', bottom: '24px', left: '50%', transform: 'translateX(-50%)',
            background: '#1a1525', border: '1px solid rgba(239,68,68,0.4)', color: '#ef4444',
            padding: '10px 22px', borderRadius: '10px', fontSize: '0.82rem',
            zIndex: 9999, transition: 'opacity 0.4s', whiteSpace: 'nowrap',
            boxShadow: '0 4px 24px rgba(0,0,0,0.5)', fontFamily: 'Inter, sans-serif'
        });
        document.body.appendChild(el);
        setTimeout(() => { el.style.opacity = '0'; }, 2800);
        setTimeout(() => el.remove(), 3300);
    }
});