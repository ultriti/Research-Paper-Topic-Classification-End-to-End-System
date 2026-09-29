const API_BASE = "";

const CORA_CLASSES = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory"
};

/* =====================================================
   INITIALIZATION
   ===================================================== */
document.addEventListener("DOMContentLoaded", () => {
    initializeNavigation();
    renderTopics();
    checkHealth();
    generateRandomGraph(); // Pre-populate random custom graph by default
});

/* =====================================================
   NAVIGATION
   ===================================================== */
function initializeNavigation() {
    document.querySelectorAll(".nav-item").forEach(button => {
        button.addEventListener("click", () => {
            const section = button.dataset.section;
            showSection(section);
        });
    });
}

function showSection(sectionId) {
    document.querySelectorAll(".page-section").forEach(section => {
        section.classList.remove("active");
    });

    document.querySelectorAll(".nav-item").forEach(button => {
        button.classList.remove("active");
    });

    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.add("active");
    }

    const navButton = document.querySelector(`.nav-item[data-section="${sectionId}"]`);
    if (navButton) {
        navButton.classList.add("active");
    }

    const names = {
        dashboard: "Dashboard",
        cora: "Cora Node Predictor",
        custom: "Custom Graph Inference",
        model: "Model & Architecture Details"
    };

    const breadcrumb = document.getElementById("breadcrumbCurrent");
    if (breadcrumb) {
        breadcrumb.textContent = names[sectionId] || "Dashboard";
    }

    const titles = {
        dashboard: "Graph Neural Network Dashboard",
        cora: "Cora Node Predictor",
        custom: "Custom Graph Inference",
        model: "Model Architecture & System Telemetry"
    };

    const titleEl = document.getElementById("pageTitle");
    if (titleEl) {
        titleEl.textContent = titles[sectionId] || "GCN Dashboard";
    }
}

/* =====================================================
   RENDER TOPICS
   ===================================================== */
function renderTopics() {
    const grid = document.getElementById("topicGrid");
    if (!grid) return;
    grid.innerHTML = "";

    Object.entries(CORA_CLASSES).forEach(([id, name]) => {
        const item = document.createElement("div");
        item.className = "topic";
        item.innerHTML = `
            <div class="topic-number">CLASS ${Number(id) + 1}</div>
            <div class="topic-name">${escapeHtml(name)}</div>
        `;
        grid.appendChild(item);
    });
}

/* =====================================================
   HEALTH CHECK TELEMETRY
   ===================================================== */
async function checkHealth() {
    const topText = document.getElementById("topStatusText");
    const sidebarText = document.getElementById("sidebarStatusText");
    const topDot = document.getElementById("topStatus");
    const sidebarDot = document.getElementById("sidebarStatus");

    try {
        const response = await fetch(`${API_BASE}/healthcheck`);
        if (!response.ok) throw new Error("API unavailable");

        const data = await response.json();

        if (topText) topText.textContent = "API Online";
        if (sidebarText) sidebarText.textContent = "API Online";
        if (topDot) topDot.style.background = "#10b981";
        if (sidebarDot) sidebarDot.style.background = "#10b981";

        const details = document.getElementById("apiDetails");
        if (details) {
            details.textContent = JSON.stringify(data, null, 2);
        }
    } catch (error) {
        if (topText) topText.textContent = "API Offline";
        if (sidebarText) sidebarText.textContent = "API Offline";
        if (topDot) topDot.style.background = "#ef4444";
        if (sidebarDot) sidebarDot.style.background = "#ef4444";

        const details = document.getElementById("apiDetails");
        if (details) {
            details.textContent = "Unable to connect to FastAPI backend.";
        }
    }
}

/* =====================================================
   PRESET CORA NODE SELECTION
   ===================================================== */
function setNodes(value) {
    const input = document.getElementById("nodeInput");
    if (input) input.value = value;
}

function setRandomNodes() {
    const count = 3;
    const randoms = [];
    for (let i = 0; i < count; i++) {
        randoms.push(Math.floor(Math.random() * 2707));
    }
    setNodes(randoms.join(", "));
    predictCora(); // Automatically trigger GCN inference when random node button is clicked!
}

/* =====================================================
   CORA PREDICTION
   ===================================================== */
async function predictCora() {
    const input = document.getElementById("nodeInput");
    const error = document.getElementById("coraError");
    const results = document.getElementById("coraResults");
    const button = document.getElementById("predictCoraBtn");
    const status = document.getElementById("resultStatus");

    if (error) error.textContent = "";

    const raw = input ? input.value.trim() : "";
    if (!raw) {
        if (error) error.textContent = "Please enter at least one node ID.";
        return;
    }

    const nodeIndices = raw
        .split(",")
        .map(val => val.trim())
        .filter(val => val !== "")
        .map(Number);

    if (
        nodeIndices.length === 0 ||
        nodeIndices.some(val => !Number.isInteger(val) || val < 0 || val > 2707)
    ) {
        if (error) error.textContent = "Node IDs must be integers between 0 and 2707.";
        return;
    }

    if (button) {
        button.disabled = true;
        button.innerHTML = `<span class="spinner"></span> <span>Running GCN...</span>`;
    }

    if (results) {
        results.innerHTML = `
            <div class="loading">
                <span class="spinner"></span>
                <span>Executing graph convolution inference...</span>
            </div>
        `;
    }

    try {
        const response = await fetch(`${API_BASE}/predict/cora_node`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_indices: nodeIndices })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        renderPredictions(data);

        if (status) {
            status.textContent = `${data.predictions.length} Node Output(s)`;
            status.style.background = "rgba(16,185,129,0.12)";
            status.style.color = "#10b981";
        }
    } catch (err) {
        if (results) {
            results.innerHTML = `
                <div class="empty-results">
                    <div class="empty-illustration">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="15" y1="9" x2="9" y2="15"></line>
                            <line x1="9" y1="9" x2="15" y2="15"></line>
                        </svg>
                    </div>
                    <h3>Prediction Failed</h3>
                    <p>${escapeHtml(err.message)}</p>
                </div>
            `;
        }
        if (status) {
            status.textContent = "Error";
            status.style.background = "rgba(239,68,68,0.12)";
            status.style.color = "#ef4444";
        }
    } finally {
        if (button) {
            button.disabled = false;
            button.innerHTML = `
                <span>Run GCN Inference</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            `;
        }
    }
}

/* =====================================================
   RENDER PREDICTIONS
   ===================================================== */
function renderPredictions(data) {
    const container = document.getElementById("coraResults");
    if (!container) return;

    if (!data.predictions || data.predictions.length === 0) {
        container.innerHTML = `<div class="empty-results">No predictions returned.</div>`;
        return;
    }

    container.innerHTML = "";

    data.predictions.forEach(prediction => {
        const probabilities = prediction.probabilities || [];
        const predictedClass = prediction.predicted_class_name;
        const confidence = probabilities[prediction.predicted_classes] || 0;

        const result = document.createElement("div");
        result.className = "prediction-result";

        let probabilityHTML = "";
        probabilities.forEach((prob, index) => {
            const percent = (prob * 100).toFixed(2);
            probabilityHTML += `
                <div class="probability-row">
                    <div class="probability-label">
                        <span>${escapeHtml(CORA_CLASSES[index] || `Class ${index}`)}</span>
                        <span>${percent}%</span>
                    </div>
                    <div class="probability-track">
                        <div class="probability-bar" style="width: ${percent}%;"></div>
                    </div>
                </div>
            `;
        });

        result.innerHTML = `
            <div class="prediction-top">
                <div>
                    <div class="node-id">PUBLICATION NODE #${prediction.node_index}</div>
                    <div class="predicted-class">${escapeHtml(predictedClass)}</div>
                </div>
                <div class="confidence">
                    <strong>${(confidence * 100).toFixed(2)}%</strong>
                    <span>Confidence</span>
                </div>
            </div>
            <div class="probability-list">
                ${probabilityHTML}
            </div>
        `;

        container.appendChild(result);
    });
}

/* =====================================================
   CUSTOM RANDOM GRAPH GENERATOR (MATCHING SCREENSHOT)
   ===================================================== */
function stepNodeCount(delta) {
    const el = document.getElementById("randomNodeCount");
    if (!el) return;
    let val = parseInt(el.value, 10) || 4;
    val = Math.max(1, Math.min(10, val + delta));
    el.value = val;
}

function generateRandomGraph() {
    const el = document.getElementById("randomNodeCount");
    const numNodes = el ? (parseInt(el.value, 10) || 4) : 4;
    
    // Generate feature vectors of 1433 dimensions for numNodes
    const features = [];
    for (let n = 0; n < numNodes; n++) {
        const vec = new Array(1433).fill(0);
        // Randomly activate 15-25 features per node (realistic bag of words)
        const activeCount = Math.floor(Math.random() * 11) + 15;
        for (let k = 0; k < activeCount; k++) {
            const idx = Math.floor(Math.random() * 1433);
            vec[idx] = 1;
        }
        features.push(vec);
    }
    
    // Generate edge list (ring / star graph structure)
    const src = [];
    const dst = [];
    for (let i = 0; i < numNodes; i++) {
        const next = (i + 1) % numNodes;
        src.push(i); dst.push(next);
        src.push(next); dst.push(i);
    }
    if (numNodes > 2) {
        for (let i = 0; i < numNodes; i++) {
            const cross = (i + 2) % numNodes;
            if (Math.random() > 0.3) {
                src.push(i); dst.push(cross);
                src.push(cross); dst.push(i);
            }
        }
    }
    
    const featureInput = document.getElementById("featureInput");
    const edgeInput = document.getElementById("edgeInput");
    
    if (featureInput) {
        featureInput.value = JSON.stringify(features);
    }
    if (edgeInput) {
        edgeInput.value = JSON.stringify([src, dst]);
    }
}

async function predictCustom() {
    const featureInput = document.getElementById("featureInput");
    const edgeInput = document.getElementById("edgeInput");
    const output = document.getElementById("customResult");

    try {
        if (!featureInput || !featureInput.value.trim()) {
            throw new Error("Node features input is empty.");
        }

        const nodeFeatures = JSON.parse(featureInput.value);
        let edgeIndices = null;

        if (edgeInput && edgeInput.value.trim()) {
            edgeIndices = JSON.parse(edgeInput.value);
        }

        if (!Array.isArray(nodeFeatures)) {
            throw new Error("Node features must be a JSON array [N x 1433].");
        }

        if (output) output.textContent = "Running custom inference...";

        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_features: nodeFeatures,
                edge_indices: edgeIndices
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Custom graph prediction failed.");
        }

        if (output) output.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        if (output) output.textContent = `Error: ${error.message}`;
    }
}

/* =====================================================
   UTILITIES
   ===================================================== */
function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function closeModal() {
    const modal = document.getElementById("predictionModal");
    if (modal) modal.classList.remove("active");
}
