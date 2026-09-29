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


document.querySelectorAll(".page-section")
    .forEach(section => {

        section.classList.remove("active");

    });


document.querySelectorAll(".nav-item")
    .forEach(button => {

        button.classList.remove("active");

    });


const section = document.getElementById(sectionId);

if (section) {
    section.classList.add("active");
}


const navButton =
    document.querySelector(
        `.nav-item[data-section="${sectionId}"]`
    );

if (navButton) {
    navButton.classList.add("active");
}


const names = {
    dashboard: "Dashboard",
    cora: "Cora Prediction",
    custom: "Custom Graph",
    model: "Model Details"
};


document.getElementById("breadcrumbCurrent")
    .textContent = names[sectionId] || "Dashboard";


const titles = {
    dashboard: "Graph Neural Network Dashboard",
    cora: "Cora Node Prediction",
    custom: "Custom Graph Prediction",
    model: "Model Details"
};


document.getElementById("pageTitle")
    .textContent = titles[sectionId] || "GCN Dashboard";


}

/* =====================================================
TOPICS
===================================================== */

function renderTopics() {


const grid =
    document.getElementById("topicGrid");

grid.innerHTML = "";


Object.entries(CORA_CLASSES).forEach(
    ([id, name]) => {

        const item =
            document.createElement("div");

        item.className = "topic";

        item.innerHTML = `
            <div class="topic-number">
                CLASS ${Number(id) + 1}
            </div>

            <div class="topic-name">
                ${name}
            </div>
        `;

        grid.appendChild(item);

    }
);


}

/* =====================================================
HEALTH CHECK
===================================================== */

async function checkHealth() {


const topText =
    document.getElementById("topStatusText");

const sidebarText =
    document.getElementById("sidebarStatusText");

const topDot =
    document.getElementById("topStatus");

const sidebarDot =
    document.getElementById("sidebarStatus");


try {

    const response =
        await fetch(`${API_BASE}/healthcheck`);


    if (!response.ok) {
        throw new Error("API unavailable");
    }


    const data =
        await response.json();


    topText.textContent = "API Online";

    sidebarText.textContent = "API Online";

    topDot.style.background = "#22c55e";
    sidebarDot.style.background = "#22c55e";


    const details =
        document.getElementById("apiDetails");

    if (details) {

        details.textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    }


} catch (error) {

    topText.textContent = "API Offline";

    sidebarText.textContent = "API Offline";

    topDot.style.background = "#ef4444";
    sidebarDot.style.background = "#ef4444";


    const details =
        document.getElementById("apiDetails");

    if (details) {

        details.textContent =
            "Unable to connect to FastAPI backend.";

    }

}


}

/* =====================================================
QUICK NODE SELECTION
===================================================== */

function setNodes(value) {


document.getElementById("nodeInput")
    .value = value;


}

/* =====================================================
CORA PREDICTION
===================================================== */

async function predictCora() {


const input =
    document.getElementById("nodeInput");

const error =
    document.getElementById("coraError");

const results =
    document.getElementById("coraResults");

const button =
    document.getElementById("predictCoraBtn");

const status =
    document.getElementById("resultStatus");


error.textContent = "";


const raw =
    input.value.trim();


if (!raw) {

    error.textContent =
        "Please enter at least one node ID.";

    return;

}


const nodeIndices =
    raw
        .split(",")
        .map(value => value.trim())
        .filter(value => value !== "")
        .map(Number);


if (
    nodeIndices.length === 0 ||
    nodeIndices.some(
        value =>
            !Number.isInteger(value) ||
            value < 0 ||
            value > 2707
    )
) {

    error.textContent =
        "Node IDs must be integers between 0 and 2707.";

    return;

}


button.disabled = true;

button.innerHTML = `
    <span class="spinner"></span>
    Running GCN...
`;


results.innerHTML = `
    <div class="loading">
        <span class="spinner"></span>
        Running graph inference...
    </div>
`;


try {

    const response =
        await fetch(
            `${API_BASE}/predict/cora_node`,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    node_indices:
                        nodeIndices
                })
            }
        );


    const data =
        await response.json();


    if (!response.ok) {

        throw new Error(
            data.detail ||
            "Prediction failed."
        );

    }


    renderPredictions(data);


    status.textContent =
        `${data.predictions.length} Result(s)`;

    status.style.background =
        "rgba(34,197,94,.08)";

    status.style.color =
        "#4dde7a";


} catch (err) {

    results.innerHTML = `
        <div class="empty-results">

            <div class="empty-icon">!</div>

            <h3>Prediction Failed</h3>

            <p>${escapeHtml(err.message)}</p>

        </div>
    `;

    status.textContent = "Error";

    status.style.background =
        "rgba(239,68,68,.08)";

    status.style.color = "#ff7373";

} finally {

    button.disabled = false;

    button.innerHTML = `
        <span>Run GCN Prediction</span>
        <span>→</span>
    `;

}


}

/* =====================================================
RENDER PREDICTIONS
===================================================== */

function renderPredictions(data) {


const container =
    document.getElementById("coraResults");


if (!data.predictions ||
    data.predictions.length === 0) {

    container.innerHTML = `
        <div class="empty-results">
            No predictions returned.
        </div>
    `;

    return;

}


container.innerHTML = "";


data.predictions.forEach(prediction => {

    const probabilities =
        prediction.probabilities || [];


    const predictedClass =
        prediction.predicted_class_name;


    const confidence =
        probabilities[
            prediction.predicted_classes
        ] || 0;


    const result =
        document.createElement("div");

    result.className =
        "prediction-result";


    let probabilityHTML = "";


    probabilities.forEach(
        (probability, index) => {

            probabilityHTML += `

                <div class="probability-row">

                    <div class="probability-label">

                        <span>
                            ${escapeHtml(
                                CORA_CLASSES[index]
                            )}
                        </span>

                        <span>
                            ${(
                                probability * 100
                            ).toFixed(2)}%
                        </span>

                    </div>

                    <div class="probability-track">

                        <div
                            class="probability-bar"
                            style="
                                width:
                                ${probability * 100}%
                            "
                        ></div>

                    </div>

                </div>
            `;

        }
    );


    result.innerHTML = `

        <div class="prediction-top">

            <div>

                <div class="node-id">
                    NODE #${prediction.node_index}
                </div>

                <div class="predicted-class">
                    ${escapeHtml(predictedClass)}
                </div>

            </div>


            <div class="confidence">

                <strong>
                    ${(confidence * 100).toFixed(2)}%
                </strong>

                <span>
                    Confidence
                </span>

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
CUSTOM GRAPH
===================================================== */

async function predictCustom() {


const featureInput =
    document.getElementById("featureInput");

const edgeInput =
    document.getElementById("edgeInput");

const output =
    document.getElementById("customResult");


try {

    const nodeFeatures =
        JSON.parse(
            featureInput.value
        );


    let edgeIndices = null;


    if (edgeInput.value.trim()) {

        edgeIndices =
            JSON.parse(
                edgeInput.value
            );

    }


    if (!Array.isArray(nodeFeatures)) {

        throw new Error(
            "Node features must be a JSON array."
        );

    }


    output.textContent =
        "Running inference...";


    const response =
        await fetch(
            `${API_BASE}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    node_features:
                        nodeFeatures,

                    edge_indices:
                        edgeIndices

                })
            }
        );


    const data =
        await response.json();


    if (!response.ok) {

        throw new Error(
            data.detail ||
            "Custom graph prediction failed."
        );

    }


    output.textContent =
        JSON.stringify(
            data,
            null,
            2
        );


} catch (error) {

    output.textContent =
        `Error: ${error.message}`;

}


}

/* =====================================================
ESCAPE HTML
===================================================== */

function escapeHtml(value) {


return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");


}

/* =====================================================
MODAL
===================================================== */

function closeModal() {


document
    .getElementById("predictionModal")
    .classList.remove("active");


}
