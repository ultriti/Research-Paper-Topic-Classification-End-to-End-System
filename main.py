from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import onnxruntime as ort

# label the topics
CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}


"""
Validation --------------------

where this file lies so we can find the odel filw 
"""

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Artifacts", "simple_gcn_cora.onnx")
DATA_DIR = os.path.join(BASE_DIR, "data", "Planetoid")
STATIC_DIR = os.path.join(BASE_DIR, "static")


# load the trained model - create a lifetime model - model runns until its closed manually
model_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


app = FastAPI()

"""
create schema for the objects
"""


# graph predict request schema - user input val
class GraphPredictRequest(BaseModel):

    # a list of nodes where each node is list of 1433 numbers ( features)

    node_features: List[List[float]]

    # optional - wch node connected to wch shape [2,num of edges]
    edge_indices: Optional[List[List[int]]] = None


# node / RP Getting check coreect index
class CoraNodeRequest(BaseModel):
    # wch node ( by index number ) from real cora dataset to classify
    node_indices: List[int]


"""
helper funtion 
"""


# softmax -[ score - added_max ---------> exp_sc / exp_score.su m]
def Softmax(scores: np.ndarray):
    # turns raw model scores ( logits ) to prababilities ( 0 - 1 )
    #  we will get max prob
    shifted = scores - scores.max(axis=1, keepdims=True)

    exp_scores = np.exp(shifted)
    # remove or short the high values

    return exp_scores / exp_scores.sum(axis=1, keepdims=1)


# run the model --------------------------------------
def run_model(
    node_features: np.ndarray, edge_index: np.ndarray, node_indices_to_return
):

    # run the onnnx model - only on graph we are given and return its predictions
    output = model_session.run(
        ["logits"],
        {
            "node_features": node_features.astype(np.float32),
            "edge_indices": edge_index.astype(np.int64),
        },
    )


    logits = output[0]  # raw scores
    probabilities = Softmax(logits)  # probablities
    predicted_classes = logits.argmax(axis=-1)  # max prob

    results = []

    for i in node_indices_to_return:
        results.append(
            {
                "node_index": i,
                "predicted_classes": int(predicted_classes[i]),
                "predicted_class_name": CORA_CLASSES[int(predicted_classes[i])],
                "probabilities": probabilities[i].tolist(),
                "logits": logits[i].tolist(),
            }
        )

    return {
        "num_nodes": node_features.shape[0],
        "num_edges": edge_index.shape[1],
        "predictions": results,
    }


# -----------------------------
# Routes ------------------------
@app.get("/")
def home_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"service": "Simple GCN CORA API", "status": "Running"}


# health check
@app.get("/healthcheck")
def health_check():
    return {"status": "healthy", "providers": model_session.get_providers()}


@app.get("/model_info")
def model_info():
    # shows basic details about the model - inputs, outputs, classes
    return {
        "Model_Name": "Simple GCN",
        "feature_dimentions": 1433,
        "num_classes": 7,
        "class_mapping": CORA_CLASSES,
        "inputs": [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in model_session.get_inputs()
        ],
        "outputs": [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in model_session.get_outputs()
        ],
    }


# PREDICTION ROUTES ----------


# GraphPredictRequest --- validation schema
@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    # classify the graph provided - optionally own edges/connection

    if not request.node_features:
        raise HTTPException(400, "node features cant be empty")

    # so for each paper (nodes) we r getting 1433 features
    for features_vector in request.node_features:
        if len(features_vector) != 1433:
            raise HTTPException(
                422, "Each nodes features vector should have at least 1433 elements"
            )

    # convert list to numpy
    node_features = np.array(request.node_features, dtype=np.float32)
    num_nodes = len(request.node_features)

    if request.edge_indices:  # if we r getting edge indices from the user
        # nearby nodes
        edge_index = np.array(request.edge_indices, dtype=np.int64)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(422, "edge indices must have shape [2, num_edges]")

    else:  # we r not getting edge indices from the user

        # no edges given just connect every node to itself ( self loop )
        node_ids = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])

    all_node_features = list(range(num_nodes))

    return run_model(node_features, edge_index, all_node_features)


        # it has 2 dimnetion ----------
        # edge_index = [
        #   [source_node...],A
        #   [destination_node...]B
        # ]


# predict real cora nodes
@app.post("/predict/cora_node")
def predict_real_cora_nodes(request: CoraNodeRequest):
    # we r importing the dataset cora load when we call this funtion / route so load balancing should be ,aintained

    from torch_geometric.datasets import Planetoid

    try:
        cora_dataset = Planetoid(root=DATA_DIR, name="Cora")[0]  # to get data

    except Exception as error:
        raise HTTPException(500, f"failed to load cora dataset ")

    # here highest limit for no of node is 2708 so more than that we just throwerror
    # check out laiers

    largest_valid_index = cora_dataset.num_nodes - 1
    invalid_index = [
        i for i in request.node_indices if i < 0 or i > largest_valid_index
    ]

    if invalid_index:
        raise HTTPException(
            400, f"node index out of bound ( must be 0 to {largest_valid_index})"
        )

    return run_model(
        cora_dataset.x.numpy(), cora_dataset.edge_index.numpy(), request.node_indices
    )


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

