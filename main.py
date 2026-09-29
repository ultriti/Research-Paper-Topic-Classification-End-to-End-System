from fastapi import FastAPI, HTTPExeception
import os
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from onnxruntime import ort

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
MODEL_PATH = os.path.join(BASE_DIR, "simple_gcn_cora.oonx")
DATA_DIR = os.path.join(BASE_DIR, "data", "Planetoid")


# load the trained model - create a lifetime model - model runns until its closed manually
model_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


app = FastAPI()

"""
create schema for the objects
"""


# graph predict request schema - user input val
def GraphPredictRequest(BaseModel):

    # a list of nodes where each node is list of 1433 numbers ( features)

    node_feature: List[List(float)]

    # optional - wch node connected to wch shape [2,num of edges]
    edge_indices: Optional[List[List(int)]] = None


# node / RP Getting check coreect index
class CoraNodeRequest(BaseModel):
    # wch node ( by index number ) from real cora dataset to classify
    node_indices: List[int]


"""
helper funtion 
"""


# softmax -[ score - added_max ---------> exp_sc / exp_score.su m]
def Softmax(scores: np.nparray):
    # turns raw model scores ( logits ) to prababilities ( 0 - 1 )
    #  we will get max prob
    shifted = scores - scores.max(axis=1, keepdims=True)

    exp_scores = np.exp(shifted)
    # remove or short the high values

    return exp_scores / exp_scores.sum(axis=1, keepdims=1)


# run the model --------------------------------------
def run_model(
    node_features: np.nparray, edge_index: np.nparray, node_indices_to_return
):

    # run the onnnx model - only on graph we are given and return its predictions
    output = model_session.run(
        ["logits"],
        {
            "num_features": node_features.astype(np.float32),
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
                "logits": logits,
            }
        )

    return {
        "num_nodes": node_features.shape[0],
        "num_edges": edge_index.shape[1],
        "predictions": results,
    }

