from fastapi import FastAPI
import os

app = FastAPI()

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


'''
where this file lies so we can find the odel filw 
'''

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR,"simple_gcn_cora.oonx")
DATA_DIR = os.path.join(BASE_DIR,"data","Planetoid")


