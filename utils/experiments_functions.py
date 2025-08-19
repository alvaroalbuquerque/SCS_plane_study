import numpy as np
import ordpy
from features_extractor.entropy_complexity import get_entropy_fisher, get_weighted_entropy_fisher
from features_extractor.entropy_complexity_correct import get_weighted_entropy_complexity, get_entropy_complexity, get_weighted_fisher_shannon
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def get_img_info(img, q_linspace = np.linspace(-2, 20, num = 221)):
    data = []
    for q in q_linspace:
        # print(f"q = {q}")
        ord_dis = ordpy.weighted_smoothness_structure(img, q=q)
        probs = ordpy.weighted_smoothness_structure_probs(img, q=q)
        w_ent_correct, w_comp_correct = get_weighted_entropy_complexity(img, 2, 2, 1, 1, q)
        data.append([q, ord_dis[0], ord_dis[1], w_ent_correct, w_comp_correct, probs[0], probs[1], probs[2]])

    data = pd.DataFrame(data, columns = ["q", "Smoothness", "Curve structure", "Weighted Entropy Correct", "Weighted Complexity Correct", "prob0", "prob1", "prob2"])
    return data

def get_img_info_simple(img, dx=2, dy=2):
    data = []
    
    ord_dis = ordpy.weighted_smoothness_structure(img, q=0)
    w_ent_correct, w_comp_correct = get_weighted_entropy_complexity(img, dx, dy, 1, 1, q=0)
    h, f = get_weighted_fisher_shannon(img, dx, dy, 1, 1, q=0)
    data.append([ord_dis[0], ord_dis[1], w_ent_correct, w_comp_correct, h, f])

    data = pd.DataFrame(data, columns = ["Smoothness", "Curve structure", "Entropy", "Complexity", "Shannon Entropy", "Fisher-Shannon Complexity"])
    return data