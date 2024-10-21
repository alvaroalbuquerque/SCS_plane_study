import numpy as np
import ordpy
from features_extractor.entropy_complexity import get_entropy_fisher, get_weighted_entropy_fisher
from features_extractor.entropy_complexity_correct import get_weighted_entropy_complexity, get_entropy_complexity
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