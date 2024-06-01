import numpy as np

def min_ade(predictions, ground_truth):
    ade = np.linalg.norm(predictions - ground_truth, axis=-1).mean(axis=-1)
    min_ade = np.min(ade)
    return min_ade

def min_fde(predictions, ground_truth):
    fde = np.linalg.norm(predictions[:, -1] - ground_truth[:, -1], axis=-1)
    min_fde = np.min(fde)
    return min_fde

def miss_rate(predictions, ground_truth, threshold=2.0):
    distances = np.linalg.norm(predictions - ground_truth, axis=-1)
    min_distances = np.min(distances, axis=-1)
    miss_rate = np.mean(min_distances > threshold)
    return miss_rate

def brier_min_fde(predictions, ground_truth, probabilities):
    fde = np.linalg.norm(predictions[:, -1] - ground_truth[:, -1], axis=-1)
    min_fde = np.min(fde)
    brier_score = (1.0 - probabilities) ** 2
    return min_fde + brier_score.mean()

def brier_min_ade(predictions, ground_truth, probabilities):
    ade = np.linalg.norm(predictions - ground_truth, axis=-1).mean(axis=-1)
    min_ade = np.min(ade)
    brier_score = (1.0 - probabilities) ** 2
    return min_ade + brier_score.mean()

def drivable_area_compliance(predictions, drivable_area_mask):
    compliance = (predictions[:, :, None] == drivable_area_mask[None, None, :]).all(axis=-1).mean()
    return compliance


