import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

def min_max_normalize(data):
     
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val < 1e-12:
        return np.zeros_like(data)    
    return (data - min_val) / (max_val - min_val)
 
pred_path = "./CelebA_new/h_pred_uniform.pt"     
true_path = "./CelebA_new/h_ori_uniform.pt"     

pred = torch.load(pred_path)   
true = torch.load(true_path)

 
pred_np = pred.detach().cpu().numpy()   
true_np = true.detach().cpu().numpy() 


# ==================== 2. normalized [0,1] ====================
pred_norm = min_max_normalize(pred_np)
true_norm = min_max_normalize(true_np)

# Mean Absolute Error 
abs_error = np.abs(pred_norm - true_norm)                      
mae = np.mean(abs_error)                                 

# related err: |pred - true| / |true| 
           
numerator = np.linalg.norm(pred_norm - true_norm, ord=2)
denominator = np.linalg.norm(true_norm, ord=2)
mean_relative_error = numerator / denominator
 

# ==================== 3.   Pearson   Spearman  ====================
pearson_r, pearson_p = pearsonr(pred_np, true_np)
spearman_r, spearman_p = spearmanr(pred_np, true_np)

# ==================== 4. input results ====================
print("========   ========")
print(f"  (MAE)      : {mae:.6f}")
print(f" (RelErr)      : {mean_relative_error:.6%}")   
print("\n======== ========")
print(f"Pearson : {pearson_r:.6f}  (p-value: {pearson_p:.4e})")
print(f"Spearman : {spearman_r:.6f}  (p-value: {spearman_p:.4e})")

