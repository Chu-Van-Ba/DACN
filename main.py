from preprocess import load_dataset, visualize_skull_stripping
from models.kmeans_graphcut_model import run_kmeans_graphcut
from models.fcm_graphcut_model import run_fcm_graphcut
from models.multi_otsu_graphcut_model import run_otsu_graphcut
import os

os.makedirs('results', exist_ok=True)
images, masks, names = load_dataset(r'C:\Users\tbgbo\Downloads\combined_dataset', num_samples=1)

#visualize_skull_stripping(images, names, n=3)
#run_kmeans_graphcut(images, masks, k=4, visualize_idx=0)
#run_fcm_graphcut(images, masks, c=4, visualize_idx=0)
run_otsu_graphcut(images, masks, n_thresholds=2, visualize_idx=0)