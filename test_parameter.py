INPUT_DIM = 4
EMBEDDING_DIM = 128
SAMPLE_SIZE = 300
K_SIZE = 20

USE_GPU = False
USE_GPU_GLOBAL = False
NUM_GPU = 1
NUM_META_AGENT = 1 #6
FOLDER_NAME = 'ae_wo_pe'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/trajectory'
length_path = f'results/length'

SAVE_GIFS = True
SAVE_TRAJECTORY = False
SAVE_LENGTH = False
DIJKSTRA_PATH = False
