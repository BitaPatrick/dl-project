# Configuration settings for the legal text decoder project

# Data
DATA_DIR = "/app/data"
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"
TEST_NEPTUN_CODES = {"EKGPBX", "TFH22P", "WFXBHI"}
TRAIN_CSV = f"{DATA_DIR}/final/train.csv"
TEST_CSV = f"{DATA_DIR}/final/test.csv"

# Classical ML baseline (scikit-learn)
VAL_SPLIT = 0.1
MAX_FEATURES = 25000
NGRAM_RANGE = (1, 2)
LOG_REG_C = 2.0
LOG_REG_MAX_ITER = 1000
MIN_DF = 2
RANDOM_STATE = 42

# Model save path (for the sklearn pipeline)
MODEL_SAVE_PATH = "/app/model.joblib"

# Transformer-based model (Hugging Face)
HF_MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
# Shorter sequences to speed up training on CPU/MPS
HF_MAX_LENGTH = 128
HF_BATCH_SIZE = 8
HF_LR = 3e-5
# Production run: 4 epochs (CPU/MPS ~2h)
HF_EPOCHS = 3
HF_WARMUP_STEPS = 0
HF_WEIGHT_DECAY = 0.01
HF_GRAD_CLIP = 1.0
HF_MODEL_PATH = f"{DATA_DIR}/model.pt"

# Unused but kept for compatibility with earlier scripts
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
