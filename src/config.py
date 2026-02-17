# ================================
# CONFIGURATION FILE
# ================================

# Data
DATA_PATH = "../data/nonlinear_sudden_rollingtorus_noise_and_redunce.csv"

# Streaming
WINDOW_SIZE = 2000

# Drift Detection
ADWIN_DELTA = 0.0005   # tuned version from notebook

# Retraining
RETRAIN_MODE = "reset"  # options: "reset"

# Logging
VERBOSE = True
