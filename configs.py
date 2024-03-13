# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter_
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
MODEL_DEVICE = "cuda:0"
MODEL_NAME = 'resnet50' # 'resnet50' 'seresnet50' 'densenet196'
MODEL_LAST_STRIDE = 1
MODEL_PRETRAIN_CHOICE = "imagenet" # imagenet for imagenet checkpoint and self for resume training
#ResNet50 Pretrained Model Path, eg "/home/gutianpei/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"
MODEL_PRETRAIN_PATH = 'resnet50-0676ba61.pth'
MODEL_PRETRAIN_PATH_SE = ''
MODEL_PRETRAIN_PATH_DENSE = ''

# -----------------------------------------------------------------------------
# APNet
# -----------------------------------------------------------------------------
APNET_LEVEL = 2
APNET_MSMT = False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
# Size of the image during training
INPUT_SIZE_TRAIN = [384, 192]
# Size of the image during test
INPUT_SIZE_TEST = [384, 192]
# Random probability for image horizontal flip
INPUT_PROB = 0.5
# Values to be used for image normalization
INPUT_PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
INPUT_PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
INPUT_PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# List of the dataset names for training, as present in pathsatalog_py
DATASETS_NAMES = ('dukemtmc') #select from "dukemtmc", "market1501" and "msmt17"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
# Number of data loading threads
DATALOADER_NUM_WORKERS = 4
# Sampler for data loading
DATALOADER_SAMPLER = 'softmax_triplet'
# Number of instance for one batch
DATALOADER_NUM_INSTANCE = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
SOLVER_OPTIMIZER_NAME = "Adam"

SOLVER_MAX_EPOCHS = 50

SOLVER_BASE_LR = 0.0004
SOLVER_BIAS_LR_FACTOR = 1

SOLVER_MOMENTUM = 0.9

SOLVER_MARGIN = 0.3

SOLVER_SMOOTH = 0.1
SOLVER_CLASSNUM = 751

# Learning rate of SGD to learn the centers of center loss
SOLVER_CENTER_LR = 0.5
# Balanced weight of center loss
SOLVER_CENTER_LOSS_WEIGHT = 0.0005

SOLVER_WEIGHT_DECAY = 0.001
SOLVER_WEIGHT_DECAY_BIAS = 0.001

SOLVER_GAMMA = 0.1
SOLVER_STEPS = [40, 80, 120, 160]

SOLVER_WARMUP_FACTOR = 0.01
SOLVER_WARMUP_ITERS = 10
SOLVER_WARMUP_METHOD = "linear"

SOLVER_CHECKPOINT_PERIOD = 1
SOLVER_LOG_PERIOD = 10
SOLVER_EVAL_PERIOD = 10
SOLVER_FINETUNE = False

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
SOLVER_IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
TEST_IMS_PER_BATCH = 128
TEST_RE_RANK = False
TEST_WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
OUTPUT_DIR = "."
