HOME_DIR=""

# Data Parameters
#DATA_DIR="data/" #ASHIM
DATA_DIR ="/uusoc/scratch/bluefish/mattiamg/ijcai2021/data" #MATTIA
#DATA_DIR = "data/"

SINGLE_MNIST_FILE="single_mnist_size_{}_seed_{}.pt"
OPERATOR_MNIST_FILE="option_{}_mnist_size_{}_operator_size_{}_{}_seed_{}.pt"
# For experiments sept
#DATA_DIR_SEPT = "/uusoc/scratch/bluefish/mattiamg/sept_exp/data"
NOISY_LABELED_PAIR="noisy_labeled_pair_data_option_{}_size_{}_seed_{}_with_single_train_size_{}_lr_{}_optim_{}_tnorm_{}_seed_{}_epochs_{}.pt"

ASSOCIATIVITY_MNIST_VAL="associativity_val_size_{}_seed_{}.pt"
COMMUTATIVITY_MNIST_VAL="commutativity_val_size_{}_seed_{}.pt"
DISTRIBUTIVITY_MNIST_VAL="distributivity_val_size_{}_seed_{}.pt"

ASSOCIATIVITY_MNIST_TEST="associativity_test_size_{}_seed_{}.pt"
COMMUTATIVITY_MNIST_TEST="commutativity_test_size_{}_seed_{}.pt"
DISTRIBUTIVITY_MNIST_TEST="distributivity_test_size_{}_seed_{}.pt"

#Model related
#MODEL_DIR="/uusoc/scratch/bluefish/mattiamg/demo_data/models"
MODEL_DIR = "/uusoc/scratch/bluefish/mattiamg/ijcai2021/models"
#MODEL_DIR = "models/"

JOINT_MODEL_DIR = "/uusoc/scratch/bluefish/mattiamg/ijcai2021/models/joint_models"
#JOINT_MODEL_DIR = "models/joint_models"
JOINT_DIGIT_MODEL_NAME = "digit_models/joint_digit_option_{}_digit_size_{}_pair_size_{}_lr_{}_optim_{}_tnorm_{}_weight_{}_seed_{}_epochs_{}.pt"
JOINT_SUM_MODEL_NAME = "sum_models/joint_sum_option_{}_digit_size_{}_pair_size_{}_lr_{}_optim_{}_tnorm_{}_weight_{}_seed_{}_epochs_{}.pt"
JOINT_PROD_MODEL_NAME = "prod_models/joint_prod_option_{}_digit_size_{}_pair_size_{}_lr_{}_optim_{}_tnorm_{}_weight_{}_seed_{}_epochs_{}.pt"
SINGLE_MODEL_NAME="single_models/single_train_size_{}_lr_{}_optim_{}_tnorm_{}_seed_{}_epochs_{}.pt"
SUM_MODEL_NAME="operator_models/option_{}_mnist_{}_sum_size_{}_lr_{}_optim_{}_tnorm_{}_seed_{}_epochs_{}.pt"
PRODUCT_MODEL_NAME="operator_models/option_{}_mnist_{}_product_size_{}_lr_{}_optim_{}_tnorm_{}_seed_{}_epochs_{}.pt"
