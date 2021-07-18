### Data folder:

This folder will contain all data files for training, development and testing, after running the commands:

```
python3 create_MNIST_train_dev_test.py --seed 20
python3 create_DIGIT_PAIR.py --mnist_seed 20 --single_size <DIGIT_SIZE> --operator_train_size <PAIR_SIZE> --operator_test_size 50000 --option 2 --seed 20 
python3 create_properties_val_test_datasets.py --mnist_seed 20 --associativity_size 50000 --commutativity_size 50000 --distributivity_size 50000 --seed 20
```
