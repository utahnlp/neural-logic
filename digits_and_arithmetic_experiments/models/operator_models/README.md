In the Pipelined Learning experiments after running the command to train the Sum and Product classifiers (with noisy labeled data): 

```
python3 operator_model_train.py --test False --data_seed <PAIR_SEED> --DIGIT_size <DIGIT_SIZE> --PAIR_size <NOISYPAIR_SIZE> --data_option 2 --PAIR_val_test_size 50000 --DigitModel_lr <Digit_MODEL_LR> --Digit_Optimizer <Digit_MODEL_OPTIMIZER> --Digit_nepochs <Digit_NEPOCH> --Digit_seed <Digit_SEED> --tnorm <TNORM> --arithmethic_operator <OPERATOR> --train_batch_size <BATCH_SIZE> --valid_batch_size 1024 --training_n_epochs <TRAIN_NEPOCHS> --training_optimizer <OPTIMIZER> --training_lr <LEARNING_RATE> --training_seed <SEED> --warm_nepochs -1
```
the best **Sum** and **Product** models are stored in this folder.
