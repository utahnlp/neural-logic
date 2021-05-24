
# Creating Main Datasets

### TRAIN, DEV, TEST datasets from MNIST [LeCun and Cortes, 2010]

<i> Run (experiments in the paper use seed 20): </i>

```
python3 create_MNIST_train_dev_test.py --seed 20
```

### DIGIT, PAIR, PairDEV, PairTEST (Note: PairDEV, PairTEST are both unique sets created (the same pair of sets) on every run for convenience)

### Dataset combinations sizes:
* <DIGIT_SIZE>: 1000, 5000, 25000
* <PAIR_SIZE>: 1000, 5000, 25000

### Other input parameters:
* operator_test_size: This is the size of PairDEV, PairTEST; we used 50,000 size for both of them. 
* option: 2 (We did not consider other data option for this paper)
* seed: We used seed 20

<i> Run (experiments in the paper use seed 20): </i>

```
python3 create_DIGIT_PAIR.py --mnist_seed 20 --single_size <DIGIT_SIZE> --operator_train_size <PAIR_SIZE> --operator_test_size 50000 --option 2 --seed 20 
```

### Properties development and test datasets. To evaluate Commutativity, Associativity, Distributivity

```
python3 create_properties_val_test_datasets.py --mnist_seed 20 --associativity_size 50000 --commutativity_size 50000 --distributivity_size 50000 --seed 20
```

# Joint Learning

### We run experiments for each logic relaxation: <TNORM> 

* Lukasiewicz
* Gödel
* S-Product
* R-Product


### for each dataset combinations sizes:
* <DIGIT_SIZE>: 1000, 5000, 25000
* <PAIR_SIZE>: 1000, 5000, 25000

### For model tuning we perform grid search over the following hyper-parameters using PairDEV set for development:
* <TRAIN_BATCH_SIZE>: 8, 16, 32, 64
* <LEARNING_RATE>: 10^−1, 5×10^−2, 10^−2, 5×10^−3, 10^−3, 5×10^−4, 10^−4, 5×10^−5, 10^−5
* \<OPTIMIZER>: 0 (SGD), 1 (Adam)
* \<LAMBDA>: 0.05, 0.1, 0.5, 1, 1.5, 2 - Lambda coeficient for Coherence constraints.
* \<NEPOCHS>: We trained each hyper-paramenter combination for 300 epochs to find the best one.
* \<SEED>: 20 

### We run the experiments three times using different seeds for each data combination with their correspondent best hyperparameters combinations from model  tuning process. we store the models from the epoch with best average development accuracy between Digit accuracy, Product Coherence accuracy and Sum Coherence accuracy:

* \<SEED>: 0, 20, 50 (extra runs in some cases with 1, 10, 60)
* \<NEPOCHS>: 1600 (We ran 1600 epochs for all reported experiments, some configurations needed this number of epochs to converge)
* \<TNORM>: t-norm in use. 

### Using Gödel t-norm: to make it learn, we "warmed-up" the system by initially by running (1 or 2) epochs of learning using S-Product t-norm.

* <WARM_UP_EPOCHS>: 1, 2 - The number of warm-up epochs. This process will use the hyper-parameters \<LEARNING_RATE>, \<OPTIMIZER> and \<LAMBDA> training with S-Product t-norm for <WARM_UP_EPOCHS> epochs. For these three parameters we used the same ones we obtained from the S-Product hyperparamenter tunning.   
* <GODEL_OPTIM>: 0, 1 - The optimizer that will be used for training with Gödel t-norm after the warming-up process.
* <GODEL_LAMBDA>: 0.05, 0.1, 0.5, 1, 1.5, 2 - Lambda coeficient for Coherence constraints for training with Gödel t-norm after the warming-up process.
* <GODEL_LR>: 0.05, 0.1, 0.5, 1, 1.5, 2 - Learning rate for training with Gödel t-norm after the warming-up process.


### Remaining input parameters:

* \<TEST>: True to save the resulting models, False otherwise
* validation_batch_size: 1024 or the biggest size allowed by the system used
* data_option: always 2
* data_seed: 20 or the one used to create the datasets in the previous section



```
python3 training_jointly.py --test <TEST> --DIGIT_size <DIGIT_SIZE> --PAIR_size <PAIR_SIZE> --data_option 2 --data_seed 20 --Dev_Test_size 50000 --train_batch_size <TRAIN_BATCH_SIZE> --validation_batch_size 1024 --tnorm <TNORM> --nepochs <NEPOCHS> --seed <SEED> --learning_rate <LEARNING_RATE> --optimizer <OPTIMIZER> --lambda_coef <LAMBDA> --warm_up_epochs <WARM_UP_EPOCHS> --Godel_Optim <GODEL_OPTIM> --Godel_lambda <GODEL_LAMBDA> --Godel_lr <GODEL_LR>
```

# Pipelined Learning

<i> We ran experiments for each logic relaxation: \<TNORM> </i> 

* Lukasiewicz
* Gödel
* S-Product (Same as R-Product when there are not constraints)

### We train the Digit model with DIGIT data alone using each t-norm.

#### Sizes of datasets combinations:
* <SIZE_DATA>: 1000, 5000, 25000


#### For Digit model tunning: we performed grid search over the following hyper-parameters:

* \<BATCH_SIZE>: 8, 16, 32, 64
* \<LEARNING_RATE>: 10^−1, 5×10^−2, 10^−2, 5×10^−3, 10^−3, 5×10^−4, 10^−4, 5×10^−5, 10^−5
* \<OPTIMIZER>: 0 (SGD), 1 (Adam)
* \<NEPOCHS>: 300
* \<SEED>: 20


#### We train the model using the best hyperparameters for each t-norm and save the one from the epoch with best development accuracy.

```
python3 digit_model_train.py --size_data <SIZE_DATA> --data_seed <SEED> --learning_rate <LEARNING_RATE> --optimizer <OPTIMIZER> --batch_size <BATCH_SIZE> --nepochs <NEPOCHS> --seed <SEED> --tnorm <TNORM> --test True
```

### We created the (noisy) labeled datasets.

<i> We use the previous Digit models (trained alone) to re-label the PAIR data: For each example in PAIR ( [Im_1 - Label_1], [Im_2 - Label_2]) we ignore the labels (Label_1, Label_2) and consider the new noisy labeled pair ( [Im_1 - Digit(Im_1)],  [Im_2 - Digit(Im_2)] ) where Digit(Im_i) is the label predicted by the Digit model on image Im_i.</i>

#### This process is done to crate dataset to for all sizes combinations:
* \<DIGIT_SIZE>: 1000, 5000, 25000
* \<PAIR_SIZE>: 1000, 5000, 25000
* dev_test_size: Size of PairDEV and PairTEST. We set this size to 50,000
* option: 2
* \<LEARNING_RATE>: Learning rate that was used to train the Digit classifier that is being used to label the data.
* \<OPTIMIZER>: Optimizer that was used to train the Digit classifier that is being used to label the data; 1 Adam, 0 SGD.
* \<TNORM>: t-norm in use. The t-norm we used to train the Digit classifier used to label the data. The resulting data will be used to train the Sum and Prod classifiers using this same t-norm as parameter.
* \<SEED>: We use the fixed seed 20.
* \<NEPOCHS>: We used 1600 epochs. This models converge much earlier but we wanted to use the same number of epochs we used in the joint training method given that we are comparing them. 


```
python3 noisy_labeling.py --digit_size <DIGIT_SIZE> --pair_size <PAIR_SIZE> --dev_test_size 50000 --option 2 --ope_seed 20 --learning_rate <LEARNING_RATE> --optimizer <OPTIMIZER> --tnorm <TNORM> --model_seed <SEED> --nepochs <NEPOCHS>
```

### We train the Sum and Product operator models using the noisy labeled dataset from the previous section:

#### The following parameters of the program are used to find the right noisy labeled dataset to train the operator models
* \<PAIR_SEED>: Seed used to create the PAIR and Noisy labeled (NOISYPAIR) data. We used 20 in our experiments.
* \<DIGIT_SIZE>: 1000, 5000, 25000. Size of the DIGIT data used to train the Digit classifier used to label the NOISYPAIR data. 
* \<NOISYPAIR_SIZE>: 1000, 5000, 25000. Size of the NOISYPAIR set.
* data_option: always 2 
* PAIR_val_test_size: Size of the PairDEV and PairTEST sets. We use 50,000.
* \<Digit_MODEL_LR> :Learning rate used to train the Digit model used to label de NOISYPAIR data
* <Digit_MODEL_OPTIMIZER>: Optimizer used to train the Digit model used to label the NOISYPAIR data. 0 if SGD, 1 if Adam.
* <Digit_NEPOCH>: Number of epochs used to train the Digit model. We use 1600 epochs. 
* <Digit_SEED>: seed used to train the Digit model. We used 20.
* <TNORM>: t-norm in use.


#### For operator models tuning we perform grid search over the following hyper-parameters using PairDEV set for development:
* <TRAIN_BATCH_SIZE>: 8, 16, 32, 64
* <LEARNING_RATE>: 10^−1, 5×10^−2, 10^−2, 5×10^−3, 10^−3, 5×10^−4, 10^−4, 5×10^−5, 10^−5
* \<OPTIMIZER>: 0 (SGD), 1 (Adam)
* \<NEPOCHS>: We trained each hyper-paramenter combination for 300 epochs to find the best one.
* \<SEED>: 20


```
python3 operator_model_train.py --test False --data_seed <PAIR_SEED> --DIGIT_size <DIGIT_SIZE> --PAIR_size <NOISYPAIR_SIZE> --data_option 2 --PAIR_val_test_size 50000 --DigitModel_lr <Digit_MODEL_LR> --Digit_Optimizer <Digit_MODEL_OPTIMIZER> --Digit_nepochs <Digit_NEPOCH> --Digit_seed <Digit_SEED> --tnorm <TNORM> --arithmethic_operator sum --train_batch_size 64 --valid_batch_size 1024 --training_n_epochs 20 --training_optimizer 0 --training_lr 0.005 --training_seed 20 --warm_nepochs -1
```


# Testing

<i>We test the three models: Digit, Sum and Product. To get individual accuracies, single accuracies, coherence accuracies, and arithmetic properties: commutativity, associativity distributivity.</i>


#### To test we have to provide the paremeter description use to train each of the models we are testing:


* <TEST_BASELINE>: True if it the pipelined setting that its being tested. False if it the joint setting.
* <SEED_DIGIT>: Seed used to train the Digit model. (It should always be the same as the one used to train the operator models).
* <SEED_OPERATOR>: Seed used to train the Sum and Prod operator models. (It should be the same as the one used to train the digit model).
* <TNORM>: T-norm in use
* <DIGIT_SIZE>: Size of the DIGIT set used.
* <DIGIT_LR>:  Learning rate used to train the Digit model used to label de NOISYPAIR data.
* <DIGIT_OPTIM>: Optimizer used to train the Digit model used to label the NOISYPAIR data. 0 if SGD, 1 if Adam.
* <DIGIT_EPOCHS>: Number of epochs used to train the Digit classifier.
* <OPE_SIZE>: Size of the NOISYPAIR set.
* <DATA_OPTION>: Data option used during training. It should always be 2.
* <OPE_LR>: Learning rate used to train the Sum and Prod operator models.
* <OPE_OPTIM>: Optimizer used to train the Sum and Prod operator models. 0 if SGD, 1 if Adam.
* <OPE_EPOCHS>: Number of epochs used to train the Sum and Prod operator classifiers.
* <JOINT_LR>: Learning rate used for the joint learning process. (Only active if test_baseline is False)
* <JONT_OPTIM>: Optimizer used used for the joint learning process. (Only active if test_baseline is False). 0 if SGD, 1 if Adam.
* <LAMBDA>: Lambda coefficient used in the joint learning process (Only active if test_baseline is False).
* <JOINT_EPOCHS>: Number of epochs used to train the joint classifier.


```
python3 models_tester.py --test_baseline <TEST_BASELINE> --seed_digit <SEED_DIGIT> --seed_operator <SEED_OPERATOR> --tnorm <TNORM> --digit_size <DIGIT_SIZE> --digit_lr <DIGIT_LR> --digit_optim <DIGIT_OPTIM> --digit_epochs <DIGIT_EPOCHS> --ope_size <OPE_SIZE> --data_option <DATA_OPTION> --ope_lr <OPE_LR> --ope_optim <OPE_OPTIM> --ope_epochs <OPE_EPOCHS> --joint_lr <JOINT_LR> --joint_optim <JONT_OPTIM> --lamb <LAMBDA> --joint_epochs <JOINT_EPOCHS>

```