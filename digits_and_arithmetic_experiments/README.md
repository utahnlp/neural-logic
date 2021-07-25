
Here are the one by one steps to create the datasets, and to train and test all the models from the results reported in the paper.
All the created datasets will be directed to be stored inside the ```/data``` directory, and the trained models (from the best epoch) will be directed to be stored in the ```/models``` directory. 

<i>Note: Check the config_ijcai.py file to change the output paths where the datasets and models will be stored in your system.</i>

From the ```/code``` folder run the following commands:

## 1. Creating datasets

#### 1.1 Splits of the original MNIST
Note that the original MNIST dataset does not provide the `train-dev` splits, we use the following command to create `train, dev, test` splits from the original MNIST data. This creates a training set of size `50000` and a development set of size `10000`.
```
python3 create_MNIST_train_dev_test.py --seed 20
```

#### 1.2 Datasets for main experiments

We have two types of classifiers, ones for images containing single digits, and other ones for classifying a pair of digits, obtained from concatenation of two single digits. 
<!-- We use the name `DIGIT` to denote all the datasets and models using a single digit and `PAIR` for datasets/models using a pair of digits. -->

Additionally, to evaluate the effectiveness of t-norms in various data regimes, we further sample training sets of different sizes. This can be done using the following command:
<!-- You can use the following command to create DIGIT, PAIR, PairDEV, PairTEST: -->
```
python3 create_DIGIT_PAIR.py --mnist_seed 20 --single_size 1000 --operator_train_size 1000 --operator_test_size 50000 --option 2 --seed 20 
```
This creates a pickle file containing the following datasets:
```
DIGIT: Sub-sampled (single digit) dataset for training
PAIR: Creates examples containing (unlabeled) pair of digits for training the operator (Sum/Product) classifiers. This is created from the sub-sampled dataset: DIGIT.
PairDEV: The development set of (unlabeled) pair of images created from the MNIST development set (created in step 1.1).
PairTEST: The test set of a pair of images created from the MNIST test set (from step 1.1).
```

#### 1.3 Evaluation Sets
To evaluate Commutativity, Associativity, Distributivity, we create corresponding evaluation sets:

```
python3 create_properties_val_test_datasets.py --mnist_seed 20 --associativity_size 50000 --commutativity_size 50000 --distributivity_size 50000 --seed 20
```

Arguments description for 1.1, 1.2, 1.3:
```
--single_size : [1000, 5000, 25000] - the size of sub-sampled DIGIT training set.
--operator_train_size : [1000, 5000, 25000] - the size of the PAIR training set.
--operator_test_size: [50000] - size of PairDEV, PairTEST. 
--option: 2 - Please do not change this.
--seed: Seed value (20) for creating train, dev splits of original MNIST.
--mnist_seed: (same as --seed)
--associativity_size: Size of the Associativity evaluation set (Similarly for Commutativity, and Distributivity).
```

## 2. Joint Learning

We jointly train the Digit, Sum and Product classifiers (we use DIGIT data to train Digit and instantiate the coherence constraints on the (unlabeled) PAIR data to obtain signal to train Sum and Product).


We run experiments for each t-norm relaxation we are considering and for all dataset sizes combinations:
```
--tnorm: [prod (S-Product), rprod (R-Product), luka (Lukasiewicz), godel (Gödel)]
--DIGIT_size: [1000, 5000, 25000]
--PAIR_size: [ 1000, 5000, 25000]
```
 #### 2.1 Model tuning
 
For model tuning we perform grid search over the following hyper-parameters using PairDEV set for development:
 ```
--train_batch_size: [8, 16, 32, 64]
--learning_rate: [10^−1, 5×10^−2, 10^−2, 5×10^−3, 10^−3, 5×10^−4, 10^−4, 5×10^−5, 10^−5]
--optimizer: [0 (SGD), 1 (Adam)]
--lambda_coef [0.05, 0.1, 0.5, 1, 1.5, 2] - Lambda coeficients for Coherence constraints.
--nepochs: We trained each hyper-paramenter combination for 300 epochs to find the best one.
 ```

#### 2.2 Experiments

We run the experiments three times using different seeds for each data combination with their corresponding best hyperparameters combinations from model tuning process. We use the models from the epoch with best average development accuracy between Digit accuracy, Product Coherence accuracy and Sum Coherence accuracy:

 ```
--seed: 20 (dataset seed) & [0, 50](experiments seed): [0, 20, 50] (extra runs in some cases with 1, 10, 60)
--nepochs: [1600] (We ran 1600 epochs for all reported experiments, some configurations needed this number of epochs to converge)
```
 
- Using Gödel t-norm: to make it learn, we "warmed-up" the system by initially running (1 or 2) epochs of learning using S-Product t-norm.

```
--warm_up_epochs: [1, 2] - The number of warm-up epochs. This process will use the hyper-parameters <LEARNING_RATE>, <OPTIMIZER> and <LAMBDA> training with S-Product t-norm for <WARM_UP_EPOCHS> epochs. For these three parameters we used the same ones we obtained from the S-Product hyperparamenter tunning.   
--Godel_Optim: [0, 1] - The optimizer that will be used for training with Gödel t-norm after the warming-up process.
--Godel_lambda: [0.05, 0.1, 0.5, 1, 1.5, 2] - Lambda coeficient for Coherence constraints for training with Gödel t-norm after the warming-up process.
--Godel_lr: [0.05, 0.1, 0.5, 1, 1.5, 2] - Learning rate for training with Gödel t-norm after the warming-up process.
```

- Other remaining input parameters:

 ```
--test: True to save the resulting models, False otherwise
--validation_batch_size: 1024 or the biggest size allowed by the system used
--data_option: always 2
--data_seed: 20 or the one used to create the datasets in the previous section
```
 
```
python3 training_jointly.py --test False --DIGIT_size 5000 --PAIR_size 5000 --data_option 2 --data_seed 20 --Dev_Test_size 50000 --train_batch_size 64 --validation_batch_size 1024 --tnorm rprod --nepochs 1000 --seed 20 --learning_rate 10^−1 --optimizer 0 --lambda_coef 0.1 --warm_up_epochs 2 --Godel_Optim 0 --Godel_lambda 0.1  --Godel_lr 0.001
```

## Pipelined Learning

We run experiments for each t-norm relaxation we are considering and for all dataset sizes combinations:

```
--tnorm: [prod (S-Product), rprod (R-Product), luka (Lukasiewicz), godel (Gödel)]
--DIGIT_size: [1000, 5000, 25000]
--PAIR_size: [ 1000, 5000, 25000]

```
#### 3.1 Digit classifier training

We train the Digit model with DIGIT data alone using each t-norm.

Digit model tunning: we performed grid search over the following hyper-parameters:
```
--batch_size: 8, 16, 32, 64
--learning_rate: 10^−1, 5×10^−2, 10^−2, 5×10^−3, 10^−3, 5×10^−4, 10^−4, 5×10^−5, 10^−5
--optimizer: 0 (SGD), 1 (Adam)
--nepochs: 300
--seed: 20 (0, 50)
```
We train the model using the best hyperparameters for each t-norm and save the one from the epoch with best development accuracy.

```
python3 digit_model_train.py --size_data 5000 --data_seed 20 --learning_rate 0.001 --optimizer 1 --batch_size 64 --nepochs 250 --seed 20 --tnorm prod --test False
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
* \<SEED>: 20 (0, 50).
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
* <Digit_SEED>: seed used to train the Digit model. We used 0, 20, 50.
* <TNORM>: t-norm in use.
 


#### For operator models tuning we perform grid search over the following hyper-parameters using PairDEV set for development:
* \<TRAIN_BATCH_SIZE>: 8, 16, 32, 64
* \<LEARNING_RATE>: 10^−1, 5×10^−2, 10^−2, 5×10^−3, 10^−3, 5×10^−4, 10^−4, 5×10^−5, 10^−5
* \<OPTIMIZER>: 0 (SGD), 1 (Adam)
* \<NEPOCHS>: We trained each hyper-paramenter combination for 300 epochs to find the best one.
* \<SEED>: 0, 20, 50
* \<TRAIN_NEPOCHS>: 300 for hyper-parameter tuning.

#### These are the parameters used for training the operator (Sum or Product):
* \<OPERATOR>: sum or prod
* \<TRAIN_NEPOCHS>:  Number of epochs to train the models
* \<WARM_EPOCHS>: Only for Godel t-norm. This is the number of epochs we "warm-up" the system using the (s)product t-norm.



```
python3 operator_model_train.py --test False --data_seed <PAIR_SEED> --DIGIT_size <DIGIT_SIZE> --PAIR_size <NOISYPAIR_SIZE> --data_option 2 --PAIR_val_test_size 50000 --DigitModel_lr <Digit_MODEL_LR> --Digit_Optimizer <Digit_MODEL_OPTIMIZER> --Digit_nepochs <Digit_NEPOCH> --Digit_seed <Digit_SEED> --tnorm <TNORM> --arithmethic_operator <OPERATOR> --train_batch_size <BATCH_SIZE> --valid_batch_size 1024 --training_n_epochs <TRAIN_NEPOCHS> --training_optimizer <OPTIMIZER> --training_lr <LEARNING_RATE> --training_seed <SEED> --warm_nepochs -1
```


## 4. Testing

<i>We test the three models: Digit, Sum and Product obtained from a given data configuration, hyperparamenter setting, and learning protocol (joint or pipeline).</i>

The following command outputs the Digit classifier accuracy on the test set, Sum/Product classifiers accuracies on PairTEST, fraction of examples in PairTEST where the coherence constraints and the commutativity,associativity and distributivity properties are satisfied for Sum and Product.
 
 ```
python3 models_tester.py --test_baseline False --seed_digit 20 --seed_operator 20 --tnorm rprod --digit_size 5000 --digit_lr 0.001 --digit_optim 1 --digit_epochs 250 --ope_size 5000 --data_option 2 --ope_lr 0.1 --ope_optim 0 --ope_epochs 250 --joint_lr 0.001 --joint_optim 0 --lamb 0.05 --joint_epochs 250

```
 

Arguments description for 4:

```
 
--test_baseline: True if it the pipelined setting that its being tested. False if it the joint setting.
--seed_digit: Seed used to train the Digit model. (It should always be the same as the one used to train the operator models).
--seed_operator: Seed used to train the Sum and Prod operator models. (It should be the same as the one used to train the digit model).
--tnorm: t-norm in use
--digit_size: Size of the DIGIT set used.
--digit_lr:  Learning rate used to train the Digit model used to label de NOISYPAIR data.
--digit_optim: Optimizer used to train the Digit model used to label the NOISYPAIR data. 0 if SGD, 1 if Adam.
--digit_epochs: Number of epochs used to train the Digit classifier.
--ope_size: Size of the NOISYPAIR set.
--data_option: Data option used during training. It should always be 2.
--ope_lr: Learning rate used to train the Sum and Prod operator models.
--ope_optim: Optimizer used to train the Sum and Prod operator models. 0 if SGD, 1 if Adam.
--ope_epochs: Number of epochs used to train the Sum and Prod operator classifiers.
--joint_lr: Learning rate used for the joint learning process. (Only active if test_baseline is False)
--joint_optim: Optimizer used used for the joint learning process. (Only active if test_baseline is False). 0 if SGD, 1 if Adam.
--lamb: Lambda coefficient used in the joint learning process (Only active if test_baseline is False).
--joint_epochs: Number of epochs used to train the joint classifier.


```
 
 

