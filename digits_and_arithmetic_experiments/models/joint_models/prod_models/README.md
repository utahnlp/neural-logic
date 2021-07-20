After the joint training by running:

```
python3 training_jointly.py --test <TEST> --DIGIT_size <DIGIT_SIZE> --PAIR_size <PAIR_SIZE> --data_option 2 --data_seed 20 --Dev_Test_size 50000 --train_batch_size <TRAIN_BATCH_SIZE> --validation_batch_size 1024 --tnorm <TNORM> --nepochs <NEPOCHS> --seed <SEED> --learning_rate <LEARNING_RATE> --optimizer <OPTIMIZER> --lambda_coef <LAMBDA> --warm_up_epochs <WARM_UP_EPOCHS> --Godel_Optim <GODEL_OPTIM> --Godel_lambda <GODEL_LAMBDA> --Godel_lr <GODEL_LR>
```

the best **Product** model is stored in this folder.
