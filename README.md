 Train

 ./bin/nn --train --learning_rate 0.01 --epochs 10000 --batch_number 10 --dataset datasets/CC_train.csv --layers 30,60,10,1 -s 1 --model model.m --verbose

Test

 ./bin/nn --test  --dataset datasets/CC_test.csv --model model.m --verbose


