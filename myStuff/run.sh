#!/bin/bash

# two variables you need to set
pdnndir=/home/mclaughlin6464/GitRepos/pdnn  # pointer to PDNN
device=cpu #gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,blas.ldflags=-libsci

# split the dataset to training, validation and testing sets
# you will see train.pickle.gz, valid.pickle.gz, test.pickle.gz
echo "Preparing datasets ..."
#FYI Uses a lot of RAM to do the loading/splittling
python data_prep.py

# train DNN model
echo "Training the DNN model ..."
python $pdnndir/cmds/run_DNN.py --train-data "milliTrain.pickle.gz" \
                                --valid-data "milliValid.pickle.gz" \
                                --nnet-spec "193:100:6" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:.1:100" --model-save-step 20 \
                                --param-output-file dnn.param --cfg-output-file dnn.cfg  --regression 1
                                #>& dnn.training.log

# classification on the testing data; -1 means the final layer, that is, the classification softmax layer
echo "Classifying with the DNN model ..."
python $pdnndir/cmds/run_Extract_Feats.py --data "milliTest.pickle.gz" \
                                          --nnet-param dnn.param --nnet-cfg dnn.cfg \
                                          --output-file "dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >& dnn.testing.log

python show_results.py dnn.classify.pickle.gz
