#!/bin/bash -l
#SBATCH -J  train
#SBATCH -o train-%j.out
#SBATCH -p gpu3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -t 2-00:00:00

## Load Needed Modules
#module load cuda/cuda-8.0

GLOBAL_PATH='/media/pilab/ssd_data/test/DeepSF/';

datadir=$GLOBAL_PATH/datasets/D1_SimilarityReduction_dataset
outputdir=$GLOBAL_PATH/test/output_nb_layers_5_pyramid_nb_filters_14_13_12_11_10
echo "TRAINING"

nb_filters=14
nb_layers=5
fc_hidden=500

## Test Theano
THEANO_FLAGS=floatX=float32,device=cuda $GLOBAL_PATH/../miniconda2/envs/theano/bin/python2.7 $GLOBAL_PATH/training/training_main.py 15 $nb_filters $nb_layers nadam '6_10' $fc_hidden 30 50 3  $datadir $outputdir
echo "EVALUATING"
## Test Theano
THEANO_FLAGS=floatX=float32,device=cuda $GLOBAL_PATH/../miniconda2/envs/theano/bin/python2.7 $GLOBAL_PATH/training/predict_main.py  15 $nb_filters $nb_layers nadam '6_10' $fc_hidden 30 50 3  $datadir $outputdir
