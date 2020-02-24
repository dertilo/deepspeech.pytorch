### singularity image

* locally build singularity image

        sudo singularity build deepspeech.sif container.def

* copy to hpc

    scp /home/tilo/code/SPEECH/deepspeech.pytorch/deepspeech.simg tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/

### on hpc
    module load singularity/2.5.2
    singularity shell --nv deepspeech.sif
#### single gpu     
    python train.py --rnn-type lstm --hidden-size 1024 --hidden-layers 5  --train-manifest libri_train_manifest.csv --val-manifest libri_val_manifest.csv --epochs 60 --num-workers 16 --cuda  --learning-anneal 1.01 --batch-size 64 --no-sortaGrad --tensorboard  --log-dir tensorboard_logdir --opt-level O1 --loss-scale 1 --id libri --checkpoint --save-folder librispeech_save/ --model-path librispeech/deepspeech_final.pth
#### multi gpu

    python -m multiproc train.py --rnn-type lstm --hidden-size 1024 --hidden-layers 5  --train-manifest libri_train_manifest.csv --val-manifest libri_val_manifest.csv --epochs 60 --num-workers 16 --cuda  --learning-anneal 1.01 --batch-size 64 --no-sortaGrad --tensorboard  --log-dir tensorboard_logdir --opt-level O1 --loss-scale 1 --id libri_multi --checkpoint --save-folder librispeech_save/ --model-path librispeech/deepspeech_final.pth

for debug
    
    python -m multiproc train.py --train-manifest libri_train_manifest_some.csv --val-manifest libri_val_manifest_some.csv --hidden-layers 2 --opt-level O1 --loss-scale 1 --id libri_multi_2layers --checkpoint --save-folder librispeech_save/2layers --model-path librispeech_models/deepspeech_2layers_final.pth
    
    
# TODO
on gpu019
    AttributeError: 'DistributedDataParallel' object has no attribute 'version'