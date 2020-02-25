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

    python -m multiproc train.py --log-dir tensorboard_logdir/libri_full --hidden-layers 5 --opt-level O1 --loss-scale 1 --id libri_full --checkpoint --save-folder librispeech_save/full --model-path librispeech_models/libri_full_final.pth
    
    python -m multiproc train.py --train-manifest ~/data/asr_data/SPANISH/spanish_train_some.csv --val-manifest ~/data/asr_data/SPANISH/spanish_train_some.csv --hidden-layers 2 --opt-level O1 --loss-scale 1 --id spanish --checkpoint --save-folder librispeech_save/2layers_spanish --model-path librispeech_models/deepspeech_2layers_spanish_final.pth
    --log-dir tensorboard_logdir/spanish
    
#### evaluation
    python test.py --model-path librispeech_pretrained_v2.pth --test-manifest data/libri_test_clean.csv --cuda --half

#### transcribing 
    python transcribe.py --model-path librispeech_models/libri_full_final.pth --audio-path LibriSpeech_dataset/train/wav/4133-6541-0035.wav
    
# RUNS
30_000 shortest examples, ran on gpu016, got 61% WER

librispeech-clean-100

    python -m multiproc train.py --train-manifest libri_train100_manifest.csv --log-dir tensorboard_logdir/libri_100 --hidden-layers 5 --opt-level O1 --loss-scale 1 --id libri_100 --checkpoint --save-folder librispeech_save/100 --model-path librispeech_models/libri_100_final.pth

librispeech-clean-100 new on gpu006

python -m multiproc train.py --train-manifest libri_train100_manifest.csv --log-dir tensorboard_logdir/libri_100_new --hidden-layers 5 --opt-level O1 --loss-scale 1 --id libri_100_new --checkpoint --save-folder librispeech_save/100_new
python -m multiproc train.py --continue-from librispeech_save/100_new/deepspeech_1.pth.tar --train-manifest libri_train100_manifest.csv --log-dir tensorboard_logdir/libri_100_new --hidden-layers 5 --opt-level O1 --loss-scale 1 --id libri_100_new --checkpoint --save-folder librispeech_save/100_new

for debug
    
    python -m multiproc train.py --log-dir tensorboard_logdir/debug --train-manifest libri_train_manifest_some.csv --val-manifest libri_train_manifest_some.csv --hidden-layers 2 --opt-level O1 --loss-scale 1 --id debug --checkpoint --save-folder librispeech_save/debug --model-path librispeech_models/deepspeech_debug.pth
    python train.py --log-dir tensorboard_logdir/debug --train-manifest libri_train_manifest_some.csv --val-manifest libri_train_manifest_some.csv --hidden-layers 2 --opt-level O1 --loss-scale 1 --id debug --checkpoint --save-folder librispeech_save/debug --model-path librispeech_models/deepspeech_debug.pth

# TODO
* librispeech-clean-100 should lead to ~30% WER!! 
    -> original running on gpu006
* reproduce CER/WER on librispeech! running on gpu019
* loading audio with torchaudio + resampling to 16kHz + test it on librispeech!