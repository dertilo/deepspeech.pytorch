### singularity image

* locally build singularity image

    sudo singularity build --writable deepspeech.simg container.def
* copy to hpc

    scp /home/tilo/code/SPEECH/deepspeech.pytorch/deepspeech.simg tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/