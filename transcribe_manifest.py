import argparse
import gzip
import warnings
from opts import add_decoder_args, add_inference_args
from transcribe import transcribe
from utils import load_model

warnings.simplefilter("ignore")

from decoder import GreedyDecoder


import torch

from data.data_loader import SpectrogramParser
import os.path
import json


def read_lines(file, mode="b", encoding="utf-8", limit=None):
    assert any([mode == m for m in ["b", "t"]])
    counter = 0
    with gzip.open(file, mode="r" + mode) if file.endswith(".gz") else open(
        file, mode="r" + mode
    ) as f:
        for line in f:
            counter += 1
            if limit and (counter > limit):
                break
            if "b" in mode:
                line = line.decode(encoding)
            yield line.replace("\n", "")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="DeepSpeech transcription")
    arg_parser = add_inference_args(arg_parser)
    arg_parser.add_argument("--manifest")
    arg_parser.add_argument(
        "--offsets",
        dest="offsets",
        action="store_true",
        help="Returns time offset information",
    )
    arg_parser = add_decoder_args(arg_parser)
    args = arg_parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(
            model.labels,
            lm_path=args.lm_path,
            alpha=args.alpha,
            beta=args.beta,
            cutoff_top_n=args.cutoff_top_n,
            cutoff_prob=args.cutoff_prob,
            beam_width=args.beam_width,
            num_processes=args.lm_workers,
        )
    else:
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index("_"))

    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

    def do_transcribe(audio_file):
        decoded_output, decoded_offsets = transcribe(
            audio_path=audio_file,
            spect_parser=spect_parser,
            model=model,
            decoder=decoder,
            device=device,
            use_half=args.half,
        )
        candidate_idx = 0
        return [x[candidate_idx] for x in decoded_output][0]

    examples_g = (
        l.replace(
            "/beegfs/home/users/t/tilo-himmelsbach/SPEECH/deepspeech.pytorch/", ""
        ).split(",")
        for l in read_lines(args.manifest, limit=10)
    )
    g = (
        (do_transcribe(audio_file), next(iter(read_lines(text_file))))
        for audio_file, text_file in examples_g
    )

    for d in g:
        print(d)
