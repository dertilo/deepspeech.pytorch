import json
import multiprocessing
import os
import subprocess
import gzip

from tqdm import tqdm

from utils import create_manifest

SAMPLE_RATE = 16_000


def read_jsonl(file, mode="b", limit=None, num_to_skip=0):
    assert any([mode == m for m in ["b", "t"]])
    with gzip.open(file, mode="r" + mode) if file.endswith(".gz") else open(
        file, mode="rb"
    ) as f:
        [next(f) for _ in range(num_to_skip)]
        for k, line in enumerate(f):
            if limit and (k >= limit):
                break
            yield json.loads(line.decode("utf-8") if mode == "b" else line)


def _preprocess_transcript(phrase):
    return phrase.strip().lower()


def process_example(wav_dir, txt_dir, audio_file, text):
    audio_file_name = audio_file.split("/")[-1]
    audio_file_name_no_suffix = os.path.splitext(audio_file_name)[0]
    wav_recording_path = os.path.join(wav_dir, audio_file_name_no_suffix + ".wav")
    subprocess.call(
        [
            "sox {}  -r {} -b 16 -c 1 {}".format(
                audio_file, str(SAMPLE_RATE), wav_recording_path
            )
        ],
        shell=True,
    )
    # process transcript
    transcript_file = os.path.join(txt_dir, audio_file_name_no_suffix + ".txt")

    with open(transcript_file, "wb") as f:
        f.write(_preprocess_transcript(text).encode("utf-8"))
        f.flush()


if __name__ == "__main__":
    from pathlib import Path

    base_path = os.path.join(os.environ["HOME"], "data/asr_data/SPANISH")

    split_type = "train"
    wav_dir = os.path.join(base_path, split_type, "wav")
    txt_dir = os.path.join(base_path, split_type, "txt")
    Path(wav_dir).mkdir(parents=True, exist_ok=True)
    Path(txt_dir).mkdir(parents=True, exist_ok=True)

    def funfun(audio_file_text):
        audio_file, text = audio_file_text
        audio_file = base_path + audio_file
        process_example(wav_dir, txt_dir, audio_file, text)

    with multiprocessing.Pool(processes=50) as p:
        result = list(
            p.imap_unordered(
                funfun, tqdm(read_jsonl(base_path + "/spanish_%s.jsonl" % split_type)),
            )
        )

    create_manifest(wav_dir, "spanish_" + split_type + "_manifest.csv", 1, 30)
