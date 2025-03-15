import argparse
from transformers import pipeline
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(
    description='Script to transcribe a custom audio file of any length using Whisper Models.')
parser.add_argument(
    "--is_public_repo",
    required=False,
    default=True,
    type=lambda x: (str(x).lower() == 'true'),
    help="If the model is available for download on Huggingface.",
)
parser.add_argument(
    "--hf_model",
    type=str,
    required=False,
    default="openai/whisper-tiny",
    help="Huggingface model name. Example: openai/whisper-tiny",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    required=False,
    default=".",
    help="Folder with the model checkpoint files.",
)
parser.add_argument(
    "--temp_ckpt_folder",
    type=str,
    required=False,
    default="temp_dir",
    help="Temporary folder to hold the model files needed for inference",
)
parser.add_argument(
    "--path_to_audio",
    type=str,
    required=True,
    help="Path to the audio file to be transcribed.",
)
parser.add_argument(
    "--language",
    type=str,
    required=False,
    default="hi",
    help="Two letter language code for the transcription language (e.g., 'hi' for Hindi).",
)
parser.add_argument(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The device to run the pipeline on (-1 for CPU, 0 for GPU, etc.).",
)

args = parser.parse_args()

if not args.is_public_repo:
    os.makedirs(args.temp_ckpt_folder, exist_ok=True)
    # Parent folder should contain the tokenizer and preprocessing files
    ckpt_parent = str(Path(args.ckpt_dir).parent)
    files_to_copy = [
        "added_tokens.json",
        "normalizer.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    ]
    for filename in files_to_copy:
        src = os.path.join(ckpt_parent, filename)
        dst = os.path.join(args.temp_ckpt_folder, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found.")

    # Copy checkpoint-specific files. Notice we copy safetensors files instead of pytorch_model.bin.
    ckpt_files = [
        "config.json",
        "training_args.bin",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        # Optionally, include other files if needed:
        "scaler.pt",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json"
    ]
    for filename in ckpt_files:
        src = os.path.join(args.ckpt_dir, filename)
        dst = os.path.join(args.temp_ckpt_folder, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found.")

    model_id = args.temp_ckpt_folder
else:
    model_id = args.hf_model

# Create the transcription pipeline
transcribe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    chunk_length_s=30,
    device=args.device,
)

# Set forced decoder ids for the target language
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(
    language=args.language, task="transcribe"
)

print('Transcription:')
print(transcribe(args.path_to_audio)["text"])

# Clean up the temporary folder if used
if not args.is_public_repo:
    shutil.rmtree(args.temp_ckpt_folder)
