import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument(
    '--model_name', 
    type=str, 
    required=False, 
    default='openai/whisper-small', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
)
parser.add_argument(
    '--language', 
    type=str, 
    required=False, 
    default='Hindi', 
    help='Language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--sampling_rate', 
    type=int, 
    required=False, 
    default=16000, 
    help='Sampling rate of audios.'
)
parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=2, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
)
parser.add_argument(
    '--train_strategy', 
    type=str, 
    required=False, 
    default='steps', 
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    required=False, 
    default=1.75e-5, 
    help='Learning rate for the fine-tuning process.'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    required=False, 
    default=20000, 
    help='Number of warmup steps.'
)
parser.add_argument(
    '--train_batchsize', 
    type=int, 
    required=False, 
    default=48, 
    help='Batch size during the training phase.'
)
parser.add_argument(
    '--eval_batchsize', 
    type=int, 
    required=False, 
    default=32, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--grad_acc', 
    type=int, 
    required=False, 
    default=1, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    required=False, 
    default=None, 
    help='Number of epochs to train for.'
)
parser.add_argument(
    '--num_steps', 
    type=int, 
    required=False, 
    default=100000, 
    help='Number of steps to train for.'
)
parser.add_argument(
    '--eval_steps', 
    type=int, 
    required=False, 
    default=100, 
    help='Number of steps to evaluate at.'
)
parser.add_argument(
    '--resume_from_ckpt', 
    type=str, 
    required=False, 
    default=None, 
    help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    required=False, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--train_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for training.'
)
parser.add_argument(
    '--train_dataset_configs', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of training dataset configs. Eg. 'hi' for the Hindi part of Common Voice",
)
parser.add_argument(
    '--train_dataset_splits', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of training dataset splits. Eg. 'train' for the train split of Common Voice",
)
parser.add_argument(
    '--train_dataset_text_columns', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="Text column name of each training dataset. Eg. 'sentence' for Common Voice",
)
parser.add_argument(
    '--eval_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for evaluation.'
)
parser.add_argument(
    '--eval_dataset_configs', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of evaluation dataset configs. Eg. 'hi_in' for the Hindi part of Google Fleurs",
)
parser.add_argument(
    '--eval_dataset_splits', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of evaluation dataset splits. Eg. 'test' for the test split of Common Voice",
)
parser.add_argument(
    '--eval_dataset_text_columns', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="Text column name of each evaluation dataset. Eg. 'transcription' for Google Fleurs",
)
parser.add_argument(
    '--testmode',
    action='store_true'
)

args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')

if len(args.train_datasets) == 0:
    raise ValueError('No train dataset has been passed')
if len(args.eval_datasets) == 0:
    raise ValueError('No evaluation dataset has been passed')

if len(args.train_datasets) != len(args.train_dataset_configs):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_configs. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_configs)} for train_dataset_configs.")
if len(args.eval_datasets) != len(args.eval_dataset_configs):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_configs. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_configs)} for eval_dataset_configs.")

if len(args.train_datasets) != len(args.train_dataset_splits):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_splits. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_splits)} for train_dataset_splits.")
if len(args.eval_datasets) != len(args.eval_dataset_splits):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_splits. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_splits)} for eval_dataset_splits.")

if len(args.train_datasets) != len(args.train_dataset_text_columns):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_text_columns. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_text_columns)} for train_dataset_text_columns.")
if len(args.eval_datasets) != len(args.eval_dataset_text_columns):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_text_columns. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_text_columns)} for eval_dataset_text_columns.")

print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
print('ARGUMENTS OF INTEREST:')
print(vars(args))
print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()

#############################       MODEL LOADING       #####################################

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if gradient_checkpointing:
    model.config.use_cache = False

############################        DATASET LOADING AND PREP        ##########################

def load_all_datasets(split):    
    combined_dataset = []
    if split == 'train':
        for i, ds in enumerate(args.train_datasets):
            dataset = load_dataset(ds, args.train_dataset_configs[i], split=args.train_dataset_splits[i])
            dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
            if args.train_dataset_text_columns[i] != "sentence":
                dataset = dataset.rename_column(args.train_dataset_text_columns[i], "sentence")
            dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
            combined_dataset.append(dataset)
    elif split == 'eval':
        for i, ds in enumerate(args.eval_datasets):
            dataset = load_dataset(ds, args.eval_dataset_configs[i], split=args.eval_dataset_splits[i])
            dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
            if args.eval_dataset_text_columns[i] != "sentence":
                dataset = dataset.rename_column(args.eval_dataset_text_columns[i], "sentence")
            dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
            combined_dataset.append(dataset)
    
    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)
    return ds_to_return

# --- NEW: Batched prepare_dataset function ---
def prepare_dataset(batch):
    # Extract audio arrays and assume all samples in the batch share the same sampling rate
    arrays = [audio_item["array"] for audio_item in batch["audio"]]
    sampling_rates = [audio_item["sampling_rate"] for audio_item in batch["audio"]]
    sample_rate = sampling_rates[0]  # all audios are cast to args.sampling_rate

    # Compute log-Mel input features for the batch of audio arrays
    features = feature_extractor(arrays, sampling_rate=sample_rate)
    batch["input_features"] = features.input_features

    # Compute input lengths in seconds for each audio sample
    batch["input_length"] = [len(array) / sample_rate for array in arrays]
    # Process transcriptions in batch
    transcriptions = batch["sentence"]
    if do_lower_case:
        transcriptions = [t.lower() for t in transcriptions]
    if do_remove_punctuation:
        transcriptions = [normalizer(t).strip() for t in transcriptions]
    # Encode target text to label ids for the batch
    tokenized = tokenizer(transcriptions)
    batch["labels"] = tokenized.input_ids

    return batch

max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(batch):
    # batch["input_length"] is a list of floats and batch["labels"] is a list of lists
    return [
        (min_input_length < length < max_input_length) and (0 < len(labels) < max_label_length)
        for length, labels in zip(batch["input_length"], batch["labels"])
    ]

print('DATASET PREPARATION IN PROGRESS...')
raw_dataset = DatasetDict()
raw_dataset["train"] = load_all_datasets('train')
raw_dataset["eval"] = load_all_datasets('eval')

if args.testmode:
    raw_dataset = DatasetDict({
        "train": raw_dataset["train"].select(range(10)),  # Select first 10 rows from train
        "eval": raw_dataset["eval"].select(range(10))     # Select first 10 rows from test
    })


# --- NEW: Process the dataset in batches to limit memory usage ---
print(f'{raw_dataset=}')
raw_dataset = raw_dataset.map(
    prepare_dataset, 
    batched=True, 
    batch_size=2,  # adjust batch size as needed
    num_proc=args.num_proc
)
print(f'map done. {raw_dataset=}')
raw_dataset = raw_dataset.filter(
    is_in_length_range,
    batched=True, 
    batch_size=2, 
    num_proc=args.num_proc,
)

###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Process audio inputs: simply pad the input features to a batch tensor
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Process label sequences and pad them
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Optionally remove an initial bos token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print('DATASET PREPARATION COMPLETED')

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad token id so that the tokenizer can decode properly
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer_score = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}

###############################     TRAINING ARGS AND TRAINING      ############################

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
        push_to_hub=True,
    )
elif args.train_strategy == 'steps':
    step_epoch_args = {"num_train_epochs": args.num_epochs} if args.num_epochs is not None else {"max_steps": args.num_steps}
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        # max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
        **step_epoch_args
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print('TRAINING IN PROGRESS...')
trainer.train()
print('DONE TRAINING')


kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_17_0",
    "dataset": "Common Voice 17.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: fa, split: test",
    "language": "fa",
    "model_name": "Whisper Large fa - Mobin Tadbir Sharif",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)

