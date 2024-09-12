import os
import torch, pickle
from transformers import AutoTokenizer
from src.custom_datautils import get_loaders

def main():
    import argparse, json

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save calibration data.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="Num validation sequences",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer (some models have only fast tokenizer).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    # Create output directory if it does not exist
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load calibration data if it exists
    train_filepath = f"{args.output_dir}/{args.dataset}_train.pkl"
    val_filepath = f"{args.output_dir}/{args.dataset}_val.pkl"
    if os.path.exists(train_filepath):
        print("Loading calibration data...")
        with open(train_filepath, "rb") as f:
            train_data = pickle.load(f)
        if os.path.exists(val_filepath):
            with open(val_filepath, "rb") as f:
                val_data = pickle.load(f)
        else:
            val_data = None
        print("Loaded calibration data from file.")
    else:
        print("Calibration data not found. Generating calibration data...")
        data = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=args.model_seqlen,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )

        if args.val_size > 0:
            all_ids = torch.randperm(len(data))
            train_ids, val_ids = all_ids[args.val_size :], all_ids[: args.val_size]
            train_data = [data[i] for i in train_ids]
            val_data = [data[i] for i in val_ids]
        else:
            train_data = data
            val_data = None

        # Save train and val data
        with open(train_filepath, "wb") as f:
            pickle.dump(train_data, f)

        if val_data is not None:
            with open(val_filepath, "wb") as f:
                pickle.dump(val_data, f)

        print("Saved calibration data.")

        # Decode the training data and the validation data and save them to json files
        train_data_decoded = []
        filepath_train_decoded = f"{args.output_dir}/{args.dataset}_train_decoded.json"
        for i, data in enumerate(train_data):
            decoded_data = tokenizer.decode(data[0])
            train_data_decoded.append(decoded_data)
        with open(filepath_train_decoded, "w") as f:
            json.dump(train_data_decoded, f)

        if val_data is not None:
            val_data_decoded = []
            filepath_val_decoded = f"{args.output_dir}/{args.dataset}_val_decoded.json"
            for i, data in enumerate(val_data):
                decoded_data = tokenizer.decode(data[0])
                val_data_decoded.append(decoded_data)
            with open(filepath_val_decoded, "w") as f:
                json.dump(val_data_decoded, f)
        
        print("Decoded and saved calibration data.")
    
    # Analyze the calibration data
    print("Analyzing calibration data...")

    # Get the number of tokens in the calibration data
    num_tokens_train = 0
    for data in train_data:
        num_tokens_train += len(data[0])

if __name__ == "__main__":
    main()
