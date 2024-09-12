import os
import random
from typing import Optional

import numpy as np, pandas as pd
import ast, torch
from datasets import load_dataset
from packaging import version
from tqdm import trange
from transformers import AutoTokenizer, LlamaTokenizer

def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=False):
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T-Sample")
    assert not eval_mode, "Only train set is supported in RedPajama"
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = []
    raw_data, input_ids, num_tokens, metadatas = list(), list(), list(), list()
    for _ in trange(nsamples, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        
        metadata = traindata[i]["meta"]
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
        metadatas.append(metadata)
        input_ids.append(inp)
        num_tokens.append(inp.shape[1])
        raw_data.append(tokenizer.decode(inp[0]))
    
    # Save the calibration set to a csv file
    data_df = pd.DataFrame({"input_ids": input_ids, "metadata": metadatas, "raw_data": raw_data, "num_tokens": num_tokens})
    data_df.to_csv("red_pajama_calibration_sample.csv", index=False, escapechar="\\")

    csv_df = pd.read_csv("red_pajama_calibration_sample.csv", escapechar="\\")
    csv_df.to_json("red_pajama_calibration_sample.json", orient="records", lines=True)
    
    return trainloader

def convert_to_dict(input_str):
    try:
        # Safely evaluate the string as a Python literal
        result_dict = ast.literal_eval(input_str)
        return result_dict
    except Exception as e:
        return None


def get_github_code(nsamples, seqlen, tokenizer, eval_mode=False):
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T")
    assert not eval_mode, "Only train set is supported in RedPajama"
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T", "github", split="train", streaming=True, revision="398f92572e94f4793e41c22ab7ea2a788d9e7de4")
    traindata = iter(traindata)

    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = list()
    raw_data, input_ids, num_tokens, metadatas = list(), list(), list(), list()
    num_java, num_python = 0, 0
    seen_repos = set()
    for _ in trange(nsamples, desc="Making GitHub code calibration set", leave=False):
        while True:
            sample = next(traindata)
            metadata = sample["meta"]
            metadata = convert_to_dict(metadata)
            if metadata is None: continue

            # Retrieve only code from GitHub without duplicates
            if "source" not in metadata or metadata["source"] != "github": continue
            if "path" not in metadata: continue

            repo_name = metadata["repo_name"]
            if repo_name in seen_repos: continue

            # Retrieve only Python and Java code
            path = metadata["path"]
            extension = path.split('.')[-1]
            if extension not in ["java", "py"]: continue
            language = 'java' if path.endswith('.java') else 'python'


            # Retrieve only code with more than seqlen tokens
            content = sample["text"]
            trainenc = tokenizer(content, return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                if language == "java" and num_java >= nsamples // 2: continue
                if language == "python" and num_python >= nsamples // 2: continue
                if language == "java": num_java += 1
                elif language == "python": num_python += 1
                seen_repos.add(repo_name)
                break

        # Retrieve only the first seqlen tokens
        i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
        metadatas.append(metadata)
        input_ids.append(inp)
        num_tokens.append(inp.shape[1])
        raw_data.append(tokenizer.decode(inp[0]))
    
    # Save the calibration set to a csv file
    data_df = pd.DataFrame({"input_ids": input_ids, "metadata": metadatas, "raw_data": raw_data, "num_tokens": num_tokens})
    data_df.to_csv("github_code_sample.csv", index=False, escapechar="\\")

    csv_df = pd.read_csv("github_code_sample.csv", escapechar="\\")
    csv_df.to_json("github_code_sample.json", orient="records", lines=True)

    return trainloader

def get_code_technical_language(nsamples, seqlen, tokenizer, eval_mode=False):

    # Retrieve only code from GitHub without duplicates
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T")
    assert not eval_mode, "Only train set is supported in RedPajama"

    nl_traindata = load_dataset("togethercomputer/RedPajama-Data-1T", "stackexchange", split="train", streaming=True, revision="398f92572e94f4793e41c22ab7ea2a788d9e7de4")
    nl_traindata = nl_traindata.filter(lambda example: "stackoverflow" in example['meta'])

    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = list()
    raw_data, input_ids, num_tokens, metadatas = list(), list(), list(), list()
    num_java, num_python = 0, 0
    seen_repos = set()

    nl_traindata = iter(nl_traindata)
    nl_nsamples = nsamples // 2
    for _ in trange(nl_nsamples, desc="Making StackOverflow calibration set", leave=False):
        while True:
            sample = next(nl_traindata)
            metadata = sample["meta"]
            metadata = convert_to_dict(metadata)
            if metadata is None: continue

            # Retrieve only code from GitHub without duplicates
            if "source" not in metadata or metadata["source"] != "stackexchange": continue

            if "language" not in metadata: continue
            language = metadata["language"]
            if language != "en": continue

            if "question_score" not in metadata: continue
            question_score = metadata["question_score"]
            if int(question_score) < 10: continue

            # Retrieve only text with more than seqlen tokens and related to Python or Java code
            content = sample["text"]
            if "Python" not in content and "Java" not in content and "python" not in content and "java" not in content: continue
            trainenc = tokenizer(content, return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break

        # Retrieve only the first seqlen tokens
        i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
        metadatas.append(metadata)
        input_ids.append(inp)
        num_tokens.append(inp.shape[1])
        raw_data.append(tokenizer.decode(inp[0]))


    github_traindata = load_dataset("togethercomputer/RedPajama-Data-1T", "github", split="train", streaming=True, revision="398f92572e94f4793e41c22ab7ea2a788d9e7de4")
    github_traindata = iter(github_traindata)

    github_nsamples = nsamples // 2
    for _ in trange(github_nsamples, desc="Making GitHub code calibration set", leave=False):
        while True:
            sample = next(github_traindata)
            metadata = sample["meta"]
            metadata = convert_to_dict(metadata)
            if metadata is None: continue

            # Retrieve only code from GitHub without duplicates
            if "source" not in metadata or metadata["source"] != "github": continue
            if "path" not in metadata: continue

            repo_name = metadata["repo_name"]
            if repo_name in seen_repos: continue

            # Retrieve only Python and Java code
            path = metadata["path"]
            extension = path.split('.')[-1]
            if extension not in ["java", "py"]: continue
            language = 'java' if path.endswith('.java') else 'python'


            # Retrieve only code with more than seqlen tokens
            content = sample["text"]
            trainenc = tokenizer(content, return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                if language == "java" and num_java >= nsamples // 4: continue
                if language == "python" and num_python >= nsamples // 4: continue
                if language == "java": num_java += 1
                elif language == "python": num_python += 1
                seen_repos.add(repo_name)
                break

        # Retrieve only the first seqlen tokens
        i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
        metadatas.append(metadata)
        input_ids.append(inp)
        num_tokens.append(inp.shape[1])
        raw_data.append(tokenizer.decode(inp[0]))

    
    # Save the calibration set to a csv file
    data_df = pd.DataFrame({"input_ids": input_ids, "metadata": metadatas, "raw_data": raw_data, "num_tokens": num_tokens})
    data_df.to_csv("mixed_sample.csv", index=False, escapechar="\\")

    csv_df = pd.read_csv("mixed_sample.csv", escapechar="\\")
    csv_df.to_json("mixed_sample.json", orient="records", lines=True)

    return trainloader

def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc


def get_ptb(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
    return testenc


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader

    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                # rare case, discovered with Yi tokenizer
                valenc.append(tmp.input_ids)
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        return valenc


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]
        return valenc


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
):
    """
    Loads and prepares data for a Transformers model.
    Args:
        name (str): The name of the dataset to load.
        This can be one of 'wikitext2', 'c4', 'ptb','pajama' for datasets loaded from Huggingface datasets,
        or 'none' for cases where a dataset is not needed, like RTN. It can also accept data path to custom file.
        nsamples (int, optional): The number of samples to load from the dataset. Defaults to 128.
        seed (int, optional): The random seed value for data shuffling and splitting. Defaults to 0.
        seqlen (int, optional): The maximum sequence length for input tokenization. Defaults to 2048.
        model_path (str, optional): The path to the pretrained model weights or full model name.
            used to detect llama to call proper tokenizer.
            see https://github.com/huggingface/transformers/issues/22222#issuecomment-1488578722 for reasons.
        eval_mode (bool, optional). defines slice selection for 'wikitext2', 'c4', 'ptb' datasets.
        leave False for train slice.
        use_fast_tokenizer: whether to use fast tokenizer
        trust_remote_code: whether to trust remote code
    Returns:
        data (torch.utils.data.DataLoader or iterable): Data iterable for the dataset.
    Note:
        the popular decapoda-research Llama models have errors in tokenizer config, specifically
        incorrect token ids for BOS, EOS. This gets corrected to ensure compatibility with transformers
        of versions 4.29 and above.
    """
    set_seed(seed)

    # for pre-tokenized datasets

    if name.lower() == "none":
        print("Not loading any dataset. (OK if you use no compression or methods like RTN.)")
        return None
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load custom data from {name}.",
                "Check data path or use one of [c4, wikitext2, ptb, pajama, none]",
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code
        )

        if name.lower() == "wikitext2":
            data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "pajama":
            data = get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "github_code":
            data = get_github_code(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "code_technical_language":
            data = get_code_technical_language(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb":
            data = get_ptb(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb_new":
            data = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4":
            data = get_c4(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4_new":
            data = get_c4_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        else:
            raise ValueError(
                f"Failed to load data from {name}.",
                "Check dataset name or path or use one of [c4, wikitext2, ptb, pajama, none]",
            )

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data
