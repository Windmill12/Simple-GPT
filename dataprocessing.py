# This file is for data processing. Including functions of processing many types of datasets

from transformers import GPT2Tokenizer
import datasets
from torch.utils.data import Dataset
import torch
import re
import json


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class AsciiTokenizer:
    def encode(self, string):
        return [ord(c) for c in string]

    def decode(self, tokens):
        chars = [chr(token) for token in tokens]
        result = ''
        for char in chars:
            result += char
        return result


# Define a text dataset, which loads the token_ids from a json file
class TextDataset(Dataset):
    def __init__(self, json_path, seq_len, device):
        self.json_path = json_path
        self.seq_len = seq_len
        self.device = device
        with open(json_path, "r") as file:
            self.input_ids = json.load(file)
        # Here I didn't assign the datasets to cuda to reduce GPU memory usage
        self.input_ids = torch.tensor(self.input_ids)
        print(f"Dataset loaded. It has {self.input_ids.size(0)} tokens!")

    def __len__(self):
        return len(self.input_ids) - self.seq_len

    def __getitem__(self, idx):
        input_token_ids = self.input_ids[idx:idx + self.seq_len]
        output_target_encoding = self.input_ids[idx + 1:idx + self.seq_len + 1]
        return input_token_ids.to(self.device), output_target_encoding.to(self.device)


# replacement rules
# Remove some special characters from the text
COMMON_RULES = {
    r'[-—*━一=]+(\n|$)': '\n',
    # standardize line breaker patterns like ------------ or ———————————————or ************* or ━━━━━━━━━━━━━━━━━━
    r'[*━]{3,}': '',  # Special cases
    # r'\([,.;:，。；：`‘\s]{1,}': '(',
    # r'[,.;:，。；：`‘\s]{1,}\)': ')',
}

# Define a mapping of punctuation characters to a standardized version for English language
EN_RULES = {
    **COMMON_RULES,
    r"[“”]|[‘’]": '"',  # Replace curly quotes with straight quotes
    r"[`´]": "'",
    r"'{2}": '"',
    r"\.{3,}": '...',
    r"\:,": ',',
    r"\：.": '.',
    r" \'": "'",
    r" \!": '!',
    r" \,": ',',
    r" \.": '.',
    r" \?": '?',
    r" - ": '-',
    r" \)": ')',
    r"\( ": '(',
    r'\(\s*[^\w\s]*\s*\)': '',  # parenthesis without any text inside
    r"(?:(?!\n)\s)+": ' ',  # Normalize whitespace while preserving new lines
    r"[\n]{3,}": '\n\n',  # standardize new lines
}


def load_and_save_bookcorpus(bookcorpus_path, tokenizer, begin_idx=0, lines=10000,
                             eval_to_train_ratio=0.1, save_path="./datasets"):
    # Preprocessing the bookcorpus datasets, reduce the time cost when loading the dataset
    dataset = datasets.load_from_disk(bookcorpus_path)
    text = ""
    texts = dataset["train"]["text"][begin_idx:begin_idx + lines]
    print("standardize the english punctuations:")
    for i, line in enumerate(texts):
        for pattern in EN_RULES.keys():
            # standardize the punctuations
            line = re.sub(pattern, EN_RULES[pattern], line)
        if i % 1000 == 0:
            print(f"processed {i / lines * 100}%")
        text += line

    token_ids = []
    sub_strings = text.split(".")
    num_of_parts = len(sub_strings)
    print("Encoding the text:")
    for i, subString in enumerate(sub_strings):
        token_ids += tokenizer.encode(subString + ". ")
        # It seems using a+=b is better than a=a+b, especially when size of a is very large.
        if i % 1000 == 0:
            print(f"processed {i / num_of_parts * 100}%")
    # divide the dataset into two parts: training set and eval set
    with open(save_path + f"/bookcorpus_train_{begin_idx}-{begin_idx + lines}.json", "w") as file:
        json.dump(token_ids[0:int(len(token_ids) * (1 - eval_to_train_ratio))], file)
    with open(save_path + f"/bookcorpus_eval_{begin_idx}-{begin_idx + lines}.json", "w") as file:
        json.dump(token_ids[int(len(token_ids) * (1 - eval_to_train_ratio)):], file)


def load_and_save_wikipedia_to_text(wikipedia_path=None, begin_idx=0, lines=10000, save_path="./datasets"
                                    , padding_length=1024):
    # This function preprocesses wikipedia dataset and save the result to a single .txt file, returns the file name
    if wikipedia_path is None:
        dataset = datasets.load_dataset("wikipedia", "20220301.en")
    else:
        dataset = datasets.load_from_disk(wikipedia_path)

    titles = dataset["train"][begin_idx:begin_idx + lines]["title"]
    texts = dataset["train"][begin_idx:begin_idx + lines]["text"]

    new_titles = []
    new_texts = []
    token_eos = "<|endoftext|>"  # The gpt2 eot token
    # The tedious part of filtering text
    print("Processing texts:")
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            print(f"Processed {i / lines * 100}%")
        # try to simplify the wikipedia texts
        text_fragments = text.split("\n\n")[:5]
        # skip if the content is too short
        if len(text) < 500:
            continue
        # skip if contains disambiguation
        if "disambiguation" in text or "disambiguation" in titles[i] or "style=" in text:
            continue
        cache_text = ""
        for ind, text_fragment in enumerate(text_fragments):
            if ind == len(text_fragments)-1:
                # Remove external links, "see also", e.t.c.
                if_line_length_0 = [len(line) > 150 for line in text_fragment.split("\n")]
                if not (True in if_line_length_0):
                    break
            if (len(text_fragment) > 200) and (token_eos not in text_fragment):
                cache_text += text_fragment + "\n\n"
        # skip if the content is too short
        if len(cache_text) < 500:
            continue
        # Ask one line to be larger than 150 to ensure quality
        if_line_length = [len(line) > 150 for line in cache_text.split("\n")]
        if not (True in if_line_length):
            continue
        # Standardize the punctuations
        for pattern in EN_RULES.keys():
            cache_text = re.sub(pattern, EN_RULES[pattern], cache_text)
        new_titles.append(titles[i])
        new_texts.append(cache_text)

    print("Concatenating texts:")
    result_texts = []
    for i, new_text in enumerate(new_texts):
        padding_txt = ""
        # use \0 as padding_txt. try to strengthen the model
        if i % 50 == 0:
            padding_txt = padding_length * chr(0)
        if i % 1000 == 0:
            print(f"Processed {i / lines * 100}%")
        result_texts.append(padding_txt + "title: " + new_titles[i] + "\n" + new_text + token_eos + "\n")
    result_text = "".join(result_texts)
    # Use .join method to reduce time cost. better than +=
    with open(save_path + f"/wikipedia_en_{begin_idx}-{begin_idx + lines}.txt", "w", encoding="utf-8") as file:
        file.write(result_text)

    return save_path + f"/wikipedia_en_{begin_idx}-{begin_idx + lines}.txt"


def load_and_save_webqa_to_text(webqa_path=None, begin_idx=0, lines=10000,
                                save_path="./datasets", padding_length=1024):
    # This function preprocesses webqa dataset
    if webqa_path is None:
        dataset = datasets.load_dataset("THUDM/webglm-qa")
    else:
        dataset = datasets.load_from_disk(webqa_path)
    questions = dataset["train"]["question"][begin_idx:begin_idx + lines]
    answers = dataset["train"]["answer"][begin_idx:begin_idx + lines]
    token_eos = "<|endoftext|>"
    result_texts = []
    ref_regular_expression = r"\[\d+\]"
    for i, answer in enumerate(answers):
        padding_txt = ""
        # use \0 as padding_txt. try to strengthen the model
        if i % 50 == 0:
            padding_txt = padding_length * chr(0)
        if i % 1000 == 0:
            print(f"Processed {i / lines * 100}%")
        standard_answer = answer
        standard_question = questions[i]
        for pattern in EN_RULES.keys():
            standard_answer = re.sub(pattern, EN_RULES[pattern], standard_answer)
            standard_question = re.sub(pattern, EN_RULES[pattern], standard_question)
        standard_answer = re.sub(ref_regular_expression, "", standard_answer)
        result_texts.append(
            padding_txt + "<User>: " + standard_question + "\n" + "<Agent>: " + standard_answer + token_eos + "\n")
    result_text = "".join(result_texts)
    with open(save_path + f"/webqa_{begin_idx}-{begin_idx + lines}.txt", "w", encoding="utf-8") as file:
        file.write(result_text)

    return save_path + f"/webqa_{begin_idx}-{begin_idx + lines}.txt"


def load_and_save_redpajama_to_text(redpajama_path=None, begin_idx=0, lines=100000, save_path="./datasets",
                                    padding_length=1024):
    if redpajama_path is None:
        dataset = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
    else:
        dataset = datasets.load_from_disk(redpajama_path)
    texts = dataset['train']['text'][begin_idx:begin_idx+lines]
    token_eos = "<|endoftext|>"
    new_texts = []
    for i, text_item in enumerate(texts):
        padding_txt = ""
        if i % 50 == 0:
            padding_txt = padding_length * chr(0)
        if i % 1000 == 0:
            print(f"Processed {i/lines * 100}%")
        for pattern in EN_RULES.keys():
            text_item = re.sub(pattern, EN_RULES[pattern], text_item)
        new_texts.append(padding_txt+text_item+token_eos+"\n")
    result_text = "".join(new_texts)
    with open(save_path + f"/redpajama_{begin_idx}-{begin_idx + lines}.txt", "w", encoding="utf-8") as file:
        file.write(result_text)

    return save_path + f"/redpajama_{begin_idx}-{begin_idx + lines}.txt"


# gopher rule for redpajama v2
def gopher_rules_pass(sample) -> bool:
    """ function returns True if the sample complies with Gopher rules """
    signals = json.loads(sample["quality_signals"])

    # rule 1: number of words between 50 and 10'000
    word_count = signals["rps_doc_word_count"][0][2]
    if word_count < 50 or word_count > 100_000:
        return False

    # rule 2: mean word length between 3 and 10
    mean_word_length = signals["rps_doc_mean_word_length"][0][2]
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # rule 2: symbol to word ratio below 0.1
    symbol_word_ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
    if symbol_word_ratio > 0.1:
        return False

    # rule 3: 90% of lines need to start without a bullet point
    n_lines = signals["ccnet_nlines"][0][2]
    n_lines_bulletpoint_start = sum(map(lambda ln: ln[2], signals["rps_lines_start_with_bulletpoint"]))
    if n_lines_bulletpoint_start / n_lines > 0.9:
        return False

    # rule 4: the ratio between characters in the most frequent 2-gram and the total number
    # of characters must be below 0.2
    top_2_gram_frac = signals["rps_doc_frac_chars_top_2gram"][0][2]
    if top_2_gram_frac > 0.2:
        return False

    # rule 5: ...

    return True


def load_and_save_redpajama2_to_text(redpajama_path=None, begin_idx=0, lines=100000, save_path="./datasets",
                                    padding_length=1024):
    # gladly to see this dataset has a quality vector
    if redpajama_path is None:
        ds = datasets.load_dataset("togethercomputer/RedPajama-Data-V2",
                                   name="sample", languages=["en"])
    else:
        ds = datasets.load_from_disk(redpajama_path)
    filtered_dataset = []
    # filter text with gopher rules
    for i in range(begin_idx, begin_idx+lines):
        sample = ds["train"][i]
        if not gopher_rules_pass(sample):
            continue
        filtered_dataset.append(sample)
    result_texts = []
    token_eos = "<|endoftext|>"
    # append the texts
    for i, sample in enumerate(filtered_dataset):
        padding_txt = ""
        if i % 50 == 0:
            padding_txt = padding_length * chr(0)
        if i % 1000 == 0:
            print(f"Processed {i/lines * 100}%")
        result_texts.append(padding_txt+sample["raw_content"]+token_eos+"\n")
    result_text = "".join(result_texts)
    # save result
    with open(save_path + f"/redpajama2_{begin_idx}-{begin_idx + lines}.txt", "w", encoding="utf-8") as file:
        file.write(result_text)

    return save_path + f"/redpajama2_{begin_idx}-{begin_idx + lines}.txt"


def load_and_save_ultrachat200k_to_text(ultrachat_path=None, begin_idx=0, lines=100000, save_path="./datasets",
                                        padding_length=1024):
    # gladly to see this dataset has a quality vector
    if ultrachat_path is None:
        ds = datasets.load_dataset("HuggingFaceH4/ultrachat_200k")
    else:
        ds = datasets.load_from_disk(ultrachat_path)
    sliced_dataset = ds["train_sft"]["messages"][begin_idx:begin_idx+lines]
    token_eos = "<|endoftext|>"
    result_texts = []
    for i, conservation in enumerate(sliced_dataset):
        conservation_text = []
        padding_text = ''
        if i % 50 == 0:
            padding_text = padding_length * chr(0)
            conservation_text.append(padding_text)
        if i % 1000 == 0:
            print(f"Processed {i/lines * 100}%")
        for sentence in conservation:
            if sentence["role"] == "user":
                conservation_text.append("<User>: " + sentence["content"] + "\n")
            if sentence["role"] == "assistant":
                conservation_text.append("<Agent>: " + sentence["content"] + "\n")
        conservation_text.append(token_eos + "\n")
        result_texts.append("".join(conservation_text))
    result_text = "".join(result_texts)
    with open(save_path + f"/ultrachat200k_{begin_idx}-{begin_idx + lines}.txt", "w", encoding="utf-8") as file:
        file.write(result_text)

    return save_path + f"/ultrachat200k_{begin_idx}-{begin_idx + lines}.txt"


def text_encode_to_json(tokenizer, text_path, save_path, eval_to_train_ratio=0.05):
    # precompile the text to reduce time cost
    with open(text_path, "r", encoding="utf-8") as file:
        long_string = file.read()
    print("Encoding the text...")
    tokens = tokenizer.tokenize(long_string)
    # split the long string and encode it one by one
    length_tokens = len(tokens)
    token_ids = []
    start_idx = 0
    for i, token in enumerate(tokens):
        if i % 10000 == 0:
            print(f"Processed {i / length_tokens * 100} %")
        token_ids += tokenizer.encode(long_string[start_idx: start_idx + len(token)])
        start_idx += len(token)
    # Process save_path. Separate the dataset
    save_path = ".".join(save_path.split(".")[:-1])
    with open(save_path + "_train.json", "w") as file:
        json.dump(token_ids[0:int(len(token_ids) * (1 - eval_to_train_ratio))], file)
    with open(save_path + "_eval.json", "w") as file:
        json.dump(token_ids[int(len(token_ids) * (1 - eval_to_train_ratio)):], file)


def read_text_json_file(file_path, tokenizer, begin_idx=0, end_idx=10000):
    # This function tests json files
    with open(file_path, "r") as file:
        token_ids = json.load(file)
    print(tokenizer.decode(token_ids[begin_idx:end_idx]))


def preprocess_wikipedia_and_save(begin_idx, lines, tokenizer):
    load_and_save_wikipedia_to_text(begin_idx=begin_idx, lines=lines)
    text_encode_to_json(tokenizer, f"./datasets/wikipedia_en_{begin_idx}-{begin_idx + lines}.txt",
                        f"./datasets/wikipedia_en_{begin_idx}-{begin_idx + lines}.json")


def process_redpajama_and_save(begin_idx, lines, tokenizer):
    load_and_save_redpajama_to_text(begin_idx=begin_idx, lines=lines)
    text_encode_to_json(tokenizer, f"./datasets/redpajama_{begin_idx}-{begin_idx + lines}.txt",
                        f"./datasets/redpajama_{begin_idx}-{begin_idx + lines}.json")


def process_redpajama2_and_save(begin_idx, lines, tokenizer):
    load_and_save_redpajama2_to_text(begin_idx=begin_idx, lines=lines)
    text_encode_to_json(tokenizer, f"./datasets/redpajama2_{begin_idx}-{begin_idx + lines}.txt",
                        f"./datasets/redpajama2_{begin_idx}-{begin_idx + lines}.json")


def process_ultrachat200k_and_save(begin_idx, lines ,tokenizer):
    load_and_save_ultrachat200k_to_text(begin_idx=begin_idx, lines=lines)
    text_encode_to_json(tokenizer, f"./datasets/ultrachat200k_{begin_idx}-{begin_idx + lines}.txt",
                        f"./datasets/ultrachat200k_{begin_idx}-{begin_idx + lines}.json")


if __name__ == "__main__":
    # read_text_json_file("./datasets/redpajama_200000-300000_train.json", tokenizer)
    process_ultrachat200k_and_save(begin_idx=100000, lines=100000, tokenizer=tokenizer)
    print("Please test functions here...")
