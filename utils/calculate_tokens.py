"""Token counting helpers.

``count_tokens`` uses ``tiktoken`` (cl100k_base) when it is installed and falls
back to a deterministic regex approximation otherwise, so that the offline
runner has no third-party dependency.
"""

import os
import re
import argparse

try:  # optional dependency
    import tiktoken  # type: ignore
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken not installed
    tiktoken = None
    _ENCODING = None

try:
    from utils.data_utils import read_json_data
except ImportError:  # executed as a script from inside utils/
    from data_utils import read_json_data  # type: ignore

_APPROX_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenizer_name():
    return "tiktoken:cl100k_base" if _ENCODING is not None else "regex-approximation"


def count_tokens(text):
    """Number of tokens in ``text`` (tiktoken when available, else regex approximation)."""
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text, disallowed_special=()))
    return len(_APPROX_RE.findall(text))


def num_tokens_from_string(input_str, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string (original API)."""
    return count_tokens(input_str)


def count_message_tokens(messages):
    return sum(count_tokens(m.get("content", "")) for m in messages)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def calculate_baseline_tokens(baseline_path):
    baseline_results = read_json_data(baseline_path)
    all_input_tokens_num, all_output_tokens_num = [], []
    for data in baseline_results:
        instruction = data["Instruction"]
        summaries = data["Summary"] if isinstance(data["Summary"], list) else [data["Summary"]]
        input_token_num = num_tokens_from_string(instruction)
        if "sc_3" in baseline_path:
            input_token_num = input_token_num * 3
        all_input_tokens_num.append(input_token_num)
        all_output_tokens_num.append(sum(num_tokens_from_string(s) for s in summaries))
    print("Avg input tokens: {}, Avg output tokens: {}, from {}".format(
        _mean(all_input_tokens_num), _mean(all_output_tokens_num), baseline_path))


def calculate_mit_debate(debate_path):
    debate_results = read_json_data(debate_path)
    all_input_tokens_num, all_output_tokens_num = [], []
    for data in debate_results:
        input_tokens_num, output_tokens_num = 0, 0
        for agent_context in data["agent_contexts"]:
            for message in agent_context:
                if message["role"] == "user":
                    input_tokens_num += num_tokens_from_string(message["content"])
                else:
                    output_tokens_num += num_tokens_from_string(message["content"])
        all_input_tokens_num.append(input_tokens_num)
        all_output_tokens_num.append(output_tokens_num)
    print("Avg input tokens: {}, Avg output tokens: {}, from {}".format(
        _mean(all_input_tokens_num), _mean(all_output_tokens_num), debate_path))


def calculate_debate_tokens(debate_path):
    debate_results = read_json_data(debate_path)
    all_output_tokens_num = []
    for data in debate_results:
        output_tokens_num = 0
        for key in data:
            if "Round" in key and isinstance(data[key], dict):
                for round_key in data[key]:
                    if "User" in round_key:
                        continue
                    value = data[key][round_key]
                    if isinstance(value, str):
                        output_tokens_num += num_tokens_from_string(value)
            if "Summary" in key and isinstance(data[key], str):
                output_tokens_num += num_tokens_from_string(data[key])
        all_output_tokens_num.append(output_tokens_num)
    print(" Avg output tokens: {}, from {}".format(_mean(all_output_tokens_num), debate_path))


def get_filelist(directory, file_list):
    if os.path.isfile(directory):
        file_list.append(directory)
    elif os.path.isdir(directory):
        for s in os.listdir(directory):
            get_filelist(os.path.join(directory, s), file_list)
    return file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Token statistics of prediction files")
    parser.add_argument("--result_path", type=str, default="./results/predictions/triviaqa/")
    args = parser.parse_args()
    for file_path in get_filelist(args.result_path, []):
        if "reflect" in file_path:
            continue
        if "mit_debate" in file_path:
            calculate_mit_debate(file_path)
        elif "baseline" in file_path:
            calculate_baseline_tokens(file_path)
        else:
            calculate_debate_tokens(file_path)
