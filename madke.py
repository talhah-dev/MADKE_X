"""Run MADKE / MADKE-X on one dataset (command-line entry point).

Examples
--------
Offline, deterministic (no API key)::

    python madke.py --dataset_name triviaqa --mode offline --cite --verify

Live, any OpenAI-compatible server (vLLM / Ollama)::

    python madke.py --dataset_name nq --mode live --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ --cite --verify --start_index 0 --end_index 500

Unpatched baseline (MADKE re-run) on the same questions::

    python madke.py --dataset_name nq --mode live --base_url ... --model ... --legacy
"""

import os
import warnings

from debate import DebateConfig, build_agents, run_debate, select_prompt
from parser import get_args
from utils.data_utils import (
    DATA_FILES,
    build_bm25_index,
    dataset_path,
    print_args,
    read_json_data,
    verify_data_checksums,
    write_json_data,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")


def main():
    args = get_args()
    print_args(args)

    prompt = select_prompt(args.dataset_name)
    cfg = DebateConfig.from_args(args)

    if args.test_data_path is None:
        args.test_data_path = dataset_path(args.dataset_name, args.data_root)
        if not args.skip_checksums:
            verify_data_checksums(args.data_root, [args.dataset_name], strict=True)
            print("SHA-256 verified for {}".format(args.test_data_path))
    elif not args.skip_checksums and args.dataset_name in DATA_FILES and \
            os.path.abspath(args.test_data_path) == os.path.abspath(dataset_path(args.dataset_name, args.data_root)):
        verify_data_checksums(args.data_root, [args.dataset_name], strict=True)

    if args.test_output_path is None:
        args.test_output_path = os.path.join("results", "predictions", args.dataset_name,
                                             "{}_{}_{}.json".format(args.dataset_name, cfg.name, args.mode))

    # Fix C2: no hard-coded [59:] slice -- process [start_index, end_index)
    test_dataset = read_json_data(args.test_data_path)
    end = min(args.end_index, len(test_dataset))
    indices = list(range(args.start_index, end))
    print("Processing {} questions (indices {}..{}) with config {}".format(len(indices), args.start_index, end - 1, cfg.name))

    if args.mode == "live":
        from models.openai_compat_agent import preflight
        ok, msg = preflight(args)
        print(("PREFLIGHT OK: " if ok else "PREFLIGHT FAILED: ") + msg)
        if not ok:
            raise SystemExit("No run started: start the model server first.")

    debaters, judge, summarizer = build_agents(args, args.agent_num, args.mode, args.dataset_name)

    # dataset-level BM25 reference statistics for the evidence governor (Algorithm 2, index B)
    bm25_index = None if cfg.legacy else build_bm25_index(test_dataset, cfg.wiki_k, cfg.google_k)

    all_dialog_process = []
    for i in tqdm(indices, total=len(indices)):
        record = run_debate(test_dataset[i], debaters, judge, summarizer, prompt, cfg, index=i, bm25_index=bm25_index)
        all_dialog_process.append(record)
        if record["status"] == "failed":
            print("Question {} FAILED at stage {}: {}".format(i, record["error"]["stage"], record["error"]["error"]))
        elif len(record["trace"]["judge_verdicts"]) < cfg.max_round:
            print("Early stop !")
        write_json_data(args.test_output_path, all_dialog_process, verbose=False)

    write_json_data(args.test_output_path, all_dialog_process)
    n_failed = sum(1 for r in all_dialog_process if r["status"] == "failed")
    print("Done. {} completed, {} failed -> {}".format(len(all_dialog_process) - n_failed, n_failed, args.test_output_path))


if __name__ == "__main__":
    main()
