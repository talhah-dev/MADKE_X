import argparse


def add_agent_backend_args(parser):
    """Flags shared by ``madke.py``, ``evaluation.py`` and ``run_experiments.py``."""
    group = parser.add_argument_group("agent backend")
    group.add_argument("--mode", type=str, default="live", choices=["offline", "live"],
                       help="offline: deterministic scripted agents (no API). live: OpenAI-compatible endpoint.")
    group.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                       help="OpenAI-compatible base URL (vLLM / Ollama / llama.cpp / OpenAI).")
    group.add_argument("--api_key", type=str, default="EMPTY", help="API key (local servers accept any value).")
    group.add_argument("--model", type=str, default=None, help="Served model name, e.g. Qwen/Qwen2.5-7B-Instruct-AWQ.")
    group.add_argument("--endpoint", type=str, default=None, help="(optional) Azure OpenAI resource name.")
    group.add_argument("--deployment_id", type=str, default=None, help="(optional) Azure OpenAI deployment id.")
    group.add_argument("--temperature", type=float, default=0.5)
    group.add_argument("--top_p", type=float, default=None)
    group.add_argument("--max_tokens", type=int, default=None)
    group.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    group.add_argument("--max_retries", type=int, default=3, help="Bounded retries per call (fix C7).")
    group.add_argument("--ca_bundle", type=str, default=None, help="Custom CA bundle for TLS verification.")
    group.add_argument("--insecure", action="store_true", help="Disable TLS verification (local self-signed servers only).")
    group.add_argument("--seed", type=int, default=0, help="Seed of the scripted agents (offline mode).")
    group.add_argument("--fail_rate", type=float, default=0.0, help="Injected call-failure rate for scripted agents.")
    return parser


def add_debate_args(parser):
    group = parser.add_argument_group("debate")
    group.add_argument("--agent_num", type=int, default=2)
    group.add_argument("--max_round", type=int, default=3)
    group.add_argument("--use_self_choose", dest="use_self_choose", action="store_false", default=True,
                       help="(original flag) passing it DISABLES self-chosen evidence.")
    group.add_argument("--no_self_choose", dest="use_self_choose", action="store_false",
                       help="Disable self-chosen evidence (alias with a clearer name).")
    group.add_argument("--wiki_k", "--wiki_num", dest="wiki_k", type=int, default=5,
                       help="Wikipedia passages in the pool (paper: 5; original code default was 10).")
    group.add_argument("--google_k", "--google_num", dest="google_k", type=int, default=5,
                       help="Google passages in the pool (paper: 5).")
    group.add_argument("--cap", type=int, default=3, help="Maximum evidence ids kept per agent (k=3).")
    group.add_argument("--textbooks_num", type=int, default=5)
    group.add_argument("--evidence_type", type=str, default="all_evidence",
                       choices=["all_evidence", "only_wiki_5", "only_google_5", "only_textbooks_5", "no_evidence"])
    group.add_argument("--no_dedup", action="store_true", help="Disable passage de-duplication.")
    group.add_argument("--no_rrf", action="store_true", help="Disable BM25+RRF re-ordering of the pool.")
    group.add_argument("--rrf_k", type=int, default=60)
    group.add_argument("--cite", action="store_true", help="Summariser must cite supporting passages as {E#}.")
    group.add_argument("--verify", action="store_true", help="Bounded verification call when the answer is unsupported.")
    group.add_argument("--legacy", action="store_true",
                       help="Reproduce the UNPATCHED MADKE behaviour (10+5 pool, buggy parsers, swapped summariser).")
    return parser


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Multi Agent Debate (MADKE / MADKE-X)")
    parser.add_argument("--dataset_name", type=str, default="triviaqa",
                        choices=["hotpotqa", "fever", "feverous", "nq", "triviaqa", "strategyqa", "wikimultihopqa",
                                 "mmlu_med", "medmcqa"])
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Raw data file. Defaults to the registered data/<dataset>/... file.")
    parser.add_argument("--test_output_path", type=str, default=None,
                        help="Prediction output (json). Defaults to results/predictions/<dataset>/<dataset>_<config>.json")
    parser.add_argument("--start_index", type=int, default=0, help="First question index (inclusive). Fix C2.")
    parser.add_argument("--end_index", type=int, default=500, help="Last question index (exclusive). Fix C2.")
    parser.add_argument("--skip_checksums", action="store_true", help="Do not verify the SHA-256 of the data file.")
    parser.add_argument("--data_root", type=str, default=".")
    add_agent_backend_args(parser)
    add_debate_args(parser)
    args = parser.parse_args(argv)
    return args
