from __future__ import annotations

import argparse
import json
from dataclasses import fields

from .config import ModelConfig, RunConfig
from .runner import benchmark, check_cached_decode
from .strategies import available_attention_strategies


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    for field in fields(ModelConfig):
        group.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default)
    group = parser.add_argument_group("run")
    for field in fields(RunConfig):
        kwargs: dict[str, object] = {"default": field.default}
        if field.name == "strategy":
            kwargs["choices"] = available_attention_strategies()
        elif isinstance(field.default, int):
            kwargs["type"] = int
        else:
            kwargs["type"] = str
        group.add_argument(f"--{field.name.replace('_', '-')}", **kwargs)


def _configs(args: argparse.Namespace) -> tuple[ModelConfig, RunConfig]:
    model = ModelConfig(**{field.name: getattr(args, field.name) for field in fields(ModelConfig)})
    run = RunConfig(**{field.name: getattr(args, field.name) for field in fields(RunConfig)})
    if run.prompt_len + run.decode_len > model.max_seq_len:
        raise ValueError("prompt_len + decode_len must not exceed max_seq_len")
    return model, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decoder-only inference research template")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "benchmark"):
        child = subparsers.add_parser(command)
        _add_config_arguments(child)
        child.add_argument("--json", action="store_true", help="emit only the JSON result record")
    args = parser.parse_args(argv)
    model, run = _configs(args)
    result = check_cached_decode(model, run) if args.command == "check" else benchmark(model, run)
    payload = json.dumps(result, sort_keys=True)
    if args.json:
        print(payload)
    elif args.command == "check":
        print(f"cached decode: {'PASS' if result['passed'] else 'FAIL'}; max_abs_error={result['max_abs_error']:.3e}")
        print(f"RESULT_JSON={payload}")
    else:
        prefill, decode = result["prefill"], result["decode"]
        print(f"prefill: {prefill['latency_ms']:.3f} ms/sequence, {prefill['tokens_per_second']:.1f} tokens/s")
        print(f"decode: {decode['latency_ms_per_token']:.3f} ms/token, {decode['tokens_per_second']:.1f} tokens/s")
        print(f"RESULT_JSON={payload}")
    return 0 if args.command != "check" or result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
