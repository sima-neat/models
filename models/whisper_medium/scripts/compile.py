import argparse
import logging
from pathlib import Path

from afe.apis.error_handling_variables import enable_verbose_error_messages
from sima_lmm.model import FileGenMode, FileGenPrecision, WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Whisper for Modalix")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num_processes", type=int, default=20)
    parser.add_argument("--use_future_token_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_log_probe", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    enable_verbose_error_messages()
    model = WhisperModel.from_hf_cache(
        hf_cache_path=args.checkpoint,
        model_name=args.checkpoint.name,
        sima_path=args.output / "sima_files",
        use_future_token_mask=args.use_future_token_mask,
        enable_log_probe=args.enable_log_probe,
    )
    model.gen_files(
        FileGenMode.ALL,
        precision={"all": {"precision": FileGenPrecision.A_BF16_W_INT8}},
        log_level=logging.INFO,
        num_processes=args.num_processes,
        part="all",
        resume=True,
    )


if __name__ == "__main__":
    main()
