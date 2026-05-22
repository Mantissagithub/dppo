from pathlib import Path
import json

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))

    def log_train(self, row):
        train_row = {"phase": "train", **row}
        self._write_jsonl(self.output_dir / "training.log", train_row)
        self._write_jsonl(self.output_dir / "train_metrics.jsonl", row)
        self._write_csv(self.output_dir / "train_metrics.csv", row)
        self._write_tensorboard("train", row)

    def log_eval(self, row):
        eval_row = {"phase": "eval", **row}
        self._write_jsonl(self.output_dir / "training.log", eval_row)
        self._write_jsonl(self.output_dir / "eval_metrics.jsonl", row)
        self._write_csv(self.output_dir / "eval_metrics.csv", row)
        self._write_tensorboard("eval", row)

    def _write_jsonl(self, path, row):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _write_csv(self, path, row):
        new_df = pd.DataFrame([row])
        if path.exists():
            old_df = pd.read_csv(path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)

    def _write_tensorboard(self, prefix, row):
        step = int(row.get("step", 0))
        for key, value in row.items():
            if key == "step" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)
        self.writer.flush()
