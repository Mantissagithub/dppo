import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


def fail_if_no_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is unavailable. this lab expects a single gpu run.")


def pick_dtype(config):
    if config.get("dtype") == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float32, "no"


def load_model_and_tokenizer(config, model_path=None):
    torch_dtype, mixed_precision = pick_dtype(config)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model_name = model_path or config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer, accelerator, torch_dtype
