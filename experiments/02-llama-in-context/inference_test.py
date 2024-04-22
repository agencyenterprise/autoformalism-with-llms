import warnings
from time import time

import transformers
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map,
)

from autoformalism_with_llms.config import config

warnings.filterwarnings("ignore")


def main(model_id):
    hf_token = config.HF_TOKEN
    # weights_location = snapshot_download(model_id, token=hf_token)
    with init_empty_weights():
        empty_model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    if "8b" in model_id.casefold():
        weights_location = "/home/mike/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/561487d18c41c76bcb5fc6cfb73a324982f04f47"
    elif "70b" in model_id.casefold():
        weights_location = "/home/mike/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70b/snapshots/a897071033c208db3c2fdba8a1f4b1b3e2fbb283"
    else:
        raise ValueError(f"Model id {model_id} not recognized")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)

    max_memory = {0: "5GiB", 1: "5GiB", "cpu": "60GiB"}
    device_map = infer_auto_device_map(empty_model, max_memory=max_memory)

    no_split_classes = empty_model._no_split_modules
    print("=" * 100)
    print(f"Model Class: {empty_model.__class__}")
    print(f"No Split Classes: {no_split_classes}")

    # bnb_quantization_config = BnbQuantizationConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,  # optional
    #     bnb_4bit_use_double_quant=True,  # optional
    #     bnb_4bit_quant_type="nf4",  # optional
    # )

    # model = load_and_quantize_model(
    #     empty_model,
    #     weights_location=weights_location,
    #     bnb_quantization_config=bnb_quantization_config,
    #     device_map=device_map,
    # )

    model = load_checkpoint_and_dispatch(
        empty_model,
        checkpoint=weights_location,
        device_map=device_map,
        no_split_module_classes=no_split_classes,
        offload_folder="./offload",
    )
    prompt = "Once upon a time"
    tokens = tokenizer(prompt, return_tensors="pt")
    t0 = time()

    output = model.generate(
        **{k: x.to("cuda:0") for k, x in tokens.items()}, max_length=10
    )
    t1 = time()
    print(tokenizer.decode(output[0]))
    print(f"Time taken: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    import sys

    model_id = sys.argv[1]
    main(model_id)
