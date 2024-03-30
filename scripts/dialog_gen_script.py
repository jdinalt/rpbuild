# Given a dataset with character meta-data, output a new dataset wtih the input and 
# newly generated dialog.

# Usage example:
# torchrun --standalone --nproc-per-node gpu ./dialog_gen_script.py --dataset-id "/home/dinalt/ai_assets/datasets/roleplay_dialog/" \
# --output-dir "/home/dinalt/rust/datasets/roleplay_dialog_v2"
#
# If started with multiple GPU's, this will create a copy of the model on each GPU and shard the input dataset
# accross all of the GPUs.
#
# After a worker completes "output_steps" batches, it will write the output to a json file prefixed with local_rank
# and global_step. Huggingface datasets is smart enough that you can just use "load_dataset() on the resulting directory
# and it will build the json files into a dataset.

import os
import time
import importlib
import transformers
from datasets import load_dataset, load_from_disk
import torch
import re
import argparse

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset
import json
import time

import rpbuild as rp
import rpbuild.data
import rpbuild.generation

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main(local_rank, args):
    transformers.set_seed(42)

    # model_id = "/home/dinalt/ai_assets/models/fhai50032_RolePlayLake-7B"
    
    dataset = load_dataset(args.dataset_id)["train"]

    # Load model and tokenizer
    causal_lm = rp.CausalLM(
        args.model_id,
        device=local_rank,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    sampler = DistributedSampler(dataset)
    
    rp.generation.infer(
        local_rank=local_rank,
        dataset=dataset,
        sampler=sampler,
        generator=rp.generation.generate_dialog,
        batch_size=1, # Dialog generation only supports a batch size of one -- for now
        output_fn=rp.generation.CharacterWriter(args.output_dir),
        output_steps=args.output_steps,
        max_steps=args.max_steps,
        generator_kwargs = dict(
            causal_lm=causal_lm,
            dataset=dataset,
            max_tokens=args.max_tokens,
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-id", type=str,
                    help="dataset path or id")
    parser.add_argument("--output-dir", type=str,
                    help="output-dir")
    parser.add_argument("--model-id", type=str, default="fhai50032/RolePlayLake-7B",
                    help="model name or path")
    parser.add_argument("--max-tokens", type=int, default=4000,
                    help="Maximum tokens to generate.")
    parser.add_argument("--max-steps", type=int, default=None,
                    help="Maximum number of generation steps per worker.")
    parser.add_argument("--output-steps", type=int, default=8,
                    help="The number of steps per output file")

    args = parser.parse_args()
    print(args)

    # Make the destination, if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    main(
        local_rank,
        args,
    )