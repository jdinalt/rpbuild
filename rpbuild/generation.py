import os
import time
import importlib
import transformers
from datasets import load_dataset, load_from_disk
import torch
import re
import jinja2
import tqdm
import json

import rpbuild as rp
import rpbuild.char
import rpbuild.data
import rpbuild.writer
import rpbuild.director
import rpbuild.roleplay
from rpbuild.model import InstructGen
from rpbuild.data import load_template

DEFAULT_MAX_TOKENS = 4000

# Some of the original meta-data is missing plists.
# This will try to build a plist and add it to the metadata, if missing.
def fix_plist(meta, causal_lm, debug=False):
    if len(meta.plist) == 0:
        if debug:
            print(f"Generating plist for {meta.name}")
        plist_generator = PListGenerator(causal_lm, debug_level=2 if debug else 1)
        meta.plist = plist_generator.generate(meta.description)
        if debug:
            print(meta.plist)
    return meta
                
# The primary inference entry-point for this task
# Given a batch of characters meta data, generate new data and attach it.
def generate_dialog(batch, causal_lm, dataset, max_tokens=DEFAULT_MAX_TOKENS):
    # Creater a writer to write the script outline.
    writer = rp.writer.Writer(
        causal_lm,
    )

    # Cleans up the dialog and tirggers "retakes" if issues with the dialog.
    dialog_filter = rp.char.DialogFilter(causal_lm)

    # For each character in the batch
    for character in batch:
        char_meta = rp.char.CharMeta.from_data(character)
        writer_selection = writer.choose_supporting_character(char_meta, rp.writer.get_random_proxy_users(dataset, char_meta, n=7))
        user_meta = writer_selection["meta"]

        # Fix missing plists
        char_meta = fix_plist(char_meta, causal_lm)
        user_meta = fix_plist(user_meta, causal_lm)
        
        # Generate a script
        scenario = writer(
            char_meta=char_meta,
            user_meta=user_meta,
        )

        # Pick random generation presets for each character.
        char_preset = rp.model.random_preset()
        user_preset = rp.model.random_preset()

        # The main character
        char = rp.char.Character(
            char_meta=char_meta,
            causal_lm=causal_lm,
            generation_config=char_preset,
            user_meta=user_meta,
            template_config=rp.char.TemplateConfig(
                chat_template=causal_lm.chat_template,
            ),
            gen_post_process=dialog_filter,
            history_token_limit=1800,
        )

        # The proxy user
        user = rp.char.Character(
            char_meta=user_meta,
            causal_lm=causal_lm,
            generation_config=user_preset,
            user_meta=char_meta,
            template_config=rp.char.TemplateConfig(
                chat_template=causal_lm.chat_template,
            ),
            gen_post_process=dialog_filter,
            history_token_limit=1800,
        )

        director = rp.director.Director(
            causal_lm,
            script=scenario,
            debug=False,
            history_token_limit=2000
        )
        
        # Create a director to give instructions to the characters for following the script
        roleplay = rp.roleplay.Roleplay(
            char=char,
            user=user,
            scenario=scenario,
            director=director,
            debug=False
        )

        # Generate dialog
        char_conversation, user_conversation = roleplay(max_tokens)

        character['pairing_reason'] = writer_selection.get('reason')
        character['scenario'] = scenario
        character['plist'] = char_meta.plist
        character['preset'] = char_preset
        character["conversation"] = char.conversation
        character["director_log"] = char.director_log

        # Attach proxy-user meta, as to allow recreation
        character["proxy"] = dict(
            name=user_meta.name,
            summary=user_meta.summary,
            description=user_meta.description,
            plist=user_meta.plist,
            example_dialog=user_meta.example_dialog,
            greeting=user_meta.greeting,
            
            system=user.conversation[0],
            preset=user_preset,
        )
    
    return batch

plist_re = re.compile(r"\[.*?\]")

class PListGenerator():
    def __init__(self, causal_lm, debug_level=1):
        self.plist_instruct = InstructGen(
            causal_lm,
            load_template("make_plist.jinja"),
            filter=self.plist_filter,
        )
        
        self.debug_level = debug_level
        
    def generate(self, description, max_retries=5):
        for i in range(max_retries):
            plist = self.plist_instruct(description=description)
            if plist:
                return plist
            if self.debug_level > 1:
                print(f"plist generation failed. Retry {i+1}")
        if self.debug_level > 0:
            print(f"plist generation failed after multiple retries.")
        return ""
        
    @staticmethod
    def plist_filter(response, **kwargs):
        m = plist_re.search(response)
        if m:
            plist = m.group()
            return plist
        else:
            print(f"plist generation failed: {response}")
        return ""

# Main inference loop
# Processes the dataset
def infer(
    local_rank,
    dataset,
    sampler,
    generator,
    generator_kwargs,

    # The size of the input batch (per GPU)
    batch_size,

    # Receives batches of outputs
    output_fn,

    # The size of the output batch
    output_steps,
    # Stop after "max_step" steps.
    max_steps=None,
):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=lambda x: x,
        num_workers=1
    )

    output_records = []
    if max_steps is None:
        max_steps = len(data_loader)
    else:
        max_steps = min(len(data_loader), max_steps)

    def is_output_step(global_step, output_steps):
        return global_step > 0 and global_step % output_steps == 0
    
    # 'process_batch' is used for recovery from failure to resume at the next unprocessed batch
    process_batch = not output_fn.exists(local_rank, output_steps)
    for global_step, batch in enumerate(tqdm.tqdm(data_loader, total=max_steps)):
        if process_batch:
            output_batch = generator(batch, **generator_kwargs)
            output_records += output_batch
            if is_output_step(global_step, output_steps):
                output_fn(local_rank, global_step, output_records)
                output_records = []
        elif is_output_step(global_step, output_steps):
            print("Skipped: ", output_fn.file_path(local_rank, global_step))
            # Does the next output exist?
            process_batch = not output_fn.exists(local_rank, global_step + output_steps)
        # Does tqdm do this?
        if global_step+1 == max_steps:
            break

    if len(output_records):
        output_fn(local_rank, global_step, output_records)

# Simple 'output_fn' for writing batches of output data to json files
class CharacterWriter():
    def __init__(self, output_path):
        self.output_path = output_path

    # Called for each batch of records to save
    def __call__(self, local_rank, global_step, output_records):
        with open(self.file_path(local_rank, global_step), 'w') as fp:
            json.dump(output_records, fp)

    # Test if records already exist for (local_rank, global_step)
    # Used to resume after failue.
    def exists(self, local_rank, global_step):
        return os.path.exists(self.file_path(local_rank, global_step))

    # Get the output file path for (local_rank, global_step)
    def file_path(self, local_rank, global_step):
        return os.path.join(self.output_path, f"{local_rank}_{global_step}.json")