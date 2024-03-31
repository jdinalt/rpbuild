import random
import re
import os
import torch
import jinja2
from transformers import StoppingCriteria
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

generation_presets = {
    "Simple-1": dict(
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.15,
    ),
    "Midnight-Enigma": dict(
        do_sample=True,
        top_k=100,
        top_p=0.37,
        temperature=0.98,
        repetition_penalty=1.18,
    ),
    "Yara": dict(
        do_sample=True,
        top_k=72,
        top_p=0.21,
        temperature=0.82,
        repetition_penalty=1.19,
    ),
    "Shortwave": dict(
        do_sample=True,
        top_k=33,
        top_p=0.64,
        temperature=1.53,
        repetition_penalty=1.07,
    ),
    "Big-O": dict(
        do_sample=True,
        top_k=85,
        top_p=0.99,
        temperature=0.87,
        repetition_penalty=1.01,
    ),
}

# All preset keys
random_preset_keys = [key for key in generation_presets.keys()]

# Causal LM abstraction
class CausalLM():
    def __init__(self, model_id, device, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        
        if self.tokenizer.pad_token is None:
            print("No PAD token defined. Setting pad token to EOS")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if self.tokenizer.padding_side == 'right':
            print("Tokenizer uses \"right\" padding; this may require moving it to \"left\" for batch generation.")
    
        self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **kwargs
                #trust_remote_code=True,
                # flash_attention_2 requires bfloat16 or float16
                #torch_dtype=torch.bfloat16,
                # One of ["flash_attention_2", "sdpa", "eager"]
                #attn_implementation="flash_attention_2",
                #device_map=device_map,
            )

        self.model.eval()
        
        if device is not None:
            self.model = self.model.to(device)
        else:
            device = 0
        self.device = device
        print(self.model)

    # Perform a "logit-query" Given a prompt and a set of short completions to the prompt,
    # this infers the probabilities of each completion in one pass through the model.
    @torch.inference_mode()
    def query(self, prompt, completions, temperature=1.0, debug=False):
        self.model.eval()
        
        if debug:
            print("Query:")
            print(f"{prompt}")
        
        input_ids_list, input_lengths = tokenizer_completions(self.tokenizer, prompt, completions)
        prompt_len = input_lengths[0]
        labels = []
        for ids in input_ids_list[1:]:
            labels.append(ids[prompt_len])
       
        label_ids = torch.tensor(labels).flatten()
        label_tokens = self.tokenizer.convert_ids_to_tokens(label_ids)
        
        input_ids = torch.tensor([input_ids_list[0]], device=self.device)
        
        model_outputs = self.model(input_ids=input_ids, return_dict=True)
        logits = model_outputs['logits'][..., -1, :]
        
        if debug:
            print("Top-10 next tokens:")
            for decoded in decode_logits(self.tokenizer, logits):
                print(f"{decoded['p']:.2f} ({decoded['logit']}) {decoded['id']} = {decoded['token']}")
            print("Label tokens:")
            for id, token in zip(label_ids, label_tokens):
                print(f"{id.item()} {token}")

        logits = logits[..., -1, label_ids] / temperature
        probabilities = torch.softmax(logits, dim=-1)
        #indices = torch.argsort(probabilities, dim=-1, descending=True, stable=False)
        
        outputs = []
        for i in range(len(labels)):
            outputs.append(
                {
                    "id": labels[i],
                    "token": label_tokens[i],
                    "p": probabilities[i].item(),
                }
            )

        if debug:
            print("Weights:")
            for output in outputs:
                print(f"{output['id']} {output['token']} {output['p']:.2f}")
        
        return outputs

    # Generate a response to a prompt (or a batch or prompts); stops when all batch elements have generated EOS or max new tokens.
    def generate_response(
        self,
        generation_config,
        prompt,
        max_new_tokens,
        past_key_values=None,
    ):
        self.model.eval()
    
        encoded_prompts = self.tokenizer(
            [prompt],
            truncation=False,
            return_length=True,
        )
        
        lengths = encoded_prompts["length"]
        tokenizer_outputs = self.tokenizer.pad(
            encoded_prompts,
            padding="longest",
            return_tensors='pt',
            return_attention_mask=True,
        )
    
        input_ids = tokenizer_outputs['input_ids'].to(self.device)
        attention_mask = tokenizer_outputs['attention_mask'].to(self.device)
        lengths = tokenizer_outputs["length"]
        stopping_criteria = EosStoppingCriteria(self.tokenizer)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
    
        total_length = len(outputs.sequences[0])
        new_tokens = outputs.sequences[0][lengths[0]:]
        output_text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )
        
        return {
            "response": output_text.strip(),
            "token_ids": outputs.sequences[0],
            "new_tokens": total_length - lengths[0],
            "past_key_values": outputs.past_key_values,
        }

    # Gen a named generation config
    def named_generation_config(self, name):
        preset = generation_presets[name]
    
        return GenerationConfig(
            max_new_tokens=512,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **preset
        )

    # Given a list of string, returns the number of tokens in each string.
    def token_count(self, prompts):
        encoded_prompts = self.tokenizer(
            prompts,
            truncation=False,
            return_length=True,
            add_special_tokens=True,
        )
        length = encoded_prompts["length"]
        return length

# Stop generation after all batch elements have generated an EOS token.
# Stores the index of the first generated EOS token for each batch element in "self.eos_index,"
# which can be used to slice off whatever extra junk was generated after it.
# Note: This is a stateful object. A new instance should be created for each call to generate().
class EosStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.eos_token = tokenizer.eos_token_id
        self.done = None
        self.eos_index = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size, seq_len = input_ids.shape
        
        # Lazy construct a bool state for each batch element
        if self.done == None:
            self.done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            self.eos_index = torch.zeros(batch_size, dtype=torch.int, device=input_ids.device)

        # Get last token ids in batch
        last_ids = input_ids[:, -1]

        # Create mask of where the last token is EOS
        done_update = self.done | (last_ids == self.eos_token)
        
        # Store the indices where we stopped at for each sequence in the batch.
        # Where the 'done' state has changed, store the seq_len (last index), else 0
        eos_index_update = torch.where(done_update ^ self.done, torch.full_like(self.eos_index, seq_len), 0)

        # Add the update to the indices
        self.eos_index += eos_index_update

        # Update the done flags
        self.done = done_update

        # Return True, if all done.
        return self.done.all()

# Get a random preset key: i.e. "Big-O"
def random_preset():
    return random_preset_keys[random.randint(0, len(random_preset_keys)-1)]

# Decode top-k logit probabilities
def decode_logits(tokenizer, logits, k=10):
    top_k_logits, ids = torch.topk(logits, k=k, dim=-1, sorted=True)
    probabilities = torch.softmax(top_k_logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    outputs = []
    for id, token, logit, p, in zip(ids[0].tolist(), tokens, top_k_logits[0].tolist(), probabilities[0].tolist()):
        outputs.append(
            {
                "id": id,
                "token": token,
                "logit": logit,
                "p": p
            }
        )
    return outputs

# Given a base prompt and a set of completions, returns the resulting token ids of the completions.
def tokenizer_completions(tokenizer, prompt, completions):
    prompt_len = len(tokenizer(prompt, add_special_tokens=True)['input_ids'])

    prompts = [ prompt ]
    for completion in completions:
        prompts.append(prompt + completion)
    outputs = tokenizer(prompts, add_special_tokens=True, return_length=True)
    input_ids = outputs['input_ids']
    lengths = outputs['length']

    prompt_len = lengths[0]
    for length in lengths[1:]:
        if length <= prompt_len:
            raise Exception("Prompt has same length as completion.")

    # Return the tokenized prompts and lengths, with the original prompt at index 0
    return input_ids, lengths

# A container for a prompt template
# causal_lm: a CausalLM object
# model_instruction_template: The model specfic instruction template.
# template: The domain specific instruction template
# filter: Post processor for generation
class InstructGen():
    def __init__(self, causal_lm, model_instruction_template, template, filter=None, generation_config="Big-O", debug_level=1):
        self.causal_lm = causal_lm
        self.debug_level = debug_level
        self.generation_config = causal_lm.named_generation_config(generation_config)
        self.environment =  jinja2.Environment()
        self.template = self.environment.from_string(template)
        self.filter = filter
        self.model_template = self.environment.from_string(model_instruction_template)

    def __call__(self, max_new_tokens=1024, **kwargs):
        instruction = self.template.render(**kwargs)
        prompt = self.model_template.render(instruction=instruction)
        if self.debug_level > 1:
            print(f"{'prompt':-^80}")
            print(prompt)

        outputs = self.causal_lm.generate_response(
            self.generation_config,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        
        response = outputs["response"]
        
        if self.debug_level > 1:
            print(f"{'response':-^80}")
            print(response)

        if self.filter is not None:
            response = self.filter(response, **kwargs)
        
        return response