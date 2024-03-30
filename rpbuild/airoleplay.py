from datasets import load_dataset, load_from_disk
import transformers
from transformers import StoppingCriteria
import datasets
import tqdm
import json
import time
import random
import re
import os
import copy
import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# One of my input datasets needs to have the example dialog split first; this enables the fix.
FIX_DIALOG_EXAMPLES = False

# Token generation target default
MAX_TOKENS = 4000

TEMPLATES_DIR = "templates"

char_token_re = re.compile(r"\{\{char\}\}")
user_token_re = re.compile(r"\{\{user\}\}")

# Identify usage of "plist" format in output
square_brackets_re = re.compile(r"[\[\]]")

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

# The primary inference entry-point for this task
# Given a batch of characters meta data, generate new data and attach it.
def generate_dialog(batch, causal_lm, dataset, max_tokens=MAX_TOKENS):
    # Creater a writer to write the script outline.
    writer = Writer(
        causal_lm,
    )

    # Cleans up the dialog and tirggers "retakes" if issues with the dialog.
    dialog_filter = DialogFilter(causal_lm)

    # For each character in the batch
    for character in batch:
        char_meta = CharMeta.from_data(character)
        writer_selection = writer.choose_supporting_character(char_meta, get_random_proxy_users(dataset, char_meta, n=7))
        user_meta = writer_selection["meta"]
        
        # Generate a script
        scenario = writer(
            char_meta=char_meta,
            user_meta=user_meta,
        )

        #dialog_filter = DialogFilter(causal_lm)
        dialog_filter=None

        # Pick random generation presets for each character.
        char_preset = random_preset()
        user_preset = random_preset()

        # The main character
        char = Character(
            char_meta=char_meta,
            causal_lm=causal_lm,
            generation_config=char_preset,
            user_meta=user_meta,
            gen_post_process=dialog_filter,
            history_token_limit=1800,
        )

        # The proxy user
        user = Character(
            char_meta=user_meta,
            causal_lm=causal_lm,
            generation_config=user_preset,
            user_meta=char_meta,
            gen_post_process=dialog_filter,
            history_token_limit=1800,
        )

        director = Director(
            causal_lm,
            script=scenario,
            debug=False,
            history_token_limit=2000
        )
        
        # Create a director to give instructions to the characters for following the script
        roleplay = Roleplay(
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
        character['preset'] = char_preset
        character["conversation"] = char.conversation
        character["director_log"] = char.director_log

        # Fix example dialog
        if FIX_DIALOG_EXAMPLES:
            character["example_dialog"] = start_token_re.split(character["example_dialog"])[1:]

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

# Character Metadata abstraction
class CharMeta():
    def __init__(
        self,
        name,
        summary,
        description,
        plist,
        greeting,
        example_dialog,
    ):
        self.name = name
        self.summary = summary
        self.description = description
        self.plist = plist
        self.greeting = greeting
        self.example_dialog = example_dialog

    @staticmethod
    def from_data(data):
        return CharMeta(
            name=data["char_name"],
            summary=data["summary"],
            description=data["description"],
            plist=data["plist"],
            greeting=data["greeting"],
            # TODO: These should be pre-split in the dataset
            example_dialog=start_token_re.split(data["example_dialog"])[1:] if FIX_DIALOG_EXAMPLES else data["example_dialog"]
        )

# Top-level driver for automated roleplay
class Roleplay():
    def __init__(self, char, user, scenario=None, director=None, debug=True):
        self.debug = debug
        self.char = char
        self.user = user
        self.scenario = scenario
        self.director = director

    # Given two characters from the dataset, have them play recipprocol roles in a conversation.
    # Generation continues until max_tokens has been reached by either context.
    def __call__(self, max_tokens=4096):
        # Run until we are at or above the token limit
        while self.generate_next()["continue"]:
            # Get the larger of the two token counts
            user_tokens = self.char.count_tokens()
            char_tokens = self.user.count_tokens()
            tokens = max(user_tokens, char_tokens)
            if self.debug:
                print(f"[proxy user tokens={user_tokens} char tokens={char_tokens}]")
    
            # Token limit reached?
            if tokens >= max_tokens:
                if self.debug:
                    print("Stopping on token count")
                break
        return self.char.conversation, self.user.conversation
        
    def generate_next(self, instruction=None):
        outputs = {
            "continue": True,
            "role": None,
            "name": None,
            "response": None,
        }
        
        # Who spoke last, from char's perspective?
        last_role = self.char.conversation[-1]["role"]
        match last_role:
            case "user":
                outputs["role"] = "assistant"
                outputs["name"] = self.char.name
                char = self.char
                user = self.user
            case "assistant":
                outputs["role"] = "user"
                outputs["name"] = self.user.name
                char = self.user
                user = self.char
            case "system":
                outputs["role"] = "assistant"
                outputs["name"] = self.char.name
                greeting = self.char.greet()
                outputs["response"] = self.user.user_says(greeting["content"])
                if self.debug:
                    print_message(outputs["response"])
                return outputs
            case _:
                raise RuntimeError(f"Unknown role at end of conversation: {last_role}")

        if instruction is None and self.director:
            instruction = self.director(
                char=char,
                user=user,
                message_count=len(char.conversation)
            )
            
            message = char.director_says(instruction)
            user.director_says(instruction)
            
            if self.debug:
                print_message(message)
            
            if re.search(r"END STORY", instruction) is not None:
                outputs["continue"] = False
                if self.debug:
                    print("Stopping on directors instruction.")
                return outputs
        
        char_outputs = char.generate(instruction)

        if char_outputs["control"] == "abort":
            if self.debug:
                print("Aborting dialog")
            outputs["continue"] = False
            return outputs

        outputs["response"] = user.user_says(char_outputs["response"])
        if self.debug:
            print_message(outputs["response"])
        
        match char_outputs["control"]:
            case "stop":
                outputs["continue"] = False
            case "continue":
                pass
            case _:
                raise RuntimeError(f"Unrecognized control {outputs['control']}")
        return outputs

# Represents a character in a roleplay.
class Character(CharMeta):
    def __init__(
        self,
        char_meta,
        causal_lm,
        generation_config,
        user_meta,
        gen_post_process=None,
        history=None,
        max_examples=1,
        chat_template=load_template("chat_prompt.jinja"),
        sys_prompt_template=load_template("char_sys_prompt.jinja"),
        debug=False,
        history_token_limit=None
    ):
        super().__init__(**char_meta.__dict__)
        self.debug = debug
        self.history_token_limit = history_token_limit
        self.user_meta = user_meta
        self._sub_char_names()
        self.causal_lm = causal_lm
        self.generation_config = self.causal_lm.named_generation_config(generation_config)
        self.gen_post_process=gen_post_process
        #self.environment =  jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        self.environment =  jinja2.Environment()
        self.chat_template = self.environment.from_string(chat_template)
        self.sys_prompt_template = self.environment.from_string(sys_prompt_template)
        self.director_log = []
        
        if history is not None:
            self.conversation=copy.deepcopy(history)
            self._compute_conversation_tokens()
        else:
            # Init conversation history.
            self.conversation = []
    
            # Create persistent context
            sys_prompt = self.system_prompt(max_examples=max_examples)
            self.system_says(sys_prompt)

    # Replace character name templates with concrete names
    def _sub_char_names(self):
        self.plist = substitute_names(
            self.plist,
            self.name,
            self.user_meta.name
        )
        
        self.description = substitute_names(
            self.description,
            self.name,
            self.user_meta.name
        )
        
        self.example_dialog = [substitute_names(example, self.name, self.user_meta.name) for example in self.example_dialog]
        
    # Add dialog to conversation history
    def push_dialog(self, role, content, name):
        message = make_message(self.causal_lm, content, name, role)
        self.conversation.append(message)
        return message

    def system_says(self, text):
        return self.push_dialog("system", text, "system")
        
    def char_says(self, text):
        return self.push_dialog("assistant", text, self.name)

    def user_says(self, text):
        return self.push_dialog("user", text, self.user_meta.name)

    def director_says(self, text):
        message = make_message(self.causal_lm, text, "Director", "user")
        message["index"] = len(self.conversation)
        self.director_log.append(message)
        return message

    def greet(self):
        greeting = substitute_names(self.greeting, self.name, self.user_meta.name)
        return self.char_says(greeting)

    def _chat_prompt(self, **kwargs):
        template_kwargs = {**self.causal_lm.tokenizer.special_tokens_map, **kwargs}
        return self.chat_template.render(
            **template_kwargs
        )

    def chat_prompt(self, instruction=None):
        messages = self.get_conversation(token_limit=self.history_token_limit, min_len=2)
        return self._chat_prompt(
            char=self.name,
            user=self.user_meta.name,
            plist=self.plist,
            messages=messages,
            instruction=instruction,
        )
        
    def system_prompt(self, max_examples=1):
        return self.sys_prompt_template.render(
            example_dialog=self.example_dialog[:max_examples],
            description=self.description,
            plist=self.plist,
            summary=self.summary,
            char=self.name,
            user=self.user_meta.name,
        )
    
    def generate(self, instruction=None, max_new_tokens=512, auto_add=True, max_retries=2):
        while True:
            prompt = self.chat_prompt(instruction)
            if self.debug:
                print_message(make_message(self.causal_lm, prompt, "Character Prompt"))
            
            gen_outputs = self.causal_lm.generate_response(
                generation_config=self.generation_config,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            
            if self.gen_post_process == None:
                post_outputs = dict(
                    control="continue",
                    response=gen_outputs["response"],
                )
                break
            
            post_outputs = self.gen_post_process(response=gen_outputs["response"], char_meta=self, user_meta=self.user_meta)
            match post_outputs["control"]:
                case "retry":
                    if max_retries == 0:
                        # TODO: We are adding the failed generation to the end for later analysis
                        # This /should/ be translated to an abort and dropped for production.
                        if auto_add:
                            self.char_says(gen_outputs["response"])
                        return dict(control="stop", response=gen_outputs["response"])
                    max_retries -= 1
                    continue
                case "abort":
                    return dict(control="abort")
            break

        outputs = dict(
            control=post_outputs["control"],
            response=post_outputs["response"]
        )
        if auto_add:
            self.char_says(outputs["response"])
        return outputs

    def print_conversation(self, include_system=True, director_log=False):
        print_conversation(self.get_conversation(include_system=include_system), self.director_log if director_log else None)
    
    def _compute_conversation_tokens(self):
        for message in self.conversation:
            token_count = message.get("tokens")
            if token_count == None:
                content = message["content"]
                token_count = self.causal_lm.token_count(content)[0]
                content["tokens"] = token_count

    # If token limit is not None, try to limit conversation to target token count
    # If min_len, ignore token_limit for the first min_len messages
    def get_conversation(self, token_limit=None, include_system=True, min_len=None):
        assert(self.conversation[0]["role"] == "system")

        if token_limit is None and include_system:
            return self.conversation
        
        input_conversation = self.conversation[1:]
        if include_system:
            total_tokens = self.conversation[0]["tokens"]
        else:
            total_tokens = 0
        
        output_conversation = []

        # Reverse iterate through messages, skipping system message
        for message in reversed(input_conversation):
            total_tokens += message["tokens"]
            if token_limit is not None and total_tokens >= token_limit and (min_len == None or len(output_conversation) >= min_len):
                if self.debug:
                    print(f"Break on token limit {total_tokens} >= {token_limit}")
                break
            output_conversation.append(message)

        if include_system:
            output_conversation.append(self.conversation[0])
        
        # Reverse the order back to normal
        output_conversation.reverse()
        return output_conversation
        
    # Get total tokens in history
    def count_tokens(self, conversation=None):
        total_tokens = 0
        if conversation is None:
            conversation = self.conversation
        
        for message in self.conversation:
            token_count = message.get("tokens")
            if token_count == None:
                content = message["content"]
                token_count = self.causal_lm.token_count(content)[0]
            total_tokens += token_count
        return total_tokens

# Generated when the Writer fails to generate a valid selection when choosing a character.
class WriterSelectionError(Exception):
    def __init__(self, message, prompt, response):            
        super().__init__(message)
        self.prompt = prompt
        self.response = response
        
# "Writer" Agent. Chooses characters and writes a plot outline for them.
class Writer():
    def __init__(self, causal_lm, generation_config="Big-O", template_str=load_template("make_scenario.jinja"), debug_level=1):
        self.causal_lm = causal_lm
        self.debug_level = debug_level
        self.generation_config = causal_lm.named_generation_config(generation_config)
        self.environment =  jinja2.Environment()
        self.template = self.environment.from_string(template_str)
        self.choose_template = self.environment.from_string(load_template("select_character.jinja"))

    def __call__(self, char_meta, user_meta):
        char_plist = substitute_names(
            char_meta.plist,
            char_meta.name,
            user_meta.name
        )
    
        user_plist = substitute_names(
            user_meta.plist,
            user_meta.name,
            char_meta.name
        )
    
        greeting = substitute_names(
            char_meta.greeting,
            char_meta.name,
            user_meta.name
        )
        
        prompt = self.template.render(
            user_plist=user_plist,
            char_plist=char_plist,
            char=char_meta.name,
            user=user_meta.name,
            greeting=greeting,
        )

        if self.debug_level > 1:
            print_message(make_message(self.causal_lm, prompt, "Writer Prompt"))
        
        outputs = self.causal_lm.generate_response(
            self.generation_config,
            prompt,
            max_new_tokens=1024,
        )
        if self.debug_level > 2:
            print(f"{'outline':-^80}")
            print(outputs["response"])
        return outputs["response"]

    # Wrapper for handling exceptions, retry, and default, if retry fails
    # The most common error appears to be not using the exact name of the character, as
    # provided, resulting in a key lookup failure.
    # Occasionally, they try to pick the main character, despite explicit instructions.
    # This suggests that the prompt still needs work.
    def choose_supporting_character(self, char_meta, user_meta_list, retry_limit=2):
        for _ in range(retry_limit):
            try:
                supporting_char = self._choose_supporting_character(char_meta, user_meta_list)
            except WriterSelectionError as e:
                print(e)
                print(f"{' prompt ':-^80}")
                print(e.prompt)
                print(f"{' response ':-^80}")
                print(e.response)
                continue
            
            return supporting_char

        if self.debug_level:
            print("choose_supporting_character(): retry limit reached. Selecting default")
        return user_meta_list[0]
        
    def _choose_supporting_character(self, char_meta, user_meta_list):
        prompt = self.choose_template.render(
            char=char_meta,
            user_list=user_meta_list,
        )
            
        if self.debug_level > 1:
            print_message(make_message(self.causal_lm, prompt, "Writer Prompt"))
        
        outputs = self.causal_lm.generate_response(
            self.generation_config,
            prompt,
            max_new_tokens=256,
        )
        
        response = outputs["response"]
        output_json = '{   "name": ' + response
        if self.debug_level > 2:
            print(f"{'response':-^80}")
            print(output_json)
        
        try:
            result = json.loads(output_json)
        except json.JSONDecodeError:
            raise WriterSelectionError("could not parse json", prompt, response)

        name = result.get("name")
        if name is None:
            raise WriterSelectionError("name not present", prompt, response)
                
        for user_meta in user_meta_list:
            if user_meta.name == name:
                result["meta"] = user_meta
                if self.debug_level > 2:
                    print(result)
                return result

        raise WriterSelectionError("name not found in candidates", prompt, response)

# Director or "GM" agent.
# Mediates the interaction between the characters and gives direction to the interaction.
class Director():
    def __init__(
        self,
        causal_lm,
        script,
        generation_config="Big-O",
        template_str=load_template("director_prompt.jinja"),
        debug=False,
        history_token_limit=None
    ):
        self.debug = debug
        self.history_token_limit = history_token_limit
        self.causal_lm = causal_lm
        self.generation_config = causal_lm.named_generation_config(generation_config)
        self.environment =  jinja2.Environment()
        self.template = self.environment.from_string(template_str)
        self.script = script

    def __call__(self, char, user, message_count, debug=False):
        messages=char.get_conversation(token_limit=self.history_token_limit, include_system=False, min_len=1)
        prompt = self.template.render(
            messages=messages,
            user_plist=user.plist,
            char_plist=char.plist,
            char=char.name,
            user=user.name,
            scenario=self.script,
            message_count=message_count,
        )

        if self.debug:
            print_message(make_message(self.causal_lm, prompt, "Director Prompt"))

        for _ in range(2):
            outputs = self.causal_lm.generate_response(
                self.generation_config,
                prompt,
                max_new_tokens=512,
            )
            response = outputs["response"]
            p = self._query(self.impersonation, response)
            #if p < 0.7:
            break

        return response

    def _query(self, template, response):
        prompt = template.render(response=response)
        outputs = self.causal_lm.query(prompt, ["Yes", "No"])
        return outputs[0]['p']

        
# Filters and steers dialog generation.
class DialogFilter():
    def __init__(self, causal_lm, debug=False):
        self.causal_lm = causal_lm
        self.debug = debug
        self.environment = jinja2.Environment()
        self.impersonation = self.environment.from_string(load_template("detect_impersonation.jinja"))
        self.end_of_story = self.environment.from_string(load_template("detect_end.jinja"))

    def __call__(self, response, char_meta, user_meta, debug=False):
        outputs = self._clean_generation(response, char_meta.name, user_meta.name)
        if outputs["control"] == "abort":
            return outputs
        control = outputs["control"]
        response = outputs["response"]

        p = self._query(self.impersonation, response, char_meta.name, user_meta.name)
        if p > 0.7:
            if self.debug:
                print(f"{'retry on impersonation p=' + str(p):#^80}")
                print(response)
            return dict(control="retry")

        p = self._query(self.end_of_story, response, char_meta.name, user_meta.name)
        if p > 0.7:
            if self.debug:
                print(f"{'end of story p=' + str(p):#^80}")
            control = "stop"
        
        return dict(
            control=control,
            response=response
        )

    def _query(self, template, response, char_name, user_name):
        prompt = template.render(response=response, char=char_name, user=user_name)
        outputs = self.causal_lm.query(prompt, ["Yes", "No"])
        return outputs[0]['p']

    @staticmethod
    def _clean_generation(response, char_name, user_name):
        control = "continue"
        # Remove instances of impersonation / speaking more than once, 3rd parties, etc.
        response = re.sub(r"\n[\w ']{4,32}:.*", "", response, flags=re.DOTALL)
        #response = re.sub(f"({char_name}|{user_name}):.*", "", response, flags=re.DOTALL)
    
        # Remove all square-brackets, which models seem to add to dialog after seeing the plist.
        response = square_brackets_re.sub("", response)
    
        # End generation early on stopping tokens.
        m = re.search(r"<END>", response) 
        if m is not None:
            if self.debug:
                print(f"Stopping early on stop token: {m.group()}")
            response = response[:m.start()]
            control = "stop"
        
        return { "response": response.strip(), "control": control }

