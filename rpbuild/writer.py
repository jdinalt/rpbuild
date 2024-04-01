import random
import re
import os
import copy
import jinja2
import json
from jinja2.exceptions import TemplateError

from rpbuild import load_template
from rpbuild.data import random_char, substitute_names, print_message
from rpbuild.char import CharMeta, make_message
from rpbuild.model import InstructTemplate

# Generated when the Writer fails to generate a valid selection when choosing a character.
class WriterSelectionError(Exception):
    def __init__(self, message, prompt, response):            
        super().__init__(message)
        self.prompt = prompt
        self.response = response

# "Writer" Agent. Chooses characters and writes a plot outline for them.
class Writer():
    def __init__(
        self,
        causal_lm,
        generation_config="Big-O",
        template=load_template("make_scenario.jinja"),
        debug_level=1
    ):
        self.causal_lm = causal_lm
        self.debug_level = debug_level
        self.generation_config = causal_lm.named_generation_config(generation_config)
        self.write_prompt_t = InstructTemplate(
            instruct_template=causal_lm.instruct_template,
            template=template,
        )
        self.select_prompt_t = InstructTemplate(
            instruct_template=causal_lm.instruct_template,
            template=load_template("select_character.jinja"),
        )

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

        prompt = self.write_prompt_t.render(
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
        result = { 
            "name": user_meta_list[0].name,
            "meta": user_meta_list[0],
            "reason": "Default",
        }
        return result
        
    def _choose_supporting_character(self, char_meta, user_meta_list):
        prompt = self.select_prompt_t.render(
            char=char_meta,
            user_list=user_meta_list,
        ) + "{\n    \"char_name\":"
    
            
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

# Get N random unique proxy-users from the global dataset
# This filters out duplicate names.
def get_random_proxy_users(dataset, char, n=10):
    candidates = []
    while True:
        proxy_user = CharMeta.from_data(random_char(dataset))
        # Skip characters with the same name as the main character and ones without plists
        if proxy_user.name == char.name or len(proxy_user.plist) == 0:
            continue
        candidates.append(proxy_user)
        n -= 1
        if n == 0:
            return candidates