import random
import re
import os
import copy
import jinja2
from jinja2.exceptions import TemplateError

from rpbuild import load_template
from rpbuild.data import print_message

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

        outputs = self.causal_lm.generate_response(
            self.generation_config,
            prompt,
            max_new_tokens=512,
        )
        response = outputs["response"]

        return response

    def _query(self, template, response):
        prompt = template.render(response=response)
        outputs = self.causal_lm.query(prompt, ["Yes", "No"])
        return outputs[0]['p']