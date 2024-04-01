import random
import re
import os
import copy
import jinja2
from jinja2.exceptions import TemplateError

from rpbuild import load_template
from rpbuild.data import print_message, impersonation_filter_re, make_message
from rpbuild.model import InstructTemplate

# Director or "GM" agent.
# Mediates the interaction between the characters and gives direction to the interaction.
class Director():
    def __init__(
        self,
        causal_lm,
        script,
        generation_config="Big-O",
        template=load_template("director_prompt.jinja"),
        debug=False,
        history_token_limit=None
    ):
        self.debug = debug
        self.history_token_limit = history_token_limit
        self.causal_lm = causal_lm
        self.generation_config = causal_lm.named_generation_config(generation_config)
        self.prompt_t = InstructTemplate(
            instruct_template=causal_lm.instruct_template,
            template=template,
        )
        self.script = script

    def __call__(self, char, user, message_count):
        messages=char.get_conversation(token_limit=self.history_token_limit, include_system=False, min_len=1)
        prompt = self.prompt_t.render(
            messages=messages,
            user_plist=user.plist,
            char_plist=char.plist,
            char=char.name,
            user=user.name,
            scenario=self.script,
            message_count=message_count,
        ) + "Director:"

        if self.debug:
            print_message(make_message(self.causal_lm, prompt, "Director Prompt"))

        outputs = self.causal_lm.generate_response(
            self.generation_config,
            prompt,
            max_new_tokens=512,
        )

        response = outputs["response"]
        
        # Remove anything which looks like a direct character prompt.
        # TODO: This is just a quick work-around. Ultimatley, reworking the
        # director prompt to avoid this would be better, as regex filtering is a
        # rather blunt instrument.
        response = impersonation_filter_re.sub("", response)
        return response
