import random
import re
import os
import copy
import jinja2
from jinja2.exceptions import TemplateError

from rpbuild import load_template
from rpbuild.data import make_message, substitute_names, print_conversation

start_token_re = re.compile(r"<START>")

FIX_DIALOG_EXAMPLES = False

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

