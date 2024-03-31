import os
from datasets import load_dataset, load_from_disk
import datasets
import random
import re

# One of my input datasets needs to have the example dialog split first; this enables the fix.
FIX_DIALOG_EXAMPLES = False
TEMPLATES_DIR = "templates"

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

char_token_re = re.compile(r"\{\{char\}\}")
user_token_re = re.compile(r"\{\{user\}\}")

# Load a template from the templates directory
def load_template(file_name):
    with open(os.path.join(TEMPLATES_DIR, file_name)) as file:
        return file.read()

# Substitute {{user}} and {{char}} in the provided text with the given values.
def substitute_names(text, char_name, user_name):
    text = char_token_re.sub(char_name, text)
    text = user_token_re.sub(user_name, text)
    return text

# Format a message dictionary object
def make_message(causal_lm, content, name=None, role=None):
    return {
        # Role is typically one of: "system" "assistant" "user"
        "role": role,

        # Message payload
        "content": content,

        # The number of tokens in the message
        "tokens": causal_lm.token_count(content)[0],

        # The name of the speaker
        "name": name,
    }
    
# Show summary of char and proxy_user
def print_char_summary(char_meta, user_meta):
    print(f"Char: {char_meta.name}\n")
    print(f"Char Summary: {char_meta.summary}\n")
    print(f"Proxy User: {user_meta.name}\n")
    print(f"Proxy User Summary: {user_meta.summary}\n")

# Dump a raw character dictionary object
# The object may only have a subset of the dictionary keys populated.
def dump_character_data(character_data):
    for feature_name in ( "char_name", "summary", "user_name", "preset", "pairing_reason", "name" ):
        feature = character_data.get(feature_name)
        if feature is not None:
            print(f"\n{feature_name}: {feature}")
    
    for feature_name in ( "description", "plist", "greeting", "scenario" ):
        feature = character_data.get(feature_name)
        if feature is not None:
            print(f"{feature_name:=^80}")
            print(f"\n{feature}")

    system = character_data.get("system")
    if system is not None:
        print_message(system)

    example_dialog = character_data.get("example_dialog")
    if example_dialog is not None:
        for i, example in enumerate(example_dialog):
            print(f"{' dialog example-' + str(i):=^80}")
            print(example)
    
    conversation = character_data.get("conversation")
    if conversation is not None:
        print(f"{' conversation ':=^80}")
        print_conversation(conversation, director_log=character_data.get("director_log"))
    
    proxy = character_data.get("proxy")
    if proxy is not None:
        print(f"{' Proxy User ':*^80}")
        dump_character_data(proxy)

# Print a message object -- diagnostics
def print_message(message):
    content = message['content']
    role = message['role']
    if not role:
        role = ""
    tokens = message.get('tokens')
    if tokens:
        tokens = f" ({tokens})"
    else:
        tokens = ""
    name = message.get('name')
    if not name:
        name = "anonymous"
    print(f"{' ' + role + ':' + name + tokens:-^80}")
    print(content)

# Diagnostic: Print a conversation.
def print_conversation(conversation, director_log=None):
    if director_log is not None:
        d_iter = iter(director_log)
        d_msg = next(d_iter)
    
    for i, message in enumerate(conversation):
        if director_log is not None and d_msg["index"] == i:
            print_message(d_msg)
            try:
                d_msg = next(d_iter)
            except StopIteration:
                director_log = None
            
        print_message(message)

# Get a random character from the dataset
def random_char(dataset):
    return dataset[random.randint(0, len(dataset)-1)]

# Convert extended conversation format to standard format
# This should allow the conversation to work with standard chat templates.
# - name are prepended to messages
# - token counts, names, are removed
# - outputs only "role" and "content"
def flatten_conversation(
    conversation,
):
    # Copy the conversation, as we don't want to add this to the dataset.
    output = []

    def apply_name(name, content):
        return name + ": " + content
    
    for message in conversation:
        role = message["role"]
        content = message["content"]
        name = message["name"]
        
        match message["role"]:
            case "system":
                pass
            case "assistant" | "user":
                content = apply_name(name, content)
            case _:
                raise RuntimeError(f"Undefined role {message['role']}")
        output.append( { "role": role, "content": content } )
            
    return output
