import re
from rpbuild.data import print_message

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