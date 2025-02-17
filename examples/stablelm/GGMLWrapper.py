import ctypes
from typing import Tuple
from langchain import PromptTemplate
import sys

class GGMLWrapper:
    def __init__(self):
        self.lib = ctypes.CDLL('../../build/examples/stablelm/libstablelm_lib.dylib')
        self.lib.generate_text.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
        self.lib.generate_text.restype = ctypes.c_char_p
        # Declare the function prototypes
        self.lib.load_model.argtypes = [ctypes.c_char_p]
        self.lib.load_model.restype = ctypes.c_bool
        

    def generate_text(self, prompt: str, seed: int, n_predict: int, n_threads: int, top_k: int, top_p: int, temp: float) -> Tuple[str, int]:
        prompt_buffer = prompt.encode('utf-8')
        result_text = self.lib.generate_text(prompt_buffer, seed, n_predict, n_threads, top_k, top_p, temp)
        return result_text
    
    def load_model(self, model_path: str) -> bool:
        model_path_buffer = model_path.encode('utf-8')
        return self.lib.load_model(model_path_buffer)

gpt = GGMLWrapper()
# template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {instruction}

# ### Response:
# """
template = """Question:{instruction}
Answer:"""
promptTemplate = PromptTemplate(template=template, input_variables=["instruction"])

if gpt.load_model('../../models/models--lambdalabs--pythia-2.8b-deduped-synthetic-instruct/ggml-pythia-3b-lambda-q4_3.bin'):
    while True:
        print("Enter instruction:")
        instruction = sys.stdin.readlines()
        instruction_text = ' '.join(instruction)
        print("Thinking ...")
        result = gpt.generate_text(promptTemplate.format(instruction=instruction_text), -1, 128, 8, 50, 1.0, 0.75)
        print(result.decode('utf-8' ))
    # result = gpt.generate_text('The meaning of life is ', -1, 64, 8, 40, 0.9, 0.8)
    # print(result.decode('utf-8' ))
    # result = gpt.generate_text('### Instruction: What is Python?\n\n### Response:\n', -1, 64, 8, 40, 0.9, 0.8)
    # print(result.decode('utf-8' ))
    # result = gpt.generate_text('### Instruction: Translate this to Japanese: All is well!\n\n### Response:\n', -1, 64, 8, 40, 0.9, 0.8)
    # print(result.decode('utf-8' ))
    # result = gpt.generate_text('### Instruction: Tell me a joke.\n\n### Response:\n', -1, 64, 8, 40, 0.9, 0.8)
    # print(result.decode('utf-8' ))
else:
    print('Error loading model')
