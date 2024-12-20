from transformers import AutoTokenizer
from openvino import compile_model
import openvino_tokenizers 
import os
import time
print(f"Process ID: {os.getpid()}")

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
"""
convert_tokenizer google-bert/bert-base-multilingual-uncased -o bert-base-multilingual-uncased --skip-special-tokens --trust-remote-code --utf8_replace_mode replace
"""
ov_tokenizer = "openvino_tokenizer.xml"
compiled_tokenzier = compile_model(ov_tokenizer)


text_input = ["I am developing a clang-based c++ compiler"]
#print("text_input: ", text_input)

hf_output = hf_tokenizer(text_input[0])
print("hf_output: ", hf_output["input_ids"])
# Existing test
ov_output = compiled_tokenzier(text_input)
print("ov_output: ", ov_output["input_ids"])

# Measure time for compiled_tokenizer
start_time = time.time()
ov_output = compiled_tokenzier(text_input)
end_time = time.time()
print(f"Time taken for compiled_tokenizer: {(end_time - start_time) * 1000:.2f} ms")

# Additional tests


