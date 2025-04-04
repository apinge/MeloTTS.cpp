## Enable mini-bart-g2p for OpenVINO
### install optimum-cil and convert mini-bart-g2p

``` 
python -m pip install git+https://github.com/huggingface/optimum.git
pip install --upgrade --upgrade-strategy eager optimum[openvino]
optimum-cli export openvino -m cisco-ai/mini-bart-g2p text2text-generation --weight-format fp16
 ```

#### Usage in Optinum-intel
Ref
https://github.com/huggingface/optimum-intel/blob/87c431c9eb777a220a417214df1b9e6a1b957108/README.md?plain=1#L101-L113

```python
from transformers import pipeline, AutoTokenizer
from optimum.intel import OVModelForSeq2SeqLM
import torch

# Automatically detect device
device = 0 if torch.cuda.is_available() else -1  # In Hugging Face pipeline, -1 means using CPU

# Load model to the appropriate device
model_id = "text2text-generation" #folder name of cisco-ai/mini-bart-g2p  
#pipe = pipeline(task="text2text-generation", model="cisco-ai/mini-bart-g2p", device=device)
model = OVModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("translation_grapheme_to_phoneme", model=model, tokenizer=tokenizer)
# Input text
text = "hello world"
# Generate results for each word
result1 = pipe(text.split())
print(result1)

text = "co-workers coworkers hunter's hunter"
result2 = pipe(text.split())
print(result2)

text = "i am absolutely thrilled to share this incredible news with everyone"
result3 = pipe(text.split())
print(result3)
```

#### Usage with OpenVino Tokenizer


```
convert_tokenizer cisco-ai/mini-bart-g2p -o mini-bart-g2p_tokenizer --with-detokenizer --skip-special-tokens --trust-remote-code --utf8_replace_mode replace
```
https://github.com/huggingface/optimum-intel/blob/87c431c9eb777a220a417214df1b9e6a1b957108/optimum/intel/openvino/modeling_seq2seq.py#L358

For stateful model
https://github.com/huggingface/optimum-intel/blob/87c431c9eb777a220a417214df1b9e6a1b957108/optimum/exporters/openvino/stateful.py#L204


# Infer model with OV native api
```python
import numpy as np

import openvino.runtime as ov
from pathlib import Path
import torch

detokenizer = {0: '<s>', 1: '<pad>', 2: '</s>', 3: '<unk>', 4: '<mask>', 5: 'e', 6: 'a', 7: 's', 8: 'i', 9: 'r', 10: 'n', 11: 'AH0', 12: 'o', 13: 'N', 14: 't', 15: 'l', 16: 'S', 17: 'L', 18: 'T', 19: 'R', 20: 'K', 21: 'c', 22: 'd', 23: 'D', 24: 'u', 25: 'IH0', 26: 'm', 27: 'M', 28: 'Z', 29: 'h', 30: 'g', 31: 'p', 32: 'ER0', 33: 'IY0', 34: 'b', 35: 'B', 36: 'P', 37: 'EH1', 38: 'AE1', 39: 'AA1', 40: 'y', 41: 'k', 42: 'IH1', 43: 'F', 44: 'f', 45: 'G', 46: 'w', 47: 'V', 48: 'v', 49: 'NG', 50: "'", 51: 'IY1', 52: 'EY1', 53: 'HH', 54: 'W', 55: 'SH', 56: 'OW1', 57: 'AO1', 58: 'OW0', 59: 'AH1', 60: 'UW1', 61: 'AY1', 62: 'JH', 63: 'z', 64: 'CH', 65: 'Y', 66: 'AA0', 67: 'ER1', 68: 'EH2', 69: 'IH2', 70: 'TH', 71: 'AY2', 72: 'AE2', 73: 'EY2', 74: 'AA2', 75: 'EH0', 76: 'j', 77: 'AW1', 78: 'OW2', 79: 'x', 80: 'IY2', 81: 'UW0', 82: 'AO2', 83: 'UH1', 84: 'AE0', 85: 'q', 86: 'AO0', 87: 'AH2', 88: 'UW2', 89: 'AY0', 90: 'OY1', 91: '-', 92: 'EY0', 93: 'DH', 94: 'AW2', 95: 'ER2', 96: 'ZH', 97: 'UH2', 98: 'AW0', 99: 'UH0', 100: 'OY2', 101: 'OY0', 102: '.'}
tokenizer = {"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"<mask>":4,"e":5,"a":6,"s":7,"i":8,"r":9,"n":10,"AH0":11,"o":12,"N":13,"t":14,"l":15,"S":16,"L":17,"T":18,"R":19,"K":20,"c":21,"d":22,"D":23,"u":24,"IH0":25,"m":26,"M":27,"Z":28,"h":29,"g":30,"p":31,"ER0":32,"IY0":33,"b":34,"B":35,"P":36,"EH1":37,"AE1":38,"AA1":39,"y":40,"k":41,"IH1":42,"F":43,"f":44,"G":45,"w":46,"V":47,"v":48,"NG":49,"'":50,"IY1":51,"EY1":52,"HH":53,"W":54,"SH":55,"OW1":56,"AO1":57,"OW0":58,"AH1":59,"UW1":60,"AY1":61,"JH":62,"z":63,"CH":64,"Y":65,"AA0":66,"ER1":67,"EH2":68,"IH2":69,"TH":70,"AY2":71,"AE2":72,"EY2":73,"AA2":74,"EH0":75,"j":76,"AW1":77,"OW2":78,"x":79,"IY2":80,"UW0":81,"AO2":82,"UH1":83,"AE0":84,"q":85,"AO0":86,"AH2":87,"UW2":88,"AY0":89,"OY1":90,"-":91,"EY0":92,"DH":93,"AW2":94,"ER2":95,"ZH":96,"UH2":97,"AW0":98,"UH0":99,"OY2":100,"OY0":101,".":102}

text = "hello"
text = text.lower()  # Input text only lower case
# turn text to input id
input_ids = [tokenizer[char] for char in text]
input_ids = [tokenizer['<s>']] + input_ids + [tokenizer['</s>']]
input_ids = torch.tensor([input_ids]).to('cpu')
attention_mask = torch.ones_like(input_ids)


# Load model    
model_folder = Path("/home/gta/qiu/mini-bart-g2p/cisco-ai/mini-bart-g2p-no_cache")
model_encoder_path = model_folder / "openvino_encoder_model.xml"
model_decoder_path = model_folder / "openvino_decoder_model.xml"

core = ov.Core()
encoder = core.compile_model(model_encoder_path)
decoder = core.compile_model(model_decoder_path)

# Prepare input data
assert torch.equal(input_ids, torch.tensor([[0, 29, 5, 15, 15, 12, 2]])), "The tensors are not equal!"
assert torch.equal(attention_mask, torch.tensor([[1, 1, 1, 1, 1, 1, 1]])), "attention masrks are not equal!"
inputs = {
    'input_ids': input_ids.to("cpu"), 
    'attention_mask': attention_mask.to("cpu"),
}



# Print encoder input and output names and types
print("Encoder inputs:")
for input in encoder.inputs:
    print(f"Name: {input.get_any_name()}, Type: {input.get_element_type()}")

print("\nEncoder outputs:")
for output in encoder.outputs:
    print(f"Name: {output.get_any_name()}, Type: {output.get_element_type()}")

# Print decoder input and output names and types
print("\nDecoder inputs:")
for input in decoder.inputs:
    print(f"Name: {input.get_any_name()}, Type: {input.get_element_type()}")

print("\nDecoder outputs:")
for output in decoder.outputs:
    print(f"Name: {output.get_any_name()}, Type: {output.get_element_type()}")

last_hidden_state = torch.from_numpy(
            encoder(inputs, share_inputs=True, share_outputs=True)["last_hidden_state"])

#last_hidden_state = torch.from_numpy(encoder_outputs[encoder.outputs[0]]).to("cpu")
decoder_input_ids = torch.tensor([[2]])#Init state


while(True):
    print("==========")
    decoder_inputs = {
                    "input_ids":decoder_input_ids.to("cpu"),
                    "encoder_hidden_states":last_hidden_state.to("cpu"),
                    "encoder_attention_mask":attention_mask.to("cpu")
                    #"decoder_attention_mask": None,
                }
    decoder_outputs = decoder(decoder_inputs, share_inputs=True, share_outputs=True)
    print("Decoder outputs:")
    logits = torch.from_numpy(decoder_outputs["logits"]).to("cpu")
    print("logits shape:", logits.shape)
    probs = logits.softmax(dim=-1)
    values, predictions = probs.topk(1, dim=-1)
    predictions = predictions.squeeze(-1)
    #print(predictions.shape)
    #print(predictions)
    #print(predictions[:, -1].item())

    if(predictions[:, -1].item()==2):
        break
    decoder_input_ids = torch.cat([torch.tensor([[2]]), predictions], dim=1)
    del predictions, values, probs, logits, decoder_outputs
    print(decoder_input_ids)
    pass

    
# Convert tensor to NumPy array
if isinstance(decoder_input_ids, torch.Tensor):
    numpy_array = decoder_input_ids.numpy().flatten()
else:
    numpy_array = np.array(decoder_input_ids).flatten()
print(numpy_array)
# Convert indices to corresponding words or symbols
tokens = [detokenizer[i] for i in numpy_array]
# Join the token list into a string
res = ' '.join(tokens)
print(res) #</s> <s> HH EH1 L OW0
```