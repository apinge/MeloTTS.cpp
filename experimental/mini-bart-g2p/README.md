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