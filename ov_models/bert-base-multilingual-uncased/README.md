## How to get openvino tokenizer
```python
pip install transfomers
pip install openvino_tokenizers
convert_tokenizer google-bert/bert-base-multilingual-uncased -o bert-base-multilingual-uncased --skip-special-tokens --trust-remote-code --utf8_replace_mode replace
```