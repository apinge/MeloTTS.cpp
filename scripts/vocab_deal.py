import re

def is_valid_token(token):
    # 检查是否是中文字符、英文字符、以"##"开头的英文字符、数字，或者是[UNK]、[CLS]、[SEP]这些特殊标记
    # 排除标点符号
    return re.match(r'^([\u4e00-\u9fff]+|[a-zA-Z0-9]+|##[a-zA-Z]+|\[UNK\]|\[CLS\]|\[SEP\])$', token) is not None

def filter_vocab(input_file, output_file, start_line=100):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile, start=0):
            if i >= start_line:
                token = line.strip()
                if is_valid_token(token):
                    outfile.write(token + ":"+str(i)+'\n')

# 使用方法
input_file = 'vocab.txt'
output_file = 'vocab_bert.txt'
filter_vocab(input_file, output_file)