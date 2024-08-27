
@echo off
echo benchmark_app  -m bert.xml   -data_shape "input_ids[1,29],attention_mask[1,29],token_type_ids[1,29]" -hint latency  
benchmark_app  -m bert.xml   -data_shape "input_ids[1,29],attention_mask[1,29],token_type_ids[1,29]" -hint latency  