from transformers import AutoTokenizer
from openvino import compile_model, save_model
from openvino.runtime import Model, PartialShape, Type, op
from openvino_tokenizers import _get_factory
from openvino_tokenizers.constants import TOKEN_IDS_INPUT_NAME, ATTENTION_MASK_INPUT_NAME, TOKEN_TYPE_IDS_INPUT_NAME, \
    TOKENIZER_NAME, DETOKENIZER_NAME
from openvino_tokenizers.hf_parser import TransformersTokenizerPipelineParser
from openvino_tokenizers.tokenizer_pipeline import SpecialTokensSplit, BasePipelineStep
from openvino_tokenizers.utils import TokenzierConversionParams, change_outputs_type
from openvino.preprocess import PrePostProcessor

def get_tokenizer_detokenizer(hf_tokenizer, params):
    pipeline = TransformersTokenizerPipelineParser(hf_tokenizer, params).parse()
    pipeline.finalize()

    string_inputs = [op.Parameter(Type.string, PartialShape(["?"])) for _ in range(pipeline.number_of_inputs)]


    processing_outputs = []
    for input_node in string_inputs:
        input_node = _get_factory().create("StringTensorUnpack", input_node.outputs()).outputs()
        ragged = []
        regex_split_outputs = []
        if isinstance(pipeline.steps[0], SpecialTokensSplit):
            input_node = pipeline.add_ragged_dimension(input_node)
            input_node = pipeline.steps[0].get_ov_subgraph(input_node)
            ragged, input_node = input_node[:2], input_node[2:]

        for step in pipeline.normalization_steps:
            input_node = step.get_ov_subgraph(input_node)

        if not ragged:
            input_node = pipeline.add_ragged_dimension(input_node)
        else:
            input_node = ragged + input_node

        for step in pipeline.pre_tokenization_steps:
            input_node = step.get_ov_subgraph(input_node)

        regex_split_outputs = input_node[2:5]
        for step in pipeline.tokenization_steps:
            input_node = step.get_ov_subgraph(input_node[:-1])

        processing_outputs.extend(input_node)

    for step in pipeline.post_tokenization_steps:
        processing_outputs = step.get_ov_subgraph(processing_outputs)

    ov_tokenizer = Model(processing_outputs, string_inputs, name=TOKENIZER_NAME)
    change_outputs_type(ov_tokenizer, Type.i64)
    output_names = hf_tokenizer.model_input_names

    ov_tokenizer_output_names = [TOKEN_IDS_INPUT_NAME, ATTENTION_MASK_INPUT_NAME]
    if len(output_names) == 3 and len(ov_tokenizer.outputs) == 3:
        ov_tokenizer_output_names.insert(1, TOKEN_TYPE_IDS_INPUT_NAME)

    filtered_outputs = []
    for i, output_name in enumerate(ov_tokenizer_output_names):
        current_output = next(
            (output for output in ov_tokenizer.outputs if output.any_name == output_name),
            False,
        )
        if current_output:
            filtered_outputs.append(current_output)
            continue

        if output_name in output_names:
            ov_tokenizer.output(i).tensor.add_names({output_name})
            filtered_outputs.append(ov_tokenizer.output(i))

    regex_split_outputs = _get_factory().create("StringTensorPack", regex_split_outputs).outputs()
    regex_split_outputs[0].tensor.add_names({"regex_string_output"})

    tokenizer_model = Model(filtered_outputs + regex_split_outputs, ov_tokenizer.get_parameters(), TOKENIZER_NAME)
    
    ### detokenizer

    vocab = pipeline.tokenization_steps[-1].vocab
    detokenizer_input = op.Parameter(Type.i64, PartialShape(["?", "?"]))
    detokenizer_outputs = (
        _get_factory()
        .create(
            "VocabDecoder",
            [*detokenizer_input.outputs(), *BasePipelineStep.create_string_constant_node(vocab).outputs()],
        )
        .outputs()
    )[2:5]
    detokenizer_outputs = _get_factory().create("StringTensorPack", detokenizer_outputs).outputs()
    detokenizer_model = Model(detokenizer_outputs, [detokenizer_input], DETOKENIZER_NAME)
    # transpose the input shape of detokenizer_model
    ppp = PrePostProcessor(detokenizer_model)
    # transpose [number_of_tokens, 1] to [1, number_of_tokens] 
    # refer to https://github.com/openvinotoolkit/openvino/issues/16331 
    # refer to https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/Make-inference-faster-via-pre-process-of-data/m-p/1397729
    ppp.input().preprocess().convert_layout([1,0])
    detokenizer_model = ppp.build()
    return tokenizer_model, detokenizer_model


hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
ov_tokenizer, ov_detokenizer = get_tokenizer_detokenizer(hf_tokenizer, TokenzierConversionParams())



bert_subword_tokenizer_path = "bert_subword_tokenizer.xml"
bert_subword_detokenizer_path = "bert_subword_detokenizer.xml"

print("Save bert tokenizer: ", bert_subword_tokenizer_path)
print("Save bert detokenizer: ", bert_subword_detokenizer_path)

save_model(ov_tokenizer, bert_subword_tokenizer_path)
save_model(ov_detokenizer, bert_subword_detokenizer_path)

compiled_tokenzier = compile_model(ov_tokenizer)
compiled_detokenzier = compile_model(ov_detokenizer)
#text_input = ["I have a new GPU!"]
text_input = ["i have installed a fortran compiler"]
print("text_input: ", text_input)
#hf_output = hf_tokenizer(text_input, return_tensors="np")
hf_output = hf_tokenizer.tokenize(text_input[0])
print("hf_output: ", hf_output)
ov_output = compiled_tokenzier(text_input)

print("ov_output: ", ov_output["regex_string_output"])
# have to transpose input ids to get the individual tokens detokenized
# shape is [number_of_tokens, 1]
print(compiled_detokenzier(ov_output["input_ids"])) 
