import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
import openvino as ov
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, PreTrainedModel
from transformers.onnx import FeaturesManager
import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nncf

class ExportModel(PreTrainedModel):
    def __init__(self, base_model, config):
        super().__init__(config)
        self.model = base_model

    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):

        #out = self.model(input_ids, attention_mask, token_type_ids)
        #return out

        out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        return {
            # "logits": out["logits"],
            # "hidden_states": torch.stack(list(out["hidden_states"]))
            "hidden_states": torch.cat(out["hidden_states"][-3:-2], -1)[0]
        }
    
class Bert():
    def __init__(self):
        model_id='bert-base-uncased'
        self.models = AutoModelForMaskedLM.from_pretrained(model_id)
        self.tokenizers = AutoTokenizer.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            self.tokenizers.save_pretrained(out_dir)
        except Exception as e:
            print(f'tokenizer loading failed with {e}')

    def prepare_calibration_data(self, dataloader, init_steps):
        data = []
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch is not None:
                with torch.no_grad():
                    inputs_dict = {}
                    inputs_dict['input_ids'] = batch['input_ids'].squeeze(0)
                    inputs_dict['token_type_ids'] = batch['token_type_ids'].squeeze(0)
                    inputs_dict['attention_mask'] = batch['attention_mask'].squeeze(0)
                    data.append(inputs_dict)
        return data

    def prepare_dataset(self, example_input=None, opt_init_steps=1, max_train_samples=1000):
        class CustomDataset(Dataset):
            def __init__(self, data_count=100, dummy_data=None):
                self.dataset = []
                for i in range(data_count):
                    self.dataset.append(dummy_data)
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self,idx):
                data = self.dataset[idx]
                return data
        """
        Prepares a vision-text dataset for quantization.
        """
        dataset = CustomDataset(data_count=1, dummy_data=example_input)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)
        calibration_data = self.prepare_calibration_data(dataloader, opt_init_steps)
        return calibration_data

    def bert_convert_to_ov(self, export_model, ov_path, language = "EN"):
        if "ZH" in language:
            model_id='bert-base-multilingual-uncased'
            text = "当需要把翻译对象表示, 可以使用这个方法。"
        elif "EN" in language:
            model_id='bert-base-uncased'
            text = "i am absolutely thrilled to share this incredible news with everyone"
        #self.models = AutoModelForMaskedLM.from_pretrained(model_id)
        #self.tokenizers = AutoTokenizer.from_pretrained(model_id)
        #self.config = AutoConfig.from_pretrained(model_id)
        
        #export_model = ExportModel(models, config)
        

        inputs = self.tokenizers(text, return_tensors="pt")

        example_input = {
                "input_ids": inputs['input_ids'],
                "token_type_ids": inputs['token_type_ids'],
                "attention_mask": inputs['attention_mask'],
            }
            
        ov_model = ov.convert_model(
            #self.models,
            export_model,
            example_input = example_input,
        )
        """
        get_input_names = lambda: ["input_ids", "token_type_ids", "attention_mask"]
        for input, input_name in zip(ov_model.inputs, get_input_names()):
            input.get_tensor().set_names({input_name})
        outputs_name = ['hidden_states']
        for output, output_name in zip(ov_model.outputs, outputs_name):
            output.get_tensor().set_names({output_name})
        
        
        reshape model
        Set the batch size of all input tensors to 1 to facilitate the use of the C++ infer
        If you are only using the Python pipeline, this step can be omitted.
        """  
        shapes = {}     
        for input_layer  in ov_model.inputs:
            shapes[input_layer] = input_layer.partial_shape
            shapes[input_layer][0] = 1
        ov_model.reshape(shapes)
         

        self.save_tokenizer(self.tokenizers, Path(ov_path))
        self.models.config.save_pretrained(Path(ov_path))
        
        ov_model_path = Path(f"{ov_path}/bert_{language}.xml")
        ov.save_model(ov_model, Path(ov_model_path))
        if True:
            calibration_data = self.prepare_dataset(example_input=example_input)
            calibration_dataset = nncf.Dataset(calibration_data)
            # quantized_model = nncf.quantize(
            #     model=ov_model,
            #     calibration_dataset=calibration_dataset,
            #     preset=nncf.QuantizationPreset.MIXED,
            #     # subset_size=len(calibration_data),
            #     )
            quantized_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=calibration_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                subset_size=len(calibration_data),
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
            )

            ov.save_model(quantized_model, Path(f"{ov_path}/bert_{language}_int8.xml"))

        



        
    def ov_bert_model_init(self, ov_path=None, language = "EN"):
        core = ov.Core()
        #if self.use_int8:
        #    ov_model_path = Path(f"{ov_path}/bert_{language}_int8.xml")
        #else:
        ov_model_path = Path(f"{ov_path}/bert_{language}_int8.xml")
        self.bert_model = core.read_model(Path(ov_model_path))
        self.bert_compiled_model = core.compile_model(self.bert_model, 'CPU')
        self.bert_request = self.bert_compiled_model.create_infer_request()
                
        self.bert_tokenizer = AutoTokenizer.from_pretrained(ov_path, trust_remote_code=True)
        self.bert_config = AutoConfig.from_pretrained(ov_path, trust_remote_code=True)
        
    def ov_bert_infer(self, input_ids=None, token_type_ids=None, attention_mask=None):
        inputs_dict = {}
        inputs_dict['input_ids'] = input_ids
        inputs_dict['token_type_ids'] = token_type_ids
        inputs_dict['attention_mask'] = attention_mask
        
        self.bert_request.start_async(inputs_dict, share_inputs=True)
        self.bert_request.wait()
        #bert_output = (self.bert_request.get_tensor("hidden_states").data.copy())
        bert_output = self.bert_request.get_output_tensor(0).data.copy()
        return bert_output

    def get_ov_bert_feature(self, text, bert_model=None):
        inputs = self.tokenizers(text, return_tensors="pt")
        res = self.ov_bert_infer(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask'])
        #breakpoint()
        res = torch.tensor(res)
        #breakpoint()
        #res = torch.tensor(res)[-3:-2][0][0]
        print("===ov_bert.var===")
        # 按行计算均值
        
        mean_per_row = torch.mean(res, dim=1)

        # 按行计算方差
        variance_per_row = torch.var(res, dim=1, unbiased=False)

        print("Mean per row:", mean_per_row)
        print("Variance per row:", variance_per_row)
        pass
        


    def get_bert_feature(self, text, device="cpu", model_id='bert-base-multilingual-uncased'):
       
        with torch.no_grad():
            inputs = self.tokenizers(text, return_tensors="pt")
            for i in inputs:
                    inputs[i] = inputs[i].to(device)
            res = self.models(**inputs, output_hidden_states=True)
            #print(res)
        #breakpoint()
            #res = self.models(**inputs, output_hidden_states=True)
            #res = torch.cat(res["hidden_states"]).cpu()
        #breakpoint()
            #res1 = res0["hidden_states"]
        
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu() 
            #breakpoint()
              
        print("===torch_bert.var===")
        #breakpoint()
            # 按行计算均值
        mean_per_row = torch.mean(res, dim=1)

            # 按行计算方差
        variance_per_row = torch.var(res, dim=1, unbiased=False)

        print("Mean per row:", mean_per_row)
        print("Variance per row:", variance_per_row)
        
        pass

 
    def get_export_model_bert_feature(self, export_model, text, device="cpu", model_id='bert-base-multilingual-uncased'):
       
        with torch.no_grad():
            inputs = self.tokenizers(text, return_tensors="pt")
            for i in inputs:
                    inputs[i] = inputs[i].to(device)
            res = export_model(**inputs, output_hidden_states=True)
            res = res["hidden_states"].cpu()
        #breakpoint()
            #res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()        
        print("===export torch_bert.var===")
            #breakpoint()
            # 按行计算均值
        mean_per_row = torch.mean(res, dim=1)

            # 按行计算方差
        variance_per_row = torch.var(res, dim=1, unbiased=False)

        print("Mean per row:", mean_per_row)
        print("Variance per row:", variance_per_row)
        
        pass
    def ov_bert_model_init(self, ov_path=None, language = "EN"):
        core = ov.Core()

        ov_model_path = Path(f"{ov_path}/bert_{language}_int8.xml")
        self.bert_model = core.read_model(Path(ov_model_path))
        self.bert_compiled_model = core.compile_model(self.bert_model, 'CPU')
        self.bert_request = self.bert_compiled_model.create_infer_request()
                
        #self.bert_tokenizer = AutoTokenizer.from_pretrained(ov_path, trust_remote_code=True)
        self.bert_config = AutoConfig.from_pretrained(ov_path, trust_remote_code=True)

            


if __name__ == "__main__":
    # from text.chinese_bert import get_bert_feature

    text = "i am absolutely thrilled to share this incredible news with everyone"
    obj = Bert()
    obj.get_bert_feature(text)
    export_model = ExportModel(obj.models, obj.config)
   

    obj.bert_convert_to_ov(export_model,"BERT_int8")
    obj.ov_bert_model_init('BERT_int8')
    obj.get_ov_bert_feature(text)



