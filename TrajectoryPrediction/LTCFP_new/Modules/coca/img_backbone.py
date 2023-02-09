import torch
from torch import nn

from helper import SEP
from collections import OrderedDict

class ImgBackbonePretrained(nn.Module):
    def __init__(self, arch='BeIT', ckpt = None) -> None:
        super().__init__()

        self.arch = arch

        if arch=='BeIT':
            from transformers import BeitForSemanticSegmentation
            self.backbone = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-base-finetuned-ade-640-640').beit

            ckpt='BeIT_grayscale_pretrain_epoch=21-step=46618.ckpt'
            self.hidden_size = self.backbone.config.hidden_size

        elif arch=='SegFormer':
            from transformers import SegformerForSemanticSegmentation
            self.backbone = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-cityscapes-1024-1024').segformer
 
            ckpt='SegFormer_grayscale_pretrain_epoch=46-step=103831.ckpt'
            self.hidden_size = self.backbone.config.hidden_sizes[-1]

        else:
            raise NotImplementedError

        CKPT_PATH = SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'checkpoints', 'img2img_pretrain', ckpt])
        state_dict = OrderedDict([(key.replace(f'model.model.{arch.lower()}.', ''), tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items() if key.startswith(f'model.model.{arch.lower()}.')])
        module_state_dict = self.backbone.state_dict()

        mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
        lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]

        load_dict = OrderedDict()
        for key, tensor in module_state_dict.items():
            # if (key in state_dict.keys()) and ('decode_head' not in key):
            if key in state_dict.keys():
                load_dict[key] = state_dict[key]
            else:
                # if key == 'model.model.classifier.classifier.weight':
                #     load_dict[key] = state_dict['model.model.classifier.weight']
                # elif key == 'model.model.classifier.classifier.bias': 
                #     load_dict[key] = state_dict['model.model.classifier.bias']
                # else:
                #     load_dict[key] = tensor
                load_dict[key] = tensor

        self.backbone.load_state_dict(load_dict)

        # if self.arch == 'BeIT':
        #     hidden_states = self.backbone.forward(pixel_values = pixel_values, bool_masked_pos = None, head_mask = None, output_attentions = None, output_hidden_states = None, return_dict = None).last_hidden_state
        # elif self.arch == 'SegFormer':
        #     hidden_states = self.backbone.forward(pixel_values=pixel_values, labels = None, output_attentions = None, output_hidden_states = None, return_dict = None)

        if self.arch == 'BeIT':
            self.forward = self.forward_beit#backbone.forward#(pixel_values = pixel_values, bool_masked_pos = None, head_mask = None, output_attentions = None, output_hidden_states = None, return_dict = None).last_hidden_state
        elif self.arch == 'SegFormer':
            self.forward = self.forward_segformer#backbone.forward#(pixel_values=pixel_values, output_attentions = None, output_hidden_states = None, return_dict = self.backbone.config.use_return_dict).last_hidden_state
            

    def forward_beit(self, pixel_values):
        output = self.backbone.forward(pixel_values = pixel_values, bool_masked_pos = None, head_mask = None, output_attentions = None, output_hidden_states = None, return_dict = None).last_hidden_state
        return output

    def forward_segformer(self, pixel_values):
        output = self.backbone.forward(pixel_values=pixel_values, output_attentions = None, output_hidden_states = None, return_dict = self.backbone.config.use_return_dict).last_hidden_state
        output = output.flatten(2).transpose(1, 2)
        return output

    
    def forward(self, pixel_values):

        raise NotImplementedError

        return self.arch_forward(pixel_values)

        if self.arch == 'BeIT':
            hidden_states = self.backbone.forward(pixel_values = pixel_values, bool_masked_pos = None, head_mask = None, output_attentions = None, output_hidden_states = None, return_dict = None).last_hidden_state
        elif self.arch == 'SegFormer':
            hidden_states = self.backbone.forward(pixel_values=pixel_values, output_attentions = None, output_hidden_states = None, return_dict = self.backbone.config.use_return_dict).last_hidden_state
        
        return hidden_states