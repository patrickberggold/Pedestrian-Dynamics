import torch
import torch.nn as nn
from transformers import BeitForSemanticSegmentation, BeitConfig, BeitFeatureExtractor, BeitForImageClassification
from torch.nn import functional as F
from transformers.models.beit.modeling_beit import BeitUperHead, BeitPooler, BeitEmbeddings, BeitModel, BeitModelOutputWithPooling
from transformers.modeling_outputs import SemanticSegmenterOutput, ImageClassifierOutput
import math

class BeITTraj(nn.Module):
    def __init__(self, mode, output_channels, num_heads, additional_info) -> None:
        super().__init__()

        self.mode = mode
        self.output_channels = output_channels
        self.additional_info = additional_info # only implemented for one setting for now
        self.evac_head_mode = 'beit_class' # 'conv', 'beit', 'beit_class'

        assert mode in ['grayscale', 'evac_only', 'class_movie', 'density_reg', 'density_class', 'denseClass_wEvac']
        # loading config is pointless if pretrained model is used
        # config = BeitConfig(image_size=640, num_labels=self.output_channels)
        if mode == 'grayscale':

            higher_res = 'sicnn'
            HPTinitHighResNet = 'xavierNormal'
            HPTbnHighResNet = False

            self.model = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            # Replace classifier to output 1-dim
            self.model.decode_head.classifier = nn.Conv2d(self.model.config.hidden_size, self.output_channels, kernel_size=1)
            # self.model = BeitModel(config=config).from_pretrained('microsoft/beit-base-finetuned-ade-640-640') # 85.7 M Trainable params
            # self.model.beit.embeddings = EmbeddingsCustom(config=self.model.config)
        
            if higher_res=='sicnn':
                self.model.high_res_net = SICNN(init_fcn = HPTinitHighResNet, bnIntoConvBlocks = HPTbnHighResNet)
            elif higher_res=='mtpc':
                self.model.high_res_net = MTPC(init_fcn = HPTinitHighResNet, bnIntoConvBlocks = HPTbnHighResNet)
            else:
                raise ValueError
        
        elif mode == 'evac_only':
            if self.evac_head_mode == 'beit_class':
                # self.model = BeitForImageClassification.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
                self.model = BeitForImageClassification_custom.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
                if not self.additional_info:
                    # self.model.classifier = nn.Linear(self.model.config.hidden_size, 1)
                    self.model.classifier = nn.Sequential(
                        nn.Linear(self.model.config.hidden_size, 64), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.Linear(64, 64), nn.Linear(64, 1)
                    )
                else:
                    # self.model.classifier = nn.Linear(self.model.config.hidden_size, 1)
                    # self.model.classifier = FloorplanInformationNetwork(self.model.config.hidden_size)
                    self.model.classifier = DistanceResMLPInformationNetwork(self.model.config.hidden_size, hidden_size=16)
                    # self.model.classifier = DistanceAttentionInformationNetwork(self.model.config.hidden_size)
                # self.model = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
                # self.model.decode_head.classifier = nn.Conv2d(self.model.config.hidden_size, 32, kernel_size=1)
                # self.model.sicnn_evac = SICNN4Evac()
                # # self.model.sicnn_evac = ABPN4Evac()
            elif self.evac_head_mode in ['conv', 'evac']:
                self.model = BeitForSemanticSegmentation_evac.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
                self.model.pred_evac_only = True
                del self.model._modules['decode_head']
                self.model.evac_head = EvacHead(self.model.config, mode='beit')
            else:
                raise NotImplementedError
        elif mode == 'class_movie':
            self.model = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            self.model.decode_head.classifier = ClassMovieClassifier(
                input_channels=self.model.config.hidden_size,
                output_channels=self.output_channels,
                num_frames=num_heads
            )
            self.end_layer_module = EndLayerModule(
                output_channels=self.output_channels,
                num_frames=num_heads
                )
        elif mode in ['density_reg', 'density_class']:
            self.model = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            self.model.decode_head.classifier = ClassMovieClassifier( 
                input_channels=self.model.config.hidden_size,
                output_channels=self.output_channels,
                num_frames=num_heads
            )
        elif mode == 'denseClass_wEvac':
            self.model = BeitForSemanticSegmentation_evac.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            self.model.pred_evac_only = False
        else:
            raise NotImplementedError
        
        # self.feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        
        # Initialize the weights
        # self.apply(_init_weights)

    def forward(self, x, *args):

        input_shape = (640, 640)
        if self.mode == 'grayscale':
            logits = self.model(x).logits
            logits = nn.ReLU()(logits)

            """ from torchvision import transforms
            import numpy as np
            from helper import get_color_from_array
            import matplotlib.pyplot as plt
            for i in range(4):
                img_gt = transforms.Resize((160, 160))(x[i])
                img_gt = img_gt.transpose(0,1).transpose(1, 2).detach().cpu().numpy()
                img_gt = (img_gt+1)/2.
            
                traj_max_timestamp = torch.max(logits[i]).item()
                traj_pred = logits[i].squeeze().detach().cpu().numpy()
                traj_pred_bigger_thresh = np.argwhere(traj_pred >= 1.0)
                pred_colors_from_timestamps = [get_color_from_array(traj_pred[x, y], traj_max_timestamp)/255. for x, y in traj_pred_bigger_thresh]
                img_gt[traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1]] = np.array(pred_colors_from_timestamps)
                plt.imshow(img_gt)
                plt.close('all') """
            x = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
            
            # x = nn.AdaptiveMaxPool2d(input_shape)(logits)
            # x = self.model.high_res_net(logits)
        elif self.mode == 'evac_only':
            if len(args) > 0:
                x = self.model(x, *args).logits
            else:
                x = self.model(x).logits
        elif self.mode == 'class_movie':
            logits = self.model(x).logits
            logits = [F.interpolate(logit_map, size=input_shape, mode="bilinear", align_corners=False) for logit_map in logits]
            x = [end_layer(logits[idx]) for idx, end_layer in enumerate(self.end_layer_module)]
            x = torch.stack(x, dim=2)
        elif self.mode in ['density_reg', 'density_class']:
            x = self.model(x).logits
            x = torch.stack(x, dim=-1).squeeze()
        elif self.mode == 'denseClass_wEvac':
            x = list(self.model(x, *args))
            x[0] = torch.stack(x[0], dim=2).squeeze()
        else:
            raise NotImplementedError

        return x


class BeitForSemanticSegmentation_evac(BeitForSemanticSegmentation):
    def __init__(self, config: BeitConfig, num_heads = 8, output_channels = 5, dropout = 0.) -> None:
        super().__init__(config)
        self.config = config
        self.config.num_labels = 1
        self.dropout = dropout
        self.pred_evac_only = False
        # self.evac_head = DistanceResMLPInformationNetwork(self.config.hidden_size, 16)
        self.evac_head = DistanceResMLPInformationNetwork_B(self.config.hidden_size)
        # self.evac_head = DistanceResMLPInformationNetwork_C(self.config.hidden_size)
        # self.evac_head = DistanceResMLPInformationNetwork_D(self.config.hidden_size)
        # self.evac_head = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, 64), nn.BatchNorm1d(64), nn.ReLU(),
        #     nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
        #     nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.Linear(64, 1)
        # )
        self.beit.pooler = BeitPooler(config)
        self.decode_head.classifier = ClassMovieClassifier(
                input_channels=self.config.hidden_size,
                output_channels=output_channels,
                num_frames=num_heads
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.conv_norm = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)

        # self.beit.embeddings = BeitEmbeddings_custom(self.config)
        self.include_before_enc = False
        if self.include_before_enc:
            self.beit = BeitModel_custom(config, add_pooling_layer=True)
        self.use_pooler_states = False
        self.set = 'meanLayNorm'
        self.use_d = False

        for m in self.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(
        self, 
        pixel_values = None,
        *args
    ):

        # return super().forward(pixel_values, head_mask, labels, output_attentions, output_hidden_states, return_dict)

        return_dict = True
        output_hidden_states = False

        if not self.include_before_enc:
            outputs = self.beit(
                pixel_values,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )
        else:
            outputs = self.beit(
                pixel_values,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
                add_args = torch.repeat_interleave(args[0][0], 48, dim=1)
            )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        pooled_outputs = outputs.pooler_output

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        if not self.use_pooler_states and self.include_before_enc:
            pooled_outputs = [x[:, 0] for x in encoder_hidden_states]
            pooled_outputs = torch.stack(pooled_outputs, dim=-1)
            if self.set == 'meanLayNorm':
                pooled_outputs = self.layer_norm(pooled_outputs.mean(dim=-1))
            elif self.set == 'convLayNorm':
                pooled_outputs = self.conv_norm(pooled_outputs.permute(0,2,1)).squeeze()
                pooled_outputs = self.layer_norm(pooled_outputs)
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [
            x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, patch_resolution, patch_resolution) for x in features
        ]

        # apply FPNs
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.pred_evac_only:
            return self.evac_head(features)

        logits = self.decode_head(features)
        if self.use_d:
            evac_time = self.evac_head(features, *args)
            return logits, evac_time

        evac_time = self.evac_head(pooled_outputs, *args)
        return logits, evac_time


class BeitModel_custom(BeitModel):
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = True) -> None:
        super().__init__(config, add_pooling_layer)
    def forward(
        self, 
        pixel_values = None,
        bool_masked_pos = None,
        head_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        add_args = None):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos)
        if add_args != None:
            # add_args = add_args.unsqueeze(1)
            embedding_output[:, 0] = add_args

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class ClassMovieClassifier(nn.ModuleList):
    def __init__(self, input_channels, output_channels, num_frames) -> None:
        classifier_list = [nn.Conv2d(input_channels, output_channels, kernel_size=1) for i in range(num_frames)]
        super().__init__(classifier_list)

        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if hasattr(m, 'weight') or hasattr(m, 'bias'):
            try:
                torch.nn.init.xavier_normal_(m.weight)
            except ValueError:
                # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                m.weight.data.uniform_(-0.2, 0.2)
                # print("Bypassing ValueError...")
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, input):
        return [nn.ReLU()(module(input)) for module in self]


class DistanceResMLPInformationNetwork_B(nn.Module):
    def __init__(self, beit_hidden_size, hidden_size=32) -> None:
        super().__init__()
        self.beit_hidden_size = beit_hidden_size

        self.classifier = nn.Linear(beit_hidden_size, 64)
        self.fc_start = nn.Linear(16, hidden_size)

        self.concat_in = nn.Linear(64+hidden_size, hidden_size)

        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc6 = nn.Sequential(nn.Linear(hidden_size,hidden_size), nn.ReLU())
        self.fc7 = nn.Linear(hidden_size, 1)

        self.skip = True

        for m in self.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, *args):
        if len(args) == 1:
            dist_info = args[0][0]
        else:
            raise NotImplementedError
        beit_out = self.classifier(x)
        dist_out = self.fc_start(dist_info)

        concat = torch.cat((beit_out, dist_out), dim=1)
        concat = self.concat_in(concat)

        if not self.skip:
            fc1 = self.fc1(concat)
            fc2 = self.fc2(fc1)
            fc3 = self.fc3(fc2)
            output = self.fc4(fc3)
        else:
            fc1 = self.fc1(concat) + concat
            fc2 = self.fc2(fc1) + fc1
            fc3 = self.fc3(fc2) + fc2
            fc4 = self.fc4(fc3) + fc3
            fc5 = self.fc5(fc4) + fc4
            fc6 = self.fc6(fc5) + fc5
            output = self.fc7(fc6)
        return output


class DistanceResMLPInformationNetwork_C(nn.Module):
    def __init__(self, beit_hidden_size, hidden_size=32) -> None:
        super().__init__()
        self.beit_hidden_size = beit_hidden_size

        self.classifier = nn.Linear(beit_hidden_size, hidden_size)
        # self.fc_start = nn.Linear(16, hidden_size)

        # self.concat_in = nn.Linear(64, hidden_size)

        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc6 = nn.Sequential(nn.Linear(hidden_size,hidden_size), nn.ReLU())
        self.fc7 = nn.Linear(hidden_size, 1)

        self.skip = True

        for m in self.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, *args):
        
        beit_out = self.classifier(x)
   
        fc1 = self.fc1(beit_out) + beit_out
        fc2 = self.fc2(fc1) + fc1
        fc3 = self.fc3(fc2) + fc2
        fc4 = self.fc4(fc3) + fc3
        fc5 = self.fc5(fc4) + fc4
        fc6 = self.fc6(fc5) + fc5
        output = self.fc7(fc6)
        return output


class DistanceResMLPInformationNetwork_D(nn.Module):
    def __init__(self, beit_hidden_size, hidden_size=16) -> None:
        super().__init__()
        self.beit_hidden_size = beit_hidden_size

        # self.classifier = nn.Linear(beit_hidden_size, 64)
        self.feat0_down = nn.Sequential(
            nn.Conv2d(in_channels=beit_hidden_size, out_channels=64, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.feat1 = nn.Sequential(
            nn.Conv2d(in_channels=beit_hidden_size, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.feat1_down = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.feat2 = nn.Sequential(
            nn.Conv2d(in_channels=beit_hidden_size, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.feat2_down = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.feat3 = nn.Sequential(
            nn.Conv2d(in_channels=beit_hidden_size, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )

        self.feat_lin = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=2), nn.BatchNorm2d(1), nn.ReLU(),
        )
        
        self.fc_start = nn.Linear(16, hidden_size)

        self.concat_in = nn.Linear(64+hidden_size, hidden_size)

        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc6 = nn.Sequential(nn.Linear(hidden_size,hidden_size), nn.ReLU())
        self.fc7 = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features, *args):
        if len(args) == 1:
            dist_info = args[0][0]
        else:
            raise NotImplementedError

        # features: [(BS, 768, 160, 160), (), (), ()]
        f0 = self.feat0_down(features[0])
        f1 = self.feat1(features[1])
        f1 = torch.cat((f0, f1), dim=1)

        f1d = self.feat1_down(f1)
        f2 = self.feat2(features[2])
        f2 = torch.cat((f1d, f2), dim=1)

        f2d = self.feat2_down(f2)
        f3 = self.feat3(features[3])
        f3 = torch.cat((f2d, f3), dim=1)
        
        beit_out = self.feat_lin(f3).flatten(start_dim=1)

        dist_out = self.fc_start(dist_info)

        concat = torch.cat((beit_out, dist_out), dim=1)
        concat = self.concat_in(concat)

        fc1 = self.fc1(concat) + concat
        fc2 = self.fc2(fc1) + fc1
        fc3 = self.fc3(fc2) + fc2
        fc4 = self.fc4(fc3) + fc3
        fc5 = self.fc5(fc4) + fc4
        fc6 = self.fc6(fc5) + fc5
        output = self.fc7(fc6)
        return output


class DistanceResMLPInformationNetwork(nn.Module):
    def __init__(self, beit_hidden_size, hidden_size=16) -> None:
        super().__init__()
        self.beit_hidden_size = beit_hidden_size

        self.classifier = nn.Linear(beit_hidden_size, 16)

        # self.dist_net = nn.Sequential(
        #     nn.Linear(16,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
        #     nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
        #     nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), # delete for ckpt "addInfo_distances"
        #     nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), # delete for ckpt "addInfo_distances"
        #     nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), # delete for ckpt "addInfo_distances"
        #     nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), # delete for ckpt "addInfo_distances"
        #     nn.Linear(hidden_size,16),
        #     # nn.Linear(16,1)
        # )
        self.fc1 = nn.Sequential(nn.Linear(16,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden_size,hidden_size), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(hidden_size,hidden_size), nn.ReLU())
        self.fc5 = nn.Linear(hidden_size,16)

        self.final = nn.Linear(32, 1)  # delete for ckpt "addInfo_distances"

        for m in self.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, *args):
        if len(args) == 1:
            dist_info = args[0][0]
        else:
            raise NotImplementedError
        beit_out = self.classifier(x)
        # dist_out = self.dist_net(dist_info)
        fc1 = self.fc1(dist_info)
        fc2 = self.fc2(fc1)
        fc3 = fc2 + self.fc3(fc2)
        fc4 = fc3 + self.fc4(fc3)
        dist_out = self.fc5(fc4)
        
        output = torch.cat((beit_out, dist_out), dim=1)
        output = self.final(output)
        return output


class DistanceAttentionInformationNetwork(nn.Module):
    def __init__(self, beit_hidden_size, hidden_size=64) -> None:
        super().__init__()
        self.beit_hidden_size = beit_hidden_size
        self.hidden_size = hidden_size
        self.hidden_sqrt = math.isqrt(hidden_size) 

        self.classifier = nn.Linear(beit_hidden_size, hidden_size)

        self.dist_net_K = nn.Sequential(
            nn.Linear(16,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size,hidden_size), nn.ReLU(),
            nn.Linear(hidden_size,hidden_size), nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
        )

        self.dist_net_Q = nn.Sequential(
            nn.Linear(16,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size,hidden_size), nn.ReLU(),
            nn.Linear(hidden_size,hidden_size), nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
        )

        self.final = nn.Linear(hidden_size, 1)  # delete for ckpt "addInfo_distances"
        # x = dist_net, y = classifier

        for m in self.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, *args):
        if len(args) == 1:
            dist_info = args[0][0]
        else:
            raise NotImplementedError

        V = self.classifier(x).view(x.size(0), self.hidden_sqrt, -1).permute(0,2,1)
        K = self.dist_net_K(dist_info).view(x.size(0), self.hidden_sqrt, -1)
        Q = self.dist_net_Q(dist_info).view(x.size(0), self.hidden_sqrt, -1).permute(0,2,1)

        KQ = torch.matmul(K, Q)
        attention = F.softmax(KQ, dim=-1)
        vector = torch.matmul(attention, V).flatten(start_dim=1)

        output = self.final(vector)
        return output


class EvacHead(nn.Module):
    def __init__(self, config, mode, dropout=0.) -> None:
        super().__init__()
        self.config = config
        assert mode in ['beit', 'conv']
        self.mode = mode
        self.hidden_states = self.config.hidden_size
        self.dropout = dropout
        if mode=='beit':
            self.evac_head = nn.Sequential(
                BeitUperHead(self.config),
                nn.BatchNorm2d(1),
                nn.Dropout(self.dropout),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(1, 1, 3),
                nn.BatchNorm2d(1),
                nn.Dropout(self.dropout),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(1, 1, 3),
                nn.BatchNorm2d(1),
                nn.Dropout(self.dropout),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(1, 1, 3),
                nn.BatchNorm2d(1),
                nn.Dropout(self.dropout),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(64,1)
            )
        
        elif mode == 'conv':
            self.striding = ConvBlock(
                in_channels=self.hidden_states,
                out_channels=self.hidden_states,
                kernel_size=1,
                dropout=self.dropout,
            )
            self.dim_reduction = ConvBlock(
                in_channels=self.hidden_states,
                out_channels=self.hidden_states//4,
                kernel_size=1,
                dropout=self.dropout,
                max_pooling=False
            )

            # self.conv1 = ConvBlock(in_channels=self.hidden_states, out_channels=self.hidden_states//2, kernel_size=3, padding=1, dropout=self.dropout)
            # self.conv2 = ConvBlock(in_channels=self.hidden_states//2, out_channels=self.hidden_states//4, kernel_size=3, padding=1, dropout=self.dropout, max_pooling=False)
            # self.conv3 = ConvBlock(in_channels=self.hidden_states//4, out_channels=64, kernel_size=3, padding=1, dropout=self.dropout, max_pooling=False)
            # self.conv4 = ConvBlock(in_channels=64, out_channels=16, kernel_size=3, padding=1, dropout=self.dropout, max_pooling=False)
            # self.conv5 = ConvBlock(in_channels=16, out_channels=1, kernel_size=3, dropout=self.dropout, max_pooling=False)
            # self.flatten = nn.Flatten()
            # self.fc = nn.Linear(64, 1)

            self.evac_head = nn.Sequential(
                ConvBlock(in_channels=self.hidden_states, out_channels=self.hidden_states//2, kernel_size=3, padding=1, dropout=self.dropout),
                ConvBlock(in_channels=self.hidden_states//2, out_channels=self.hidden_states//4, kernel_size=3, padding=1, dropout=self.dropout, max_pooling=False),
                ConvBlock(in_channels=self.hidden_states//4, out_channels=64, kernel_size=3, padding=1, dropout=self.dropout, max_pooling=False),
                ConvBlock(in_channels=64, out_channels=16, kernel_size=3, padding=1, dropout=self.dropout, max_pooling=False),
                ConvBlock(in_channels=16, out_channels=1, kernel_size=3, dropout=self.dropout, max_pooling=False),
                nn.Flatten(),
                nn.Linear(64, 1)
            )

        # elif mode == 'beit_class':
            
    
    def apply_striding(self, x, stridings):
        for i in range(stridings):
            x = self.striding(x)
        x = self.dim_reduction(x)
        return x

    def forward(self, x):
        if self.mode == 'beit':
            x = self.evac_head(x)
            return x

        elif self.mode == 'conv':
            laterals = []
            for idx, lat in enumerate(x):
                stridings = len(x) - idx - 1
                laterals.append(self.apply_striding(lat, stridings))

            laterals = torch.cat(laterals, dim=1)
            x = self.evac_head(laterals)
            # x = self.conv1(laterals)
            # x = self.conv2(x)
            # x = self.conv3(x)
            # x = self.conv4(x)
            # x = self.conv5(x)
            # x = self.fc(self.flatten(x))
            
            return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        max_pooling = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.max_pooling = max_pooling

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)
        if self.max_pooling:
            output = self.pool(output)
        return output


class EndLayerModule(nn.ModuleList):
    def __init__(self, output_channels, num_frames) -> None:
        classifier_list = [nn.Conv2d(output_channels, output_channels, kernel_size=1)  for i in range(num_frames)]
        super().__init__(classifier_list)
    
    def forward(self, input):
        return [module(input) for module in self]


class BeitForImageClassification_custom(BeitForImageClassification):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        # self.classifier = DistanceInformationNetwork(config.hidden_size)
        
        """ self.floorplan_dims = 2
        self.per_area_input_nodes = 4
        self.max_areas_per_floorplan = 12
        self.dimension_layer_output = 16
        self.areas_layer_output = 32

        # self.dimension_layer = nn.Linear(self.floorplan_dims, self.dimension_layer_output)
        # self.dim_norm = nn.BatchNorm1d(self.dimension_layer_output)
        
        self.origin_input_layers = nn.ModuleList([nn.Linear(self.per_area_input_nodes, self.areas_layer_output, bias=False) for i in range(self.max_areas_per_floorplan)])
        self.origins_norm = nn.BatchNorm1d(self.areas_layer_output * self.max_areas_per_floorplan)
        
        self.destination_input_layers = nn.ModuleList([nn.Linear(self.per_area_input_nodes, self.areas_layer_output, bias=False) for i in range(self.max_areas_per_floorplan)])
        self.destination_norm = nn.BatchNorm1d(self.areas_layer_output * self.max_areas_per_floorplan)

        self.info_norm = nn.BatchNorm1d(2 * self.max_areas_per_floorplan * self.areas_layer_output)
        
        self.floorplan_output = nn.Linear(2*self.areas_layer_output*self.max_areas_per_floorplan+self.dimension_layer_output, 1)
        self.floorplan_norm = nn.BatchNorm1d(2*self.areas_layer_output*self.max_areas_per_floorplan+self.dimension_layer_output)

        # init
        self.origin_input_layers.apply(self._initialize_weights)
        self.destination_input_layers.apply(self._initialize_weights)
        self.floorplan_output.apply(self._initialize_weights) """

    def _initialize_weights(self, m):
        # if isinstance(m, nn.Linear):
        if hasattr(m, 'weight') or hasattr(m, 'bias'):
            m.weight.data.uniform_(-0.2, 0.2)
            # m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, pixel_values = None, *args):
        
        """ if len(args) == 1:
            add_info = args[0]
        
        # input_dimensions = torch.cat((add_info[0].unsqueeze(1), add_info[1].unsqueeze(1)), dim=1)
        # input_vector_dimensions = nn.ReLU()(self.dim_norm(self.dimension_layer(input_dimensions)))
        # input_dimensions = self.dropout_layer(input_dimensions)

        input_origins = [layer(add_info[2][:, i]) for i, layer in enumerate(self.origin_input_layers)]
        input_origins = torch.stack(input_origins, dim=1).flatten(start_dim=1, end_dim=-1) # bs x 12 x 16
        input_origins = nn.ReLU()(self.origins_norm(input_origins))
        # input_origins = self.dropout_layer(input_origins)

        input_destinations = [layer(add_info[3][:, i]) for i, layer in enumerate(self.destination_input_layers)]
        input_destinations = torch.stack(input_destinations, dim=1).flatten(start_dim=1, end_dim=-1) # bs x 12 x 16
        input_destinations = nn.ReLU()(self.destination_norm(input_destinations))
        # input_destinations = self.dropout_layer(input_destinations)

        # input_vector_areas = torch.matmul(input_origins, input_destinations.transpose(-2, -1)) / math.sqrt(128) # bs x 16 x 16
        # info_vector = torch.cat((input_vector_dimensions, input_origins, input_destinations), axis=1)
        info_vector = torch.cat((input_origins, input_destinations), axis=1)
        info_vector = nn.ReLU()(self.info_norm(info_vector))
        # info_vector = info_vector.unsqueeze(1)
       
        self.beit.embeddings.cls_token = info_vector.unsqueeze(1) """

        outputs = self.beit(
            pixel_values,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=self.config.use_return_dict,
        )

        pooled_output = outputs.pooler_output if self.config.use_return_dict else outputs[1]

        if len(args) > 0:
            logits = self.classifier(pooled_output, *args)
        else:
            logits = self.classifier(pooled_output)

        return ImageClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FloorplanInformationNetwork(nn.Module):
    def __init__(self, beit_hidden_size) -> None:
        super().__init__()
        
        self.beit_hidden_size = beit_hidden_size
        self.floorplan_dims = 2
        self.per_area_input_nodes = 4
        self.max_areas_per_floorplan = 12
        self.dimension_layer_output = 16
        self.areas_layer_output = 32
        self.beit_layer_output_dim = self.dimension_layer_output+2*self.max_areas_per_floorplan*self.areas_layer_output
        self.dropout = 0.2
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.dimension_layer = nn.Linear(self.floorplan_dims, self.dimension_layer_output)
        self.dim_norm = nn.BatchNorm1d(self.dimension_layer_output)
        """
        self.origin_input_layers = nn.ModuleList([nn.Linear(self.per_area_input_nodes, self.areas_layer_output, bias=False) for i in range(self.max_areas_per_floorplan)])
        self.origins_norm = nn.BatchNorm1d(self.areas_layer_output * self.max_areas_per_floorplan)
        self.destination_input_layers = nn.ModuleList([nn.Linear(self.per_area_input_nodes, self.areas_layer_output, bias=False) for i in range(self.max_areas_per_floorplan)])
        self.destination_norm = nn.BatchNorm1d(self.areas_layer_output * self.max_areas_per_floorplan)
        
        # self.beit_network = nn.Linear(self.beit_hidden_size, )

        self.beit_layer = nn.Linear(self.beit_hidden_size, self.beit_layer_output_dim)
        self.beit_norm = nn.BatchNorm1d(self.beit_layer_output_dim)

        self.output_layer = nn.Linear(2*self.beit_layer_output_dim, 1) """

        ### NEW ###
        self.dimension_layer = nn.Linear(self.floorplan_dims, self.dimension_layer_output)
        self.dim_norm = nn.BatchNorm1d(self.dimension_layer_output)
        
        self.origin_input_layers = nn.ModuleList([nn.Linear(self.per_area_input_nodes, self.areas_layer_output, bias=False) for i in range(self.max_areas_per_floorplan)])
        self.origins_norm = nn.BatchNorm1d(self.areas_layer_output * self.max_areas_per_floorplan)
        
        self.destination_input_layers = nn.ModuleList([nn.Linear(self.per_area_input_nodes, self.areas_layer_output, bias=False) for i in range(self.max_areas_per_floorplan)])
        self.destination_norm = nn.BatchNorm1d(self.areas_layer_output * self.max_areas_per_floorplan)
        
        self.floorplan_output = nn.Linear(2*self.areas_layer_output*self.max_areas_per_floorplan+self.dimension_layer_output, 1)
        self.floorplan_norm = nn.BatchNorm1d(2*self.areas_layer_output*self.max_areas_per_floorplan+self.dimension_layer_output)
        self.classifier = nn.Linear(beit_hidden_size, 1)
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x, *args):
        # input: (BSx768), (BSx1), (BSx1), (BSx12x4), (BSx12x4)
        if len(args) == 1:
            add_info = args[0]
        else:
            raise NotImplementedError

        """
        input_vector_beit = nn.ReLU()(self.beit_norm(self.beit_layer(x)))

        input_dimensions = torch.cat((add_info[0].unsqueeze(1), add_info[1].unsqueeze(1)), dim=1)
        input_vector_dimensions = nn.ReLU()(self.dim_norm(self.dimension_layer(input_dimensions)))
        input_dimensions = self.dropout_layer(input_dimensions)
        
        input_origins = [layer(add_info[2][:, i]) for i, layer in enumerate(self.origin_input_layers)]
        input_origins = torch.stack(input_origins, dim=1).flatten(start_dim=1, end_dim=-1)
        input_origins = nn.ReLU()(self.origins_norm(input_origins))
        input_origins = self.dropout_layer(input_origins)

        input_destinations = [layer(add_info[3][:, i]) for i, layer in enumerate(self.destination_input_layers)]
        input_destinations = torch.stack(input_destinations, dim=1).flatten(start_dim=1, end_dim=-1)
        input_destinations = nn.ReLU()(self.destination_norm(input_destinations))
        input_destinations = self.dropout_layer(input_destinations)

        input_vector_areas = torch.cat((input_origins, input_destinations), dim=1)

        hidden_vector = torch.cat((input_vector_dimensions, input_vector_areas, input_vector_beit), dim=1)
        output = self.output_layer(hidden_vector).squeeze()"""


        ### NEW ###
        
        input_dimensions = torch.cat((add_info[0].unsqueeze(1), add_info[1].unsqueeze(1)), dim=1)
        input_vector_dimensions = nn.ReLU()(self.dim_norm(self.dimension_layer(input_dimensions)))
        input_dimensions = self.dropout_layer(input_dimensions)

        input_origins = [layer(add_info[2][:, i]) for i, layer in enumerate(self.origin_input_layers)]
        input_origins = torch.stack(input_origins, dim=1).flatten(start_dim=1, end_dim=-1) # bs x 12 x 16
        input_origins = nn.ReLU()(self.origins_norm(input_origins))
        input_origins = self.dropout_layer(input_origins)

        input_destinations = [layer(add_info[3][:, i]) for i, layer in enumerate(self.destination_input_layers)]
        input_destinations = torch.stack(input_destinations, dim=1).flatten(start_dim=1, end_dim=-1) # bs x 12 x 16
        input_destinations = nn.ReLU()(self.destination_norm(input_destinations))
        input_destinations = self.dropout_layer(input_destinations)

        # input_vector_areas = torch.matmul(input_origins, input_destinations.transpose(-2, -1)) / math.sqrt(128) # bs x 16 x 16
        info_vector = torch.cat((input_vector_dimensions, input_origins, input_destinations), axis=1)
        add_info_output = self.floorplan_output(info_vector)
        
        beit_output = self.classifier(x)
        output = torch.cat((beit_output, add_info_output), dim=1)
        output = self.output_layer(output)

        return output


######################################################## PAPER APPOACH 1 ########################################################
class SICNN(nn.Module):
    def __init__(self, f0_layers: int = 4, f8_layers: int = 8, init_fcn: str = 'uniform.2', bnIntoConvBlocks: bool = False) -> None:
        super().__init__()
        # paper: based on A_Convolutional_Neural_Network_for_Image_Super-Res
        self.init_fcn = init_fcn
        # upper branch
        self.f0 = nn.Sequential(
            ConvBlock(1, 64, bnIntoConvBlocks),
        )
        for i in range(1, f0_layers):
            self.f0.add_module(str(i), ConvBlock(64, 64, bnIntoConvBlocks))

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=1, dilation=2), # 78
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=1, dilation=2), # 37
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=1, dilation=2), # 
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=1, dilation=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        )
        self.f_cs = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # lower branch
        self.f1_upscale = nn.AdaptiveMaxPool2d((800, 800))
        self.f8 = nn.Sequential(
            ConvBlock(1, 64, bnIntoConvBlocks),
        )
        for i in range(1, f8_layers):
            self.f8.add_module(str(i), ConvBlock(64, 64, bnIntoConvBlocks))

        self.f_ss = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # branch fusion
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        )

        self.final_conv = ConvBlock(1, 1)

        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if hasattr(m, 'weight') or hasattr(m, 'bias'):
            if self.init_fcn == 'xavierNormal':
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
            elif self.init_fcn == 'uniform.2': m.weight.data.uniform_(-0.2, 0.2)
            elif self.init_fcn == 'uniform.4': m.weight.data.uniform_(-0.4, 0.4)
            # m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()



    def forward(self, traj_masks):
        # 160 x 160 traj mask ==> 800 x 800 (scale factor 5)
        # upper branch
        f0 = self.f0(traj_masks)
        deconv = self.deconv(f0)
        f_cs = self.f_cs(deconv)
        # lower branch
        f1_upscaled = self.f1_upscale(traj_masks)
        f8 = self.f8(f1_upscaled)
        f_ss = self.f_ss(f8)
        # concatination
        scale_fusion = torch.cat((f_ss, f_cs), dim=1)
        f_ms = self.concat_conv(scale_fusion)
        # residual connection
        f_gf = f_ms + f1_upscaled

        output = self.final_conv(f_gf)
        output = nn.ReLU()(output)
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_c, num_f, withBN: bool = False) -> None:
        super().__init__()
        self.withBN = withBN
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=num_f, kernel_size=3, padding=1)
        if withBN:
            self.bn = nn.BatchNorm2d(num_f)

    def forward(self, x):
        x = self.conv(x)
        if self.withBN:
            x = self.bn(x)
        x = nn.ReLU()(x)
        return x
######################################################## PAPER APPOACH 1 ########################################################

#################################################### PAPER APPOACH 1 4 EVAC #####################################################
class SICNN4Evac(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # upper branch
        self.f0 = nn.Sequential(
            ConvBlock(32, 32, True),
        )
        for i in range(1, 4):
            self.f0.add_module(str(i), ConvBlock(32, 32, True))

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2), # 160 -> 78
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2), # 78 -> 38
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2), # 38 -> 18
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 18 -> 16
        )
        self.f_cs = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # lower branch
        self.f1_downscale = nn.AdaptiveMaxPool2d((16, 16))
        self.f8 = nn.Sequential(
            ConvBlock(32, 32, True),
        )
        for i in range(1, 4):
            self.f8.add_module(str(i), ConvBlock(32, 32, True))

        self.f_ss = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # branch fusion
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2lin = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3), # 16 -> 14
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3), # 14 -> 12
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3),  # 12 -> 10
        )

        self.final_layer = nn.Linear(100, 1)

        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if hasattr(m, 'weight') or hasattr(m, 'bias'):

            m.weight.data.uniform_(-0.2, 0.2)
            # try:
            #     torch.nn.init.xavier_normal_(m.weight)
            # except ValueError:
            #     # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
            #     m.weight.data.uniform_(-0.2, 0.2)
            #     # print("Bypassing ValueError...")
            # elif self.init_fcn == 'uniform.4': m.weight.data.uniform_(-0.4, 0.4)
            # m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, traj_masks, *args):
        
        if len(args) == 1:
            add_info = args[0]

        # 160 x 160 traj mask ==> 800 x 800 (scale factor 5)
        # upper branch
        f0 = self.f0(traj_masks)
        deconv = self.conv_down(f0)
        f_cs = self.f_cs(deconv)
        # lower branch
        f1_downscaled = self.f1_downscale(traj_masks)
        f8 = self.f8(f1_downscaled)
        f_ss = self.f_ss(f8)
        # concatination
        scale_fusion = torch.cat((f_ss, f_cs), dim=1)
        f_ms = self.concat_conv(scale_fusion)
        # residual connection
        f_gf = f_ms + f1_downscaled

        output = self.conv2lin(f_gf).squeeze()
        output = nn.Flatten(start_dim=1)(output)
        output = self.final_layer(output)
        return output


class SICNN4EvacDense(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # downscaling to 80x80
        self.conv80 = nn.Sequential(
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2), # 160 -> 80
            nn.ReLU(),
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
        )
        self.downscale80 = nn.AdaptiveMaxPool2d((80, 80))

        # downscaling to 40x40
        self.conv40 = nn.Sequential(
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2), # 80 -> 40
            nn.ReLU(),
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
        )
        self.downscale40 = nn.AdaptiveMaxPool2d((40, 40))

        # downscaling to 20x20
        self.conv20 = nn.Sequential(
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2), # 40 -> 20
            nn.ReLU(),
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
        )
        self.downscale20 = nn.AdaptiveMaxPool2d((20, 20))

        # downscaling to 10x10
        self.conv10 = nn.Sequential(
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2), # 20 -> 10
            nn.ReLU(),
            ConvBlock(32, 32, True),
            ConvBlock(32, 32, True),
        )
        self.downscale10 = nn.AdaptiveMaxPool2d((10, 10))

       # filter reduction
        self.conv2lin = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(100, 1)

        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if hasattr(m, 'weight') or hasattr(m, 'bias'):

            # m.weight.data.uniform_(-0.2, 0.2)
            try:
                torch.nn.init.xavier_normal_(m.weight)
            except ValueError:
                # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                m.weight.data.uniform_(-0.2, 0.2)
                # print("Bypassing ValueError...")
            # elif self.init_fcn == 'uniform.4': m.weight.data.uniform_(-0.4, 0.4)
            # m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, traj_masks):
        conv80 = self.conv80(traj_masks)
        downscale80 = self.downscale80(traj_masks)
        out80 = conv80+downscale80

        conv40 = self.conv40(out80)
        downscale40 = self.downscale40(out80)
        out40 = conv40+downscale40

        conv20 = self.conv20(out40)
        downscale20 = self.downscale20(out40)
        out20 = conv20+downscale20

        conv10 = self.conv10(out20)
        downscale10 = self.downscale10(out20)
        out10 = conv10+downscale10

        outLin = self.conv2lin(out10)
        outLin = nn.Flatten(start_dim=1)
        output = self.final_layer(outLin)

        return output
#################################################### PAPER APPOACH 1 4 EVAC #####################################################


######################################################## PAPER APPOACH 2 ########################################################
# http://staff.ustc.edu.cn/~huanghb/LiuB_POF.pdf
class MTPC(nn.Module):
    def __init__(self, feat_extr_layers: int = 4, reconstr_layers: int = 8, init_fcn: str = 'uniform.2', bnIntoConvBlocks: bool = False) -> None:
        super().__init__()
        # maybe add BN later...
        self.init_fcn = init_fcn
        self.ratio = 5 # 160 -> 800
        self.bnIntoConvBlocks = True # bnIntoConvBlocks

        # feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        )
        for i in range(1, feat_extr_layers):
            self.conv1.add_module(str(i), ConvBlock(64, 64, self.bnIntoConvBlocks))

        self.block1 = EMSRB()
        self.block2 = EMSRB()

        # reconstruction
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=self.ratio**2, kernel_size=1)
        )
        for i in range(1, reconstr_layers):
            self.conv2.add_module(str(i), ConvBlock(self.ratio**2, self.ratio**2, self.bnIntoConvBlocks))
        
        self.shuffle = nn.PixelShuffle(upscale_factor=self.ratio)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        # weight path
        """ self.conv2_weights = nn.Conv2d(in_channels=192, out_channels=self.ratio**2, kernel_size=1)
        self.conv3_weights = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3) """

        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if hasattr(m, 'weight') or hasattr(m, 'bias'):
            if self.init_fcn == 'xavierNormal':
                try:
                    torch.nn.init.xavier_normal_(m.weight)
                except ValueError:
                    # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                    m.weight.data.uniform_(-0.2, 0.2)
                    # print("Bypassing ValueError...")
            elif self.init_fcn == 'uniform.2': m.weight.data.uniform_(-0.2, 0.2)
            elif self.init_fcn == 'uniform.4': m.weight.data.uniform_(-0.4, 0.4)
            # m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        # feature extraction
        conv1_out = self.conv1(x)
        block1_out = self.block1(conv1_out)
        block2_out = self.block2(block1_out)

        # reconstruction
        feat_out = torch.cat((conv1_out, block1_out, block2_out), dim=1)
        out_conv2 = self.conv2(feat_out)
        out_conv2 = self.shuffle(out_conv2)
        output = self.conv3(out_conv2)
        output = nn.ReLU()(output)

        # weight path
        """ out_conv2_weights = self.conv2_weights(feat_out)
        out_conv2_weights = self.shuffle(out_conv2_weights)
        output_weights = self.conv3_weights(out_conv2_weights)
        mask_weights = output_weights.ge(0.0).float()
        output = output * mask_weights """

        return output

class EMSRB(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.skip_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.lay1_conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.lay1_conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)

        self.lay2_conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.lay2_conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)

        self.concat_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        
    def forward(self, x):
        out_skip_conv1 = self.skip_conv1(x)

        out_lay1_conv3 = self.lay1_conv3(x)
        out_lay1_conv3 = nn.ReLU()(out_lay1_conv3)
        out_lay1_conv5 = self.lay1_conv5(x)
        out_lay1_conv5 = nn.ReLU()(out_lay1_conv5)

        concat_1 = torch.cat((out_lay1_conv3, out_lay1_conv5), dim=1)

        out_lay2_conv3 = self.lay2_conv3(concat_1)
        out_lay2_conv3 = nn.ReLU()(out_lay2_conv3)
        out_lay2_conv5 = self.lay2_conv5(concat_1)
        out_lay2_conv5 = nn.ReLU()(out_lay2_conv5)

        concat_1 = torch.cat((out_lay2_conv3, out_lay2_conv5), dim=1)
        out_concat_conv1 = self.concat_conv1(concat_1)

        output = torch.cat((out_concat_conv1, out_skip_conv1), dim=1)

        return output

class NormAndNonlinear(nn.Sequential):
    def __init__(self, bn_channels: int):
        super().__init__(
            nn.BatchNorm2d(bn_channels),
            nn.ReLU()
        )
######################################################## PAPER APPOACH 2 ########################################################



######################################################## PAPER APPOACH 3 ########################################################
# ABPN (Attention based Back Projection Network for image super-resolution) https://arxiv.org/pdf/1910.04476v1.pdf
# git: https://github.com/Holmes-Alan/ABPN/blob/master/model.py
class ABPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass

class ConvBlockABPN(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlockABPN, self).__init__()

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)

        return self.act(out)


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlockABPN(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlockABPN(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlockABPN(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlockABPN(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlockABPN(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlockABPN(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlockABPN(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + x
        out = self.act2(out)

        return out


class Space_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Space_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale
        # downscale = scale + 4

        self.K = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
        #self.bn = nn.BatchNorm2d(output_size)
        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)


    def forward(self, x):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        # Q = F.interpolate(Q, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(x)
        # V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)
        # if self.type == 'softmax':
        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)

        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = x + W
        #output = self.bn(output)
        return output
######################################################## PAPER APPOACH 3 ########################################################


##################################################### PAPER APPOACH 3 4 EVAC ####################################################
class Same4Evac(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(Same4Evac, self).__init__()

        # self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        # self.conv2 = ConvBlockABPN(output_size, output_size, kernel_size, stride, padding, bias=True)
        # self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        
        self.conv1 = ConvBlockABPN(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        
        self.local_weight1 = ConvBlockABPN(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight2 = ConvBlockABPN(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x) # down
        sr = self.conv2(lr) # up
        residue = self.local_weight1(x) - lr # down
        h_residue = self.conv3(residue) # up
        hr_weight = self.local_weight2(sr) #  same
        return hr_weight + h_residue


class DownBlock4Evac(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock4Evac, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = ConvBlockABPN(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlockABPN(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight1 = ConvBlockABPN(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight2 = ConvBlockABPN(output_size, output_size, kernel_size, stride, padding, bias=True)

    def forward(self, x):
        sr = self.conv1(x) # same
        lr = self.conv2(sr) # down
        residue = self.local_weight1(x) - lr # down
        l_residue = self.conv3(residue) # same
        lr_weight = self.local_weight2(sr) # down
        return lr_weight + l_residue

class ABPN4Evac(nn.Module):
    def __init__(self):
        super(ABPN4Evac, self).__init__()
        kernel_size = 2
        pad = 0
        stride = 2
        input_dim = 32
        dim = 32

        self.feat1 = ConvBlockABPN(input_dim, dim, 3, 1, 1) # kernel_size, stride, padding
        # self.SA0 = Space_attention(dim, dim, 1, 1, 0, 1)
        self.feat2 = ConvBlockABPN(dim, dim, kernel_size, stride, pad)
        # BP 1
        self.same1 = Same4Evac(dim, dim, kernel_size, stride, pad)
        self.down1 = DownBlock4Evac(dim, dim, kernel_size, stride, pad)
        self.SA1 = Time_attention(dim, dim, 1, 1, 0, 1)
        self.feat3 = ConvBlockABPN(dim, dim, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(32)
        # BP 2
        self.same2 = Same4Evac(dim, dim, kernel_size, stride, pad)
        self.down2 = DownBlock4Evac(dim, dim, kernel_size, stride, pad)
        self.SA2 = Time_attention(dim, dim, 1, 1, 0, 1)
        self.feat4 = ConvBlockABPN(dim, dim, kernel_size, stride, pad)
        self.bn2 = nn.BatchNorm2d(32)
        # BP 3
        self.same3 = Same4Evac(dim, dim, kernel_size, stride, pad)
        self.down3 = DownBlock4Evac(dim, dim, kernel_size, stride, pad)
        self.SA3 = Time_attention(dim, dim, 1, 1, 0, 1)
        self.feat5 = ConvBlockABPN(dim, dim, kernel_size, stride, pad)
        self.bn3 = nn.BatchNorm2d(32)
        # BP 4
        self.same4 = Same4Evac(dim, dim, kernel_size, stride, pad)
        self.down4 = DownBlock4Evac(dim, dim, kernel_size, stride, pad)
        self.SA4 = Time_attention(dim, dim, 1, 1, 0, 1)
        self.bn4 = nn.BatchNorm2d(32)
        # reduce feature space
        self.toLin = ConvBlockABPN(dim, 1, 1, 1, 0)
        self.lin = nn.Linear(100, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, traj_masks):
        # 160 -> 160
        f1 = self.feat1(traj_masks)
        # sa0 = self.SA0(f1)
        f2 = self.feat2(f1)

        # 160 -> 80
        same1 = self.same1(f1)
        down1 = self.down1(same1)
        down1 = self.SA1(f2, down1)
        down1 = self.bn1(down1)

        # 80 -> 40
        f3 = self.feat3(down1)
        same2 = self.same2(down1)
        down2 = self.down2(same2)
        down2 = self.SA2(f3, down2)
        down2 = self.bn2(down2)

        # 40 -> 20
        f4 = self.feat4(down2)
        same3 = self.same3(down2)
        down3 = self.down3(same3)
        down3 = self.SA3(f4, down3)
        down3 = self.bn3(down3)

        # 20 -> 10
        f5 = self.feat5(down3)
        same4 = self.same4(down3)
        down4 = self.down4(same4)
        down4 = self.SA4(f5, down4)
        down4 = self.bn4(down4)

        # to linear
        down4_red = self.toLin(down4).squeeze()
        down4_red = nn.Flatten(start_dim=1)(down4_red)
        output = self.lin(down4_red).squeeze()

        return output


class Time_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Time_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale
        # downscale = scale + 4

        self.K = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
        #self.bn = nn.BatchNorm2d(output_size)
        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)


    def forward(self, x, y):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        # Q = F.interpolate(Q, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(y)
        # V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        #attention = x
        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)

        # if self.type == 'softmax':
        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)
        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = y + W
        #output = self.bn(output)
        return output
##################################################### PAPER APPOACH 3 4 EVAC ####################################################