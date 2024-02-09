import torch
import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
from torch import nn
# from typing import Optional
import torch.nn.functional as F

class GoalPredictionModel(nn.Module):
    def __init__(self, option='attnMap_cat'):
        super().__init__()
        self.vel_max_real = 1.61
        self.resize_factor = 1
        self.delta_x_max_pix = self.vel_max_real / 0.03125 * 0.5 / self.resize_factor + 2 # pix2real_factor = 0.03125, time_frame_length = 0.5 sec, 2 pix potential rounding errors
        self.pred_length = 12
        self.overall_mean, self.overall_std = 1280 // self.resize_factor / 2, 150 / self.resize_factor
        self.step_mutliplier = self.delta_x_max_pix * self.pred_length
        # normalize the step multiplier
        self.step_mutliplier = self.step_mutliplier / self.overall_std
        # encode semanticMaps_egocentric
        self.unet = UNet(in_channels=1, out_channels=64)

        if option == 'noObs':
            # try again
            self.forwardOption = 0
            # merge goalMaps_egocentric and semanticMaps_egocentric
            self.merged_maps_layers = nn.Sequential(
                UNetConvBlock(128, 128), UNetConvBlock(128, 128), UNetConvBlock(128, 128)
            )
            self.final = nn.Linear(128, 2)
        elif option == 'crossAttnObs':
            # try again
            self.forwardOption = 1
            self.encode_coords = nn.Linear(2, 512) 
            # Transormer cross attention
            self.cross_attention = TransformerCrossAttention(input_dim=512, hidden_dim=128, num_heads=8)
        elif option == 'noObs+':
            self.forwardOption = 2
            self.merged_maps_layers = nn.Sequential(
                UNetConvBlock(128, 128), UNetConvBlock(128, 128), UNetConvBlock(128, 128)
            ) 
            self.final = nn.Linear(128, 2)
        elif option == 'crossAttnObs+':
            self.forwardOption = 3
            self.encode_coords = nn.Linear(2, 512) 
            # Transormer cross attention
            self.cross_attention = TransformerCrossAttention(input_dim=512, hidden_dim=128, num_heads=8)
        elif option == 'tf_decoder':
            # try again (with zero pads)
            self.forwardOption = 4
            self.encode_coords = nn.Linear(2, 512)
            self.map_projection = nn.Linear(128, 512)
            decLayer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
            self.tf_decoder = nn.TransformerDecoder(decoder_layer=decLayer, num_layers=1)
            self.output_projection = nn.Linear(512, 2)
        elif option == 'tf_decoder+max':
            raise ValueError('not a good option')
            self.forwardOption = 5
            self.encode_coords = nn.Linear(2, 512)
            self.map_projection = nn.Linear(128, 512)
            decLayer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
            self.tf_decoder = nn.TransformerDecoder(decoder_layer=decLayer, num_layers=1)
            self.output_projection = nn.Linear(512, 2)
        elif option == 'tf_decoder_vae':
            raise ValueError('not a good option')
            self.forwardOption = 6
            self.encode_coords = nn.Linear(2, 512)
            self.map_projection = nn.Linear(128, 512)
            decLayer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
            self.tf_decoder = nn.TransformerDecoder(decoder_layer=decLayer, num_layers=1)
            self.angle_projection = nn.Linear(512, 1)
            self.abs_velocity_projection = nn.Linear(512, 1)
            self.mu_net = nn.Linear(512, 20)
            self.logvar_net = nn.Linear(512, 20)
            self.output_projection = nn.Linear(20, 2)
        elif option == 'noObs_heavy':
            self.forwardOption = 7
            # merge goalMaps_egocentric and semanticMaps_egocentric
            self.merged_maps_layers = nn.Sequential(
                UNetConvBlock(128, 256), UNetConvBlock(256, 256), UNetConvBlock(256, 256)
            )
            self.final = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Linear(128, 2))
        elif option == 'catCoordDstMap':
            self.forwardOption = 8
            self.encode_coords = nn.Linear(2, 512)
            # decLayer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
            # self.tf_decoder = nn.TransformerDecoder(decoder_layer=decLayer, num_layers=1)
            window_size = 64
            self.decrease_map = nn.Sequential(UNetConvBlock(64, 64, kernel_size=2, stride=2), UNetConvBlock(64, 64, kernel_size=2, stride=2), UNetConvBlock(64, 512, kernel_size=2, stride=2))
            self.output_projection = nn.Linear(3*512, 2)
        elif option in ['coordAttnDstAttnMap', 'coordAttnMapAttnDst']:
            self.forwardOption = 9 if option == 'coordAttnDstAttnMap' else 10
            self.encode_coords = nn.Linear(2, 256)
            self.map_conv = nn.Conv2d(64, 256, kernel_size=8, stride=8)
            decLayer_dst = nn.TransformerDecoderLayer(d_model=256, nhead=8)
            self.tf_decoder_dst = nn.TransformerDecoder(decoder_layer=decLayer_dst, num_layers=1)
            decLayer_map = nn.TransformerDecoderLayer(d_model=256, nhead=8)
            self.tf_decoder_map = nn.TransformerDecoder(decoder_layer=decLayer_map, num_layers=1)
            self.output_projection = nn.Linear(256, 2)
        elif option == 'attnMap_cat':
            self.forwardOption = 11
            self.encode_coords = nn.Linear(2, 512)
            self.map_conv = nn.Conv2d(64, 512, kernel_size=8, stride=8)
            decLayer_map = nn.TransformerDecoderLayer(d_model=512, nhead=8)
            self.tf_decoder_map = nn.TransformerDecoder(decoder_layer=decLayer_map, num_layers=1)
            self.output_projection = nn.Linear(1024, 2)
        
        if self.forwardOption in [0, 1, 2, 3, 4, 5, 6, 7]:
            # encode goalMaps_egocentric
            self.resnet18 = Resnet18Adaption(in_channels=1)            
        

    def forward(self, obs_coords, goal_information, semanticMaps_egocentric):
        if self.forwardOption in [0, 1, 2, 3, 4, 5, 6, 7]:
            out = self.forwardOptions_0to7(obs_coords, goal_information, semanticMaps_egocentric)
        # these were not so good
        elif self.forwardOption == 8:
            coords_encoded = self.encode_coords(obs_coords)
            dests_encoded = (self.encode_coords(goal_information[:,:2]) + self.encode_coords(goal_information[:, 2:])) / 2
            # unet forward pass --> semanticMaps_egocentric
            semanticMaps_encoded = self.unet(semanticMaps_egocentric)
            semanticMaps_encoded = self.decrease_map(semanticMaps_encoded).squeeze()
            all_encoded = torch.cat((coords_encoded, dests_encoded, semanticMaps_encoded), dim=1)
            out = self.output_projection(all_encoded)
        elif self.forwardOption in [9, 10]:
            coords_encoded = self.encode_coords(obs_coords)
            dests_encoded = (self.encode_coords(goal_information[:,:2]) + self.encode_coords(goal_information[:, 2:])) / 2
            # unet forward pass --> semanticMaps_egocentric
            semanticMaps_encoded = self.unet(semanticMaps_egocentric)
            flattenedMap = self.map_conv(semanticMaps_encoded).flatten(-2)
            if self.forwardOption == 9:
                # cross attention dest
                dec_out_dst = self.tf_decoder_dst(coords_encoded.unsqueeze(0), dests_encoded.unsqueeze(0))
                # cross attention map
                dec_out_map = self.tf_decoder_map(dec_out_dst, flattenedMap.permute(2, 0, 1)).squeeze()
            elif self.forwardOption == 10:
                # cross attention map
                dec_out_map = self.tf_decoder_map(coords_encoded.unsqueeze(0), flattenedMap.permute(2, 0, 1))
                # cross attention dest
                dec_out_dst = self.tf_decoder_dst(dec_out_map, dests_encoded.unsqueeze(0)).squeeze()
            out = self.output_projection(dec_out_map)
        elif self.forwardOption == 11:
            coords_encoded = self.encode_coords(obs_coords)
            dests_encoded = (self.encode_coords(goal_information[:,:2]) + self.encode_coords(goal_information[:, 2:])) / 2
            # unet forward pass --> semanticMaps_egocentric
            semanticMaps_encoded = self.unet(semanticMaps_egocentric)
            flattenedMap = self.map_conv(semanticMaps_encoded).flatten(-2)
            dec_out_map = self.tf_decoder_map(coords_encoded.unsqueeze(0), flattenedMap.permute(2, 0, 1)).squeeze()
            proj_in = torch.cat((dec_out_map, dests_encoded), dim=1)
            out = self.output_projection(proj_in)
        
        return out


    def forwardOptions_0to7(self, obs_coords, goal_information, semanticMaps_egocentric):
        # resnet forward pass --> goalMaps_egocentric
        goalMaps_encoded = self.resnet18(goal_information)
        goalMaps_encoded = nn.AdaptiveAvgPool2d(output_size=(semanticMaps_egocentric.shape[-2], semanticMaps_egocentric.shape[-1]))(goalMaps_encoded)

        # unet forward pass --> semanticMaps_egocentric
        semanticMaps_encoded = self.unet(semanticMaps_egocentric)
        # goal decoding
        maps_encoded = torch.cat((goalMaps_encoded, semanticMaps_encoded), dim=1)
        if self.forwardOption == 0:
            maps_encoded = self.merged_maps_layers(maps_encoded)
            maps_encoded = nn.AdaptiveAvgPool2d(output_size=(1,1))(maps_encoded)
            maps_encoded = maps_encoded.view(obs_coords.shape[0], 128)
            out = self.final(maps_encoded)
        
        elif self.forwardOption == 1:
            coord_embed = self.encode_coords(obs_coords)
            out = self.cross_attention(coord_embed, maps_encoded)
        
        elif self.forwardOption == 2:
            maps_encoded = self.merged_maps_layers(maps_encoded)
            maps_encoded = nn.AdaptiveAvgPool2d(output_size=(1,1))(maps_encoded)
            maps_encoded = maps_encoded.view(obs_coords.shape[0], 128)
            out = obs_coords + self.final(maps_encoded)
        
        elif self.forwardOption == 3:
            coord_embed = self.encode_coords(obs_coords)
            out = obs_coords + self.cross_attention(coord_embed, maps_encoded)
        
        elif self.forwardOption == 4:
            coord_embed = self.encode_coords(obs_coords).unsqueeze(0)
            maps_encoded = self.map_projection(maps_encoded.flatten(-2).permute(0, 2, 1)).permute(1, 0, 2)
            dec_out = self.tf_decoder(coord_embed, maps_encoded).squeeze(0)
            out = self.output_projection(dec_out)
        
        elif self.forwardOption == 5:
            coord_embed = self.encode_coords(obs_coords).unsqueeze(0)
            maps_encoded = self.map_projection(maps_encoded.flatten(-2).permute(0, 2, 1)).permute(1, 0, 2)
            dec_out = self.tf_decoder(coord_embed, maps_encoded).squeeze(0)
            out = obs_coords + self.output_projection(dec_out)
        
        elif self.forwardOption == 6:
            coord_embed = self.encode_coords(obs_coords).unsqueeze(0)
            maps_encoded = self.map_projection(maps_encoded.flatten(-2).permute(0, 2, 1)).permute(1, 0, 2)
            dec_out = self.tf_decoder(coord_embed, maps_encoded).squeeze(0)
            mu = self.mu_net(dec_out)
            logvar = self.logvar_net(dec_out)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std
            out = self.output_projection(z), mu, logvar
        
        elif self.forwardOption == 7:
            maps_encoded = self.merged_maps_layers(maps_encoded)
            maps_encoded = nn.AdaptiveAvgPool2d(output_size=(1,1))(maps_encoded)
            maps_encoded = maps_encoded.view(obs_coords.shape[0], 256)
            out = self.final(maps_encoded)
        return out


class Resnet18Adaption(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.layer4 = UNetConvBlock(256, 64, 1)
        del self.resnet18._modules['avgpool']
        del self.resnet18._modules['fc']
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = UNetConvBlock(in_channels, 64)
        self.conv2 = UNetConvBlock(64, 256)
        self.conv3 = UNetConvBlock(256, 512)

        # Decoder
        self.deconv1 = UNetDeconvBlock(512, 256)
        self.deconv2 = UNetDeconvBlock(256, 64)
        self.deconv3 = UNetDeconvBlock(64, out_channels)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        dc3 = self.deconv1(c3) + c2
        dc2 = self.deconv2(dc3) + c1
        x = self.deconv3(dc2)
        return x


class UNetConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(UNetConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

class UNetDeconvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(UNetDeconvBlock, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )


""" class TransformerCrossAttentionEncoder(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
 """


class TransformerCrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(TransformerCrossAttention, self).__init__()

        # Multihead attention layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # Positional encoding for coordinates
        max_len = 512
        d_model = input_dim
        self.positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Map encoding projection layer
        self.map_projection = nn.Linear(hidden_dim, input_dim)

        # Output projection layer
        self.output_projection = nn.Linear(input_dim, 2)

    def forward(self, coordinates, map_encoding):
        # Add positional encoding to coordinates
        position_indices = torch.arange(coordinates.size(0), device=coordinates.device).unsqueeze(1)
        positional_encoding = self.positional_encoding[position_indices].to(position_indices.device)
        coordinates_with_position = coordinates.unsqueeze(1) + positional_encoding

        # Project map encoding to the same input dimension as coordinates
        projected_map_encoding = self.map_projection(map_encoding.flatten(-2).permute(0, 2, 1))

        # generate mask that ensure that each i-th coordinate can only attend to the i-th map encoding
        # attn_mask = self.generate_mask(size=coordinates_with_position.size(0))

        # Apply masked multihead attention
        output, _ = self.attention(coordinates_with_position.permute(1, 0, 2),
                                   projected_map_encoding.permute(1, 0, 2),
                                   projected_map_encoding.permute(1, 0, 2),
                                   attn_mask=None
                                   )

        # Project the output to the goal coordinates
        predicted_coordinates = self.output_projection(output.permute(1, 0, 2)).squeeze(1)

        return predicted_coordinates

    def generate_mask(self, size):
        mask = torch.zeros(size, size)
        for i in range(size):
            mask[i, i] = 1
        return mask.bool()
    

def build_pos_enc(d_model, max_len):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe