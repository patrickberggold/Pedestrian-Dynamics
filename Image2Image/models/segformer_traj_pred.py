import torch
from transformers import SegformerForSemanticSegmentation
from torch.nn import functional as F

class SegFormerTraj(torch.nn.Module):
    def __init__(self, mode, output_channels) -> None:
        super().__init__()

        self.mode = mode
        self.output_channels = output_channels
        # self.relu_at_end = relu_at_end
        # self.p_dropout = p_dropout
        # self.num_heads = num_heads

        assert mode == 'grayscale'
        # https://xieenze.github.io/segformer.pdf
        # self.model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
        self.model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-cityscapes-1024-1024')
        # Replace classifier to output 1-dim
        self.model.decode_head.classifier = torch.nn.Conv2d(self.model.config.decoder_hidden_size, self.output_channels, kernel_size=1)
        # self.model = BeitModel(config=config).from_pretrained('microsoft/beit-base-finetuned-ade-640-640') # 85.7 M Trainable params

        # self.feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

    def forward(self, x):
        # x = self.feature_extractor(images=x, return_tensors='pt')
        input_shape = (640, 640)
        logits = self.model(x).logits
        logits = torch.nn.ReLU()(logits)
        x = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)

        # TODO for using pretrained: either exchange layers or resize to 224x244
        # TODO for resizing/transforming: use my own, or transformers FeatureExtractor or a mix?

        return x
