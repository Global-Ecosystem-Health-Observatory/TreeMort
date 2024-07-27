# modified code from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb

import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, num_labels=1):
        super(LinearClassifier, self).__init__()
        self.in_channels = in_channels
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, kernel_size=1)

    def forward(self, embeddings):
        batch_size, seq_length, hidden_size = embeddings.size()
        height = width = int(seq_length ** 0.5)  # Assuming square image patches
        embeddings = embeddings.reshape(batch_size, height, width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)
        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.conv1 = torch.nn.Conv2d(4, 3, kernel_size=1)  # Convert 4 channels to 3
        self.classifier = LinearClassifier(config.hidden_size, config.num_labels)

    def forward(
        self,
        pixel_values,
        output_hidden_states=False,
        output_attentions=False,
        labels=None,
    ):
        # Convert 4-channel input to 3-channel input
        pixel_values = self.conv1(pixel_values)

        # use frozen features
        outputs = self.dinov2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
