from . import ocr_model
from . import patch_merging_2d
from . import simple_mamba
from . import ss_conv_ssm
from . import util
import torch
from torch import nn
from typing import Any

class MedmambaStack(ocr_model.OcrModel):
    """Rough copy of MedMamba with context injection.
    """
    def __init__(
        self,
        d_feature: int,
        d_label: int,
        d_input_stack: int,
        stack: list[str],
        stack_options: list[dict[str, Any]],
        final_mamba_options: dict[str, Any],
    ) -> None:
        """Initialize an instance of MambaCnnStack

        Args:
            d_feature: The number of input channels.
            d_label: The number of output channels.
            d_input_stack: The number of input channels for the stack.
            stack: A list of strings equal to "ss_conv_ssm" or patch_merging_2d"
                defining the layer architecture.
            stack_options: A list of dictionaries that will be passed to the
                stack layers as initialization options.
        """
        super().__init__()
        self.d_feature = d_feature
        self.d_label = d_label

        self.input_projection = nn.Linear(self.d_feature, d_input_stack)

        self.layers = nn.ModuleList()
        channel_num = d_input_stack
        for layer_index, layer_type in enumerate(stack):
            if layer_type == "ss_conv_ssm":
                self.layers.append(ss_conv_ssm.SsConvSsm(
                    d_hidden=channel_num,
                    d_label=self.d_label,
                    **stack_options[layer_index],
                ))
            elif layer_type == "patch_merging_2d":
                self.layers.append(patch_merging_2d.PatchMerging2D(
                    dim=channel_num,
                    **stack_options[layer_index],
                ))
                channel_num *= 2
        
        self.final_mamba_layer = simple_mamba.SimpleMAMBA(
            d_input=channel_num + self.d_label,
            d_output=self.d_label,
            **final_mamba_options,
        )

    def forward(
        self,
        features: list[torch.Tensor],
        labels: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        x = [feature.permute(1, 2, 0) for feature in features]
        assert x[0].shape[2] == self.d_feature

        x = [self.input_projection(x_) for x_ in x]

        x = [x_.permute(2, 0, 1) for x_ in x]

        for layer in self.layers():
            if isinstance(layer, ss_conv_ssm.SsConvSsm):
                x = layer(x, labels)
            elif isinstance(layer, patch_merging_2d.PatchMerging2D):
                x = [layer(x_) for x_ in x]
        
        x = [x_.permute(2, 1, 0) for x_ in x]
        x = [x_.flatten(0, 1) for x_ in x]
        feature_sizes = [x_.shape[0] for x_ in x]
        label_sizes = [label.shape[0] for label in labels]
        x = util.inject_sequence_labels(x, labels)
        x = self.final_mamba_layer(x)

        current_index = -1
        output = []
        for i in range(len(feature_sizes)):
            current_index += feature_sizes[i]
            output.append(x[current_index:current_index + label_sizes[i]])
            current_index += label_sizes[i]
        return output
