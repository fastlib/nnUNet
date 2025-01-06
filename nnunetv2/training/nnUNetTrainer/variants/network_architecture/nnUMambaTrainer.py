
from typing import Tuple, Union, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetWithClassificationTrainer import nnUNetWithClassificationTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.classificationBranchTrainer import ClassificationTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.models.mamba import get_umamba_enc_2d_from_plans

class nnUNetTrainerUMambaEnc(nnUNetWithClassificationTrainer):

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_umamba_enc_2d_from_plans(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels, num_output_channels, enable_deep_supervision)

        print("UMambaEnc: {}".format(model))

        return model


class nnUNetTrainerUMambaEncClassification(ClassificationTrainer):

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_umamba_enc_2d_from_plans(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels, num_output_channels, enable_deep_supervision)

        print("UMambaEnc: {}".format(model))

        return model