from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    model_id: str = "openvla/openvla-7b-finetuned-libero-spatial"
    unnorm_key: str = "libero_spatial"
    image_key: str = "observation.images.image"

    def __post_init__(self):
        super().__post_init__()

        if not self.input_features:
            self.input_features = {
                self.image_key: PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 224, 224),
                )
            }

        if not self.output_features:
            self.output_features = {
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(7,),
                )
            }

        self.validate_features()

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return [0]

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return AdamWConfig(lr=5e-4)

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        if self.n_obs_steps != 1:
            raise ValueError("현재는 관측 이미지 1개를 사용합니다.")

        if self.image_key not in self.input_features:
            raise ValueError("설정에 이미지 입력이 없습니다.")

        if self.input_features[self.image_key].shape != (3, 224, 224):
            raise ValueError("이미지 형태는 (3, 224, 224)여야 합니다.")

        if "action" not in self.output_features:
            raise ValueError("설정에 행동 출력이 없습니다.")

        if self.output_features["action"].shape != (7,):
            raise ValueError("행동 출력은 숫자 7개여야 합니다.")