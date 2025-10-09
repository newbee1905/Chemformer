from argparse import Namespace
from molbart.models import BARTGRPOModel, Chemformer

class ChemformerGRPO(Chemformer):
    """
    A Chemformer wrapper for setting up and running GRPO training.
    It loads a pre-trained SFT model into a BARTGRPOModel for fine-tuning.
    """
    def __init__(self, config: Namespace):
        super().__init__(config)

    def build_model(self, args: Namespace) -> None:
        """
        Builds the BARTGRPOModel and loads pre-trained weights from a checkpoint.
        """
        if not self.model_path:
            raise ValueError("A pre-trained model checkpoint must be provided via 'model_path' for GRPO training.")

        print(f"Loading SFT weights from {self.model_path} into BARTGRPOModel.")

        extra_hparams = {
            "rl_learning_rate": args.rl.learning_rate,
            "ppo_epochs": args.rl.ppo_epochs,
            "clip_param": args.rl.clip_param,
            "kl_coeff": args.rl.kl_coeff,
            "g_samples": args.rl.g_samples,
        }

        self.model = BARTGRPOModel.load_from_checkpoint(
            self.model_path,
            strict=False,
            decode_sampler=self.sampler,
            **extra_hparams
        )

        self.model.sampler.gumbel_noise = True

# vim: ts=4 sw=4 expandtab
