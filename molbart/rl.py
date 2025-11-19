import hydra
from molbart.models.chemformer_grpo import ChemformerGRPO
import molbart.utils.data_utils as util

@hydra.main(version_base=None, config_path="config", config_name="rl")
def main(args):
    """ Main script to run GRPO training using the ChemformerGRPO wrapper. """
    
    util.seed_everything(args.seed)
    print("Initializing ChemformerGRPO for Reinforcement Learning.")
    chemformer_grpo = ChemformerGRPO(args)

    print("Starting GRPO training...")
    chemformer_grpo.fit()

    print("GRPO training complete.")

if __name__ == "__main__":
    main()
