import hydra
import torch

import molbart.utils.data_utils as util
from molbart.models import Chemformer, ChemformerGRPO

import time

import joblib

@hydra.main(version_base=None, config_path="config", config_name="inference_score")
def main(args):
    util.seed_everything(args.seed)

    print("Running model inference and scoring.")

    print(args)
    # chemformer = Chemformer(args)
    chemformer = ChemformerGRPO(args)
    chemformer.model.sampler.gumbel_noise = False

    chemformer.score_model(
        n_unique_beams=args.n_unique_beams,
        dataset=args.dataset_part,
        output_scores=args.output_score_data,
        output_sampled_smiles=args.output_sampled_smiles,
    )
    print("Model inference and scoring done.")
    return

    # model = chemformer.model
    # tokenizer = chemformer.tokenizer
    # vocabulary_size = len(tokenizer)
    # device = chemformer.device

    # model.eval()

    # batch_size = 64
    # encoder_seq_len = 256
    # decoder_seq_len = 256

    # # encoder_input = torch.randint(0, vocabulary_size, (encoder_seq_len, batch_size), device=device)
    # # decoder_input = torch.randint(0, vocabulary_size, (decoder_seq_len, batch_size), device=device)

    # # encoder_pad_mask = torch.zeros((batch_size, encoder_seq_len), dtype=torch.bool, device=device)
    # # decoder_pad_mask = torch.zeros((batch_size, decoder_seq_len), dtype=torch.bool, device=device)

    # # random_batch = {
    # #     "encoder_input": encoder_input,
    # #     "encoder_pad_mask": encoder_pad_mask.transpose(0, 1),
    # #     "decoder_input": decoder_input,
    # #     "decoder_pad_mask": decoder_pad_mask.transpose(0, 1)
    # # }
    # # print("Generated random input and target tensors.")

    # joblib_filename = "inference_data.joblib"
    # inference_data = joblib.load(joblib_filename)

    # random_batch = {
    #     "encoder_input": torch.tensor(inference_data["encoder_input"]).to(device),
    #     "encoder_pad_mask": torch.tensor(inference_data["encoder_pad_mask"]).transpose(0, 1).to(device),
    #     "decoder_input": torch.tensor(inference_data["encoder_input"]).to(device),
    #     "decoder_pad_mask": torch.tensor(inference_data["encoder_pad_mask"]).transpose(0, 1).to(device),
    # }

    # warm_up_iterations = 3
    # print(f"\nPerforming {warm_up_iterations} warm-up iterations...")
    # for i in range(warm_up_iterations):
    #     # with torch.no_grad():
    #     _ = model.forward(random_batch)

    # print("Warm-up complete.")

    # num_iterations = 10
    # execution_times = []
    # peak_memory_usages = []
    # print(f"\nStarting benchmark for {num_iterations} iterations...")

    # for i in range(num_iterations):
    #     if device.startswith("cuda"):
    #         torch.cuda.reset_peak_memory_stats(device)
    #     start_time = time.perf_counter()

    #     # with torch.no_grad():
    #     _ = model.forward(random_batch)

    #     end_time = time.perf_counter()

    #     peak_mem_mb = 0
    #     if device.startswith("cuda"):
    #         peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    #         peak_mem_mb = peak_mem_bytes / 1024**2 # Convert bytes to MB
        
    #     execution_times.append(end_time - start_time)
    #     peak_memory_usages.append(peak_mem_mb)

    #     print(f"  Iteration {i+1}/{num_iterations}: Time={execution_times[-1]:.4f}s, Peak Memory={peak_memory_usages[-1]:.2f}MB")

    # with torch.no_grad():
    #     output = model.forward(random_batch)
    #     predicted_token_log_probs = output["token_output"]

    # print("Prediction finished.")

    # # inference_data = {
    # #     "encoder_input": encoder_input.cpu().numpy(),
    # #     "encoder_pad_mask": encoder_pad_mask.cpu().numpy(),
    # #     "decoder_input": decoder_input.cpu().numpy(),
    # #     "decoder_pad_mask": decoder_pad_mask.cpu().numpy(),
    # #     "predicted_token_log_probs": predicted_token_log_probs.cpu().numpy(),
    # #     "hyperparameters": {
    # #         "vocabulary_size": vocabulary_size,
    # #         "d_model": model.hparams.d_model,
    # #         "num_layers": model.hparams.num_layers,
    # #         "num_heads": model.hparams.num_heads,
    # #         "d_feedforward": model.hparams.d_feedforward,
    # #         "pad_token_idx": model.hparams.pad_token_idx,
    # #         "dropout": model.hparams.dropout,
    # #         "activation": model.hparams.activation,
    # #         "max_seq_len": model.hparams.max_seq_len
    # #     }
    # # }
    # # joblib_filename = "inference_data.joblib"
    # # joblib.dump(inference_data, joblib_filename)
    # # print(f"Saved random tensors and model output to '{joblib_filename}'")

    # weights_filename = "chemformer_bart_model.pth"
    # torch.save(model.state_dict(), weights_filename)
    # print(f"Saved model weights to '{weights_filename}'")
    # print("--- Process Complete ---")


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 expandtab
