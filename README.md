Running instructions:
   1. python src/data/simulator.py 
   Outputs: data/processed/simulated_noisy_votes.csv, data/processed/ground_truth.csv, data/true_params
   true_annotator_parameters.csv 

   2. python src/data/prepare_tensors.py
   Outputs: data/tokenized/rlhf_tokens.pt
   Optimise to CUDA if you want for Windows

   3. python src/training/train_joint.py
   Outputs: The terminal will display the climbing Golden Dataset Accuracy and the current Weighted DPO Loss.

   or 

   3. python src/training/train_joint_wo_sft.py
   Here we can quickly see the epochs and the training dont on the terminal, it is a stubbed model notn having too much context of the language itself as doesnt have SFT, so lesser accuracy.

   notebooks/ is for personal analysis and plots.

We get 75% accuracy with the SFT gpt2 model.