# DARLA
PyTorch implementation of the DARLA reinforcement learning pipeline, using PPO to learn a policy from the ß-VAE's latent state

### DARLA Paper
https://arxiv.org/pdf/1707.08475.pdf

### Pipeline
1. Learn disentangled features of the environment using a random agent in an unsupervised domain
2. Learn a policy for the source domain (in this case with PPO) using the learned state representation from step 1
3. Test the policy from step 2 on the target domain
