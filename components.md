Wake (W) Component:
In the wake component of the training loop, we process the real images using the discriminator, and obtain the latent space from them. We add Gaussian noise to the image. This component does image reconstruction, and latent space regularization, ensuring the model's robustness and that the information is preserved during wakefulness.

NREM Perturbed Dreaming (N) Component:
During NREM phase of dreaming, the discriminator model is updated with occluded (partially hidden) images generated from the latent representation. This phase aims to enhance the model's ability to handle incomplete input, contributing to better generalization (akin to forming new connections in the brain) in non-REM sleep-like states.

REM Adversarial Dreaming (R) Component:
In the REM adversarial dreaming component, a mix of latent noise and previous latent representations generates "dreamed" images. The discriminator evaluates these images, and the generator learns to compete with the discriminator to score more, resulting in more realistic images.