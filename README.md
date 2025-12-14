# tiny-recursive-models
a clean and readable implementation of **Tiny Recursive Models by Samsung SAIL** from scratch in pytorch.

---

<img width="777" height="758" alt="trm-img" src="https://github.com/user-attachments/assets/393f5932-bd5b-4670-902b-4c8e6d56f5bc" />


Tiny Recursive Models (TRMs) work by refining their predictions step by step instead of producing an answer in a single forward pass like standard Transformers. The model maintains two internal states: a prediction state and a latent state. Starting from learnable initial values, the latent state is repeatedly refined using the input and current prediction, and then the prediction itself is updated using the refined latent state. The same small Transformer block is reused at every step, allowing the model to iteratively improve its understanding while keeping the number of parameters low. This recursive refinement process is inspired by how humans revise their thoughts over multiple passes rather than reasoning all at once.




