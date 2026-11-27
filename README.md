1# asitpofborscht <img src="sitp/src/assets/borscht.webp" width="32">

This repository hosts the https://sitp.ai textbook under [./sitp](./sitp) where you will train [`nanogpt`](https://github.com/karpathy/nanogpt)
*using* the `borscht` teaching deep learning framework and then *implement* the framework itself.

While the SITP textbook spends a lot more time on intuition and motivation via visualizations, figures, and prose,
you can get a condensed "just show me the code" version of the textbook by reading the [literary programmed](http://www.literateprogramming.com/)
source below.

![](./sitp/src/assets/models_banner.webp)
## Deep Neural Networks
Deep neural networks you train throughout the SITP textbook, which are located in [`./models`](./models):

**Part I (1969-2001). From Language Compilers to Neural Language Models**<br>
In whickkh we learn the elements of differentiable programming: specifying a parameterized architecture, scoring such paramaters with a measure, and optimizing the score.
- Model 1: Bigram [(Markov 1913)](), [(Shannon 1948)]()
- Model 2: Discriminative Perceptron [(Rosenblatt 1957)]()
- Model 3: Probabilistic Perceptron [(Rosenblatt 1957)]()
- Model 4: Multilayer Peceptron [(Bengio et al., 2003)](), [(Mikolov et al., 2013)]()

**Part II (2012-2019). From Static to Dynamic Embeddings**<br>
In which we rediscover the various inductive biases explored during the age of research for ourselves.
We implement the same character-level networks from `karpathy/makemore`, presented in the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.
- Model 5: Hello Recurrence (RNN)  [(Mikolov et al., 2010)]()
- Model 6: Hello Long Short-Term Memory (LSTM) [(Graves et al., 2014)]()
- Model 7: Hello Gated Recurrence (GRU) [(Cho 2014)]()
- Model 8: Hello Encoder-Decoder [(Sutskever et al., 2014)]()
- Model 9: Hello Attention (ANN) [(Graves et al, 2013)]() [(Bahdanau et al., 2015)]()
- Model 10: Hello Convolution (CNN) [(van den Oord et al., 2016)]()
- Model 11: Attention is All You Need [(Vaswani et al., 2017)]()
- Model 12: Hello Decoder Only Transformers (GPT-1) [Liu et al., 2018]() [(Radford et al., 2018)]()

**Part III (2019-2025). From Language Models to Large Language Models**<br>
In which we rediscover the post-training during the age of scale. We implement `karpathy/mingpt` and `karpathy/nanogpt`, and `karpathy/nanochat`
- Law: Hello Scaling Laws [(Kaplan et al., 2020)](https://arxiv.org/pdf/2001.08361)
- Model 13: Hello High School Student (GPT-2 1.5B [(Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) via Llama-1/2 [(Touvron et al., 2023)](https://arxiv.org/pdf/2302.13971), [(Touvron et al., 2023)](https://arxiv.org/pdf/2307.09288) 
- Model 14: Hello College Student (GPT-3 175B [(Brown et al., 2020)](https://arxiv.org/pdf/2005.14165)) via Llama-3 [(AI@Meta 2024)](https://arxiv.org/pdf/2407.21783)
- Model 15: Hello PhD Student (GPT-4 ~1.8T [(OpenAI 2023)](https://arxiv.org/pdf/2303.08774)) via Whale v3/v4 [(DeepSeek-AI 2025)](https://arxiv.org/pdf/2412.19437), [(DeepSeek-AI 2026)](https://arxiv.org/pdf/2606.19348)
- Model 16: Hello Reasoner (o1/o3 [(OpenAI 2024)](https://arxiv.org/pdf/2412.16720)) via Whale r1 [(DeepSeek-AI 2025)](https://arxiv.org/pdf/2501.12948) and Kimi k1.5 [(Kimi Team 2025)](https://arxiv.org/pdf/2501.12599)
- Model 17: Hello Expert (GPT-5 ~6-8T?)

<!-- - Model 11: Reinforcement Learning with Human Feedback [(Christiano et al., 2017)]()
- Model 12: Reinforcement Learning with Veriable Rewards (nanoreasoner) -->


<!-- After Model 12, you might be interested in the first research blog series
From SITP to SOTA [FS2S Part 1 (ARCH/OPT): From modded-nanogpt to dsv3 and k3](). -->

![](./sitp/src/assets/gpu_banner.webp)
## Deep Learning Framework
Deep neural framework you both use and implement throughout the SITP textbook, located in [`./borscht`](./borscht):
Ensure you have Python and Rust toolchains on your machine by running [`borscht/setup.sh`](./borscht/setup.sh) in your shell.

<!-- ```sh
docker ps
docker exec -it
cuda oxide doctor
cuda oxide run
``` -->

**Part I: From Arrays to NDArrays (`borscht.Tensor`)**<br>
In which we transition from software 1.0 arrays to software 2.0 ndarrays.<br>
Want to read the textbook? See SITP [Chapter 2. The Massively Parallel Tensor](https://sitp.ai/1.html#2-the-massively-parallel-multidimensional-tensor)
- File 1. Hello borscht ([`__init__.py`](./borscht/python/borscht/__init__.py))
- File 2. Hello Tensor ([`tensor_managed.py`)](./borscht/python/borscht/tensor_managed.py))
- File 3. Accelerate Tensor
  - File 3A: Analysis: From Asymptotic Analysis of Growth to Empirical Measurement of Time ([`tensor_nativecpu.py`](./borscht/python/borscht/tensor_nativecpu.py))
  - File 3B: Three Language Problem: From Virtual to Physical Machines (CPython API: [`pyproject.toml`](./borscht/pyproject.toml) Rust FFI: [`Cargo.toml`](./borscht/Cargo.toml))
  - File 3C: Optimization (GEMM1): From Rust to RISC-V (PyO3: [`lib.rs`](./borscht/rust/lib.rs), [`kernelscpu.rs`](./borscht/rust/kernelscpu/mod.rs))
- File 4. Accelerate+ Tensor
  - File 4A: Analysis: From Asymptotic Analysis of RAM Roofline Analysis of Memory Hierarchies
  - File 4B: Optimization (GEMM2): Loop Ordering
  - File 4C: Optimization (GEMM3): Threaded
- File 5. Accelerate++ Tensor
  - File 5A: Three Language Problem: From Latency-Oriented to Throughput-Oriented Machines ([`tensor_nativegpu.py`](./borscht/python/borscht/tensor_nativegpu.py))
  - File 5B: Optimization (GEMM4): From Rust to CUDA Rust [`kernelsgpu.rs`](./borscht/rust/kernelsgpu/mod.rs)
- File 6. Abstracted Accelerated Tensor
  - File 6A: `Allocator`, `Storage` [`tensor_native.py`](./borscht/python/borscht/tensor_native.py)
  - File 6B: `Dispatcher`, `Iterator`
- File ?? From `borscht.Tensor` to `torch.Tensor`

**Part II: From Numerical Linear Algebra Library to Deep Learning Framework (`borscht.nn`, `borscht.autograd`, `borscht.optim`)**<br>
In which we add support for first-order gradient-based optimization via automatic differentiation.<br>
Want to read the textbook? See SITP [Chapter 4. The Engine of Automatic Differentiation](https://sitp.ai/2#4-the-engine-of-automatic-differentiation)
- File 7. Hello Neural Network
  - File 7A. ([`nn/module.py`](./borscht/python/borscht/nn/module.py))
  - FIle 7B. ([`nn/functional.py`](./borscht/python/borscht/nn/functional.py))
- File 8. Hello Autograd ([`tensor_autograd.py`](./borscht/python/borscht/tensor_autograd.py))
  - File 8A. Adding Forward Ops
  - File 8B. Adding Backward Ops
- File 9. Hello Optimizer
  - File 9A. `borscht.optim`
- File 10. Hello Convolutional Neural Network
- File 11. Hello Recurrent Neural Network
- File 12. Hello Transformer
- File ??. From `borscht.nn` to [`torch.nn`]()
- File ??. From `borscht.optim` to [`torch.optim`]()
- File ??. From `borscht.autograd` to [`torch.autograd`]()

**Part III: From CPU Kernels to GPU Kernels (`beets`)**<br>
In which we accelerate our kernel library to reach speed of light throughputs with tensor cores.<br>
Want to read the textbook? See SITP [Chapter 5. The Speed of Light]()
- File ??. From Volta (sm70) to Ampere (sm80)
- File ??. From Ampere (sm80) to Hopper (sm90)
- File ??. From Hopper (sm90) to Blackwell (sm100)
- File ??. From GPUs to TPUs
- File ??. From `beets` to [`quack`](https://github.com/Dao-AILab/quack) and [`flash-attention`](https://github.com/dao-ailab/flash-attention)

**Part IV: From Interpreters to Compilers (`bt2`)**<br>
- File ??. Hello Tensor Compiler
- File ??. From `bortsch bynamo` and `borscht binductor` to [`torch dynamo`](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo) and [`torch inductor`](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)

**Part V: Runtimes (`bllm`)**<br>
- File ??. Hello Inference Engine
- File ??. From `bllm` to [`vllm`](https://github.com/vllm-project/vllm)


<!-- After Model 12, you might be interested in the second and third research blog series From SITP to SOTA
- [FS2S Part 1 (SYS): From Hopper to Blackwell and Rubin]()
- [FS2S Part 2 (DISTSYS): From FSDP to MP]() -->

## License

Following [GPU MODE](https://x.com/marksaroufim/status/2064442386374369597),
this project is licensed under the June 9 Researcher Reciprocity License.

The license adapts the Open RAIL-S structure and adds one specific use restriction: training, fine-tuning, distillation, synthetic-data generation for training, embedding for training, or otherwise using this project to improve an AI model or AI service requires Researcher Reciprocity.

If you train on it, you let us generate.

Covered AI model and service providers may not use this project while imposing terms that prevent SITP and borscht project contributors, or authorized researchers from generating outputs, evaluating models, benchmarking, publishing research, or exploring their own research ideas on materially equal terms to ordinary users.

<!-- ## Quick Start: SITP Book

If you want to *modify* the SITP book, follow the installation instructions below:

The book is built with [mdbook](https://rust-lang.github.io/mdBook/), the Rust ecosystem's standard static site generator for markdown. The steps below install the whole set, and mirror the Netlify
build in `netlify.toml` — that file is the source of truth for what production uses, so keep the two
in sync when you change either. Beyond the Rust and Lean toolchains below, you need `python3` on
PATH: the `nb`, `refine`, and `lean` preprocessors are Python scripts under `sitp/preprocessors/`
(standard library only, so there is nothing to `pip install`).

Expect the first build to be slow: aquascope runs every annotated Rust block through Miri, and Lean
elaborates every `{{#lean}}` example. Both cache afterwards, aquascope in `sitp/.aquascope-cache`. -->