# asitpofborscht <img src="sitp/src/assets/borscht.webp" width="32">

This repository hosts the https://sitp.ai textbook under [./sitp](./sitp) where you will train [`nanogpt`](https://github.com/karpathy/nanogpt)
*using* the `borscht` teaching deep learning framework and then *implement* the framework itself.

While the SITP textbook spends a lot more time on intuition and motivation via visualizations, figures, and prose,
you can get a condensed "just show me the code" version of the textbook by reading the [literary programmed](http://www.literateprogramming.com/)
source below.

![](./sitp/src/assets/models_banner.webp)
## Deep Neural Networks
Deep neural networks you train throughout the SITP textbook, which are located in [`./models`](./models):
- Model 1: Bigram
- Model 2: Discriminative Perceptron
- Model 3: Probabilistic Perceptron
- Model 4: Multilayer Peceptron
- Model 5: Convolutional Neural Network
- Model 6: Recurrent Neural Network
- Model 7: Long Short-Term Memory Network
- Model 8: Encoder-Decoder Pre-trained Transformers (BERT)
- Model 9: Decoder Only Pre-trained Transformers (mingpt)
- Model 10: Decoder Only Pre-trained Transformers (nanogpt)
- Model 11: Post-trained Transformer (nanochat)
- Model 12: Post-trained Transformer (nanoreasoner)

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

<!-- - File 5. Abstracting Acceleration with `Dispatcher`, `Storage`, `Allocator`: [`tensor.py`](./borscht/python/borscht/tensor.py)
- File 6. Adding Forward Ops
- File 7. Hello Autograd
- File 8. Adding Backward Ops
- File 9. From Volta to Ampere (sm80)
- File 10. From Ampere (sm80) to Hopper (sm90)
- File 11. Hello Tensor Compiler
- File 12. Hello Inference Engine -->

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