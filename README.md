# asitpofborscht <img src="https://sitp.ai/assets/borscht.png" width="32">

This repository hosts the https://sitp.ai textbook and the `borscht` teaching deep learning framework ([`nanogpt`](https://github.com/karpathy/nanogpt) complete!)<br>

If you prefer to "just show me the code" mode, you can read the [literary programmed](http://www.literateprogramming.com/)
source of the deep neural network architecture under [./models](./models) and the deep learning framework under [./borscht](./borscht/)
Reading the source provides a condensed version of the SITP textbook, which spend a lot more time on intuition and motivation
via visualizations, figures, and prose.

**Reading Order for Deep Neural Networks**
- Model 1: Bigram
- Model 2: Discriminative Perceptron
- Model 3: Probabilistic Perceptron
- Model 4: MLP

**Reading Order for Deep Learning Framework**
- File 1. Hello borscht: [`python/borscht/__init__.py`](./borscht/python/borscht/__init__.py)
- File 2. Hello Tensor: [`python/borscht/tensor_managed.py`](./borscht/python/borscht/tensor_managed.py)
- File 3. Python + Rust: [`python/borscht/tensor_nativecpu.py`](./borscht/python/borscht/tensor_nativecpu.py), [`pyproject.toml`](./borscht/pyproject.toml), [`Cargo.toml`](./borscht/Cargo.toml), [`rust/lib.rs`](./borscht/rust/lib.rs)
- File 4. Accelerate Tensor on CPU:  [`rust/kernelscpu.rs`](./borscht/rust/kernelscpu.rs)
<!-- - File 5. Python + Rust + CUDA Rust: 
- File 6. Accelerate Tensor on GPU: [`python/borscht/tensor_nativegpu.py`](), [`rust/kernelsgpu.rs`]()
- File 7. Abstract Acceleration with `Dispatcher`, `Storage`, `Allocator`: [`borscht/python/borscht/tensor.py`](./borscht/python/borscht/tensor.py) -->

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