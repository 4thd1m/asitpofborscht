![](./assets/syl.png)

# Singularity Systems Syllabus

<div class="dropcap">

Welcome to <span class="smallcaps">Singularity Systems</span>, the online course for a self-guided accelerated introduction to deep neural networks and deep learning systems. This page is the syllabus for the course, which consists of lectures, a textbook<span class="sidenote-number"></span><span class="sidenote front-note">*All text (every single word) has been written by a human, for a human, as this book is not a reference book, but rather a cognitive helmet of understanding for a human. You are highly encouraged if not expected to read with an LLM, but be aware (that is, to beware) of how much you cognitively offload and outsource your understanding.*</span>, and labs.
If you are interested in learning about the course's philosophy, see the [course thesis](./thesis.md).

</div>

## Syllabus

<div class="toc">

- [Week 1. Hello Tensor 👋](#week-1-hello-tensor-)
- [Week 2. I Perceive You 👀](#week-2-i-perceive-you-)
- [Week 3. MetaCircularEvaLuAtor! ↩️](#week-3-metacircularevaluator-️)
- [Week 4. GPUs Go Brrr 🏎️](#week-4-gpus-go-brrr-️)
- Week 5. King - Man + Queen = 👩 (SOON)
- Week 6. Plato's Cave 𐦂𖨆𐀪𖠋 (SOON)
- Week 7. AutoNabla (≧∇≦) (SOON)
- Week 8. Don't Think 🤔 (TBD)
- Week 9. Just Do 🏃‍♂️ (TBD)
- Week 10. Compilers, They're InTense 😰 (TBD)

</div>

**Week 1. Hello Tensor 👋**

In week 1, you will be gently introduced to language modeling with the bigram model,
using the stochastic and infinitely continuous software 2.0 programming language <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;"> `borscht.Tensor`. You will need to watch one lecture, read two chapters, and write one lab.

- Watch [▶️ Lecture 1: Hello Tensor]()
- Read [📖 Chapter 1.1 From Certain to Uncertain Knowledge](./1.md#11-from-certain-to-uncertain-knowledge)
- Read [📖 Chapter 1.2 Next Token Prediction is Classification and Compression](./1.md#12-next-token-prediction-is-classification-and-compression)
- Write [✍️ Lab 1. Elements of Tensors]()

Optional reading consists of SITP's Intermezzo chapters, which formalizes the notions of uncertainty from probability,
as well as historical papers that first introduced statistical language modeling.

- Read [📖 Intermezzo 1.1 From Programs to Proof](./1.md#i11-from-programs-to-proof)
- Read [📖 Intermezzo 1.4 Probability Spaces and Distributions](./1.md#i14-probability-spaces-and-distributions)
- Read [An Example of Statistical Investigation of the Text Eugene
Onegin Concerning the Connection of Samples in Chains (Markov 1913)](https://alpha60.de/research/markov/DavidLink_AnExampleOfStatistical_MarkovTrans_2007.pdf)
- Read [A Mathematical Theory of Communication (Shannon 1948)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- Read [A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence (McCarthy et al. 1955)](http://jmc.stanford.edu/articles/dartmouth/dartmouth.pdf)

> Note: reading <span class="smallcaps">SITP</span> Chapter 1.1 and Chapter 1.2 consists of watching four embedded lectures from Stanford CS109 Probability for Computer Scientists.
> - Watch Embedded [Stanford CS109 Probability for Computer Scientists Lecture 3: What is Probability?](./1.md#probability)
> - Watch Embedded [Stanford CS109 Probability for Computer Scientists Lecture 6: Random Variables and Expectation](./1.md#random-variable-1)
> - Watch Embedded [Stanford CS109 Probability for Computer Scientists Lecture 11: Joint Distributions](./1.md#joint-distribution)
> - Watch Embedded [Stanford CS109 Probability for Computer Scientists Lecture 4: Conditional Probability and Bayes](./1.md#conditional-distribution)

**Week 2. I Perceive You 👀**

In week 2, you will be introduced to the workhorse model of machine learning, the perceptron.
The perceptron forms a single neuron in a neural network.
You will need to watch one lecture, read two chapters, and write one lab.

- Watch [▶️ Lecture 2: I Perceive You]()
- Read [📖 Chapter 1.3 Parameterizing Classification with Discriminant Functions](./1.md#13-parameterizing-classification-with-discriminant-functions)
- Read [📖 Chapter 1.4 Parameterizing Classification with Probabilistic Models](./1.md#14-parameterizing-classification-with-probabilistic-models)
- Write [✍️ Lab 2.]()

Optional reading consists of continuing to read SITP's Intermezzo chapter,
which formalizes the notion of dimensionality from linear algebra, as well as historical papers on the perceptron, and statistical modeling.

- Read [📖 Intermezzo 1.2 From Programs to Proof](./1.md#i12-from-generics-to-algebras)
- Read [📖 Intermezzo 1.3 Vector Spaces and Linear Transformations](./1.md#i13-vector-spaces-and-linear-transformations)
- Read [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain (Rosenblatt 1958)](https://homepages.math.uic.edu/~lreyzin/papers/rosenblatt58.pdf)
- Read [Statistical Modeling: The Two Cultures (Breiman 2001)](https://www2.math.uu.se/~thulin/mm/breiman.pdf)
- Read [The Epic Story of Maximum Likelihood (Stigler 2008)](https://arxiv.org/pdf/0804.2996)

> Note: reading <span class="smallcaps">SITP</span> Chapter 1.3 and Chapter 1.4 consists of watching four embedded lectures from Stanford CS109 Probability for Computer Scientists.


**Week 3. MetaCircularEvaLuAtor! ↩️**

In week 3, you will be implementing your own <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;"> `borscht.Tensor`
which will run the bigram and perceptron model you trained in the first two weeks.
This implementation of `borscht.Tensor` will be incrementally evolved to support GPU acceleration, and training deep neural networks.
You will  need to watch one lecture, read three chapters, and write one lab.

- Watch [▶️ Lecture 3: MetaCircularEvaLuAtor!]()
- Read [📖 Chapter 2.1 Virtualizing Shapes with Strides](./1.md#21-virtualizing-shapes-with-strides)
- Read [📖 Chapter 2.2 Accelerating Basic Linear Algebra on CPUs](./1.md#22-accelerating-basic-linear-algebra-on-cpus)
- Read [📖 Chapter 2.3 From Performance Primitives to Linear Solvers](./1.md#23-from-performance-primitives-to-linear-solvers)
- Write [✍️ Lab 3. Tensor++ and Matrix Decompositions](./lab.md#lab-3-tensor-and-matrix-decompositions)

Optional reading consists of reading historical papers on array programming languages and numerical scientific computing.

- Read [A Set of Level 3 Basic Linear Algebra Subprograms (Dongarra et al. 1990)](https://dl.acm.org/doi/pdf/10.1145/77626.79170)
- Read [Anatomy of High-Performance Matrix Multiplication (Goto and van de Geijn 2008a)](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- Read [The History of Fortran I, II, and III (Backus 1978)](https://softwarepreservation.computerhistory.org/FORTRAN/paper/p165-backus.pdf)
- Read [A History of MATLAB (Moler and Little 2020)](https://dl.acm.org/doi/pdf/10.1145/3386331)
- Read [Array Programming with NumPy (Harris et al. 2020)](https://www.nature.com/articles/s41586-020-2649-2)

**Week 4. GPUs Go Brrr 🏎️**

In week 4, you will be accelerating the implementation of <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;"> `borscht.Tensor`
by implementing GPU kernels. You will need to watch one lecture, read one chapter, and write one lab.

- Watch [▶️ Lecture 4: GPUs Go Brrr]()
- Read [📖 Chapter 2.4 Accelerating Matrix Multiplication on GPUs](./1.md#24-accelerating-matrix-multiplication-on-gpus)
- Write [✍️ Lab 4. Accelerating Matmul with Tiles and Layouts](./lab.md#lab-4-accelerating-matmul-with-tiles-and-layouts)

**Week 5. King - Man + Queen = 👸**

*Modeling Language with Multi Layer Perceptrons*

- Watch [▶️ Lecture 5]()
- Read [📖 Chapter 3.1 Learning Static Representations with FFNs](./2.md#31-learning-static-representations-with-ffns)
- Read [📖 Chapter 3.2 Iterative Optimization](./2.md#32-iterative-optimization)
- Write [Lab 5. Embedding Projector]()

**Week 6. Plato's Cave 𐦂𖨆𐀪𖠋**

*Modeling Language with Conv Nets, Recurrent Nets, and Transformers*

- Watch [▶️ Lecture 6]()
- Read [📖 Chapter 3.3 Learning Representations with CNNs](./2.md#33-learning-representations-with-cnns)
- Read [📖 Chapter 3.4 Learning Representations with RNNs](./2.md#34-learning-representations-with-rnns)
- Read [📖 Chapter 3.5 Learning Representations with GPTs](./2.md#35-learning-dynamic-representations-with-gpts)
- Write [✍️ Lab 6. Vision Language Action Model]()

**Week 7. AutoNabla (≧∇≦)**

*Implementing* <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;"> `borscht.Tensor.backward()`

- Watch [▶️ Lecture 7]()
- Read [📖 Chapter 4.1 Automatic Differentiation](./2.md#41-automatic-differentation)
- Read [📖 Chapter 4.2 Gradient Based Optimization](./2.md#42-gradient-based-optimization)
- Write [✍️ Lab 7. Custom Autograd Backwards for VLA](./lab.md#lab-7-custom-autograd-backwards-for-vla)

**Week 8. Don't Think 🤔 (TODO)**

*Reasoning Models (todo)*

**Week 9. Just Do 🏃‍♂️ (TODO)**

*Agents (todo)*

**Week 10. Compilers, They're InTense 😰 (TODO)**

*Tensor Compilers (todo)*

## Labs

**Lab 1: Elements of Tensors**

**Lab 2**

**Lab 3: Tensor++ and Matrix Decompositions**

- Extend `Tensor` which we developed in [Chapter 2.1 Virtualizing Shapes with Strides](./1.md#21-virtualizing-shapes-with-strides), [Chapter 2.2 Accelerating Basic Linear Algebra on CPUs](./1.md#22-accelerating-basic-linear-algebra-on-cpus) with native implementations of the operations from Lab 1
- Extend `borscht.linalg.solve()` which we developed in [Chapter 2.3 From Performance Primitives to Linear Solvers](./1.md#23-from-performance-primitives-to-linear-solvers) with matrix decompositions

**Lab 4: Accelerating Matmul with Tiles and Layouts**

- Extend `Tensor.__matmul__` which we developed in [Chapter 2.4 Accelerating Matrix Multiplication on GPUs](./1.md#24-accelerating-matrix-multiplication-on-gpus) with an alternatively accelerated matrix multiplication using cuteDSL

**Lab 5**

**Lab 6: Vision Language Action Model**

**Lab 7: Custom Autograd Backwards for VLA**

- Extend `Tensor.backward()` which we developed in [Chapter 4.1 Automatic Differentiation](./2.md#41-automatic-differentation) with accelerated VLA operations from Lab 6