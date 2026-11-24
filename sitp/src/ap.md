# Appendix

> In which we implement the engine of automatic differentiation, the magic spell AI researchers cast to receive $\nabla \mathscr{L}(\theta)$ just from the specifcation of $\mathscr{L}$.

<div class="toc">

- [4.1 Automatic Differentiation](#41-automatic-differentation)
  - [4.1.1 Symbolic, Numerical, and Algorithmic Differentiation](#411-symbolic-numerical-and-algorithmic-differentiation)
  - [4.1.2 Forward Mode with <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;">`borscht.forward()`](#412-forward-mode-with-borschtforward)
  - [4.1.3 Backward Mode with <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;">`borscht.backward()`](#413-backward-mode-with-borschtbackward)
  - [4.1.4 Stochastic Gradient Descent with <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;">`borscht.optim.sgd()`]()
  - [4.1.5 Adam with <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;">`borscht.optim.adamw()`]()

</div>


#### 4.1.1 Symbolic, Numerical, and Algorithmic Differentiation

<small>[$\hookleftarrow$ Table of Contents](#ii-table-of-contents)</small>



<div class="dropcap">

Lorem ipsum dolor sit amet,


<span class="lecnote">*<span class="fa-svg"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Free 6.2.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free --><path d="M512 256c0 141.4-114.6 256-256 256S0 397.4 0 256S114.6 0 256 0S512 114.6 512 256zM188.3 147.1c-7.6 4.2-12.3 12.3-12.3 20.9V344c0 8.7 4.7 16.7 12.3 20.9s16.8 4.1 24.3-.5l144-88c7.1-4.4 11.5-12.1 11.5-20.5s-4.4-16.1-11.5-20.5l-144-88c-7.4-4.5-16.7-4.7-24.3-.5z"/></svg></span> [MIT 18.S096 Matrix Calculus for Machine Learning Lecture 5 Part 2: Forward Automatic Differentiation via Dual Numbers](https://www.youtube.com/watch?v=5F6roh4pmJU&list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE&index=12)*</span>

<span class="lecnote">*<span class="fa-svg"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Free 6.2.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free --><path d="M512 256c0 141.4-114.6 256-256 256S0 397.4 0 256S114.6 0 256 0S512 114.6 512 256zM188.3 147.1c-7.6 4.2-12.3 12.3-12.3 20.9V344c0 8.7 4.7 16.7 12.3 20.9s16.8 4.1 24.3-.5l144-88c7.1-4.4 11.5-12.1 11.5-20.5s-4.4-16.1-11.5-20.5l-144-88c-7.4-4.5-16.7-4.7-24.3-.5z"/></svg></span> [MIT 18.S096 Matrix Calculus for Machine Learning Lecture 5 Part 3: Differentiation on Computational Graphs](https://www.youtube.com/watch?v=r9_5dxtDTOk&list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE&index=12)*</span>

<span class="lecnote">*<span class="fa-svg"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Free 6.2.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free --><path d="M512 256c0 141.4-114.6 256-256 256S0 397.4 0 256S114.6 0 256 0S512 114.6 512 256zM188.3 147.1c-7.6 4.2-12.3 12.3-12.3 20.9V344c0 8.7 4.7 16.7 12.3 20.9s16.8 4.1 24.3-.5l144-88c7.1-4.4 11.5-12.1 11.5-20.5s-4.4-16.1-11.5-20.5l-144-88c-7.4-4.5-16.7-4.7-24.3-.5z"/></svg></span> [MIT 18.S096 Matrix Calculus for Machine Learning Lecture 8 Part 2: Automatic Differentiation on Computational Graphs](https://www.youtube.com/watch?v=43kl6wi1H6k&list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE&index=18)*</span>

consectetur adipiscing elit. Donec aliquam varius mi, pharetra scelerisque nulla vehicula ut. Cras ac faucibus dolor. Nunc tempus efficitur ultricies. Nullam non fermentum elit, et tincidunt ante. Donec hendrerit metus ac urna semper, in facilisis leo egestas. Integer sit amet quam ultrices, dapibus sem a, volutpat velit. Pellentesque eget ipsum consequat sem luctus blandit a et nibh. Mauris gravida felis ut dignissim volutpat. Quisque volutpat pulvinar enim in venenatis. Integer posuere tincidunt leo vel congue. Sed ex velit, posuere ut nibh nec, ultrices malesuada nibh. Fusce scelerisque mollis neque id fringilla. Etiam rhoncus quam a leo ultrices rutrum non vitae est. Donec bibendum, arcu sagittis consequat vestibulum, lectus elit bibendum libero, non rhoncus lorem lorem a neque. Sed ex felis, rutrum non consequat eget, tincidunt ac lacus.

</div>

<iframe width="560" height="315" loading="lazy" src="https://www.youtube.com/embed/YG15m2VwSjA?si=z6eEC7e_4qDckMeB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

#### 4.1.2 Forward Mode

<small>[$\hookleftarrow$ Table of Contents](#ii-table-of-contents)</small>



<div class="dropcap">

Consider the function $f\colon \reals, \reals \to \reals$ where $f \colonequals \exp(x_1) * \sin^2x_2$,
and translate it to it's computational counterpart in python with one-dimensional `Tensor`s:

</div>

```python
import picograd as pg

def f(x1: pg.Tensor, x2: pg.Tensor) -> pg.Tensor:
  a = pg.exp(x1)
  b = pg.sin(x2)
  c = b**2
  d = a*c
  return d
```
<small>Figure 1. Python source for the function $f\colon \reals, \reals \to \reals$ where $f \colonequals \exp(x_1) * \sin^2x_2$</small>

Here we've broken up the function to render the subexpressions more clearly.
But this isn't necessary — automatic differentiation will work if the function was expressed in one line.
In part one, the development of `picograd` followed that of [`numpy`](https://numpy.org/) —
an array programming language similar to Matlab but embedded in the host language of Python,
that could evaluate functions of the form $f(\mathbf{x})$
where `Tensor` objects stored their values $f(\mathbf{x})$ with the `value:` field
and the function types that produced their values with `Op`.
For instance, evaluating the specified function `f` from above with 9 and 10

```python
if __name__ == "__main__":
  print(f(9, 10))
```

populates the `Tensor.value` fields.
In part one of the book we verified this with a REPL-interface,
but we can also represent the entire expression being evaluated with a
graph of vertices and edges where the vertices are `Tensors`
(along with their `Op`s and `value`s) and the edges are their data dependencies:

Here you can see that even if the function was specified in one line,
the graph of the expression always parses into `Tensor` vertices, and data dependency edges.
You may have noticed the `Tensor.grad` fields, which supposedly store the values of derivatives $f'(\mathbf{x})$.
The question now remains in how to populate these fields.

Taking a step back to differential calculus,
deriving the derivative of $f \colonequals \exp(x_1) * \sin^2x_2$
involves the application of the chain rule where $f(x) := g(h(x)) \implies f'(x) = g'(h(x))h'(x)$.
Evaluating the derivative of the function with respect to its inputs $f'(x_1)$ and $f'(x_2)$ results in

$$f'(x_1) = \frac{\partial}{\partial x_1} \lbrack \exp(x_1) * \sin^2x_2 \rbrack
                                  = \frac{\partial}{\partial x_1} \lbrack \exp(x_1)\rbrack * \sin^2x_2
                                  = \exp(x_1)(1)*(\sin^2x_2)$$
$$f'(x_2) = \frac{\partial}{\partial x_2} \lbrack \exp(x_1) * \sin^2x_2 \rbrack
                                  = \exp(x_1) * \frac{\partial}{\partial x_2} \lbrack \sin^2x_2 \rbrack
                                  = \exp(x_1)* 2\sin x_2\cos x_2$$

symbolic and numeric differentiattion
symbolic differentiation has *performance* issues since a large unrolled expression must be constructed in order to differentiate[^0],
whereas numerical differentiation has *correctness* issues since evaluating finite differences requires evaluating functions to a precision point resulting in numerical instability.
(trace through EXAMPLE for both. talking nets widrow)


To populate the `Tensor.grad` fields, the simplest idea would be
to literally translate the manual derivation of the derivative into code.
The translation from math to code involves a design decision:
*should we evaluate from outputs to inputs (symbolically outside-in, graphically right-to-left) or from inputs to outputs (symbolically inside-out, graphically left-to-right)?*
Although the former order seems more natural with symbolic expressions,
there's nothing illegal about the latter.

```python
import picograd as pg

def f(x1: pg.Tensor, x2: pg.Tensor) -> pg.Tensor:
  a = pg.exp(x1)
  b = pg.sin(x2)
  c = b**2
  d = a*c
  return d

# dict[f(x), f'(x)] of local derivatives (adjoints)
dd_da, dd_dc = [c, a] # d(a,c):=a*c ==> d'(a)=c, d'(c)=a
da_dx1 = pg.exp(x1) # a(x1):=exp(x1) ==> a'(x1)=exp(x1)
dc_db = 2*b # c(b):=b^2 ==> c'(b)=2b
db_dx2 = pg.cos(x2) # b(x2):=sin(x2) ==> b'(x2)=cos(x2)

# outputs to inputs: outside-in symbolically, right-to-left graphically
dd_dd = pg.Tensor(1) # base case
dd_da, dd_dc = [dd_dd*dd_da, dd_dd*dd_dc]
dd_dx1 = dd_da*da_dx1 # DONE for the x1->d path

dd_db = dd_dc*dc_db
dd_dx1 = dd_db*db_dx2 # DONE for x2->path

# inputs to outputs: inside-out symbolically, left-to-right graphically
dx1_dx1, dx2_dx2 = [pg.Tensor(1), pg.Tensor(1)] # base case
da_dx1 = da_dx1*dx1_dx1
dd_dx1 = dd_da*da_dx1 # DONE for the x1->d path

db_dx2 = db_dx2*dx2_dx2
dc_dx2 = dc_dc*db_dx2
dd_dx2 = dd_dc*dc_dx_2 # DONE for the x2->d path
```

*Do you notice any difference in the number of evaluations between the two orders?*

#### 4.1.3 Backward Mode

<small>[$\hookleftarrow$ Table of Contents](#ii-table-of-contents)</small>



<div class="dropcap">

The outputs-to-input ordering takes 6 arithmetic operations (including the destructuring),
whereas the input-to-output ordering take 7 arithmetic operations.
This is because the former can *reuse* `dd_dd`
as a dynamic programming solution to a subproblem for the two inputs, whereas the latter cannot.
And taking a step back, we only want to reuse the output because the shape of the function
is of $f\colon\reals^2 \to \reals$. Alternatively, if $f$ had type $f\colon\reals \to \reals^2$,
then the input-to-output ordering would be able to reuse results.
This distinction is referred to as "forward-mode" vs "reverse-mode", and reflects
the fact that for some function $f\colon\reals^n \to \reals^m$ 
the time complexity of forward-mode differentiation is proportional to $n$,
whereas that of forward-mode differentiation is proportional to $m$.
If the expression graph *fans-in* so that $n > m$, reverse-mode is preferred.
If the expression graph *fans-out* so that $m > n$, forward-mode is preferred.
However, if we take a step with a graph-theory lens, we can see that the derivative is the *sum of paths*,
where each path is a product of local derivatives from the input source to the output sink.
From a combinatorics perspective, we are calculating all the possible (ors) ways (ands)
on how the inputs perturb the output. That is:
</div>

$$\frac{\partial f}{\partial x_i } = \sum\prod \frac{\partial v_j}{\partial v_k} $$

and as long as the operations along this path are *associative* — $(AB)C = A(BC)$ —
then we can choose the order in how we perform these path products to minimize the number of operations.
Finding the optimal ordering is an NP-hard problem because ____.
For instance, if the expression graph is diamond-shaped, evaluating the derivative
with forward-mode for the left-half and reverse-mode for the right-half would be more performant.
In practice, we use reverse-mode as a heuristic, since most of the functions
that are differentiated (so they can be optimized) in the field of machine learning are neural networks of the form $f\colon\reals^n \to \reals$


<small>[$\hookleftarrow$ Table of Contents](#ii-table-of-contents)</small>

<iframe  width="560" height="315" loading="lazy" src="https://www.youtube.com/embed/VkHfRKewkWw?si=ctIS6a1CxJrUBCxI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<iframe width="560" height="315" loading="lazy" src="https://www.youtube.com/embed/tIeHLnjs5U8?si=_DkzRHubxKOI4kW1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<iframe width="560" height="315" loading="lazy" src="https://www.youtube.com/embed/SmZmBKc7Lrs?si=OeqPIs5no9Fv1H_u" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

*How can we generalize this into an algorithm?*<br/>
All we need are 1. mappings from $f(\mathbf{x}) \to f'(\mathbf{x})$ and 2. a topological sort

For the derivative rules, the same way that optimizing compilers
implement an optimization "manually" once which then gets reused many times,
the authors of deep learning frameworks also implement derivatives manually
which then become reused many times through automatic differentiation.
In theory, we can differentiate any expression with f'(x) with only a few derivative
rules for addition and multiplication, but in practice most frameworks provide sugar for complex derivatives.

For topological sort, we can simply reversed the ordering produced by a depth-first-search:
```python
def toposort(self):
  order: list[Op] = []
  visited: set[Op] = set()

  def dfs(node: Op) -> None:
    if node in visited: return
    visited.add(node)
    for src in node.src: dfs(src)
    order.append(node)

  dfs(self)
  return order

class Tensor():
  def backward():
    for t in reversed(topo):
      t.backward()
```


We will now use this idea to modify the interpretation of our deep learning framework
to not only evaluate $f(\mathbf{x})$, but $f'(\mathbf{x})$ as well. This is done
by dynamically overloading the operators at runtime[^0] to trace the expression graph
```python
chain_rules = PatternMatcher([
  (Pattern(OpCode.MATMUL, name="input"), lambda output_grad, input: (_____,)),
  (Pattern(OpCode.MATVEC, name="input"), lambda output_grad, input: (_____,)),
  (Pattern(OpCode.RECIPROCAL, name="input"), lambda output_grad, input: (-output_grad * input * input,)),
  (Pattern(OpCode.SIN, name="input"), lambda output_grad, input: ((math.pi/2 - input.src[0]).sin() * output_grad,)),
  (Pattern(OpCode.LOG2, name="input"), lambda output_grad, input: (output_grad / (input.src[0] * math.log(2)),)),
  (Pattern(OpCode.EXP2, name="input"), lambda output_grad, input: (input * output_grad * math.log(2),)),
  (Pattern(OpCode.SQRT, name="input"), lambda output_grad, input: (output_grad / (input*2),)),
  (Pattern(OpCode.ADD), lambda output_grad: (1.0*output_grad, 1.0*output_grad)),
  (Pattern(OpCode.MUL, name="input"), lambda output_grad, input: (input.src[1]*output_grad, input.src[0]*output_grad)),
])

class Tensor:
  def _forward(self, f:Callable, *other:Tensor) -> Tensor: #extra_args=(), **kwargs)
    out_tensor = evaluator.eval_uop([self, other], out_uop)

  def backward(self, grad:Tensor|None=None) -> Tensor:
    """
    backward performs by collecting tensors, computing gradients with automatic differentiation, and updating said tensors.
    """
    # 1. collect all tensors that requires grad by topologically sorting the graph of uops and filter
    all_uops = self.uop.toposort()
    tensors_require_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and t.uop in all_uops and t.requires_grad]
    uops_require_grad = [t.uop for t in tensors_require_grad]
    assert grad is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in tensors_require_grad)): raise RuntimeError("only float Tensors have gradient")
    
    # 2. compute the gradient with a map of tensors to partials
    if grad is None: grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False) # base case is 1.0
    tens2grads = Tensor._automatically_differentiate(self.uop, grad.uop, set(uops_require_grad)) # skipping materializing zerod grads for now
    grads = [Tensor(g, device=t.device) for t,g in zip(tens2grads.keys, tens2grads.values)] # initialize tensor grads on device
    
    # 3. update the tensors that require grad with the gradient's partials
    for t,g in zip(tensors_require_grad, grads):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g) # accumulate if t.grad exists
    return self

  @staticmethod
  def _automatically_differentiate(root:Op, root_grad:Op, targets:set[Op]) -> dict[Op, Op]:
    """
    _differentiate backpropagates partials on a topologically sorted expression graph with the chain rule
    and produces the gradient in the form of a map of ops to their partials (which, in turn, are ops)
    """
    tens2grads = {root: root_grad}

    # 1. topological sort
    in_target_path: dict[Op, bool] = {}
    for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.src)
    dfs = list(root.toposort()) # lambda node: node.op not in {OpCode.DETACH, OpCode.ASSIGN} and in_target_path[node])) # don't flow through DETACH/ASSIGN or anything not in target path

    # 2. backpropagation with the chain rule
    for tensor in reversed(dfs):
      if tensor not in tens2grads: continue

      local_grads: tuple[Op|None, ...]|None = cast(tuple[Op, ...]|None, chain_rules.rewrite(tensor, ctx=tens2grads[tensor]))
      if local_grads is None: raise RuntimeError(f"failed to compute gradient for {tensor.op}\n\nin {str(tensor)[0:1000]}...")
      assert len(local_grads) == len(tensor.src), f"got {len(local_grads)} gradient, expected {len(tensor.src)}"

      for tensor,local_grad in zip(tensor.src, local_grads): # <--------------------- MOOOSE: why are we accumulating inside ad()? don't we do it in backward()??
        if local_grad is None: continue
        if tensor in tens2grads: tens2grads[tensor] = tens2grads[tensor] + local_grad # accumulate if tensor exists
        else: tens2grads[tensor] = local_grad # o/w initialize
```

To implement automatic differentiation with `Tensor.backward()`,
there is a design decision to be made
— the choice of implementing it dynamically or just-in-time[^3],
similar to the decision of how to implement types for general programming languages[^4].
This stands in contrast to the alternative of performing a just-in-time, source-to-source transformation.

Let's now move onto automatically differentiating the functions of neural networks,
specifically the FFN language model from earlier. (johnson/ryan adams ordering)
n^2 vs n^3 

<iframe src="https://videolectures.net/embed/videos/deeplearning2017_johnson_automatic_differentiation?part=1" width="100%" frameborder="0" allowfullscreen style="aspect-ratio:16/9"></iframe>


<!-- So far we've been training feedforward neural networks using the magical

<!--

blabbing about htdp and netsor

So, similar to <span class="smallcaps">HTDP</span>'s [Intermezzo](https://htdp.org/2026-5-28//Book/i1-2.html)<span class="sidenote-number"></span><span class="sidenote">*Also see a Functional Introduction to Computer Science (Ragde) [§3. Syntax and Semantics](https://cs.uwaterloo.ca/~plragde/flaneries/FICS/Syntax_and_semantics.html)*</span>, after introducing <img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;">[`borscht`](https://github.com/j4orz/asitpofborscht) as if it were a natural language appealing to our existing knowledge for intuition, we now define the language's *form* with a **syntax**<span class="defnote">**syntax**</span> via **grammar**<span class="defnote">**grammar**</span> and it's *meaning* with a **semantics**<span class="defnote">**semantics**</span> via **inference rules**<span class="defnote">**inference rules**</span>. The only difference is that rather than define a language of software 1.0 which has features for assignment, control flow, and functions, we will do so with a language of software 2.0 referred to as <span class="smallcaps">NETSOR</span> and implemented by
<img class="icon" src="./assets/borscht.png" alt="" style="aspect-ratio: 736 / 510;">[`borscht.Tensor`](https://github.com/j4orz/asitpofborscht) which has features for dimension, and stochastics. -->

<!-- The first was the mask of probability.
Whether these next token predictors are bigram models, logistic regression models, or the GPT-2 like neural nets that we will encounter in
[Part II. Deep Neural Networks](./2.md),
they are all functions that produce an output distribution $p$ over the next word $w_{t+1}$ given some input sentence $(w_1, \dots, w_t)$ as the history. That is, they are functions of type $f: \Sigma^{*} \to \Delta^{|\Sigma|}$, which correspond to multidimensional lists of numbers.
That is, **probability can be seen *as* array programming *of* lists**. (bigram)
More succinctly, a distribution has type $p: \N^3 \to [0,1]^{|V|}$ corresponds to $x \in \reals^{n \times m \times p}$

The second was the mask of linear algebra. Biologically inspired by a neuron's synapses and soma, the discipline of deep learning implements such function bodies $f$ by parameterizing them with weights $\mathbf{w}$ and bias $b$
such that their final classifying predictions correspond to linearly separated geometrical points.
That is, **the architecture of neural networks can be seen *as* array programming *of* points in the space of prediction**.
More succinctly, single layer neural net $f(\mathbf{x}):= \sum_{i=1}^{d} w_ix_i$,
which corresponds to the dot product of $\mathbf{w}^{\top}\mathbf{x}$.

The primary texts consulted in the writing of this chapter were
[(Bertsekas and Tsitsiklis 2008)](https://www.mit.edu/~dimitrib/probbook.html),
[(Chan 2021)](https://probability4datascience.com/),
[(Wasserman 2004)](https://www.stat.cmu.edu/~brian/valerie/617-2022/0%20-%20books/2004%20-%20wasserman%20-%20all%20of%20statistics.pdf),
-->

<!-- <div class="dual">

```python
def affirm_or_negate_life() -> bool:
  return True

def meaning_of_life() -> int:
  return 42

def best_dessert() -> float:
  return 3.14

def best_computer_science_book() -> String:
  return "Structure and Interpretation of Computer Programs"

if __name__ == "__main__":
  philosophy = affirm_or_negate_life()
  teleology = meaning_of_life()
  dessert = best_dessert()
  book = best_computer_science_book()
  print(f"Should life be affirmed or negated?: {philosophy}")
  print(f"What is the meaning of life?: {teleology}")
  print(f"What is the best dessert?: {dessert}")
  print(f"What is the best computer science book?: {book}")
```

```rust
fn affirm_or_negate_life() -> bool {
  return true
}

fn meaning_of_life() -> i32 {
  return 42
}

fn best_dessert() -> f32 {
  return 3.14
}

fn best_computer_science_book() -> char {
  return 'S'
}

fn main() {
  let philosophy = affirm_or_negate_life()
  let teleology = meaning_of_life()
  let dessert = best_dessert()
  let book = best_computer_science_book()
  println!("Should life be affirmed or negated?: {philosophy}")
  println!("What is the meaning of life?: {teleology}")
  println!("What is the best dessert?: {dessert}")
  println!("What is the best computer science book?: {book}")
}
```

</div>

<div class="dual">

```python
class Coordinate() {
  def __init__(self):
    self.x = 4
    self.y = 2
}
```

```rust
struct TwoDimensionalCoordinate {
  x: i32,
  y: i32
}

impl TwoDimensionalCoordinate {
  fn new(x: i32, y: i32) -> Self {
    Self { x: 4, y: 2 }
  }
}

enum Tensor {
  TwoDimensional(TwoDimensionalCoordinate),
  ThreeDimensional(ThreeDimensionalCoordinate)
}

fn add_tensor(x: Tensor, y: Tensor) -> Tensor {
  match x {

  }
}
```

</div>

 -->