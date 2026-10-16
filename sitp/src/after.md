![](./assets/pref.jpeg)

<small>*Presenting an early outline of <span class="smallcaps">SITP</span> at [Toronto School of Foundation Modeling Season 1 (2025)](https://tsfm.ca/schedule)*</small>

# Afterword

<div class="dropcap">

Congratulations on completing your adventure through the whirlwind tour of The Structure and Interpretation of Tensor Programs.
To continue deepening your knowledge of deep learning and deep learning systems,
we provide explicit offramps that bridge you from `nanochat` to `marin` as well as from `borscht` to `torch` and `vllm`.

Good luck on your journey.<br>
I'll see you at work.

</div>

## Afterward Table of Contents

<div class="toc toc-root">

- [A. From `nanochat` to `marin`](#1-from-nanochat-to-llama-and-deepseek)
- [B. From `borscht` to `torch`](#2-from-borscht-to-torch)

</div>

## A. From `nanochat` to `marin`

<div class="toc">

- [A Mixture of Experts]()

</div>


## B. From `borscht` to `torch`

<div class="dropcap">

Borscht is an array programming language with support for automatic differentiation and gpu acceleration following PyTorch1's eager mode outlined in [(Paszke et al., 2019)](https://arxiv.org/pdf/1912.01703) as well as compilation and distributed training following PyTorch2's graph mode outlined in [(Ansel et al., 2024)](https://dl.acm.org/doi/pdf/10.1145/3620665.3640366).

</div>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        # gather the word embeddings of the previous 3 words
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
            embs.append(tok_emb)

        # concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        logits = self.mlp(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
```

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.



```python
if __name__ == '__main__':
    # todo: parser
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    model = MLP(config)
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # B.1.1: feed into the model
        logits, loss = model(X, Y)

        # B.1.2: calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        break
```

<div class="toc">

- [B.1 Eager Mode with PyTorch1]()
  - [B.1.1 Forward]()
  - [B.1.2 Backward]()
  - [B.1.3 Step]()
- [B.2 Graph Mode with PyTorch2]()
- [B.3 Inference with vLLM]()

</div>

### B.1 Eager Mode with PyTorch1

#### B.1.1 Forward

<div class="dropcap">

Forward

</div>

<br><br><br><br>

#### B.1.2 Backward

<div class="dropcap">

Backward

</div>

<br><br><br><br>

#### B.1.3 Step

<div class="dropcap">

Step

</div>

<br><br><br><br>

### B.2 Graph Mode with PyTorch2

### B.3 Inference with vLLM