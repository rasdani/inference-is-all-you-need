# inference-is-all-you-need


## Setup

```bash
mamba create env -f environment.yml
```

## Run

In a seperate terminal, run the verfier server:

```bash
CUDA_VISIBLE_DEVICES="2,3" python verifier.py
```

Then, run the solver:
```bash
CUDA_VISIBLE_DEVICES="0,1" python solver.py
```


## Evaluate 

```bash
python evaluate.py results/your_results.jsonl
```


## Papers and resources used

- https://arxiv.org/pdf/2408.03314v1
- https://arxiv.org/pdf/2407.21787
- https://arxiv.org/pdf/2305.20050
- https://arxiv.org/pdf/2408.15240
- https://arxiv.org/pdf/2406.18629


Sponsored by ellamind:

https://ellamind.com/
