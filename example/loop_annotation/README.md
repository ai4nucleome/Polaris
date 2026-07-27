# Loop Annotation Example (GM12878 bulk Hi-C, 250M valid read pairs)

This example annotates chromatin loops on chromosomes 15–17 of GM12878 Hi-C data with **Polaris**, and shows the three equivalent ways to run it. All commands are also available in [`loop_annotation.ipynb`](./loop_annotation.ipynb).

## 1. Download the example contact map

```bash
wget "https://huggingface.co/rr-ss/Polaris/resolve/main/example/loop_annotation/GM12878_250M.bcool?download=true" -O GM12878_250M.bcool
```

Polaris accepts a multi-resolution cooler (`.mcool`) or the band-cooler (`.bcool`) format at 5 kb resolution.

> **Single-cell Hi-C (scHi-C):** the loop-calling command is **identical** to bulk. You only need to prepare the input first, by aggregating contact maps from cells of the same type into a single pseudo-bulk `.mcool` (e.g. by summing per-cell `.cool` files with `cooler merge`, or from a `.scool`). Polaris can then annotate loops from as few as ~25 cells. See the [top-level README](../README.md).

## 2. Annotate loops

**Method 1 — one step (`loop pred`):**
```bash
polaris loop pred -c chr15,chr16,chr17 -i GM12878_250M.bcool -o GM12878_250M_chr151617_loops.bedpe -t 0.5
```

**Method 2 — two steps (`loop score` → `loop pool`):** produces the same loops, but also keeps the intermediate per-pixel score file.
```bash
polaris loop score -c chr15,chr16,chr17 -i GM12878_250M.bcool -o GM12878_250M_chr151617_loop_score.bedpe -t 0.5
polaris loop pool  -i GM12878_250M_chr151617_loop_score.bedpe -o GM12878_250M_chr151617_loops_method2.bedpe -t 0.5
```

**Large maps (`loop scorelf`):** a memory-efficient scorer for very large, high-coverage, or high-resolution `.mcool` files; use it in place of `loop score`, then pool as above.
```bash
polaris loop scorelf -c chr15,chr16,chr17 -i GM12878_250M.bcool -o GM12878_250M_chr151617_loop_score.bedpe -t 0.5
polaris loop pool    -i GM12878_250M_chr151617_loop_score.bedpe -o GM12878_250M_chr151617_loops_method2.bedpe -t 0.5
```

Parameters: `-c` restricts the analysis to the listed chromosomes (omit it to run on all autosomes); `-t` is the loop-score threshold. Run `polaris loop <subcommand> --help` for the full list.

## 3. Output

Each `*_loops.bedpe` file is tab-separated, one loop per line:

| Field | Detail |
|---|---|
| Chr1 / Chr2 | chromosome names |
| Start1 / Start2 | anchor start coordinates |
| End1 / End2 | anchor end coordinates (End = Start + resolution) |
| Score | Polaris loop score, in [0, 1] |

The precomputed outputs of this example are provided in this folder:
- `GM12878_250M_chr151617_loops.bedpe` — loops from Method 1
- `GM12878_250M_chr151617_loops_method2.bedpe` — loops from Method 2 (identical to Method 1)
- `GM12878_250M_chr151617_loop_score.bedpe` — the intermediate per-pixel score file

Next, see the [APA example](../APA) to visualize the aggregate signal at these loops.
