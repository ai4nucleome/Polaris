# Example Usage: Loop Annotation and APA

This folder contains worked examples of **Polaris** for chromatin loop annotation and aggregate peak analysis (APA), with sample data and expected outputs. Each subfolder has its own step-by-step `README`:

- [**`loop_annotation/`**](./loop_annotation) — annotate loops from a contact map (three equivalent workflows) and interpret the output.
- [**`APA/`**](./APA) — aggregate peak analysis of the annotated loops.

For the complete documentation and CLI reference, see the [Polaris documentation](https://nucleome-polaris.readthedocs.io/en/latest/).

## Input data

Polaris takes a chromosomal contact map as input, in either the multi-resolution cooler (`.mcool`) or band-cooler (`.bcool`) format, at 5 kb resolution by default. Depending on the assay:

- **Bulk Hi-C / Micro-C / DNA SPRITE:** use a processed `.mcool` (e.g. from the [4DN data portal](https://data.4dnucleome.org)), or convert your own data with [cooler](https://cooler.readthedocs.io).
- **Single-cell Hi-C (scHi-C):** the Polaris command is **identical** to bulk — you only prepare the input differently. First aggregate contact maps from cells of the same type into a single **pseudo-bulk** `.mcool` (e.g. by summing per-cell `.cool` files with `cooler merge`, or from a `.scool`), then run Polaris exactly as in the bulk example. Polaris can annotate loops from as few as ~25 cells.

> ❗ **GPU / memory note:** we recommend running Polaris on a **GPU**. If you encounter a `CUDA OUT OF MEMORY` error, reduce the `--batchsize` parameter (the default 128 needs ~36 GB of GPU memory; 24 needs less than 10 GB).

## Command-line subcommands

| Command | Purpose |
|---|---|
| `polaris loop pred`    | Annotate loops directly (scoring + clustering in one step). |
| `polaris loop score`   | Output a per-pixel loop-score file. |
| `polaris loop pool`    | Cluster a loop-score file into discrete loops. |
| `polaris loop scorelf` | Memory-efficient scoring for very large / high-coverage / high-resolution maps. |
| `polaris util pileup`  | Aggregate peak analysis (APA) of a loop set. |

Run `polaris <command> --help`, or see the [CLI reference](https://nucleome-polaris.readthedocs.io/en/latest/CLI_reference.html) for the full list of parameters.

## Quick start

```bash
# 1. download the example bulk Hi-C contact map (GM12878, 250M valid read pairs)
wget "https://huggingface.co/rr-ss/Polaris/resolve/main/example/loop_annotation/GM12878_250M.bcool?download=true" -O ./loop_annotation/GM12878_250M.bcool

# 2. annotate loops on chr15-17
polaris loop pred -c chr15,chr16,chr17 -i ./loop_annotation/GM12878_250M.bcool -o ./loop_annotation/GM12878_250M_chr151617_loops.bedpe -t 0.5

# 3. aggregate peak analysis of the annotated loops
polaris util pileup --savefig ./APA/GM12878_250M_chr151617_loops.pileup.png --p2ll True \
    ./loop_annotation/GM12878_250M_chr151617_loops.bedpe ./loop_annotation/GM12878_250M.bcool
```

See [`loop_annotation/README.md`](./loop_annotation) and [`APA/README.md`](./APA) for the full walkthroughs and expected outputs.
