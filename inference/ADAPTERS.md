# ADAPTERS — model & benchmark contract

Companion to `PLAN.md`. Defines the **only** code you write to add a model or benchmark. Everything
else (queue, SLURM pool, worker, output layout, logging, scheduling, report) is generic and never
imports diffusers / mmdet / OneIG / DPG — only adapters do.

Grounded in the current code: `scripts/common.py` is already a generic diffusers runner (config =
`pipeline:` module-path + `generation_params:`), and `scripts/generate_*.py` / `eval_*.py` already
isolate the per-bench prompt-source and scorer. Adapters are those pieces repackaged behind two ABCs.

---

## 0. Shared data types

```python
# cfgbench/benchmarks/base.py
@dataclass(frozen=True)
class PromptSpec:
    id: str                 # stable id within the benchmark (e.g. "00042", or OneIG row id)
    text: str               # the prompt string fed to the model
    category: str           # subcategory bucket for summaries (e.g. "two_object", "text", "reasoning")
    meta: dict              # bench-specific extras carried through to eval (e.g. GenEval's full metadata row)

@dataclass(frozen=True)
class GenItem:              # one generated sample, handed to evaluate()
    item_id: str            # "<model>/<config>/<bench>/<category>/<prompt_id>/s<idx>"
    prompt: PromptSpec
    image_path: Path        # outputs/.../<prompt_id>/sample_<idx>.png
    sidecar_path: Path      # outputs/.../<prompt_id>/sample_<idx>.json
    sample_idx: int
```

`PromptSpec.meta` is the escape hatch: anything an evaluator needs that isn't text/category travels
here (GenEval needs the full metadata object for object/color/position checks; OneIG/DPG need nothing
extra). Core treats it as opaque.

---

## 1. ModelAdapter

```python
# cfgbench/models/base.py
class ModelAdapter(ABC):
    name: str                                   # registry key, e.g. "sd35"
    @abstractmethod
    def load(self, config: dict) -> "ModelHandle": ...

class ModelHandle(ABC):
    settings: GenSettings                       # resolved params, copied into each sidecar JSON
    @abstractmethod
    def generate(self, prompt: str, *, seed: int, n: int = 1, **override) -> list["PIL.Image"]: ...
    def close(self) -> None: ...                # free VRAM; default no-op
```

- `config` is the parsed method YAML (see §2). It encodes both **which pipeline/method** and its
  **generation params** — so the method (CFG/SEG/PAG/…) is not a separate abstraction; it lives in the
  config and is applied inside `load`.
- `generate` returns `n` PIL images for one prompt at a given seed. Determinism: seed the generator
  per sample (`seed + i`) exactly as today.
- The worker calls `load(config)` **once per shard**, then `generate(...)` per prompt, writes the image
  + sidecar JSON, then `close()`. Core owns paths, timing, sidecar, retries.

### 1a. The generic diffusers adapter (covers sd35, cosmos2, and most future models)

Because every `diffusers` pipeline shares `pipe(prompt, num_inference_steps=, guidance_scale=,
generator=, num_images_per_prompt=, **extra).images`, one adapter handles all of them, driven entirely
by config. It is a thin repackage of `common.load_config/load_pipeline/resolve_settings`:

```python
# cfgbench/models/diffusers_adapter.py
class DiffusersModelAdapter(ModelAdapter):
    def __init__(self, name): self.name = name
    def load(self, config):
        spec, gen_params = load_config(config["__path__"])     # existing common.load_config
        settings = resolve_settings(gen_params)                # existing GenSettings merge
        pipe = load_pipeline(spec, pick_device())              # imports `pipeline:` module factory
        return _DiffusersHandle(pipe, settings)

class _DiffusersHandle(ModelHandle):
    def generate(self, prompt, *, seed, n=1, **override):
        kw = self.settings.call_kwargs(); kw.update(override)
        return [self.pipe(prompt, **kw, num_images_per_prompt=1,
                          generator=make_generator(self.device, seed + i)).images[0]
                for i in range(n)]
```

`sd35` and `cosmos2` are just `DiffusersModelAdapter("sd35")` / `("cosmos2")`. Their differences
(method patches, scheduler, dtype) already live in `pipelines/<model>/*.py` + the YAML — no per-model
Python needed here.

### 1b. Adding a model — three cases

1. **Another diffusers checkpoint, existing method code** → *zero code*. Add `configs/methods/<model>/*.yaml`
   pointing `pipeline:` at a factory module, register `DiffusersModelAdapter("<model>")` (one line).
2. **New guidance method on an existing model** → add `pipelines/<model>/<method>.py` exposing
   `pipeline(device)` (+ the method logic in the consolidated `pipeline_*_methods.py`) and a config yaml.
   Still no new adapter.
3. **Non-diffusers backend** (API model, a different framework) → subclass `ModelAdapter` + `ModelHandle`,
   implement `load`/`generate`, register it. Nothing else in core changes.

---

## 2. Config format (unchanged from today)

Per-method YAML, `configs/methods/<model>/<name>.yaml`:

```yaml
# configs/methods/sd35/seg.yaml   (== current configs/sd35/seg_sigma10_cfg3.yaml)
pipeline: pipelines.sd35.seg          # import path; module exposes pipeline(device) factory
generation_params:
  num_inference_steps: 25
  guidance_scale: 3.0
  seg_scale: 3.0
  seg_blur_sigma: 10.0                 # floats: always write 1.0e+10 (YAML 1.1 str-parse trap)
  seg_applied_layers: ["d8"]
```

- Keys in `common.IMAGE_LEVEL_KEYS` (steps, guidance_scale, height, width, seed, …) map to
  `GenSettings`; everything else (seg_scale, method, oseg_scale, …) is forwarded verbatim as pipeline
  kwargs via `GenSettings.extra`. **No change to this contract** — existing configs work as-is.
- The campaign references configs by bare name (`seg`); core resolves `configs/methods/<model>/<name>.yaml`
  and skips a (model, name) pair when the file is absent (e.g. sd35-only methods).

---

## 3. BenchmarkAdapter

```python
# cfgbench/benchmarks/base.py
class BenchmarkAdapter(ABC):
    name: str                                   # registry key: "geneval" | "oneig" | "dpg"
    eval_needs_gpu: bool = True                 # routes eval to a GPU job vs login node
    default_samples_per_prompt: int = 1

    @abstractmethod
    def prompts(self) -> list[PromptSpec]: ...

    @abstractmethod
    def evaluate(self, items: list[GenItem], workdir: Path) -> dict[str, dict]:
        """item_id -> metrics dict. `workdir` is a private scratch dir for bridging/temp."""

    def summarize(self, metrics: dict[str, dict], prompts: list[PromptSpec]) -> dict:
        """nested summary. Default = mean of each numeric metric grouped by category, plus overall.
        Override for non-mean aggregation (GenEval macro, OneIG composite)."""
```

- **Core owns generation + output paths + sidecars.** The adapter does **not** write images; it only
  supplies prompts and consumes generated items. This is the key difference from today's `generate_*.py`
  (which also chose layout) and is what makes the output format uniform (PLAN §"Output format").
- `evaluate` typically: build a **bridge dir** in `workdir` (symlinks from the uniform layout into the
  layout the upstream scorer expects), run the scorer (subprocess), parse its output back into per-item
  metrics. Per-item granularity is achievable for all three current benches.
- Returned metrics are merged into each sample's sidecar JSON `metrics` block by core; `summarize`
  output is written to `_summary.json` at category / bench / run levels.

### 3a. Reference: GenEval adapter

- **prompts()**: read `$GENEVAL_PROMPTS` (`geneval-bench/prompts/evaluation_metadata.jsonl`), one
  `{prompt, tag, ...}` per line → `PromptSpec(id=f"{i:05d}", text=prompt, category=tag, meta=row)`.
  Categories = the 6 tags (`single_object,two_object,counting,colors,position,color_attr`).
- **evaluate(items, workdir)**: bridge → `workdir/{NNNNN}/samples/{i}.png` (symlink) +
  `{NNNNN}/metadata.jsonl` (from `prompt.meta`). Run
  `evaluate_images.py <bridge> --outfile results.jsonl --model-path $GENEVAL_MODELS --model-config $MM`.
  Parse `results.jsonl` (one line per image: `correct`, `tag`) → per-item `{"correct": 0/1, "tag": ...}`.
  GPU (Mask2Former + CLIP). `eval_needs_gpu = True`. `default_samples_per_prompt = 4`.
- **summarize()**: per-tag mean of `correct`; **Overall = macro = mean of per-tag means** (matches
  upstream `summary_scores.py`). Override (not the default mean-over-all-items).

### 3b. Reference: OneIG adapter

- **prompts()**: read `$ONEIG_CSV` (`OneIG-Benchmark/OneIG-Bench.csv`, cols `category,id,prompt_en`).
  Keep categories `{General_Object,Text_Rendering,Knowledge_Reasoning}` → `category` set to the short
  form `{object,text,reasoning}`. `PromptSpec(id=row.id, text=row.prompt_en, category=short)`.
- **evaluate(items, workdir)**: bridge → `workdir/imgs/<short>/<model>/<id>.webp` (symlinks; OneIG wants
  `<dir>/<class>/<model>/<id>`). Run the three score modules with `cwd=$ONEIG_ROOT`:
  `python -m scripts.alignment.alignment_score --class_items object`,
  `python -m scripts.text.text_score`, `python -m scripts.reasoning.reasoning_score`
  (`--mode EN --model_names <m> --image_grid 1,1 --image_dirname ...`).
  **Concurrency gotcha**: OneIG hardcodes output to `$ONEIG_ROOT/results/` + shared `tmp_*` dirs →
  concurrent eval runs corrupt each other (`PIL.UnidentifiedImageError`). Fix for autonomy: give each
  worker a **private ONEIG_ROOT view** — a dir with `scripts/`, `qwen_vl_utils.py`, model caches
  symlinked but its own writable `results/`+`tmp/` (set via env / `cwd`). Then per-eval is isolated and
  the pool can run OneIG evals in parallel.
  Parse CSVs: object = alignment score; **text = last column `text score`** (not `ED`); reasoning =
  reasoning score. Map rows→item by id. All 0–1.
- **summarize()**: mean per category (object/text/reasoning); optionally the OneIG composite. `eval_needs_gpu = True`. `default_samples_per_prompt = 1`.

### 3c. Reference: DPG adapter

- **prompts()**: read `$DPG_PROMPTS` (`ELLA/dpg_bench/prompts/<id>.txt`, long paragraphs). One category
  (`dpg`). `PromptSpec(id=stem, text=file_contents, category="dpg")`. N=90.
- **evaluate(items, workdir)**: bridge → flat `workdir/imgs/<id>.png` (symlinks; compute_dpg_bench
  listdir's a flat dir). Run
  `accelerate launch --num_processes 1 --mixed_precision fp16 compute_dpg_bench.py
   --image-root-path <imgs> --resolution 1024 --pic-num 1 --vqa-model mplug --res-path dpg_score.txt`
  with `cwd=$DPG_ROOT` (single-proc — upstream's `--multi_gpu --num_processes 8` breaks on 1×V100).
  Parse `dpg_score.txt` per-image lines `<path>, <score>, <score>` → per-item `{"dpg": col[1]}`
  (`line.rsplit(",", 2)[1]`). GPU (mPLUG). `eval_needs_gpu = True`. `default_samples_per_prompt = 1`.
- **summarize()**: mean of `dpg` × 100, report N. (Default mean works; ×100 scaling in override.)

### 3d. Adding a benchmark — checklist

1. New file `cfgbench/benchmarks/<name>.py` subclassing `BenchmarkAdapter`.
2. Implement `prompts()` (read your prompt source → `PromptSpec`s with sensible `category`).
3. Implement `evaluate(items, workdir)`: bridge uniform layout → scorer layout, run scorer, parse →
   per-item metrics. Use a **private workdir** for any tool with shared/hardcoded output dirs.
4. Override `summarize()` only if aggregation isn't a plain per-category mean.
5. Set `eval_needs_gpu`, `default_samples_per_prompt`.
6. Register one line in `benchmarks/registry.py`.

Nothing in core, worker, pool, or report needs to change.

---

## 4. Registries

```python
# cfgbench/models/registry.py
MODELS = {"sd35": DiffusersModelAdapter("sd35"),
          "cosmos2": DiffusersModelAdapter("cosmos2")}
# cfgbench/benchmarks/registry.py
BENCHMARKS = {"geneval": GenEvalAdapter(), "oneig": OneIGAdapter(), "dpg": DPGAdapter()}
```

Lookups by name from the campaign spec. External roots stay env-overridable (`$GENEVAL_ROOT`,
`$ONEIG_ROOT`, `$DPG_ROOT`, `$GENEVAL_PROMPTS`, `$ONEIG_CSV`, `$DPG_PROMPTS`) with `Path.home()`
defaults — **never hardcode `/home/aaturevich`**.

---

## 5. Metric / summary JSON written by core (recap)

Per sample sidecar `metrics` block = exactly the dict the adapter's `evaluate` returns for that item.
`_summary.json` (category / bench / run) = exactly the adapter's `summarize` output. Example
category-level GenEval:

```json
{"category": "two_object", "n": 100, "correct_mean": 0.62,
 "by_metric": {"correct": 0.62}, "model": "sd35", "config": "seg"}
```

The report builder consumes these `_summary.json` files (no re-scoring), replacing the current
`summarize_*.py` + `build_report.py` scraping of heterogeneous layouts.
