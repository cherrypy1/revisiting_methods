# Запуск бенчмарков SD3.5 guidance

Инструкция описывает, как прогнать все реализованные методы guidance для
`Stable Diffusion 3.5 medium` на трёх бенчмарках: GenEval,
OneIG-Benchmark (три категории) и DPG-Bench. Все пути/имена — для удалённого
сервера `hse-hpc`; измените переменные окружения под себя, если нужно.

## Подготовка кода

1. Синхронизируйте локальный репозиторий с `~/geneval` на сервере
   (`rsync -av --exclude outputs --exclude models /home/tiom4eg/prog/geneval/
   hse-hpc:~/geneval/`).
2. Убедитесь, что кастомные пайплайны лежат в
   `~/diffusers/src/diffusers/pipelines/stable_diffusion_3/`. Ожидаемые файлы:
   `pipeline_stable_diffusion_3_{seg,oseg,apg,cfgpp,cfg0s,tcfg,sag}.py`
   (PAG берётся из `diffusers.pipelines.pag`).
3. Для каждого метода конфиг лежит в
   `~/geneval/generation/configs/sd35/<method>.py` и задаёт класс пайплайна и
   гипер‑параметры guidance.

## Выделение ресурсов

```bash
salloc -p normal --gres=gpu:1 -c 4 -t 12:00:00
ssh <job-id-node>
module purge && module load gnu14/14.1
source ~/.venv/bin/activate
cd ~/geneval
```

Либо отправьте batch‑джобу:

```bash
sbatch --export=ALL,METHOD=cfg,BENCH=geneval scripts/slurm_job.sh
sbatch --export=ALL,BENCH=geneval scripts/slurm_job.sh all   # все методы
```

## Запуск одного метода на одном бенчмарке

```bash
bash scripts/bench.sh <method> <geneval|oneig|dpg>
```

Пример:

```bash
bash scripts/bench.sh seg_sigma10_cfg3 geneval
bash scripts/bench.sh apg              oneig
bash scripts/bench.sh cfgpp            dpg
```

Доступные методы (имена файлов из `generation/configs/sd35/` без `.py`):
`no_cfg`, `cfg`, `seg_sigma10_cfg3`, `seg_sigmainf_cfg3`, `oseg`, `pag`,
`sag`, `apg`, `cfgpp`, `cfg0s`, `tcfg`.

## Переменные окружения (при необходимости переопределить)

| переменная | дефолт | для чего |
|---|---|---|
| `PROJECT_ROOT` | `/home/aaturevich/cfg_evaluation` | корень репозитория (наш код) |
| `VENV` | `/home/aaturevich/.venv` | venv с diffusers + mmdet |
| `RUN_TAG` | `$(date +%d%m%Y)` | тег прогона (идёт в имена папок) |
| `OUT_ROOT` | `$PROJECT_ROOT/outputs` | куда писать результаты |
| `GENEVAL_ROOT` | `~/geneval-bench` | внешний репо GenEval-бенчмарка (детектор/mmdet/промпты/скорер) |
| `GENEVAL_PROMPTS` | `$GENEVAL_ROOT/prompts/evaluation_metadata.jsonl` | промпты GenEval |
| `GENEVAL_MODELS` | `$GENEVAL_ROOT/models` | веса Mask2Former |
| `GENEVAL_MM_CONFIG` | `$GENEVAL_ROOT/mmdetection/configs/mask2former/…` | py‑конфиг mmdet |
| `ONEIG_CSV` | `~/OneIG-Benchmark/OneIG-Bench.csv` | каталог промптов OneIG |
| `ONEIG_ROOT` | `~/OneIG-Benchmark` | репо OneIG |
| `DPG_PROMPTS` | `~/ELLA/dpg_bench/prompts` | промпты DPG |
| `DPG_ROOT` | `~/ELLA` | репо ELLA |

## GenEval

`scripts/bench.sh <method> geneval` делает три шага:

1. `generation/diffusers_generate.py` — рендерит 4 сэмпла на промпт в
   `outputs/geneval/<method>_<RUN_TAG>/`.
2. `evaluation/evaluate_images.py` — Mask2Former + CLIP‑цветовой классификатор,
   пишет `results.jsonl`.
3. `evaluation/summary_scores.py` — сводная таблица в `summary.txt`.

## OneIG-Benchmark (три категории)

`scripts/bench.sh <method> oneig` генерирует 2x2 сетки и раскладывает их по
`outputs/oneig/<RUN_TAG>/{object,text,reasoning}/<method>/<id>.webp`.
Оценка потом делается родным OneIG‑пайплайном:

```bash
cd ~/OneIG-Benchmark
# отредактируйте run_overall.sh: IMAGE_DIR=..../outputs/oneig/<RUN_TAG>
# и MODEL_NAMES=("<method>")
bash run_overall.sh
```

Категории проекта — `General_Object`, `Text_Rendering`, `Knowledge_Reasoning`
(остальные пропускаются, чтобы сэкономить диск).

## DPG-Bench

`scripts/bench.sh <method> dpg` генерирует 4 сэмпла на промпт, выкладывает
их горизонтальным тайлом в `outputs/dpg/<method>_<RUN_TAG>/images/<id>.png`.
Затем запустите оценку:

```bash
cd ~/ELLA
bash dpg_bench/dist_eval.sh /home/aaturevich/geneval/outputs/dpg/<method>_<RUN_TAG> 1024
```

mPLUG VQA‑модель подтянется автоматически (первая итерация будет долгой).

## Частые грабли

- **Нет места на диске:** чистите `~/.cache/huggingface` или большие
  старые папки в `outputs/` перед запуском массового прогона.
- **`ImportError: is_bitsandbytes_available`:** конфиги уже содержат
  обходной monkey‑patch через `_patch.py`. Если меняете конфиг — сохраните вызов
  `patch_diffusers_no_bnb()` в начале.
- **Старая glibc:** всегда `module purge && module load gnu14/14.1` перед
  активацией venv.
- **OOM:** V100 32GB хватает на SD3.5‑medium при 1024x1024 и batch=1; при
  методах с тремя forward‑пассами (OSEG+CFG) возможны просадки — при нужде
  уменьшите `H/W` до 768.

## Обновление `PROGRESS.md`

По завершении каждого крупного этапа (прогон всех методов на одном
бенчмарке, обновление оценок) кратко фиксируйте итог в `PROGRESS.md`.
