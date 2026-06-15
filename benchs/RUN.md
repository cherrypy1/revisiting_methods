# Benchmark Workflow

Основной запуск делаем интерактивно на GPU-ноде. Установленный `diffusers` не
патчим: методы лежат в `pipelines/<model>/`, параметры в
`configs/<model>/<method>.yaml`.

## 0. Синхронизация

На локальной машине из корня репозитория:

```bash
rsync -az --delete \
  --exclude outputs \
  --exclude prompts \
  --exclude geneval-bench \
  --exclude OneIG-Benchmark \
  --exclude ELLA \
  --exclude .venv_bench \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  benchs/ hse-hpc:~/cfg_evaluation/
rsync -az reqs.txt PROGRESS.md hse-hpc:~/cfg_evaluation/
```

На кластере рабочая папка ожидается такой:

```text
~/cfg_evaluation/
  configs/
  generation/
  pipelines/
  scripts/
  RUN.md
  outputs/
  prompts/
  geneval-bench/
  OneIG-Benchmark/
  ELLA/
  .venv_bench/
```

## 1. Окружение

```bash
ssh hse-hpc
salloc -p normal --gres=gpu:1 -c 4 -t 12:00:00
ssh <job-id-node>
module purge && module load gnu14/14.1
source ~/.venv/bin/activate
cd ~/cfg_evaluation
```

Зависимости:

```bash
pip install -U pip setuptools wheel
pip install -r reqs.txt
bash scripts/setup/install_benchmark_eval.sh
```

Если нужен только один evaluator, можно запускать точечно:

```bash
bash scripts/setup/install_geneval_eval.sh
bash scripts/setup/install_dpg_eval.sh
bash scripts/setup/install_oneig_eval.sh
```

Все benchmark evaluator deps ставятся в отдельный `~/cfg_evaluation/.venv_bench`.
Основной `.venv` используется только для генерации. Это изолирует старые
`requests` от `openmim/openxlab` и `transformers==4.50.0` от OneIG.

В `reqs.txt` уже добавлен PyTorch wheel index для `torch==2.5.1+cu121`.
В файле не должно быть локального editable `diffusers`, локального `/home/.../mmcv`
или локального wheel для `bitsandbytes`. `lerobot` выключен, потому что он не
нужен для этих benchmark-ов и конфликтует с `huggingface_hub==0.36.2`.
`llm2vec` тоже выключен, потому что требует старый `transformers<=4.44.2`.
`openxlab/opendatalab/modelscope/openmim` выключены в основном окружении:
они не нужны для генерации и конфликтуют с современным `requests`. Для evaluator
они ставятся отдельно в `.venv_bench`.
Если `bitsandbytes` понадобится, ставь его отдельно.

## 2. Prompt Sets

Один раз скачать benchmark repos и подготовить три фиксированных набора:

```bash
bash scripts/prepare_benchmarks.sh
```

Будут созданы:

```text
prompts/smoke_test/
  geneval.jsonl      # 1 prompt per tag
  oneig.csv          # 1 prompt per category
  dpg/
prompts/evaluation/
  geneval.jsonl      # 5 prompts per tag, 30 total
  oneig.csv          # 6 prompts per category, about 30 total
  dpg/               # about 30 prompt ids
prompts/full_test/
  geneval.jsonl      # symlink to full GenEval metadata
  oneig.csv          # symlink to full OneIG CSV
  dpg/               # symlinks to full DPG CSV/prompts
```

## 3. Запуск

Формат у всех режимов одинаковый:

```bash
bash scripts/smoke_test.sh <model> <bench...> <method...> [-- extra args]
bash scripts/evaluation.sh <model> <bench...> <method...> [-- extra args]
bash scripts/full_test.sh <model> <bench...> <method...> [-- extra args]
```

`bench...`: `geneval`, `oneig`, `dpg` или `all`. Всё после списка бенчей до
`--` считается методами. Во всех режимах генерируется 4 картинки на промпт:
GenEval `--n_samples 4`, OneIG `2x2`, DPG `--pic-num 4`.

Примеры:

```bash
bash scripts/smoke_test.sh flux2_klein_base all cfg -- --skip-eval
bash scripts/evaluation.sh flux2_klein_base geneval oneig cfg cfgpp pag sag
bash scripts/evaluation.sh flux2_klein_base dpg seg_sigma10 seg_sigmainf oseg
bash scripts/full_test.sh sd35 all cfg cfgpp pag sag
```

Дополнительные аргументы после `--` передаются в каждый запуск `bench.py`:

```bash
bash scripts/evaluation.sh flux2_klein_base geneval cfg -- --skip-eval
bash scripts/full_test.sh flux2_klein_base dpg cfg -- --resolution 1024
```

Результаты:

```text
outputs/smoke_test/<model>/
outputs/evaluation/<model>/
outputs/full_test/<model>/
```

Итоговые числа руками переносим в `PROGRESS.md`.

## 4. Batch Если Нужно

Основной режим - `salloc`, но `slurm_job.sh` поддерживает тот же интерфейс:

```bash
sbatch scripts/slurm_job.sh smoke_test flux2_klein_base all cfg
sbatch scripts/slurm_job.sh evaluation flux2_klein_base geneval dpg cfg pag sag
sbatch scripts/slurm_job.sh full_test sd35 all cfg
```

Через env:

```bash
sbatch --export=ALL,MODE=evaluation,MODEL=flux2_klein_base,BENCHES="geneval dpg",METHODS="cfg pag sag" scripts/slurm_job.sh
```

## 5. Проверки После Правок

```bash
bash -n scripts/smoke_test.sh
bash -n scripts/evaluation.sh
bash -n scripts/full_test.sh
bash -n scripts/slurm_job.sh
bash -n scripts/prepare_benchmarks.sh
python -m py_compile scripts/evaluate.py scripts/bench.py scripts/run_methods.py scripts/common.py
python scripts/evaluate.py --mode smoke_test --model flux2_klein_base --benches geneval dpg --methods cfg pag --dry-run
python scripts/evaluate.py --mode evaluation --model sd35 --benches all --methods cfg --dry-run
```

После проверок удалить `__pycache__`.
