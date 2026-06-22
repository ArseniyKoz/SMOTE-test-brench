# SMOTE Test Bench

Benchmark стенд для сравнения oversampling-методов семейства SMOTE на бинарных imbalanced classification задачах.

Проект запускает один и тот же набор классификаторов на исходной train-выборке и на train-выборке после oversampling, считает метрики, сохраняет артефакты эксперимента и при необходимости логирует результаты в ClearML.

## Что Делает Pipeline

1. Загружает датасет из ClearML Dataset Registry.
2. Проверяет, что задача бинарная и подходит для stratified holdout/CV.
3. Делит данные на train/test.
4. Считает baseline CV через `NoSMOTE`.
5. Считает CV для выбранного oversampling-метода.
6. Обучает классификаторы на original train и resampled train.
7. Сравнивает качество на holdout.
8. Сохраняет JSON/CSV/NPZ артефакты и manifest run-а.

Поддерживаемый контракт параллельности: можно запускать независимые `ExperimentRunner` instances в разных потоках. Один общий runner между потоками не считается поддерживаемым API.

## Требования

- Python `>=3.10,<3.13`
- доступ к ClearML Server для реальных benchmark runs
- настроенный ClearML client (`clearml-init`)

Зависимости:
- runtime: [requirements.txt](./requirements.txt)
- dev/test: [requirements-dev.txt](./requirements-dev.txt)
- packaging metadata: [pyproject.toml](./pyproject.toml)

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

ClearML инициализируется один раз на окружение:

```bash
clearml-init
```

## Конфигурация

Основной experiment config:

- [configs/experiment/base_experiment.yaml](./configs/experiment/base_experiment.yaml)

Registry файлов:

- [configs/data/datasets.yaml](./configs/data/datasets.yaml) - ClearML dataset IDs и metadata
- [configs/methods.yaml](./configs/methods.yaml) - oversampling methods

Минимальная структура experiment config:

```yaml
methods:
  - AKMeans_SMOTE
  - KMeans_SMOTE

datasets:
  - Adult
  - Haberman

datasets_params:
  preprocessed: false

experiment_config:
  cv_folds: 5
  test_size: 0.2
  random_state: 42
  priority_metrics:
    - balanced_accuracy
    - f1_macro
    - f1_class_0
    - f1_class_1
    - g_mean
    - roc_auc_macro
    - precision_macro
    - recall_macro
  selected_classifiers:
    - RandomForest
    - LogisticRegression
    - kNN
  max_resampled_multiplier: 5.0
  max_plot_samples: 5000
  enable_tsne_plots: false
```

## Методы

`configs/methods.yaml` поддерживает два источника:

- `smote_variants` - классы из библиотеки `smote-variants`
- `local` - локальные реализации из `src/methods/classic/*`

Пример local method:

```yaml
AKMeans_SMOTE:
  source: local
  module: src.methods.classic.akmeans_smote
  class: AKMeansSMOTE
  params: {}
```

Все методы, включая third-party методы из `smote_variants`, проходят runner-level проверку результата `fit_resample`.

## Данные

Каждый датасет в [configs/data/datasets.yaml](./configs/data/datasets.yaml) задает:

- `data_id` - raw ClearML Dataset ID
- `prep_data_id` - optional preprocessed ClearML Dataset ID
- `source`, `license`, `sensitive_attributes`, `intended_use`, `limitations` - metadata для аудита датасетов
- `preprocessing_provenance` - optional доказательство, что preprocessing был сделан без leakage

По умолчанию используется raw `data_id`.

`prep_data_id` считается небезопасным, если в config включено `datasets_params.preprocessed: true`, но у датасета нет:

```yaml
preprocessing_provenance:
  train_only: true
```

Обойти проверку можно только явно:

```yaml
datasets_params:
  preprocessed: true
  allow_unsafe_preprocessed: true
```

Это не делает данные безопасными автоматически. Флаг только фиксирует осознанное решение использовать заранее подготовленный датасет.

CSV внутри ClearML Dataset должен:

- называться `<dataset_name>.csv`
- хранить target в последнем столбце
- содержать ровно две target-категории после загрузки

## Запуск

Dry validation без запуска экспериментов:

```bash
python main.py --dry-validate
```

Запуск default config:

```bash
python main.py
```

Запуск с другим config относительно `configs/`:

```bash
python main.py --config experiment/experiment_test.yaml
```

CLI overrides:

```bash
python main.py \
  --datasets Adult,Haberman \
  --methods SMOTE,ADASYN \
  --classifiers RandomForest,LogisticRegression \
  --cv-folds 3
```

Отключить построение plot artifacts:

```bash
python main.py --no-plots
```

## Safety Checks

Runner останавливает эксперимент с `ValueError`, если `fit_resample` вернул невалидный результат:

- `X` не двумерный
- `y` не одномерный
- длины `X/y` не совпадают
- изменилось число features
- появились `NaN` или `inf`
- размер resampled output превысил `max_resampled_multiplier`

Plot safety:

- `enable_tsne_plots` по умолчанию `false`
- `max_plot_samples` по умолчанию `5000`
- если scatter plots включены и данных больше лимита, run падает явно, а не пытается строить тяжелый plot

## Артефакты

Для каждого run создается директория:

```text
<results_dir>/<run_id>/
```

`run_id` включает timestamp, microseconds, git sha и короткий UUID, чтобы независимые runner instances не конфликтовали.

Сохраняемые файлы:

- `manifest.json` - список generated files и metadata run-а
- `experiment_results_<dataset>_<method>.json` - metrics, metadata, ссылки на prediction arrays
- `predictions_<dataset>_<method>.npz` - `y_pred` и `y_pred_proba`
- `results_summary_<dataset>_<method>.csv` - compact summary по priority metrics
- `summary.csv` / `summary.json` - aggregate results после batch run

JSON artifacts не содержат raw numpy arrays. Большие prediction arrays лежат в `.npz`, а JSON хранит только имя artifact и key.

## Метрики

Default priority metrics ориентированы на imbalanced binary classification:

- `balanced_accuracy`
- `f1_macro`
- `f1_class_0`
- `f1_class_1`
- `g_mean`
- `roc_auc_macro`
- `precision_macro`
- `recall_macro`

Weighted metrics остаются доступными, но не являются default, потому что могут скрывать плохое качество на minority class.

## Тесты

Локальная проверка:

```bash
python -m compileall -q main.py configs experiments src tests
python -m pytest -q
```

Если system Python управляется дистрибутивом и запрещает `pip install`, используйте `.venv`.

На Python 3.14 полный install может упереться в transitive dependency `tensorflow` из `smote-variants`. Для этого проекта целевой диапазон Python ограничен `>=3.10,<3.13`.

## Структура

```text
SMOTE-test-bench/
├─ configs/
│  ├─ data/
│  │  └─ datasets.yaml
│  ├─ experiment/
│  │  ├─ base_experiment.yaml
│  │  └─ experiment_test.yaml
│  ├─ methods.yaml
│  ├─ schemas.py
│  └─ validation.py
├─ data/
│  └─ dataset_to_clearML.py
├─ experiments/
│  ├─ experiment_runner.py
│  └─ results_printer.py
├─ src/
│  ├─ evaluation/
│  │  └─ basic_evaluator.py
│  ├─ methods/
│  │  ├─ base.py
│  │  ├─ registry.py
│  │  └─ classic/
│  └─ utils/
│     ├─ data_loader.py
│     ├─ preprocessing.py
│     └─ visualise.py
├─ tests/
├─ main.py
├─ requirements.txt
├─ requirements-dev.txt
└─ pyproject.toml
```
