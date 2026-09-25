# llm-finetune-case: QLoRA-дообучение под извлечение вакансий

Практический кейс fine-tuning: Qwen2.5-3B-Instruct учится извлекать данные
о вакансиях из постов Telegram-каналов (строгий JSON). Задача взята из
собственного инструмента (tg-vacancy-collector, этап 2 — классификация).

> **Статус.** Полный **воспроизводимый пайплайн** (данные → обучение → оценка
> → адаптер). Обучение уже прогнано локально на Apple M1 Pro (Qwen2.5-0.5B,
> PEFT-LoRA): **eval_loss 0.40** — отчёт с метриками и примерами в
> [`RESULTS.md`](RESULTS.md) и `results/`. Целевой сценарий (3B, QLoRA на GPU)
> — в `train_unsloth.py` для Colab. Датасет синтетический: цель — показать сам
> процесс, а не production-качество.

## Состав

- `make_dataset.py` — генератор датасета (500 примеров ChatML, синтетика:
  вакансии в 3 стилях + трудные негативы: резюме, курсы, продажи, спам;
  дедупликация, train/val сплит). Без внешних зависимостей.
- `data/train.jsonl`, `data/val.jsonl` — готовый датасет
- `train_unsloth.py` — целевой скрипт обучения для Google Colab (3B, QLoRA,
  Unsloth; требует GPU/CUDA)
- `train_local.py` — прогон без GPU (Apple Silicon MPS / CPU): обычный
  PEFT-LoRA на уменьшенной модели, чтобы воспроизвести цикл там, где CUDA нет
- `model_card.md` — карточка модели с реальными метриками локального прогона

## Быстрый локальный прогон (без GPU)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
python make_dataset.py            # data/ (уже готова в репозитории)
python train_local.py             # Qwen2.5-0.5B, LoRA на MPS/CPU
# результат: out/metrics.json, out/samples.txt, out/adapter/
# метрики зафиксированного прогона лежат в results/
```

Параметры настраиваются через переменные окружения: `BASE_MODEL`, `EPOCHS`,
`MAX_LEN`, `BATCH`, `GRAD_ACCUM`, `MAX_TRAIN`.

## Целевой сценарий на GPU (Colab, 3B QLoRA)

1. colab.research.google.com → Runtime → Change runtime type → T4 GPU
2. Скопируй блоки из `train_unsloth.py` по ячейкам, загрузи
   `train.jsonl`/`val.jsonl` через панель Files
3. Прогони ячейки 1–6 (~30–40 минут). Запиши train/val loss по эпохам
4. huggingface.co → token (write) → ячейка 7 запушит адаптер в твой репозиторий
5. Вставь `model_card.md` как README модели на HF

## Зачем

- Показать владение полным циклом fine-tuning: данные → обучение → оценка →
  публикация (а не только вызов готового API)
- Проект портфолио, демонстрирующий воспроизводимый пайплайн QLoRA
- Задел для этапа 2 tg-vacancy-collector (локальный классификатор вместо
  платного API)
