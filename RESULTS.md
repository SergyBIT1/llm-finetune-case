# Результаты прогона

Отчёт о фактическом запуске пайплайна. Подтверждает, что цикл
«данные → обучение → оценка → адаптер» рабочий и воспроизводимый.

## Локальный прогон (Apple M1 Pro, MPS)

Скрипт: `train_local.py` · база: `Qwen/Qwen2.5-0.5B-Instruct` (PEFT-LoRA, fp32).
GPU-версия (`train_unsloth.py`, 3B + QLoRA) требует CUDA и запускается в Colab.

| Параметр            | Значение                                  |
|---------------------|-------------------------------------------|
| База                | Qwen2.5-0.5B-Instruct                     |
| Метод               | LoRA r=16, alpha=32, dropout=0            |
| Trainable params    | 8.8M / 502M (1.75%)                       |
| Эпохи               | 1                                         |
| Batch × grad_accum  | 1 × 8                                     |
| max_len             | 256                                       |
| Train / Val         | 200 (подвыборка) / 60                     |
| Устройство          | MPS (Apple Silicon)                       |

### Метрики (`results/metrics.json`)

- train_loss: **2.25 → 0.61** по ходу эпохи
- финальный train_loss: **1.22**
- **eval_loss: 0.40**

### Примеры ответов адаптера (`results/samples.txt`)

```
ПОСТ: Срочно нужен ассистент в онлайн-школу! Удалёнка, оплата 35 000 р. @school_hr
→ {"is_vacancy": true, "title": "ассистент", "salary": "35 000 р.",
   "format": "удаленно", "contact": "@school_hr"}          ← вакансия распознана

ПОСТ: Заработок на криптовалюте без вложений! Пиши + в личку
→ {"is_vacancy": false, ...}                                ← спам отсеян

ПОСТ: В типографию требуется оператор ПК, офис м. Тульская, от 40 тыс.
→ {"is_vacancy": true, "title": "оператор ПК", "salary": "от 40 тыс.",
   "format": "офис"}                                        ← поля извлечены
```

## Как воспроизвести

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
python make_dataset.py     # data/ уже в репозитории
python train_local.py      # out/metrics.json, out/samples.txt, out/adapter/
```

## Статус

- ✅ Датасет генерируется детерминированно (500 примеров, дедуп, train/val)
- ✅ Обучение проходит, loss снижается, eval_loss измерен
- ✅ Адаптер сохраняется и даёт валидный JSON на контрольных примерах
- 🔜 Для сильной модели: прогон 3B + QLoRA на GPU (Colab) и публикация на HF

Ограничение: прогон на 0.5B / 1 эпоха / подвыборка — демонстрация пайплайна,
не production-качество. Подробности — в `model_card.md`.
