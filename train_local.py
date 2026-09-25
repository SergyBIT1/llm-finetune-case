#!/usr/bin/env python3
"""Локальный LoRA-прогон без GPU (Apple Silicon MPS / CPU).

Зачем отдельный скрипт: `train_unsloth.py` рассчитан на CUDA (Unsloth +
bitsandbytes 4bit) и запускается в Google Colab. На машине без NVIDIA он не
работает. Этот вариант обучает тот же датасет обычным PEFT-LoRA на MPS/CPU,
чтобы показать полный цикл (данные -> обучение -> оценка -> адаптер) там, где
GPU нет. Модель по умолчанию меньше (0.5B) — чтобы прогон реально завершался.

Запуск:
    python train_local.py                    # Qwen2.5-0.5B-Instruct, MPS/CPU
    BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct python train_local.py
    EPOCHS=2 python train_local.py

Выход:
    out/adapter/            — обученный LoRA-адаптер (можно грузить через PEFT)
    out/metrics.json        — train/val loss по эпохам (для model_card)
    out/samples.txt         — ответы модели на контрольных примерах
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "out"

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
EPOCHS = float(os.getenv("EPOCHS", "3"))
MAX_LEN = int(os.getenv("MAX_LEN", "512"))
LR = float(os.getenv("LR", "2e-4"))
BATCH = int(os.getenv("BATCH", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
# Ограничение размера train-выборки (0 = весь). На CPU/MPS без дискретного GPU
# полный прогон 3B/большого сплита может занимать часы — этим держим прогон
# в разумных рамках, не меняя сам пайплайн.
MAX_TRAIN = int(os.getenv("MAX_TRAIN", "0"))


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    device = pick_device()
    print(f"Устройство: {device} | база: {BASE_MODEL} | эпох: {EPOCHS}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device in ("mps", "cpu") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(
        "json",
        data_files={"train": str(DATA / "train.jsonl"),
                    "val": str(DATA / "val.jsonl")},
    )

    def tokenize(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        out = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
        return out

    tokenized = ds.map(tokenize, remove_columns=ds["train"].column_names)
    if MAX_TRAIN > 0:
        tokenized["train"] = tokenized["train"].select(
            range(min(MAX_TRAIN, len(tokenized["train"])))
        )
        print(f"train ограничен до {len(tokenized['train'])} примеров")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=collator,
    )

    train_result = trainer.train()

    # Метрики по эпохам из истории логов
    epoch_metrics = []
    for rec in trainer.state.log_history:
        row = {}
        if "loss" in rec and "epoch" in rec:
            row = {"epoch": round(rec["epoch"], 2), "train_loss": round(rec["loss"], 4)}
        if "eval_loss" in rec and "epoch" in rec:
            row = {"epoch": round(rec["epoch"], 2), "eval_loss": round(rec["eval_loss"], 4)}
        if row:
            epoch_metrics.append(row)

    OUT.mkdir(exist_ok=True)
    (OUT / "adapter").mkdir(exist_ok=True)
    model.save_pretrained(str(OUT / "adapter"))
    tokenizer.save_pretrained(str(OUT / "adapter"))

    metrics = {
        "base_model": BASE_MODEL,
        "device": device,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "max_len": MAX_LEN,
        "final_train_loss": round(train_result.training_loss, 4),
        "history": epoch_metrics,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Метрики ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Быстрая проверка на контрольных примерах
    model.config.use_cache = True
    model.eval()
    SYSTEM = ("Ты извлекаешь данные о вакансиях из постов Telegram-каналов. "
              "Верни строго JSON с полями: is_vacancy (bool), title, salary, "
              "format (удаленно/офис/гибрид), employment, contact. Если пост не "
              "является вакансией — is_vacancy=false, остальные поля null. "
              "Только JSON, без пояснений.")
    tests = [
        "🔥 Срочно нужен ассистент в онлайн-школу! Удалёнка, 3-4 часа в день, "
        "оплата 35 000 р. Писать @school_hr",
        "Заработок на криптовалюте без вложений! Пиши + в личку",
        "В типографию требуется оператор ПК, офис м. Тульская, 5/2, от 40 тыс.",
    ]
    lines = []
    for t in tests:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": t}]
        # В свежих transformers apply_chat_template с return_dict=True отдаёт
        # BatchEncoding; берём из него input_ids (голый тензор в .generate).
        enc = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(device)
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        ans = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
        lines.append(f"ПОСТ: {t[:70]}\nОТВЕТ: {ans}\n{'-'*60}")
    (OUT / "samples.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Примеры ===\n" + "\n".join(lines))
    print(f"\nГотово. Адаптер: {OUT/'adapter'}  Метрики: {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()
