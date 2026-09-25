# QLoRA-дообучение Qwen2.5-3B-Instruct на задачу извлечения вакансий.
# ЗАПУСКАТЬ В GOOGLE COLAB (GPU: T4). Копируй блоки между "# %%" в ячейки.
# Если API Unsloth/TRL изменился — сверься с их актуальным Colab-примером,
# структура та же: модель -> LoRA -> данные -> SFTTrainer -> инференс -> push.

# %% [Ячейка 1] Установка (~2 мин)
# !pip install unsloth
# Затем Runtime -> Restart runtime, если Colab попросит.

# %% [Ячейка 2] Загрузка модели в 4bit
from unsloth import FastLanguageModel

MAX_LEN = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
)

# %% [Ячейка 3] LoRA-адаптер
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# %% [Ячейка 4] Данные: сначала загрузи train.jsonl и val.jsonl в Colab
# (панель слева -> Files -> Upload). Файлы из папки data/ этого проекта.
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "train.jsonl", "val": "val.jsonl"})

def to_text(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(to_text)
print(ds)
print(ds["train"][0]["text"][:600])

# %% [Ячейка 5] Обучение (~15-30 мин на T4)
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir="out",
        seed=42,
        report_to="none",
        fp16=True,
    ),
)
stats = trainer.train()
print(stats)
# Запиши себе: train_loss и eval_loss по эпохам (понадобится для model card
# и ответа работодателю). Чекпоинт выбираем по минимальному eval_loss.

# %% [Ячейка 6] Ручная проверка на свежих примерах
FastLanguageModel.for_inference(model)

SYSTEM = ("Ты извлекаешь данные о вакансиях из постов Telegram-каналов. "
          "Верни строго JSON с полями: is_vacancy (bool), title, salary, "
          "format (удаленно/офис/гибрид), employment, contact. Если пост не "
          "является вакансией — is_vacancy=false, остальные поля null. "
          "Только JSON, без пояснений.")

TESTS = [
    "🔥 Срочно нужен ассистент в онлайн-школу! Удалёнка, 3-4 часа в день, "
    "оплата 35 000 р. Писать @school_hr",
    "Заработок на криптовалюте без вложений! Пиши + в личку",
    "Ищу работу контент-менеджером, опыт 2 года, портфолио есть. ЛС открыты",
    "В типографию требуется оператор ПК, офис м. Тульская, 5/2, от 40 тыс, "
    "оформление по ТК. Резюме на hr@print.ru",
]
for t in TESTS:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": t}]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=120, temperature=0.1)
    print("ПОСТ:", t[:70])
    print("ОТВЕТ:", tokenizer.decode(out[0][inputs.shape[1]:],
                                     skip_special_tokens=True))
    print("-" * 60)

# %% [Ячейка 7] Публикация адаптера на Hugging Face
# Нужен аккаунт huggingface.co и токен с правом write:
# huggingface.co/settings/tokens
# from huggingface_hub import login
# login()  # вставишь токен
#
# REPO = "ТВОЙ_НИК/qwen2.5-3b-tg-vacancy-extractor-lora"
# model.push_to_hub(REPO, tokenizer=tokenizer)
# После пуша зайди в репозиторий на HF и вставь model_card.md как README.
