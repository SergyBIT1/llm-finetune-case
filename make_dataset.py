#!/usr/bin/env python3
"""Генератор синтетического датасета: извлечение вакансий из постов Telegram.

Задача модели: по тексту поста вернуть строгий JSON:
  is_vacancy, title, salary, format, employment, contact
Негативные примеры: реклама курсов, продажи, резюме соискателей, новости,
спам — модель должна отличать их от вакансий.

Запуск:  python3 make_dataset.py
Выход:   data/train.jsonl, data/val.jsonl (ChatML messages для SFT)

Только стандартная библиотека. Детерминирован (seed).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).resolve().parent / "data"

SYSTEM = (
    "Ты извлекаешь данные о вакансиях из постов Telegram-каналов. "
    "Верни строго JSON с полями: is_vacancy (bool), title, salary, format "
    "(удаленно/офис/гибрид), employment, contact. Если пост не является "
    "вакансией — is_vacancy=false, остальные поля null. Только JSON, без "
    "пояснений."
)

TITLES = [
    "оператор ПК", "оператор ввода данных", "контент-менеджер",
    "контент-менеджер маркетплейсов", "ассистент руководителя",
    "менеджер маркетплейсов", "специалист по документообороту",
    "Python-разработчик", "разработчик Telegram-ботов", "модератор контента",
    "специалист поддержки", "аналитик данных", "помощник рекрутера",
    "специалист по автоматизации", "оператор 1С", "копирайтер",
    "SMM-менеджер", "дизайнер карточек", "тестировщик", "бухгалтер по первичке",
]
COMPANIES = [
    "агентство «Вектор»", "ООО «Прайм Трейд»", "студия WebCraft",
    "интернет-магазин «Дом и Сад»", "ИП Смирнова", "компания «Логистик Про»",
    "онлайн-школа SkyLearn", "бренд одежды NORD", "сервис «Финансист»",
    "клиника «Здоровье+»", None, None,  # часто компании нет в посте
]
SALARIES = [
    ("30 000 ₽", "30 000 ₽"), ("от 45 000 руб.", "от 45 000 руб."),
    ("50-70 тыс. руб.", "50-70 тыс. руб."), ("до 120к", "до 120к"),
    ("40к на руки", "40к на руки"), ("сдельная", "сдельная"),
    ("от 500 ₽/час", "от 500 ₽/час"), ("по договоренности", "по договоренности"),
    (None, None),
]
FORMATS = [("удаленно", "удаленно"), ("удалённая работа", "удаленно"),
           ("офис (Москва)", "офис"), ("гибрид", "гибрид"), (None, None)]
EMPLOYMENTS = [("полная занятость", "полная"), ("частичная занятость", "частичная"),
               ("подработка", "подработка"), ("проектная работа", "проектная"),
               (None, None)]
CONTACTS = ["@hr_maria", "@job_admin", "@recruit_anna", "hr@company.ru",
            "писать в лс", None]
REQ_SNIPPETS = [
    "Требования: внимательность, ответственность, уверенный ПК.",
    "Нужен опыт работы с Excel и Google Таблицами.",
    "Опыт не требуется, всему научим.",
    "Требуется грамотная письменная речь.",
    "Знание Python и API будет преимуществом.",
    "Опыт работы с WB/Ozon от полугода.",
]
EMOJIS = ["🔥", "❗️", "💼", "📌", "🚀", ""]


def vacancy_post() -> tuple[str, dict]:
    title = random.choice(TITLES)
    company = random.choice(COMPANIES)
    sal_text, sal_label = random.choice(SALARIES)
    fmt_text, fmt_label = random.choice(FORMATS)
    emp_text, emp_label = random.choice(EMPLOYMENTS)
    contact = random.choice(CONTACTS)
    emo = random.choice(EMOJIS)

    style = random.randint(1, 3)
    lines: list[str] = []
    if style == 1:  # структурированный пост
        head = f"{emo} ВАКАНСИЯ: {title.upper()}" if random.random() < 0.5 else f"{emo} Ищем: {title}"
        lines.append(head)
        if company:
            lines.append(f"Компания: {company}")
        if fmt_text:
            lines.append(f"Формат: {fmt_text}")
        if emp_text:
            lines.append(f"Занятость: {emp_text}")
        if sal_text:
            lines.append(f"Оплата: {sal_text}")
        lines.append(random.choice(REQ_SNIPPETS))
        if contact:
            lines.append(f"Контакт: {contact}")
    elif style == 2:  # сплошной абзац
        parts = [f"{company or 'В команду'} требуется {title}."]
        if fmt_text:
            parts.append(f"Работа: {fmt_text}.")
        if emp_text:
            parts.append(f"{emp_text.capitalize()}.")
        if sal_text:
            parts.append(f"Оплата {sal_text}.")
        parts.append(random.choice(REQ_SNIPPETS))
        if contact:
            parts.append(f"Отклик: {contact}")
        lines.append(" ".join(parts))
    else:  # короткий пост
        s = f"{emo} Нужен {title}"
        if fmt_text:
            s += f", {fmt_text}"
        if sal_text:
            s += f", {sal_text}"
        s += "."
        if contact:
            s += f" Пишите {contact}"
        lines.append(s)

    label = {
        "is_vacancy": True,
        "title": title,
        "salary": sal_label,
        "format": fmt_label,
        "employment": emp_label,
        "contact": contact,
    }
    return "\n".join(lines), label


NEGATIVE_TEMPLATES = [
    # реклама курсов
    "🔥 Хочешь зарабатывать на маркетплейсах от 150к? Запишись на бесплатный вебинар! Осталось {n} мест. Жми на ссылку в шапке!",
    "Научим профессии {prof} с нуля за 2 месяца. Скидка 40% до конца недели! Диплом и трудоустройство.",
    # продажи
    "Продам ноутбук Lenovo, состояние отличное, {n}0 000 руб. Самовывоз, торг.",
    "Отдам даром котёнка, 2 месяца, к лотку приучен. Москва.",
    # резюме соискателя (сложный негатив!)
    "Ищу работу: {prof}, опыт 3 года, рассматриваю удалёнку. Портфолио по запросу, пишите в лс.",
    "Резюме: {prof}. Опыт с Excel и 1С. Ищу подработку на вечер. Контакт: @user{n}",
    # новости/статьи
    "Рынок труда в {year}: аналитики отмечают рост спроса на специалистов по ИИ на {n}%.",
    "Как я автоматизировал свою рутину: тред. Сохраняй, чтобы не потерять.",
    # спам/крипта
    "Пассивный доход от {n}00$ в день! Проверенная схема, пиши «+» в личку.",
    "Раздача призов от нашего канала! Подпишись и выиграй iPhone.",
    # обсуждения
    "Коллеги, подскажите, какой сервис рассылок сейчас лучший? Поделитесь опытом.",
    "Опрос: сколько часов в день вы реально работаете на удалёнке?",
]


def negative_post() -> tuple[str, dict]:
    t = random.choice(NEGATIVE_TEMPLATES)
    text = t.format(prof=random.choice(TITLES), n=random.randint(2, 9),
                    year=random.choice([2024, 2025, 2026]))
    label = {"is_vacancy": False, "title": None, "salary": None,
             "format": None, "employment": None, "contact": None}
    return text, label


def to_example(text: str, label: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
    ]}


def main(n_pos: int = 260, n_neg: int = 240, val_ratio: float = 0.12) -> None:
    seen: set[str] = set()
    examples: list[dict] = []
    attempts = 0
    while len(examples) < n_pos and attempts < n_pos * 20:
        attempts += 1
        text, label = vacancy_post()
        if text in seen:
            continue
        seen.add(text)
        examples.append(to_example(text, label))
    n_pos_real = len(examples)
    attempts = 0
    negs = 0
    while negs < n_neg and attempts < n_neg * 20:
        attempts += 1
        text, label = negative_post()
        if text in seen:
            continue
        seen.add(text)
        examples.append(to_example(text, label))
        negs += 1

    random.shuffle(examples)
    n_val = int(len(examples) * val_ratio)
    val, train = examples[:n_val], examples[n_val:]

    OUT.mkdir(exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (OUT / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Всего примеров: {len(examples)} (вакансий {n_pos_real}, негативов {negs})")
    print(f"train: {len(train)}  val: {len(val)}  (дубликаты отсеяны: сет {len(seen)})")
    print("\n--- пример (вакансия) ---")
    sample = next(e for e in train if '"is_vacancy": true' in e["messages"][2]["content"])
    print(sample["messages"][1]["content"])
    print("->", sample["messages"][2]["content"])
    print("\n--- пример (негатив) ---")
    sample = next(e for e in train if '"is_vacancy": false' in e["messages"][2]["content"])
    print(sample["messages"][1]["content"])
    print("->", sample["messages"][2]["content"])


if __name__ == "__main__":
    main()
