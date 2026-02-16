import argparse
import os
import re

from data_processing.util import PlainArchive, ngrams
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH, LeanMinHash
from tqdm import tqdm


# Функция для токенизации текста. Возвращает список токенов.
def re_tokenize(text):
    return re.findall(r'[а-яё-]+|[a-z-]+|\d+|\S', text, re.I)


# Функция для вычисления отпечатка записи. Возвращает словарь с отпечатком.
def calc_fingerprint(record, ngram_size: int = 1, num_perm: int = 128):
    tokens = re_tokenize(record["text"])
    if ngram_size > 1:
        tokens = {" ".join(t) for t in ngrams(tokens, ngram_size)}
    tokens = [token.encode('utf-8') for token in tokens]

    minhash = MinHash(num_perm=num_perm)
    minhash.update_batch(tokens)

    lean_minhash = LeanMinHash(minhash)
    buf = bytearray(lean_minhash.bytesize())
    lean_minhash.serialize(buf)

    return {"minhash": buf}


# Основная функция программы.
def main(
        input_path,
        output_path,
        num_perm
):
    # Загрузка датасета.
    dataset = load_dataset("rulm/jsonl_loader.py", data_files={"train": [input_path]})["train"]
    # Вычисление отпечатков для каждой записи в датасете.
    dataset = dataset.map(
        function=calc_fingerprint,
        fn_kwargs={
            "num_perm": num_perm,
            "ngram_size": 1,
        },
        num_proc=os.cpu_count(),
        desc="Fingerprinting..."
    )

    # Создание архива для записи результатов.
    archive = PlainArchive(output_path)

    # Установка порога схожести и веса ложных срабатываний для LSH.
    threshold = 0.95
    false_positive_weight = 0.05
    lsh = MinHashLSH(
        threshold=threshold,
        weights=(false_positive_weight, 1 - false_positive_weight),
        num_perm=num_perm,
    )

    # Проверка записей на дубликаты и запись уникальных в архив.
    for idx, record in tqdm(enumerate(dataset)):
        minhash = LeanMinHash.deserialize(record["minhash"])

        is_dup = False
        for other_idx in lsh.query(minhash):
            other_record = dataset[other_idx]
            other_minhash = LeanMinHash.deserialize(other_record["minhash"])
            if minhash.jaccard(other_minhash) > threshold:
                is_dup = True
                break

        # Если запись не является дубликатом или источником является "math", добавляем в архив.
        if not is_dup or record["meta"]["source"] == "math":
            record.pop("minhash")
            text = record["text"]
            meta = record["meta"]
            archive.add_data(text=text, meta=meta)

        # Вставляем запись в LSH для последующего поиска дубликатов.
        lsh.insert(idx, minhash)


# Точка входа в программу.
if __name__ == "__main__":
    # Парсинг аргументов командной строки.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num-perm", type=int, default=128)
    args = parser.parse_args()
    # Запуск основной функции программы.
    main(**vars(args))
