import functools
import random
from typing import Tuple
import duckdb


class SentenceDB:
    def __init__(self, db_file: str, read_only: bool) -> None:
        self.db_file = db_file
        self.db = duckdb.connect(db_file, read_only=read_only)

    def __del__(self):
        self.db.close()

    def from_txt(self, file: str, rewrite: bool) -> None:
        """
        txt only has one column: sentence
        """
        if rewrite:
            self.db.execute("DROP TABLE IF EXISTS sentences")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sentences (id INTEGER PRIMARY KEY, sentence VARCHAR)"
        )

        with open(file, "r") as f:
            # 用于批量收集数据的列表
            batch = []
            batch_size = 1000  # 可以根据需要调整批量大小
            next_id = 1
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    batch.append((next_id, line))
                    next_id += 1
                    if len(batch) >= batch_size:
                        self.db.execute("BEGIN;")
                        self.db.executemany(
                            "INSERT INTO sentences (id, sentence) VALUES (?, ?)", batch
                        )
                        self.db.execute("COMMIT;")
                        batch = []
            if batch:
                self.db.execute("BEGIN;")
                self.db.executemany(
                    "INSERT INTO sentences (id, sentence) VALUES (?, ?)", batch
                )
                self.db.execute("COMMIT;")

        self.show_info()

    def show_info(self):
        print(f"Database info: {self.db_file}")
        table_name = "sentences"
        print("Columns and data types:")
        result = self.db.execute(f"DESCRIBE {table_name}").fetchall()
        for row in result:
            print(row)

        print(f"Number of rows: {self.count}")

    @functools.cached_property
    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        return self.db.execute(f"SELECT * FROM sentences WHERE id={idx}").fetchone()

    def get(self):
        rnd_id = random.randint(0, self.count)
        return self[rnd_id]


if __name__ == "__main__":
    langs = ['en', 'latin', 'ko', 'ja', 'zh_cn', 'zh_tw']
    dst_dir = 'assets/merged'
    for lang in langs:
        db_file = f"{dst_dir}/{lang}.duckdb"    
        txt_file = f"{dst_dir}/{lang}.txt"
        db = SentenceDB(db_file, read_only=False)
        db.from_txt(txt_file, rewrite=True)
        del db
