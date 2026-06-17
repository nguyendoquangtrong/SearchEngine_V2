import json
import re
from collections import defaultdict


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9à-ỹăâêôơưđ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


class MovieMetadataIndex:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.movies = json.load(f)

        self.index = {
            "title": defaultdict(set),
            "origin_name": defaultdict(set),
            "actor": defaultdict(set),
            "director": defaultdict(set),
            "category": defaultdict(set),
            "country": defaultdict(set),
            "year": defaultdict(set),
            "quality": defaultdict(set),
            "lang": defaultdict(set),
            "type": defaultdict(set),
            "status": defaultdict(set),
        }

        self.movie_by_name = {}
        self._build()

    def _add(self, field, value, movie_name):
        if value is None:
            return

        values = value if isinstance(value, list) else [value]

        for v in values:
            v_norm = normalize_text(v)

            # Bỏ dữ liệu rỗng hoặc quá ngắn
            if not v_norm:
                continue

            if v_norm in ["đang cập nhật", ""]:
                continue

            # Bỏ actor/director/category/country quá ngắn như "h", "j"
            if field in ["actor", "director", "category", "country"] and len(v_norm) < 3:
                continue

            # Bỏ title/origin_name quá ngắn
            if field in ["title", "origin_name"] and len(v_norm) < 2:
                continue

            self.index[field][v_norm].add(movie_name)

    def _build(self):
        for movie in self.movies:
            movie_name = movie.get("origin_name") or movie.get("title")
            if not movie_name:
                continue

            self.movie_by_name[movie_name] = movie

            for field in self.index.keys():
                self._add(field, movie.get(field), movie_name)

    def get_movie(self, movie_name):
        return self.movie_by_name.get(movie_name)