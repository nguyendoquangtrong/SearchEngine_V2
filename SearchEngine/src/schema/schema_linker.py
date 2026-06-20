import re
from rapidfuzz import process, fuzz


FIELD_WEIGHTS = {
    "title": 4.0,
    "origin_name": 4.0,
    "actor": 2.5,
    "director": 2.5,
    "category": 1.4,
    "country": 1.0,
    "year": 1.2,
    "quality": 0.5,
    "lang": 0.5,
    "type": 0.5,
    "status": 0.4,
}


BAD_ENTITY_VALUES = {
    "",
    "h",
    "j",
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "v",
    "mr",
    "mrs",
    "ms",
    "dr",
    "alf",
}


class MovieJSONSchemaLinker:
    def __init__(self, metadata_index):
        self.meta = metadata_index

        self.field_thresholds = {
            "title": 85,
            "origin_name": 85,
            "actor": 90,
            "director": 88,
            "category": 75,
            "country": 80,
            "quality": 90,
            "lang": 85,
            "type": 85,
            "status": 85,
        }

    def _is_valid_choice(self, value, field):
        if not value:
            return False

        value = value.strip().lower()

        if value in BAD_ENTITY_VALUES:
            return False

        if len(value) < 4 and field in ["actor", "director", "category", "country"]:
            return False

        if field in ["actor", "director"]:
            tokens = [
                t for t in value.split()
                if t not in BAD_ENTITY_VALUES
            ]

            # Tránh actor/director chỉ có 1 token quá ngắn
            if len(tokens) == 1 and len(tokens[0]) < 5:
                return False

        if field in ["title", "origin_name"]:
            if len(value) < 2:
                return False

        return True

    def _get_scorer(self, field):
        # Person/title nên match theo token, tránh match một phần nhỏ như "jr", "alf"
        if field in ["title", "origin_name", "actor", "director"]:
            return fuzz.token_set_ratio

        # Category/country/lang có thể dùng partial vì query tiếng Việt thường ngắn
        return fuzz.partial_ratio

    def _fuzzy_field_link(self, query, field, limit=5):
        choices = [
            c for c in self.meta.index[field].keys()
            if self._is_valid_choice(c, field)
        ]

        if not choices:
            return []

        scorer = self._get_scorer(field)

        matches = process.extract(
            query,
            choices,
            scorer=scorer,
            limit=limit
        )

        threshold = self.field_thresholds.get(field, 85)
        results = []

        for value, score, _ in matches:
            if score >= threshold:
                movies = list(self.meta.index[field][value])
                results.append({
                    "field": field,
                    "value": value,
                    "score": score / 100,
                    "movies": movies
                })

        return results

    def _extract_year(self, query):
        return re.findall(r"\b(19\d{2}|20\d{2})\b", query)

    def link(self, query):
        query_norm = query.lower()
        linked_entities = []

        for field in [
            "title",
            "origin_name",
            "actor",
            "director",
            "category",
            "country",
            "quality",
            "lang",
            "type",
            "status",
        ]:
            linked_entities.extend(
                self._fuzzy_field_link(query_norm, field)
            )

        for year in self._extract_year(query_norm):
            if year in self.meta.index["year"]:
                linked_entities.append({
                    "field": "year",
                    "value": year,
                    "score": 1.0,
                    "movies": list(self.meta.index["year"][year])
                })

        return {
            "query": query,
            "linked_entities": linked_entities
        }


def compute_schema_scores(schema_result):
    movie_scores = {}

    for link in schema_result["linked_entities"]:
        field = link["field"]
        field_weight = FIELD_WEIGHTS.get(field, 1.0)
        link_score = link["score"]

        for movie in link["movies"]:
            movie_scores[movie] = movie_scores.get(movie, 0.0) + (
                field_weight * link_score
            )

    return movie_scores