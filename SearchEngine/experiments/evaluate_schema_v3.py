from src.core.search_engine_v3 import MovieSearchEngineV3

TEST_QUERIES = [
    # ==========================
    # GROUP 1: EXACT QUOTE
    # ==========================
    {
        "group": "G1_EXACT_QUOTE",
        "query": "I am going to make him an offer he can't refuse",
        "expected": "The Godfather",
    },
    {
        "group": "G1_EXACT_QUOTE",
        "query": "My mama always said life was like a box of chocolates",
        "expected": "Forrest Gump",
    },
    {
        "group": "G1_EXACT_QUOTE",
        "query": "I ate his liver with some fava beans and a nice Chianti",
        "expected": "The Silence of the Lambs",
    },
    {
        "group": "G1_EXACT_QUOTE",
        "query": "Keep your friends close, but your enemies closer",
        "expected": "The Godfather Part III",
    },
    {
        "group": "G1_EXACT_QUOTE",
        "query": "Here's looking at you, kid",
        "expected": "Casablanca",
    },

    # ==========================
    # GROUP 2: SEMANTIC / PLOT
    # ==========================
    {
        "group": "G2_SEMANTIC_PLOT",
        "query": "two completely opposite families one extremely rich and the other always lives in poverty",
        "expected": "Parasite",
    },
    {
        "group": "G2_SEMANTIC_PLOT",
        "query": "a computer hacker learning the truth about his simulated reality",
        "expected": "The Matrix",
    },
    {
        "group": "G2_SEMANTIC_PLOT",
        "query": "a banker wrongly convicted of murder escapes prison",
        "expected": "The Shawshank Redemption",
    },
    {
        "group": "G2_SEMANTIC_PLOT",
        "query": "entering dreams to steal information from a target",
        "expected": "Inception",
    },
    {
        "group": "G2_SEMANTIC_PLOT",
        "query": "two gangsters, a boxer, and a stolen briefcase",
        "expected": "Pulp Fiction",
    },
    {
        "group": "G2_SEMANTIC_PLOT",
        "query": "a girl trying to save her parents who turned into pigs",
        "expected": "Spirited Away",
    },
    {
        "group": "G2_SEMANTIC_PLOT",
        "query": "brother and sister struggling to survive during World War II in Japan",
        "expected": "Grave Of The Fireflies",
    },

    # ==========================
    # GROUP 3: VISUAL / SCENE
    # ==========================
    {
        "group": "G3_VISUAL_SCENE",
        "query": "a woman screaming in a motel shower black and white",
        "expected": "Psycho",
    },
    {
        "group": "G3_VISUAL_SCENE",
        "query": "a glowing mechanical suit flying in the sky",
        "expected": "Iron Man 3",
    },
    {
        "group": "G3_VISUAL_SCENE",
        "query": "giant robots fighting monsters in the ocean",
        "expected": "Pacific Rim: Uprising",
    },
    {
        "group": "G3_VISUAL_SCENE",
        "query": "seven warriors defending a village in the rain",
        "expected": "Seven Samurai",
    },
    {
        "group": "G3_VISUAL_SCENE",
        "query": "a dark knight standing on a tall building in Gotham",
        "expected": "The Dark Knight",
    },

    # ==========================
    # GROUP 4: LEXICAL TRAP / NOISE
    # ==========================
    {
        "group": "G4_LEXICAL_TRAP",
        "query": "a guy with short term memory loss taking polaroid pictures",
        "expected": "Memento",
    },
    {
        "group": "G4_LEXICAL_TRAP",
        "query": "two magicians competing and sabotaging each other",
        "expected": "The Prestige",
    },
    {
        "group": "G4_LEXICAL_TRAP",
        "query": "I am going to make him an offer he cannot refuse",
        "expected": "The Godfather",
    },
]


def normalize_movie_name(name):
    return name.lower().strip()


def find_rank(results, expected):
    expected_norm = normalize_movie_name(expected)

    for idx, movie in enumerate(results, start=1):
        movie_norm = normalize_movie_name(movie)

        if movie_norm == expected_norm:
            return idx

    return None


def reciprocal_rank(results, expected):
    rank = find_rank(results, expected)

    if rank is None:
        return 0.0

    return 1.0 / rank


def precision_at_5(results, expected):
    rank = find_rank(results, expected)

    if rank is not None and rank <= 5:
        return 1.0 / 5

    return 0.0


def recall_at_5(results, expected):
    rank = find_rank(results, expected)

    if rank is not None and rank <= 5:
        return 1.0

    return 0.0


def evaluate_system(search_fn, system_name, top_n=5):
    rows = []

    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected"]
        group = item["group"]

        try:
            results = search_fn(query, top_n)
        except Exception as e:
            print(f"[ERROR] {system_name} failed on query: {query}")
            print(e)
            results = []

        rank = find_rank(results, expected)

        rows.append({
            "system": system_name,
            "group": group,
            "query": query,
            "expected": expected,
            "results": results,
            "rank": rank,
            "rr": reciprocal_rank(results, expected),
            "precision@5": precision_at_5(results, expected),
            "recall@5": recall_at_5(results, expected),
        })

    return rows


def compute_summary(rows):
    n = len(rows)

    return {
        "mrr": sum(r["rr"] for r in rows) / n,
        "precision@5": sum(r["precision@5"] for r in rows) / n,
        "recall@5": sum(r["recall@5"] for r in rows) / n,
    }


def summarize(rows, system_name):
    summary = compute_summary(rows)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {system_name}")
    print("=" * 80)
    print(f"MRR         : {summary['mrr']:.4f}")
    print(f"Precision@5 : {summary['precision@5']:.4f}")
    print(f"Recall@5    : {summary['recall@5']:.4f}")

    groups = sorted(set(r["group"] for r in rows))

    print("\nGROUP SUMMARY")
    for group in groups:
        group_rows = [r for r in rows if r["group"] == group]
        group_summary = compute_summary(group_rows)

        print(
            f"{group}: "
            f"MRR={group_summary['mrr']:.4f}, "
            f"Precision@5={group_summary['precision@5']:.4f}, "
            f"Recall@5={group_summary['recall@5']:.4f}"
        )


def print_final_table(result_map):
    print("\n" + "=" * 100)
    print("FINAL TABLE: V1 VS SCHEMA LINKING")
    print("=" * 100)

    table_pairs = [
        ("Thuần BM25", "BM25_ONLY", "BM25_SCHEMA"),
        ("Thuần CLIP Text", "CLIP_TEXT_ONLY", "CLIP_TEXT_SCHEMA"),
        ("Thuần SBERT", "SBERT_ONLY", "SBERT_SCHEMA"),
        ("Thuần CLIP Image", "CLIP_IMAGE_ONLY", "CLIP_IMAGE_SCHEMA"),
        ("Đa phương thức", "MULTIMODAL_V2", "MULTIMODAL_SCHEMA"),
    ]

    header = (
        f"{'Luồng mô hình':<22} | "
        f"{'V1 MRR':>8} | {'V1 P@5':>8} | {'V1 R@5':>8} | "
        f"{'Schema MRR':>10} | {'Schema P@5':>10} | {'Schema R@5':>10}"
    )

    print(header)
    print("-" * len(header))

    for label, v1_key, schema_key in table_pairs:
        v1 = compute_summary(result_map[v1_key])
        schema = compute_summary(result_map[schema_key])

        print(
            f"{label:<22} | "
            f"{v1['mrr']:>8.4f} | {v1['precision@5']:>8.4f} | {v1['recall@5']:>8.4f} | "
            f"{schema['mrr']:>10.4f} | {schema['precision@5']:>10.4f} | {schema['recall@5']:>10.4f}"
        )


def shorten_query(query, max_len=42):
    if len(query) <= max_len:
        return query

    return query[:max_len - 3] + "..."


def print_group_query_rr_table(result_map, group_name, mode="normal"):
    """
    In bảng RR theo từng query cho từng nhóm.
    Lưu ý:
    - RR là điểm của từng query.
    - MRR là trung bình RR của cả nhóm.
    """

    if mode == "normal":
        title = f"QUERY-LEVEL RR TABLE - {group_name} - NORMAL"
        systems = [
            ("BM25", "BM25_ONLY"),
            ("CLIP Text", "CLIP_TEXT_ONLY"),
            ("SBERT", "SBERT_ONLY"),
            ("CLIP Image", "CLIP_IMAGE_ONLY"),
            ("Đa phương thức", "MULTIMODAL_V2"),
        ]
    elif mode == "schema":
        title = f"QUERY-LEVEL RR TABLE - {group_name} - SCHEMA LINKING"
        systems = [
            ("BM25", "BM25_SCHEMA"),
            ("CLIP Text", "CLIP_TEXT_SCHEMA"),
            ("SBERT", "SBERT_SCHEMA"),
            ("CLIP Image", "CLIP_IMAGE_SCHEMA"),
            ("Đa phương thức", "MULTIMODAL_SCHEMA"),
        ]
    else:
        raise ValueError("mode must be 'normal' or 'schema'")

    base_rows = [r for r in result_map["BM25_ONLY"] if r["group"] == group_name]

    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)

    header = (
        f"{'Query':<45} | "
        f"{'BM25':>8} | "
        f"{'CLIP Text':>10} | "
        f"{'SBERT':>8} | "
        f"{'CLIP Image':>11} | "
        f"{'Đa phương thức':>15}"
    )

    print(header)
    print("-" * len(header))

    group_values = {label: [] for label, _ in systems}

    for idx, base_row in enumerate(base_rows):
        query = shorten_query(base_row["query"])

        values = []

        for label, system_key in systems:
            system_group_rows = [
                r for r in result_map[system_key]
                if r["group"] == group_name
            ]

            rr = system_group_rows[idx]["rr"]
            values.append(rr)
            group_values[label].append(rr)

        print(
            f"{query:<45} | "
            f"{values[0]:>8.4f} | "
            f"{values[1]:>10.4f} | "
            f"{values[2]:>8.4f} | "
            f"{values[3]:>11.4f} | "
            f"{values[4]:>15.4f}"
        )

    print("-" * len(header))

    mrr_values = []
    for label, _ in systems:
        vals = group_values[label]
        mrr = sum(vals) / len(vals) if vals else 0.0
        mrr_values.append(mrr)

    print(
        f"{'MRR trung bình nhóm':<45} | "
        f"{mrr_values[0]:>8.4f} | "
        f"{mrr_values[1]:>10.4f} | "
        f"{mrr_values[2]:>8.4f} | "
        f"{mrr_values[3]:>11.4f} | "
        f"{mrr_values[4]:>15.4f}"
    )


def print_all_group_query_rr_tables(result_map):
    groups = [
        "G1_EXACT_QUOTE",
        "G2_SEMANTIC_PLOT",
        "G3_VISUAL_SCENE",
        "G4_LEXICAL_TRAP",
    ]

    for group in groups:
        print_group_query_rr_table(result_map, group, mode="normal")
        print_group_query_rr_table(result_map, group, mode="schema")


def print_detailed_results(rows, system_name):
    print("\n" + "=" * 80)
    print(f"DETAILED RESULTS: {system_name}")
    print("=" * 80)

    for idx, row in enumerate(rows, start=1):
        print("\n" + "-" * 80)
        print(f"Query {idx}: {row['query']}")
        print(f"Group     : {row['group']}")
        print(f"Expected  : {row['expected']}")

        print("\nResults:")
        for i, movie in enumerate(row["results"], start=1):
            marker = "✅" if normalize_movie_name(movie) == normalize_movie_name(row["expected"]) else "  "
            print(f"{marker} {i}. {movie}")

        print(
            "\nMetrics: "
            f"rank={row['rank']}, "
            f"RR={row['rr']:.4f}, "
            f"Precision@5={row['precision@5']:.4f}, "
            f"Recall@5={row['recall@5']:.4f}"
        )


def main():
    print("Loading SearchEngine V3...")
    engine = MovieSearchEngineV3()

    systems = {
        # V1 / Original single streams
        "BM25_ONLY": lambda query, top_n: engine.search_bm25_only(
            query_text=query,
            top_n=top_n
        ),
        "CLIP_TEXT_ONLY": lambda query, top_n: engine.search_clip_text_only(
            query_text=query,
            top_n=top_n
        ),
        "SBERT_ONLY": lambda query, top_n: engine.search_sbert_only(
            query_text=query,
            top_n=top_n
        ),
        "CLIP_IMAGE_ONLY": lambda query, top_n: engine.search_image_only(
            query_text=query,
            top_n=top_n
        ),

        # V1 / Original multimodal
        "MULTIMODAL_V2": lambda query, top_n: engine.search(
            query_text=query,
            system_type="PT2",
            top_n=top_n
        ),

        # Schema linking variants
        "BM25_SCHEMA": lambda query, top_n: engine.search_bm25_schema(
            query_text=query,
            top_n=top_n,
            debug=False
        ),
        "CLIP_TEXT_SCHEMA": lambda query, top_n: engine.search_clip_text_schema(
            query_text=query,
            top_n=top_n,
            debug=False
        ),
        "SBERT_SCHEMA": lambda query, top_n: engine.search_sbert_schema(
            query_text=query,
            top_n=top_n,
            debug=False
        ),
        "CLIP_IMAGE_SCHEMA": lambda query, top_n: engine.search_image_schema(
            query_text=query,
            top_n=top_n,
            debug=False
        ),
        "MULTIMODAL_SCHEMA": lambda query, top_n: engine.search_v3(
            query_text=query,
            system_type="PT2",
            top_n=top_n,
            debug=False
        ),
    }

    result_map = {}

    for system_name, search_fn in systems.items():
        print("\n" + "#" * 100)
        print(f"Running system: {system_name}")
        print("#" * 100)

        rows = evaluate_system(
            search_fn=search_fn,
            system_name=system_name,
            top_n=5
        )

        result_map[system_name] = rows
        summarize(rows, system_name)

    print_final_table(result_map)

    # In bảng RR theo từng query cho từng nhóm, gồm Normal và Schema Linking.
    print_all_group_query_rr_tables(result_map)

    # Muốn xem chi tiết hệ thống nào thì mở comment dòng dưới.
    # print_detailed_results(result_map["MULTIMODAL_SCHEMA"], "MULTIMODAL_SCHEMA")


if __name__ == "__main__":
    main()