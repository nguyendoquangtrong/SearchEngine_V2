import concurrent.futures

from src.core.search_engine_v2 import MovieSearchEngine
from src.schema.schema_linker import MovieJSONSchemaLinker, compute_schema_scores
from src.schema.metadata_index_builder import MovieMetadataIndex
from configs.config import *


class MovieSearchEngineV3(MovieSearchEngine):
    def __init__(self):
        super().__init__()

        print("⏳ Đang khởi tạo Metadata Schema Linker...")
        self.metadata_index = MovieMetadataIndex(CLEAN_EN_JSON_PATH)
        self.schema_linker = MovieJSONSchemaLinker(self.metadata_index)
        print("✅ Metadata Schema Linker đã sẵn sàng!")

    # ==============================================================
    # BASIC SCHEMA SEARCH
    # ==============================================================

    def schema_search_only(self, query, top_n=10):
        """
        Chỉ tìm bằng Schema Linking.
        Dùng để debug xem query link được vào metadata nào.
        """
        schema_result = self.schema_linker.link(query)
        schema_scores = compute_schema_scores(schema_result)
        ranked = sorted(schema_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "query": query,
            "schema_result": schema_result,
            "results": ranked[:top_n]
        }

    def _schema_candidates(self, query, top_n=30):
        """
        Trả về:
        - schema_movies: danh sách phim từ schema linking
        - schema_scores: điểm schema theo từng phim
        - schema_result: thông tin entity đã link được
        """
        schema_result = self.schema_linker.link(query)
        schema_scores = compute_schema_scores(schema_result)
        ranked = sorted(schema_scores.items(), key=lambda x: x[1], reverse=True)

        schema_movies = [movie for movie, _ in ranked[:top_n]]

        return schema_movies, dict(ranked), schema_result

    def _print_schema_debug(self, schema_result):
        """
        In thông tin schema linking để debug.
        """
        print("\n🔗 [Schema Linking]")

        if schema_result["linked_entities"]:
            for item in schema_result["linked_entities"]:
                print(
                    f"  - {item['field']} = {item['value']} "
                    f"(score={item['score']:.2f}) -> {item['movies'][:5]}"
                )
        else:
            print("  - Không tìm thấy linked entity rõ ràng.")

    # ==============================================================
    # CONTEXT HELPERS
    # ==============================================================

    def _get_schema_weight(self, intent):
        """
        Trọng số schema theo intent cho full multimodal pipeline.
        """
        if intent == "EXACT_QUOTE":
            return 0.5

        if intent == "VISUAL_SCENE":
            return 0.7

        return 1.4

    def _get_fallback_context(self, movie_name):
        """
        Lấy summary/description fallback cho phim nếu BM25 hoặc SBERT không trả về context.
        """
        for meta, doc in zip(self.bm25_meta, self.bm25_docs):
            if meta["movie_name"] == movie_name and meta.get("type") == "summary":
                return doc

        movie = self.metadata_index.get_movie(movie_name)
        if movie:
            content = movie.get("content", "")
            if content:
                return str(content)

        return "No specific dialogue context found."

    def _get_metadata_context(self, movie_name):
        """
        Lấy metadata của phim để đưa vào CrossEncoder reranking.
        """
        movie = self.metadata_index.get_movie(movie_name)

        if not movie:
            return ""

        def join_value(value):
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)

            return str(value) if value is not None else ""

        metadata_context = (
            f"Title: {join_value(movie.get('title'))}. "
            f"Original title: {join_value(movie.get('origin_name'))}. "
            f"Year: {join_value(movie.get('year'))}. "
            f"Actors: {join_value(movie.get('actor'))}. "
            f"Director: {join_value(movie.get('director'))}. "
            f"Category: {join_value(movie.get('category'))}. "
            f"Country: {join_value(movie.get('country'))}. "
            f"Content: {join_value(movie.get('content'))}."
        )

        return metadata_context

    # ==============================================================
    # FULL MULTIMODAL + SCHEMA SEARCH
    # ==============================================================

    def search_v3(self, query_text, system_type="PT2", top_n=5, k=60, debug=True):
        """
        SearchEngine V3:
        BM25 + CLIP Image + SBERT/CLIP Text + Schema Linking
        Sau đó fusion bằng Weighted RRF và rerank bằng CrossEncoder + metadata context.
        """
        intent, w_bm25, w_img, w_txt, enable_visual_boost = self.router.analyze(query_text)
        w_schema = self._get_schema_weight(intent)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_bm25 = executor.submit(self._thread_bm25, query_text)
            future_txt = executor.submit(self._thread_text, query_text, system_type)
            future_schema = executor.submit(self._schema_candidates, query_text)

            if w_img > 0:
                future_img = executor.submit(self._thread_image, query_text)

            bm25_movies, bm25_contexts = future_bm25.result()
            txt_movies, txt_contexts = future_txt.result()
            schema_movies, schema_scores, schema_result = future_schema.result()

            if w_img > 0:
                img_movies, img_distances = future_img.result()
            else:
                img_movies, img_distances = [], {}

        if debug:
            self._print_schema_debug(schema_result)

        # ==========================
        # 1. Weighted RRF fusion
        # ==========================
        rrf = {}

        for rank, movie in enumerate(bm25_movies):
            rrf[movie] = rrf.get(movie, 0.0) + w_bm25 / (k + rank + 1)

        for rank, movie in enumerate(img_movies):
            rrf[movie] = rrf.get(movie, 0.0) + w_img / (k + rank + 1)

        for rank, movie in enumerate(txt_movies):
            rrf[movie] = rrf.get(movie, 0.0) + w_txt / (k + rank + 1)

        for rank, movie in enumerate(schema_movies):
            rrf[movie] = rrf.get(movie, 0.0) + w_schema / (k + rank + 1)

        for movie, score in schema_scores.items():
            rrf[movie] = rrf.get(movie, 0.0) + 0.03 * score

        candidates = [
            movie for movie, _ in sorted(
                rrf.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        ]

        if not candidates:
            return []

        # ==========================
        # 2. CrossEncoder reranking + metadata context
        # ==========================
        final_scores = []

        for movie in candidates:
            if movie in bm25_contexts:
                ctx = bm25_contexts[movie]
            elif movie in txt_contexts:
                ctx = txt_contexts[movie]
            else:
                ctx = self._get_fallback_context(movie)

            metadata_ctx = self._get_metadata_context(movie)

            pair = [
                query_text,
                f"Movie: {movie}. Metadata: {metadata_ctx} Retrieval context: {ctx}"
            ]

            rerank_score = float(self.rerank_model.predict([pair])[0])

            # Schema-aware rerank bonus
            rerank_score += 0.25 * schema_scores.get(movie, 0.0)

            # Adaptive visual boost
            if enable_visual_boost and movie in img_movies:
                img_rank = img_movies.index(movie)
                distance = img_distances.get(movie, 1.0)

                if distance < 0.65:
                    print(
                        f"    ✨ Kích hoạt Visual Boost cho '{movie}' "
                        f"(Rank {img_rank}, Lệch {distance:.2f})"
                    )

                    if img_rank == 0:
                        rerank_score += 3.0
                    elif img_rank < 3:
                        rerank_score += 1.5
                    elif img_rank < 10:
                        rerank_score += 0.5

            final_scores.append((movie, rerank_score))

        final = sorted(final_scores, key=lambda x: x[1], reverse=True)

        if debug:
            print("\n🏆 [Final V3 Ranking]")
            for movie, score in final[:top_n]:
                print(f"  - {movie}: {score:.4f}")

        return [movie for movie, _ in final[:top_n]]

    # ==============================================================
    # SCHEMA-AWARE SINGLE STREAM SEARCH
    # Áp dụng Schema Linking cho từng mô hình thuần:
    # BM25, CLIP Text, SBERT, CLIP Image
    # ==============================================================

    def _schema_aware_rerank_for_single_stream(
        self,
        query_text,
        base_movies,
        base_contexts=None,
        schema_movies=None,
        schema_scores=None,
        img_distances=None,
        stream_name="base",
        top_n=5,
        k=60,
        w_base=1.0,
        w_schema=1.0,
        rrf_schema_bonus=0.03,
        rerank_schema_bonus=0.25,
        enable_visual_boost=False,
        debug=False
    ):
        """
        Pipeline chung cho các mô hình thuần + Schema Linking.

        Quy trình:
        1. Lấy candidates từ mô hình thuần.
        2. Lấy candidates từ Schema Linking.
        3. Fusion bằng RRF.
        4. Lấy metadata context của từng phim.
        5. CrossEncoder rerank query với metadata + retrieval context.
        6. Cộng schema bonus nếu phim match metadata.
        """
        if base_contexts is None:
            base_contexts = {}

        if schema_movies is None:
            schema_movies = []

        if schema_scores is None:
            schema_scores = {}

        if img_distances is None:
            img_distances = {}

        # ==========================
        # 1. RRF fusion
        # ==========================
        rrf = {}

        for rank, movie in enumerate(base_movies):
            rrf[movie] = rrf.get(movie, 0.0) + w_base / (k + rank + 1)

        for rank, movie in enumerate(schema_movies):
            rrf[movie] = rrf.get(movie, 0.0) + w_schema / (k + rank + 1)

        for movie, score in schema_scores.items():
            rrf[movie] = rrf.get(movie, 0.0) + rrf_schema_bonus * score

        candidates = [
            movie for movie, _ in sorted(
                rrf.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        ]

        if not candidates:
            return []

        # ==========================
        # 2. Metadata-aware CrossEncoder rerank
        # ==========================
        final_scores = []

        for movie in candidates:
            if movie in base_contexts:
                ctx = base_contexts[movie]
            else:
                ctx = self._get_fallback_context(movie)

            metadata_ctx = self._get_metadata_context(movie)

            pair = [
                query_text,
                f"Movie: {movie}. Metadata: {metadata_ctx} Retrieval context: {ctx}"
            ]

            rerank_score = float(self.rerank_model.predict([pair])[0])

            # Schema-aware bonus giống full V3
            rerank_score += rerank_schema_bonus * schema_scores.get(movie, 0.0)

            # Riêng CLIP Image có thể giữ visual boost
            if enable_visual_boost and movie in base_movies:
                img_rank = base_movies.index(movie)
                distance = img_distances.get(movie, 1.0)

                if distance < 0.65:
                    if img_rank == 0:
                        rerank_score += 3.0
                    elif img_rank < 3:
                        rerank_score += 1.5
                    elif img_rank < 10:
                        rerank_score += 0.5

            final_scores.append((movie, rerank_score))

        final = sorted(final_scores, key=lambda x: x[1], reverse=True)

        if debug:
            print(f"\n🏆 [Final Ranking - {stream_name.upper()} + Schema]")
            for movie, score in final[:top_n]:
                print(f"  - {movie}: {score:.4f}")

        return [movie for movie, _ in final[:top_n]]

    def search_bm25_schema(self, query_text, top_n=5, debug=False):
        """
        BM25 + Schema Linking + Metadata-aware Reranking.
        """
        base_movies, base_contexts = self._thread_bm25(query_text)

        schema_movies, schema_scores, schema_result = self._schema_candidates(
            query=query_text,
            top_n=30
        )

        if debug:
            print("\n" + "=" * 80)
            print("[BM25 + SCHEMA LINKING + METADATA RERANK]")
            print("=" * 80)
            self._print_schema_debug(schema_result)

        return self._schema_aware_rerank_for_single_stream(
            query_text=query_text,
            base_movies=base_movies,
            base_contexts=base_contexts,
            schema_movies=schema_movies,
            schema_scores=schema_scores,
            stream_name="bm25",
            top_n=top_n,
            w_base=1.0,
            w_schema=1.0,
            rrf_schema_bonus=0.03,
            rerank_schema_bonus=0.25,
            enable_visual_boost=False,
            debug=debug
        )

    def search_clip_text_schema(self, query_text, top_n=5, debug=False):
        """
        CLIP Text + Schema Linking + Metadata-aware Reranking.
        PT1 = CLIP Text theo search_engine_v2.
        """
        base_movies, base_contexts = self._thread_text(query_text, "PT1")

        schema_movies, schema_scores, schema_result = self._schema_candidates(
            query=query_text,
            top_n=30
        )

        if debug:
            print("\n" + "=" * 80)
            print("[CLIP TEXT + SCHEMA LINKING + METADATA RERANK]")
            print("=" * 80)
            self._print_schema_debug(schema_result)

        return self._schema_aware_rerank_for_single_stream(
            query_text=query_text,
            base_movies=base_movies,
            base_contexts=base_contexts,
            schema_movies=schema_movies,
            schema_scores=schema_scores,
            stream_name="clip_text",
            top_n=top_n,
            w_base=1.0,
            w_schema=1.0,
            rrf_schema_bonus=0.03,
            rerank_schema_bonus=0.25,
            enable_visual_boost=False,
            debug=debug
        )

    def search_sbert_schema(self, query_text, top_n=5, debug=False):
        """
        SBERT + Schema Linking + Metadata-aware Reranking.
        PT2 = SBERT theo search_engine_v2.
        """
        base_movies, base_contexts = self._thread_text(query_text, "PT2")

        schema_movies, schema_scores, schema_result = self._schema_candidates(
            query=query_text,
            top_n=30
        )

        if debug:
            print("\n" + "=" * 80)
            print("[SBERT + SCHEMA LINKING + METADATA RERANK]")
            print("=" * 80)
            self._print_schema_debug(schema_result)

        return self._schema_aware_rerank_for_single_stream(
            query_text=query_text,
            base_movies=base_movies,
            base_contexts=base_contexts,
            schema_movies=schema_movies,
            schema_scores=schema_scores,
            stream_name="sbert",
            top_n=top_n,
            w_base=1.0,
            w_schema=1.0,
            rrf_schema_bonus=0.03,
            rerank_schema_bonus=0.25,
            enable_visual_boost=False,
            debug=debug
        )

    def search_image_schema(self, query_text, top_n=5, debug=False):
        """
        CLIP Image + Schema Linking + Metadata-aware Reranking.
        """
        base_movies, img_distances = self._thread_image(query_text)

        schema_movies, schema_scores, schema_result = self._schema_candidates(
            query=query_text,
            top_n=30
        )

        if debug:
            print("\n" + "=" * 80)
            print("[CLIP IMAGE + SCHEMA LINKING + METADATA RERANK]")
            print("=" * 80)
            self._print_schema_debug(schema_result)

        return self._schema_aware_rerank_for_single_stream(
            query_text=query_text,
            base_movies=base_movies,
            base_contexts={},
            schema_movies=schema_movies,
            schema_scores=schema_scores,
            img_distances=img_distances,
            stream_name="clip_image",
            top_n=top_n,
            w_base=1.0,
            w_schema=1.0,
            rrf_schema_bonus=0.03,
            rerank_schema_bonus=0.25,
            enable_visual_boost=True,
            debug=debug
        )