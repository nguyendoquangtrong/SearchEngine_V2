import pickle
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer, CrossEncoder , util
import torch

from config import BM25_PATH, CHROMA_HOST, CHROMA_PORT
from helpers import tokenize


class IntentRouter:
    def __init__(self, sbert_model):
        """
        Sử dụng kỹ thuật Zero-Shot Semantic Classification bằng SBERT.
        Tái sử dụng mô hình SBERT đã load trên RAM để không gây tốn thêm bộ nhớ.
        """
        self.sbert_model = sbert_model

        # 1. Định nghĩa "Profile" của 3 ý định bằng ngôn ngữ tự nhiên
        self.intents = {
            "EXACT_QUOTE": "exact dialogue, character quote, speaking, saying a specific sentence, memorable verbatim lines",
            "VISUAL_SCENE": "visual scene, aesthetics, camera angle, physical action, wearing, colors, background, landscape, fighting",
            "PLOT_SEMANTIC": "movie plot, semantic theme, summary, abstract concepts, backstory, genre, character development"
        }

        self.intent_labels = list(self.intents.keys())

        # 2. Encode sẵn các Profile này thành Vector để so sánh siêu tốc
        print("⏳ Đang khởi tạo Không gian Vector cho Bộ định tuyến Ý định...")
        self.intent_embeddings = self.sbert_model.encode(
            list(self.intents.values()),
            convert_to_tensor=True
        )

    def analyze(self, query_text):
        """
        Phân tích câu query và trả về cấu hình trọng số RRF phù hợp.
        Chỉ dùng thuật toán Vector Semantic để hiểu ngữ cảnh, không dùng từ khóa thủ công.
        """
        # 1. Biến câu tìm kiếm thành vector
        query_embedding = self.sbert_model.encode(query_text, convert_to_tensor=True)

        # 2. Tính toán độ tương đồng Cosine Similarity giữa Query và 3 Profile Ý định
        cosine_scores = util.cos_sim(query_embedding, self.intent_embeddings)[0]

        # 3. Lấy ra Ý định có điểm tương đồng cao nhất
        best_idx = torch.argmax(cosine_scores).item()
        best_intent = self.intent_labels[best_idx]
        confidence = cosine_scores[best_idx].item()

        # 4. Fallback (Luật an toàn): Nếu AI phân vân (độ tự tin quá thấp), mặc định đó là kể chuyện (Plot)
        if confidence < 0.35:
            best_intent = "PLOT_SEMANTIC"

        print(f"\n🧠 [AI Intent Router] Đã phân loại: {best_intent} (Độ tự tin: {confidence:.2f})")

        # 5. TRẢ VỀ CẤU HÌNH TRỌNG SỐ TƯƠNG ỨNG
        if best_intent == "EXACT_QUOTE":
            return "EXACT_QUOTE", 2.0, 0.0, 1.0, False

        elif best_intent == "VISUAL_SCENE":
            return "VISUAL_SCENE", 0.3, 1.8, 1.5, True

        else:  # PLOT_SEMANTIC
            return "PLOT_SEMANTIC", 0.8, 0.2, 2.0, False

class MovieSearchEngine:
    def __init__(self):
        print("⏳ Đang tải Bộ não AI và kết nối Database...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        with open(BM25_PATH, 'rb') as f:
            self.bm25_model, self.bm25_meta, self.bm25_docs = pickle.load(f)

        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.img_collection = self.chroma_client.get_collection("image_clip_collection")
        self.txt_clip_collection = self.chroma_client.get_collection("text_clip_collection")
        self.txt_sbert_collection = self.chroma_client.get_collection("text_sbert_collection")
        self.router = IntentRouter(self.sbert_model)
        print("✅ Bộ máy tìm kiếm đã sẵn sàng!")

    def _thread_bm25(self, query_text):
        tok_q = tokenize(query_text)
        bm25_scores = self.bm25_model.get_scores(tok_q)
        movies, context_dict = [], {}
        for s, meta, doc in sorted(zip(bm25_scores, self.bm25_meta, self.bm25_docs), key=lambda x: x[0], reverse=True):
            if s <= 0: break
            m = meta['movie_name']
            if m not in movies:
                movies.append(m)
                context_dict[m] = doc
            if len(movies) == 20: break
        return movies, context_dict

    def _thread_image(self, query_text):
        vec_img = self.clip_model.encode(query_text).tolist()

        # Lấy thêm 'distances' từ ChromaDB để check độ an toàn
        res_img = self.img_collection.query(
            query_embeddings=[vec_img],
            n_results=150,
            include=['metadatas', 'distances']
        )

        movies = []
        movie_distances = {}  # Lưu khoảng cách để chặn ảo giác

        if res_img['metadatas'] and res_img['metadatas'][0]:
            for m, dist in zip(res_img['metadatas'][0], res_img['distances'][0]):
                name = m['movie_name']
                if name not in movies:
                    movies.append(name)
                    movie_distances[name] = dist
                if len(movies) == 20: break

        # TRẢ VỀ ĐÚNG 2 GIÁ TRỊ NHƯ HÀM SEARCH MONG ĐỢI
        return movies, movie_distances

    def _thread_text(self, query_text, system_type):
        movies, context_dict = [], {}
        if system_type == "PT1":
            vec_txt = self.clip_model.encode(query_text).tolist()
            res_txt = self.txt_clip_collection.query(query_embeddings=[vec_txt], n_results=150)
        else:
            vec_txt = self.sbert_model.encode(query_text).tolist()
            res_txt = self.txt_sbert_collection.query(query_embeddings=[vec_txt], n_results=150)

        if res_txt['metadatas'] and res_txt['metadatas'][0]:
            for meta, doc in zip(res_txt['metadatas'][0], res_txt['documents'][0]):
                name = meta['movie_name']
                if name not in movies: movies.append(name)
                if name not in context_dict: context_dict[name] = doc
                if len(movies) == 20: break
        return movies, context_dict

    def search(self, query_text, system_type="PT2", top_n=5, k=60):
        # 1. GỌI AI ROUTER ĐỂ PHÂN TÍCH Ý ĐỊNH VÀ LẤY TRỌNG SỐ
        intent, w_bm25, w_img, w_txt, enable_visual_boost = self.router.analyze(query_text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_bm25 = executor.submit(self._thread_bm25, query_text)
            future_txt = executor.submit(self._thread_text, query_text, system_type)

            # TỐI ƯU HIỆU NĂNG: Chỉ chạy luồng Ảnh nếu trọng số > 0
            if w_img > 0:
                future_img = executor.submit(self._thread_image, query_text)

            bm25_movies, bm25_contexts = future_bm25.result()
            txt_movies, txt_contexts = future_txt.result()

            # Lấy kết quả Ảnh (Cần đảm bảo hàm _thread_image đã sửa để trả về cả list movies và list distances như tôi hướng dẫn ở tin nhắn trước)
            if w_img > 0:
                img_movies, img_distances = future_img.result()
            else:
                img_movies, img_distances = [], {}

        # 2. DÙNG TRỌNG SỐ ĐỘNG CHO THUẬT TOÁN RRF (THAY VÌ FIX CỨNG)
        rrf = {}
        for rank, m in enumerate(bm25_movies): rrf[m] = rrf.get(m, 0) + w_bm25 / (k + rank + 1)
        for rank, m in enumerate(img_movies):  rrf[m] = rrf.get(m, 0) + w_img / (k + rank + 1)
        for rank, m in enumerate(txt_movies):  rrf[m] = rrf.get(m, 0) + w_txt / (k + rank + 1)

        candidates = [m for m, s in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:20]]
        if not candidates: return []

        def get_fallback_context(movie_name):
            for meta, doc in zip(self.bm25_meta, self.bm25_docs):
                if meta['movie_name'] == movie_name and meta.get('type') == 'summary': return doc
            return "No specific dialogue context found."

        final_scores = []
        for m in candidates:
            if m in bm25_contexts:
                ctx = bm25_contexts[m]
            elif m in txt_contexts:
                ctx = txt_contexts[m]
            else:
                ctx = get_fallback_context(m)

            rerank_score = self.rerank_model.predict([query_text, f"Movie: {m}. Content: {ctx}"])

            # 3. ADAPTIVE VISUAL BOOST (PHẦN THƯỞNG CÓ ĐIỀU KIỆN)
            if enable_visual_boost and m in img_movies:
                img_rank = img_movies.index(m)
                distance = img_distances.get(m, 1.0)

                # CHỈ THƯỞNG NẾU ẢNH THỰC SỰ CHÍNH XÁC (Tránh Negative Transfer)
                # Cosine distance trong Chroma: 0.0 là hoàn hảo, > 0.65 là bắt đầu ảo giác
                if distance < 0.65:
                    print(f"    ✨ Kích hoạt Visual Boost cho '{m}' (Rank {img_rank}, Lệch {distance:.2f})")
                    if img_rank == 0:
                        rerank_score += 3.0
                    elif img_rank < 3:
                        rerank_score += 1.5
                    elif img_rank < 10:
                        rerank_score += 0.5

            final_scores.append((m, rerank_score))

        final = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return [m for m, s in final[:top_n]]

    # ==============================================================
    # CÁC HÀM TRUY XUẤT THÔ (DÙNG ĐỂ TEST ĐỘC LẬP / ABLATION STUDY)
    # Sếp nhớ lùi lề vào trong class MovieSearchEngine nhé!
    # ==============================================================

    def search_bm25_only(self, query_text, top_n=5):
        tok_q = tokenize(query_text)
        bm25_scores = self.bm25_model.get_scores(tok_q)
        movies = []
        for s, meta in sorted(zip(bm25_scores, self.bm25_meta), key=lambda x: x[0], reverse=True):
            if s <= 0: break
            m = meta['movie_name']
            if m not in movies: movies.append(m)
            if len(movies) == top_n: break
        return movies

    def search_image_only(self, query_text, top_n=5):
        vec_img = self.clip_model.encode(query_text).tolist()
        res_img = self.img_collection.query(query_embeddings=[vec_img], n_results=50)
        movies = []
        if res_img['metadatas'] and res_img['metadatas'][0]:
            for m in res_img['metadatas'][0]:
                name = m['movie_name']
                if name not in movies: movies.append(name)
                if len(movies) == top_n: break
        return movies

    def search_sbert_only(self, query_text, top_n=5):
        vec_txt = self.sbert_model.encode(query_text).tolist()
        res_txt = self.txt_sbert_collection.query(query_embeddings=[vec_txt], n_results=50)
        movies = []
        if res_txt['metadatas'] and res_txt['metadatas'][0]:
            for meta in res_txt['metadatas'][0]:
                name = meta['movie_name']
                if name not in movies: movies.append(name)
                if len(movies) == top_n: break
        return movies

    def search_clip_text_only(self, query_text, top_n=5):
        vec_txt = self.clip_model.encode(query_text).tolist()
        res_txt = self.txt_clip_collection.query(query_embeddings=[vec_txt], n_results=50)
        movies = []
        if res_txt['metadatas'] and res_txt['metadatas'][0]:
            for meta in res_txt['metadatas'][0]:
                name = meta['movie_name']
                if name not in movies: movies.append(name)
                if len(movies) == top_n: break
        return movies