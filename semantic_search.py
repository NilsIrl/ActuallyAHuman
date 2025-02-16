from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, documents):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        doc_embs = self.model.encode(documents, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_emb, doc_embs)
        scores = cosine_scores.tolist()[0]
        results = list(zip(documents, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results