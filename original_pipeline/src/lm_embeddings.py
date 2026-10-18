"""Explicitly chunked LM Studio embeddings with local pinned tokenizers."""
import hashlib,json,pathlib
import numpy as np
from embedding_tokenizers import EmbeddingTokenizer, pool_chunk_vectors
from local_inference import ROOT,digest,request

REVISIONS={'mx':'b33106f585b9ce46904ad7443a3b52b7a63e231c','nomic':'e9b6763023c676ca8431644204f50c2b100d9aab'}
MODELS={'mx':'vtse-mx','nomic':'vtse-nomic'}
PREFIX={'mx':{'query':'Represent this sentence for searching relevant passages: ','document':''},
        'nomic':{'query':'search_query: ','document':'search_document: '}}

class LMEmbedding:
    def __init__(self,key):
        self.key=key;self.model=MODELS[key]
        self.tokenizer=EmbeddingTokenizer(key)
        self.last_metadata=[]
    def encode(self,texts,role):
        vectors=[];metadata=[]
        for text in texts:
            chunks=self.tokenizer.chunks(text,'code' if role=='query' else 'document')
            if not chunks:
                vectors.append(np.zeros(1024 if self.key=='mx' else 768));metadata.append({'empty':True});continue
            payload={'model':self.model,'input':[c['text'] for c in chunks],'encoding_format':'float'}
            result=request('/v1/embeddings',payload,ROOT/'results/embedding_cache'/self.key/(digest(payload)+'.json'))
            data=sorted(result['response']['data'],key=lambda r:r['index'])
            if len(data)!=len(chunks):raise RuntimeError('Embedding response count mismatch')
            if [r['index'] for r in data]!=list(range(len(chunks))):
                raise RuntimeError('Embedding response indices are missing or duplicated')
            vector=pool_chunk_vectors([r['embedding'] for r in data],chunks,self.key)
            vectors.append(vector)
            metadata.append({'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'role':role,
                'content_tokens':self.tokenizer.count(text,False),
                'chunk_content_tokens':[c['content_tokens'] for c in chunks],
                'hf_payload_tokens':[c['payload_tokens_with_specials'] for c in chunks],
                'backend_usage':result['response'].get('usage'),'cache_key':digest(payload)})
        self.last_metadata=metadata
        return np.asarray(vectors,dtype=np.float64)
