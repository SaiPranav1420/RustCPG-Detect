"""
embeddings.py — GraphCodeBERT Embedding Extractor for RustCPG-Detect
Encodes BasicBlock IR text into 768-dim semantic vectors.

Usage:
    from src.embeddings import BERTEmbedder
    embedder = BERTEmbedder(device='cuda')
    embedding = embedder.encode(basic_block.text)   # → np.ndarray (768,)
    embeddings = embedder.encode_batch(texts)        # → np.ndarray (N, 768)
"""

import numpy as np
import torch


class BERTEmbedder:
    """
    Wraps microsoft/graphcodebert-base for IR text encoding.
    Mean-pools across token embeddings to produce one 768-dim vector per block.

    Why GraphCodeBERT over regular BERT?
    - Pre-trained on source code (not natural language)
    - Data-flow aware: understands variable references and def-use chains
    - IR-aware tokenizer handles LLVM IR tokens correctly
    """

    MODEL_NAME = 'microsoft/graphcodebert-base'
    EMBED_DIM  = 768

    def __init__(self, device: str = None, max_length: int = 512):
        """
        Args:
            device     : 'cuda', 'cpu', or None (auto-detect)
            max_length : max tokens per block (BERT limit = 512)
        """
        from transformers import AutoTokenizer, AutoModel

        self.device     = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        print(f"Loading {self.MODEL_NAME} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        print(f"Model loaded ✅  ({self.EMBED_DIM}-dim embeddings)")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single BasicBlock's IR text.

        Args:
            text : raw IR text (newline-separated instructions)
        Returns:
            np.ndarray of shape (768,)
        """
        inputs = self.tokenizer(
            text,
            return_tensors  = 'pt',
            max_length      = self.max_length,
            truncation      = True,
            padding         = 'max_length'
        ).to(self.device)

        with torch.no_grad():
            outputs          = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state         # [1, seq, 768]
            attention_mask   = inputs['attention_mask'].unsqueeze(-1)  # [1, seq, 1]
            # Mean pool — average only non-padding tokens
            mean_emb = (token_embeddings * attention_mask).sum(1) \
                       / attention_mask.sum(1).clamp(min=1e-8)   # [1, 768]

        return mean_emb.squeeze().cpu().numpy().astype(np.float32)

    def encode_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of BasicBlock texts in batches.

        Args:
            texts      : list of IR text strings
            batch_size : number of texts per GPU batch
        Returns:
            np.ndarray of shape (N, 768)
        """
        from tqdm.auto import tqdm

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc='Encoding blocks', unit='batch'):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors  = 'pt',
                max_length      = self.max_length,
                truncation      = True,
                padding         = True
            ).to(self.device)

            with torch.no_grad():
                outputs          = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask   = inputs['attention_mask'].unsqueeze(-1)
                mean_emb = (token_embeddings * attention_mask).sum(1) \
                           / attention_mask.sum(1).clamp(min=1e-8)

            all_embeddings.append(mean_emb.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)   # [N, 768]

    def encode_function(self, function) -> list:
        """
        Encode all BasicBlocks of a Function object.

        Args:
            function : parsed Function object (from src.parser)
        Returns:
            list of np.ndarray (768,), one per BasicBlock
        """
        texts = [bb.text for bb in function.basic_blocks]
        if not texts:
            return []
        embeddings = self.encode_batch(texts)
        return [embeddings[i] for i in range(len(texts))]
