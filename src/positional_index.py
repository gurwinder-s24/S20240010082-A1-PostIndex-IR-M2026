# -*- coding: utf-8 -*-
"""
Punjabi Positional Indexer
Two-tier storage setup:
  - In-memory vocabulary (RAM): term -> {df, cf, offset, size}
  - Secondary storage postings (SSD/Disk): data/postings.bin
"""

import os
import sys
import re
import pickle
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "corpus.db")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")
POSTINGS_PATH = os.path.join(DATA_DIR, "postings.bin")

# Gurmukhi punctuation, dandas (full stops), and separators
PUNJABI_DANDAS = re.compile(r"[।॥\|]+")
CLEAN_PUNCT = re.compile(r"[^\w\s\u0A00-\u0A7F]")


def tokenize(text):
    """
    Tokenizes Punjabi text into (term, word_position) tuples.
    Normalizes dandas (। , ॥ , |) and cleans punctuation while preserving Gurmukhi Unicode.
    """
    if not text:
        return []

    # Replace dandas with space
    text = PUNJABI_DANDAS.sub(" ", text)
    # Remove latin pseudo-dandas like standalone 'I' or 'l'
    text = re.sub(r"(?<=\S)[Il](?=\s|$)", "", text)
    # Strip foreign punctuation
    text = CLEAN_PUNCT.sub(" ", text)

    tokens = []
    pos = 0
    for word in text.split():
        word = word.strip().strip("'\"-—–()[]{}:;,.<>?/")
        # Remove trailing English/Roman artifacts if any
        word = re.sub(r"[Il|]+$", "", word)
        if len(word) >= 1:
            tokens.append((word, pos))
            pos += 1
    return tokens


class DiskPostingsReader:
    """
    Secondary Memory Reader:
    Maintains an open file handle to postings.bin and retrieves postings lists
    on-demand using byte offsets and lengths from Main Memory.
    """
    def __init__(self, postings_path=POSTINGS_PATH):
        self.postings_path = postings_path
        self._fp = None
        if os.path.exists(postings_path):
            self._fp = open(postings_path, "rb")

    def read_postings(self, offset, size):
        """Perform random-access seek on SSD/Disk and unpack postings list."""
        if not self._fp:
            if not os.path.exists(self.postings_path):
                return []
            self._fp = open(self.postings_path, "rb")

        self._fp.seek(offset)
        raw_bytes = self._fp.read(size)
        return pickle.loads(raw_bytes)

    def close(self):
        if self._fp and not self._fp.closed:
            self._fp.close()

    def __del__(self):
        self.close()


class PositionalIndex:
    """
    Complete Positional Index manager implementing:
      - In-Memory Vocabulary (RAM)
      - Disk Postings (Secondary Storage)
      - Positional Boolean Search
    """
    def __init__(self, vocab_path=VOCAB_PATH, postings_path=POSTINGS_PATH, db_path=DB_PATH):
        self.vocab_path = vocab_path
        self.postings_path = postings_path
        self.db_path = db_path

        # Main Memory (RAM) Storage
        # Maps term -> {"df": int, "cf": int, "offset": int, "size": int}
        self.vocabulary = {}

        # Secondary Storage Reader
        self.reader = None
        self.total_docs = 0
        self.total_postings = 0

    def load(self):
        """Loads vocabulary into RAM and binds disk postings reader."""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}. Run --build first.")
        if not os.path.exists(self.postings_path):
            raise FileNotFoundError(f"Postings file not found: {self.postings_path}. Run --build first.")

        t0 = time.time()
        with open(self.vocab_path, "rb") as f:
            data = pickle.load(f)
            self.vocabulary = data["vocabulary"]
            self.total_docs = data.get("total_docs", 0)
            self.total_postings = data.get("total_postings", 0)

        self.reader = DiskPostingsReader(self.postings_path)
        load_time = time.time() - t0
        return load_time

    def get_postings(self, term):
        """
        Retrieves the complete postings list for a term from Secondary Storage (SSD/Disk)
        using the in-memory vocabulary byte offset pointer.
        Returns: list of (doc_id, [pos1, pos2, ...])
        """
        if term not in self.vocabulary:
            return []
        meta = self.vocabulary[term]
        return self.reader.read_postings(meta["offset"], meta["size"])

    def get_stats(self):
        """Returns corpus and memory statistics."""
        vocab_bytes = os.path.getsize(self.vocab_path) if os.path.exists(self.vocab_path) else 0
        postings_bytes = os.path.getsize(self.postings_path) if os.path.exists(self.postings_path) else 0

        return {
            "total_documents": self.total_docs,
            "unique_terms": len(self.vocabulary),
            "total_postings": self.total_postings,
            "vocab_ram_bytes": vocab_bytes,
            "vocab_ram_mb": vocab_bytes / (1024 * 1024),
            "postings_disk_bytes": postings_bytes,
            "postings_disk_mb": postings_bytes / (1024 * 1024)
        }

    def get_document_meta(self, doc_id):
        """Fetches document title and snippet from SQLite corpus database."""
        if not os.path.exists(self.db_path):
            return {"id": doc_id, "title": f"Document #{doc_id}", "snippet": "", "word_count": 0}

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, title, clean_text, word_count FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            d_id, title, text, wc = row
            snippet = text[:250] + ("..." if len(text) > 250 else "")
            return {
                "id": d_id,
                "title": title,
                "snippet": snippet,
                "word_count": wc or len(text.split()),
                "full_text": text
            }
        return {"id": doc_id, "title": f"Document #{doc_id}", "snippet": "", "word_count": 0}

    def search(self, query_string, top_k=50):
        """
        Executes Boolean AND positional search across secondary memory postings.
        Returns documents containing all query terms, ranked by total term frequency,
        along with exact term positions in each document.
        """
        query_terms = [t for t, _ in tokenize(query_string)]
        if not query_terms:
            return []

        # Check if any query term is unknown in vocabulary
        for t in query_terms:
            if t not in self.vocabulary:
                return []

        # Sort terms by Document Frequency (DF) ascending for optimal intersection order
        sorted_terms = sorted(query_terms, key=lambda t: self.vocabulary[t]["df"])

        # Fetch postings list from secondary memory for first term
        first_term = sorted_terms[0]
        first_postings = self.get_postings(first_term)
        # doc_map: doc_id -> { term: [positions...] }
        doc_map = {doc_id: {first_term: pos_list} for doc_id, pos_list in first_postings}

        # Intersect with subsequent terms' postings from disk
        for term in sorted_terms[1:]:
            term_postings = self.get_postings(term)
            term_docs = {}
            for doc_id, pos_list in term_postings:
                if doc_id in doc_map:
                    term_docs[doc_id] = pos_list

            # Keep only intersecting documents
            doc_map = {d_id: doc_map[d_id] for d_id in term_docs}
            for d_id, pos_list in term_docs.items():
                doc_map[d_id][term] = pos_list

            if not doc_map:
                break

        if not doc_map:
            return []

        # Rank matched documents by Total Term Frequency
        results = []
        for doc_id, terms_data in doc_map.items():
            meta = self.get_document_meta(doc_id)
            total_freq = sum(len(positions) for positions in terms_data.values())

            matched_terms_info = {
                t: {"positions": pos_list, "count": len(pos_list)}
                for t, pos_list in terms_data.items()
            }

            results.append({
                "doc_id": doc_id,
                "title": meta["title"],
                "word_count": meta["word_count"],
                "snippet": meta["snippet"],
                "total_frequency": total_freq,
                "matched_terms": matched_terms_info
            })

        results.sort(key=lambda x: x["total_frequency"], reverse=True)
        return results[:top_k]


def build_positional_index(db_path=DB_PATH, vocab_out=VOCAB_PATH, postings_out=POSTINGS_PATH):
    """
    Builds the Two-Tier Positional Index from SQLite corpus:
      - Extracts term positions for all authentic Wikipedia documents.
      - Writes complete postings lists to Secondary Storage (postings.bin).
      - Computes DF & CF and writes in-memory vocabulary to vocab.pkl.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} not found! Run dump_parser.py first.")

    os.makedirs(os.path.dirname(vocab_out), exist_ok=True)
    os.makedirs(os.path.dirname(postings_out), exist_ok=True)

    print("=" * 70)
    print("BUILDING TWO-TIER POSITIONAL INDEX (RAM Vocabulary + Secondary Postings)")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    total_docs = cur.fetchone()[0]
    print(f"Reading {total_docs:,} authentic Punjabi Wikipedia documents from SQLite...")

    # Inverted Index In-Memory Accumulator
    # term -> dict: doc_id -> [pos1, pos2, ...]
    inverted_index = {}
    term_cf = {}  # term -> total occurrences (CF)

    cur.execute("SELECT id, clean_text FROM documents ORDER BY id")
    processed = 0
    total_tokens_indexed = 0
    t0 = time.time()

    while True:
        rows = cur.fetchmany(2000)
        if not rows:
            break
        for doc_id, text in rows:
            tokens = tokenize(text)
            total_tokens_indexed += len(tokens)
            for word, pos in tokens:
                if word not in inverted_index:
                    inverted_index[word] = {}
                    term_cf[word] = 0

                term_cf[word] += 1
                doc_dict = inverted_index[word]
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = []
                doc_dict[doc_id].append(pos)

            processed += 1
        elapsed = time.time() - t0
        print(f"  Tokenized {processed:,}/{total_docs:,} documents ({len(inverted_index):,} unique terms) in {elapsed:.1f}s...", end="\r")

    conn.close()
    print(f"\n[OK] Tokenization complete! Total terms: {len(inverted_index):,}, Total tokens: {total_tokens_indexed:,}")

    # Write Postings to Secondary Storage (postings.bin)
    print(f"\nWriting postings lists to Secondary Memory: {postings_out}...")
    vocabulary = {}
    total_postings_count = 0

    with open(postings_out, "wb") as pf:
        current_offset = 0
        for term, doc_dict in inverted_index.items():
            df = len(doc_dict)
            cf = term_cf[term]

            # Convert to sorted list of (doc_id, [pos...])
            postings_list = sorted(doc_dict.items(), key=lambda x: x[0])
            total_postings_count += df

            # Serialize binary payload
            payload = pickle.dumps(postings_list, protocol=pickle.HIGHEST_PROTOCOL)
            size = len(payload)

            pf.write(payload)

            # Store pointers in In-Memory Vocabulary
            vocabulary[term] = {
                "df": df,
                "cf": cf,
                "offset": current_offset,
                "size": size
            }
            current_offset += size

    postings_file_size = os.path.getsize(postings_out)
    print(f"[OK] Secondary storage postings written: {postings_file_size / (1024*1024):.2f} MB ({total_postings_count:,} postings)")

    # Save In-Memory Vocabulary
    print(f"Saving Main Memory vocabulary to: {vocab_out}...")
    with open(vocab_out, "wb") as vf:
        pickle.dump({
            "vocabulary": vocabulary,
            "total_docs": total_docs,
            "total_postings": total_postings_count
        }, vf, protocol=pickle.HIGHEST_PROTOCOL)

    vocab_file_size = os.path.getsize(vocab_out)
    print(f"[OK] In-memory vocabulary written: {vocab_file_size / (1024*1024):.2f} MB ({len(vocabulary):,} unique terms)")

    print("\n" + "=" * 70)
    print("[SUCCESS] TWO-TIER POSITIONAL INDEX BUILD COMPLETE!")
    print(f"  - Total Documents Indexed : {total_docs:,}")
    print(f"  - Unique Vocabulary Terms : {len(vocabulary):,}")
    print(f"  - Total Postings Count    : {total_postings_count:,}")
    print(f"  - Main Memory (RAM) Vocab : {vocab_file_size / (1024*1024):.2f} MB")
    print(f"  - Secondary Memory (Disk) : {postings_file_size / (1024*1024):.2f} MB")
    print(f"  - Total Build Duration    : {time.time() - t0:.2f} seconds")
    print("=" * 70)

    return total_docs, len(vocabulary), total_postings_count
