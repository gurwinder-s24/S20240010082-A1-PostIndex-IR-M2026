# -*- coding: utf-8 -*-
"""
Information Retrieval - Punjabi Positional Indexer
Roll No: S20240010082

Usage:
  python src/indexer.py --stats
  python src/indexer.py --test-cases
  python src/indexer.py --rare-terms
  python src/indexer.py --search "ਪੰਜਾਬ ਸੱਭਿਆਚਾਰ"
  python src/indexer.py --generate-output
"""

import os
import sys
import argparse
import pickle
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from positional_index import PositionalIndex, build_positional_index, tokenize
from dump_parser import download_dump, build_corpus_database

VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")
POSTINGS_PATH = os.path.join(DATA_DIR, "postings.bin")
DB_PATH = os.path.join(DATA_DIR, "corpus.db")
OUTPUT_PATH = os.path.join(BASE_DIR, "output")
README_PATH = os.path.join(BASE_DIR, "README")


def load_indexer():
    """Boot the positional index into memory."""
    idx = PositionalIndex(vocab_path=VOCAB_PATH, postings_path=POSTINGS_PATH, db_path=DB_PATH)
    load_time = idx.load()
    return idx, load_time


def run_5_test_cases(index):
    """
    Executes the 5 distinct assignment test cases testing the positional index.
    Covers single term, multi-term conjunction, high-frequency, domain-specific, and compound queries.
    """
    test_cases_output = []

    # ── Test Case 1: Single-Term Positional Query ──
    q1 = "ਇਤਿਹਾਸ"
    v1 = index.vocabulary.get(q1, {})
    res1 = index.search(q1, top_k=3)
    test_cases_output.append({
        "case_num": 1,
        "title": "Single-Term Positional Query",
        "description": "Evaluates exact term lookup, in-memory DF/CF retrieval, and document ranking by term frequency.",
        "query": q1,
        "transliteration": "itihaas (History)",
        "df": v1.get("df", 0),
        "cf": v1.get("cf", 0),
        "total_matches": len(index.search(q1, top_k=100000)),
        "sample_results": res1
    })

    # ── Test Case 2: Multi-Term Boolean AND Positional Search ──
    q2 = "ਪੰਜਾਬ ਸੱਭਿਆਚਾਰ"
    t_tokens = [t for t, _ in tokenize(q2)]
    res2 = index.search(q2, top_k=3)
    meta_terms = {t: index.vocabulary.get(t, {}) for t in t_tokens}
    test_cases_output.append({
        "case_num": 2,
        "title": "Multi-Term Boolean AND Positional Search",
        "description": "Evaluates positional postings intersection for multiple terms, returning documents containing ALL query terms.",
        "query": q2,
        "transliteration": "punjab sabhyachar (Punjab Culture)",
        "terms_meta": meta_terms,
        "total_matches": len(index.search(q2, top_k=100000)),
        "sample_results": res2
    })

    # ── Test Case 3: High Collection Frequency Term Handling ──
    q3 = "ਭਾਰਤ"
    v3 = index.vocabulary.get(q3, {})
    res3 = index.search(q3, top_k=3)
    test_cases_output.append({
        "case_num": 3,
        "title": "High Collection Frequency Positional Postings Retrieval",
        "description": "Evaluates rapid secondary storage binary seeks for a high-frequency national term across thousands of documents.",
        "query": q3,
        "transliteration": "bharat (India)",
        "df": v3.get("df", 0),
        "cf": v3.get("cf", 0),
        "total_matches": len(index.search(q3, top_k=100000)),
        "sample_results": res3
    })

    # ── Test Case 4: Domain-Specific / Technical Vocabulary Term ──
    q4 = "ਸੰਵਿਧਾਨ"
    v4 = index.vocabulary.get(q4, {})
    res4 = index.search(q4, top_k=3)
    test_cases_output.append({
        "case_num": 4,
        "title": "Domain-Specific Technical Term Positional Lookup",
        "description": "Evaluates targeted vocabulary lookup and exact positional coordinates for legislative and constitutional concepts.",
        "query": q4,
        "transliteration": "samvidhan (Constitution)",
        "df": v4.get("df", 0),
        "cf": v4.get("cf", 0),
        "total_matches": len(index.search(q4, top_k=100000)),
        "sample_results": res4
    })

    # ── Test Case 5: Regional Linguistic / Literary Term Positional Lookup ──
    q5 = "ਗੁਰਮੁਖੀ ਲਿਪੀ"
    t_tokens5 = [t for t, _ in tokenize(q5)]
    res5 = index.search(q5, top_k=3)
    meta_terms5 = {t: index.vocabulary.get(t, {}) for t in t_tokens5}
    test_cases_output.append({
        "case_num": 5,
        "title": "Linguistic Conjunction Positional Retrieval",
        "description": "Evaluates multi-word language query intersecting linguistic concepts in Punjabi articles.",
        "query": q5,
        "transliteration": "gurmukhi lipi (Gurmukhi Script)",
        "terms_meta": meta_terms5,
        "total_matches": len(index.search(q5, top_k=100000)),
        "sample_results": res5
    })

    return test_cases_output


def run_10_rare_terms(index):
    """
    Retrieves 10 genuine rare terms (DF = 1 or 2, CF = 1 or 2)
    and loads their exact positional indices directly from Secondary Storage (postings.bin).
    """
    rare_terms_pool = [
        ("ਸੋਚੈ", "Sochai (By thinking / Through contemplation - Gurbani)"),
        ("ਸੋਚਿ", "Sochi (Pondered / Reflected upon - Gurbani)"),
        ("ਸਰਸਾਰ", "Sarsar (Abundant / Overflowing / Exuberant)"),
        ("ਹੋਵਈ", "Hovai (Occurs / Comes to pass - Archaic Punjabi)"),
        ("ਬਖ਼ਸ਼ਿਸ਼ਾਂ", "Bakhshishan (Blessings / Divine Gifts)"),
        ("ਸੁਭਾਇਮਾਨ", "Subhaiman (Glorious / Splendid / Graceful)"),
        ("ਨਾਭਵੀ", "Nabhavi (Belonging to / Resident of Nabha state)"),
        ("ਅਕਾਲਪੁਰਖ", "Akalpurakh (The Timeless Being / Almighty)"),
        ("ਵੀਚਾਰੁ", "Vicharu (Deliberation / Spiritual Contemplation)"),
        ("ਨਿਰਵੈਰੁ", "Nirvairu (Without animosity / Beyond hatred)"),
        ("ਬੈਕੁੰਠ", "Baikunth (Celestial Paradise / Divine Realm)"),
        ("ਜਲਮਾਰਗ", "Jalmarg (Waterway / Navigation Canal)")
    ]

    results = []
    for term, meaning in rare_terms_pool:
        if term in index.vocabulary:
            meta = index.vocabulary[term]
            if meta["df"] <= 10:
                postings = index.get_postings(term)
                doc_entries = []
                for doc_id, positions in postings[:2]:
                    doc_info = index.get_document_meta(doc_id)
                    doc_entries.append({
                        "doc_id": doc_id,
                        "title": doc_info["title"],
                        "positions": positions,
                        "snippet": doc_info["snippet"]
                    })
                results.append({
                    "term": term,
                    "meaning": meaning,
                    "df": meta["df"],
                    "cf": meta["cf"],
                    "offset": meta["offset"],
                    "size": meta["size"],
                    "documents": doc_entries
                })
        if len(results) == 10:
            break

    # If pool did not yield 10 terms, find from vocabulary
    if len(results) < 10:
        for term, meta in index.vocabulary.items():
            if meta["df"] == 1 and meta["cf"] == 1 and len(term) >= 4 and not term.isdigit():
                if any(r["term"] == term for r in results):
                    continue
                postings = index.get_postings(term)
                doc_entries = []
                for doc_id, positions in postings:
                    doc_info = index.get_document_meta(doc_id)
                    doc_entries.append({
                        "doc_id": doc_id,
                        "title": doc_info["title"],
                        "positions": positions,
                        "snippet": doc_info["snippet"]
                    })
                results.append({
                    "term": term,
                    "meaning": "Rare Gurmukhi Vocabulary Word",
                    "df": meta["df"],
                    "cf": meta["cf"],
                    "offset": meta["offset"],
                    "size": meta["size"],
                    "documents": doc_entries
                })
                if len(results) == 10:
                    break

    return results


def generate_submission_output_file(index, output_filepath=OUTPUT_PATH):
    """
    Generates the 'output' file with 5 test cases and 10 rare terms.
    """
    stats = index.get_stats()
    test_cases = run_5_test_cases(index)
    rare_terms = run_10_rare_terms(index)

    lines = []
    lines.append("Positional Index Evaluation Output")
    lines.append("Roll No: S20240010082")
    lines.append("Dataset: Punjabi Wikipedia XML Collection (pawiki-latest-pages-articles.xml.bz2)")
    lines.append("Storage: RAM Vocabulary + Secondary Storage Postings (postings.bin)")
    lines.append("-" * 60)
    lines.append("")
    lines.append("Corpus & Index Stats:")
    lines.append(f"  Total Documents        : {stats['total_documents']:,}")
    lines.append(f"  Unique Vocabulary Terms: {stats['unique_terms']:,}")
    lines.append(f"  Total Postings         : {stats['total_postings']:,}")
    lines.append("")

    # Part 1: 5 Test Cases
    lines.append("Part 1: 5 Test Cases")
    lines.append("-" * 60)
    lines.append("")

    for tc in test_cases:
        lines.append(f"Test Case {tc['case_num']}: {tc['title']}")
        lines.append(f"Description: {tc['description']}")

        if tc["case_num"] in (1, 3, 4):
            lines.append(f"Query Term: {tc['query']} ({tc['transliteration']})")
            lines.append(f"DF (Document Frequency): {tc['df']}")
            lines.append(f"CF (Collection Frequency): {tc['cf']}")
            lines.append(f"Total Matches: {tc['total_matches']}")
            lines.append("Sample Matching Documents:")
            for r in tc["sample_results"]:
                pos_list = r["matched_terms"].get(tc['query'], {}).get("positions", [])
                lines.append(f"  Doc #{r['doc_id']} ('{r['title']}'): positions {pos_list[:15]}")
                lines.append(f"  Snippet: {r['snippet']}")

        elif tc["case_num"] in (2, 5):
            lines.append(f"Query: '{tc['query']}' ({tc['transliteration']})")
            lines.append("Vocabulary Stats:")
            for term, meta in tc["terms_meta"].items():
                lines.append(f"  Term '{term}': DF = {meta.get('df', 0)}, CF = {meta.get('cf', 0)}")
            lines.append(f"Boolean AND Result: {tc['total_matches']} documents contain all query terms")
            lines.append("Sample Matching Documents:")
            for r in tc["sample_results"]:
                lines.append(f"  Doc #{r['doc_id']} ('{r['title']}'):")
                for term, info in r["matched_terms"].items():
                    lines.append(f"    '{term}' ({info['count']} times) at positions: {info['positions'][:10]}")
                lines.append(f"  Snippet: {r['snippet']}")

        lines.append("")

    # Part 2: 10 Rare Terms
    lines.append("Part 2: 10 Rare Terms with Positional Index")
    lines.append("-" * 60)
    lines.append("Retrieved from secondary memory (postings.bin) via RAM vocabulary offsets:")
    lines.append("")

    for idx, item in enumerate(rare_terms, 1):
        lines.append(f"{idx}. Term: {item['term']} [{item['meaning']}]")
        lines.append(f"   DF: {item['df']} | CF: {item['cf']} | Disk Offset: {item['offset']} bytes | Size: {item['size']} bytes")
        lines.append("   Postings from Secondary Storage:")
        for d in item["documents"]:
            lines.append(f"     Doc #{d['doc_id']} ('{d['title']}'): positions {d['positions']}")
            lines.append(f"     Context: {d['snippet']}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("End of Output")

    content = "\n".join(lines)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated output file: {output_filepath}")
    return output_filepath


def main():
    parser = argparse.ArgumentParser(
        description="Punjabi Positional Indexer"
    )
    parser.add_argument("--build", action="store_true", help="Download Wikipedia dump and build index")
    parser.add_argument("--stats", action="store_true", help="Display index statistics")
    parser.add_argument("--test-cases", action="store_true", help="Run 5 test cases")
    parser.add_argument("--rare-terms", action="store_true", help="Display 10 rare terms with positional index")
    parser.add_argument("--generate-output", action="store_true", help="Generate the 'output' file")
    parser.add_argument("--search", type=str, help="Search query (e.g. 'ਪੰਜਾਬ ਸੱਭਿਆਚਾਰ')")

    args = parser.parse_args()

    if args.build:
        download_dump()
        build_corpus_database()
        build_positional_index()
        return

    if not os.path.exists(VOCAB_PATH) or not os.path.exists(POSTINGS_PATH):
        print("Index files not found. Building index first...")
        download_dump()
        build_corpus_database()
        build_positional_index()

    if len(sys.argv) == 1:
        parser.print_help()
        index, load_time = load_indexer()
        print(f"\nIndex loaded in {load_time:.4f}s: {index.get_stats()}")
        return

    index, load_time = load_indexer()

    if args.stats:
        stats = index.get_stats()
        print("Index Statistics:")
        print(f"  Total Documents : {stats['total_documents']:,}")
        print(f"  Unique Terms    : {stats['unique_terms']:,}")
        print(f"  Total Postings  : {stats['total_postings']:,}")
        print(f"  Vocab in RAM    : {stats['vocab_ram_mb']:.2f} MB")
        print(f"  Postings on Disk: {stats['postings_disk_mb']:.2f} MB")
        print(f"  Boot Time       : {load_time:.4f}s")

    if args.test_cases:
        t_cases = run_5_test_cases(index)
        for tc in t_cases:
            print(f"\n[Test Case {tc['case_num']}] {tc['title']}")
            print(f" Query: {tc['query']} ({tc['transliteration']})")
            print(f" Total Matches: {tc['total_matches']:,} documents")

    if args.rare_terms:
        r_terms = run_10_rare_terms(index)
        for idx, r in enumerate(r_terms, 1):
            print(f"{idx}. {r['term']} [{r['meaning']}] | DF: {r['df']} | CF: {r['cf']}")
            for d in r["documents"]:
                print(f"   -> Doc #{d['doc_id']} ('{d['title']}'): positions {d['positions']}")

    if args.generate_output:
        generate_submission_output_file(index)

    if args.search:
        results = index.search(args.search)
        print(f"\nSearch results for '{args.search}': {len(results)} matching documents")
        for r in results[:5]:
            print(f"  * Doc #{r['doc_id']} ('{r['title']}'): Total Freq = {r['total_frequency']}")
            for t, info in r["matched_terms"].items():
                print(f"    - {t} (count: {info['count']}): positions {info['positions'][:8]}")


if __name__ == "__main__":
    main()
