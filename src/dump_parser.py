# -*- coding: utf-8 -*-
"""
Punjabi Wikipedia XML Dump Parser
Streams and extracts authentic articles from pawiki-latest-pages-articles.xml.bz2
and stores clean text into SQLite (data/corpus.db).
"""

import os
import sys
import re
import bz2
import time
import urllib.request
import sqlite3
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DUMP_URL = "https://dumps.wikimedia.org/pawiki/latest/pawiki-latest-pages-articles.xml.bz2"
DUMP_PATH = os.path.join(DATA_DIR, "pawiki-latest-pages-articles.xml.bz2")
DB_PATH = os.path.join(DATA_DIR, "corpus.db")


def download_dump(url=DUMP_URL, dest=DUMP_PATH):
    """Download the official Wikipedia dump if not already present."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 10 * 1024 * 1024:
        print(f"[OK] Official Wikipedia dump found at: {dest} ({os.path.getsize(dest)/(1024*1024):.2f} MB)")
        return dest

    print(f"Downloading official Punjabi Wikipedia dump from:\n  {url}")
    print(f"Destination: {dest}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AcademicIRResearcher/1.0 (academic; student@iiits.in)"}
    )
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        t0 = time.time()
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            if total > 0:
                pct = (downloaded / total) * 100
                print(f"\rDownloaded {downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s", end="")
            else:
                print(f"\rDownloaded {downloaded/(1024*1024):.1f} MB at {speed:.2f} MB/s", end="")
    print(f"\n[OK] Dump download complete: {dest}")
    return dest


def clean_wikitext(text):
    """
    Clean MediaWiki markup to extract plain readable text.
    Handles templates, citations, internal links, categories, and HTML.
    """
    if not text:
        return ""

    # Remove XML/HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove citations / references
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/>]*/>", "", text)
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove templates {{...}} (up to 3 levels of nesting)
    for _ in range(3):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text, flags=re.DOTALL)
    # Remove File, Image, Category links
    text = re.sub(r"\[\[(?:Category|ਸ਼੍ਰੇਣੀ|ਤਸਵੀਰ|File|Image):.*?\]\]", "", text, flags=re.IGNORECASE)
    # Extract piped links [[Target|Display Text]] -> Display Text
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Remove external links [http... label] -> label
    text = re.sub(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]", r"\1", text)
    # Remove headers (== Header ==)
    text = re.sub(r"=+\s*(.*?)\s*=+", r"\1", text)
    # Remove bold & italics markup
    text = text.replace("'''", "").replace("''", "")
    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_wikipedia_dump(dump_path=DUMP_PATH, max_articles=None):
    """
    Iteratively streams and parses the Wikipedia XML bz2 dump.
    Yields (page_id, title, clean_text) for Namespace 0 (main content), non-redirect articles.
    """
    if not os.path.exists(dump_path):
        download_dump(dest=dump_path)

    print(f"Parsing official Punjabi Wikipedia dump: {dump_path}...")
    article_count = 0
    total_pages = 0
    t0 = time.time()

    with bz2.open(dump_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag.split("}")[-1]
            if tag == "page":
                total_pages += 1
                ns = elem.find("{http://www.mediawiki.org/xml/export-0.11/}ns")
                redirect = elem.find("{http://www.mediawiki.org/xml/export-0.11/}redirect")

                # Filter strictly: Namespace 0 (Articles) & NOT a redirect
                if ns is not None and ns.text == "0" and redirect is None:
                    title_elem = elem.find("{http://www.mediawiki.org/xml/export-0.11/}title")
                    id_elem = elem.find("{http://www.mediawiki.org/xml/export-0.11/}id")
                    revision = elem.find("{http://www.mediawiki.org/xml/export-0.11/}revision")
                    text_elem = revision.find("{http://www.mediawiki.org/xml/export-0.11/}text") if revision is not None else None

                    title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else ""
                    page_id = int(id_elem.text) if (id_elem is not None and id_elem.text) else article_count + 1
                    raw_text = text_elem.text if (text_elem is not None and text_elem.text) else ""

                    clean_text = clean_wikitext(raw_text)

                    # Only keep substantive articles (> 30 characters)
                    if len(clean_text) >= 30 and title:
                        article_count += 1
                        yield (article_count, page_id, title, clean_text)

                        if max_articles and article_count >= max_articles:
                            elem.clear()
                            break

                elem.clear()

                if total_pages % 20000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Processed {total_pages:,} pages ({article_count:,} content articles extracted) in {elapsed:.1f}s...")

    print(f"[OK] Completed dump parse in {time.time() - t0:.2f}s: {article_count:,} authentic articles.")


def build_corpus_database(db_path=DB_PATH, max_articles=None):
    """
    Parses the Wikipedia XML dump and stores authentic articles into a high-performance SQLite database.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            page_id INTEGER,
            title TEXT NOT NULL,
            clean_text TEXT NOT NULL,
            word_count INTEGER
        )
    """)
    conn.commit()

    print(f"\nBuilding SQLite Corpus database at: {db_path}...")
    batch = []
    batch_size = 2000
    total_words = 0
    t0 = time.time()

    for doc_id, page_id, title, clean_text in parse_wikipedia_dump(max_articles=max_articles):
        wc = len(clean_text.split())
        total_words += wc
        batch.append((doc_id, page_id, title, clean_text, wc))

        if len(batch) >= batch_size:
            cur.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", batch)
            conn.commit()
            print(f"  Stored {doc_id:,} articles ({total_words:,} words)...", end="\r")
            batch = []

    if batch:
        cur.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", batch)
        conn.commit()

    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(id)")
    conn.commit()

    cur.execute("SELECT COUNT(*), SUM(word_count) FROM documents")
    total_docs, total_w = cur.fetchone()
    conn.close()

    print(f"\n[SUCCESS] Corpus database ready!")
    print(f"  - Total Authentic Wikipedia Articles: {total_docs:,}")
    print(f"  - Total Words in Collection          : {total_w:,}")
    print(f"  - Database Size                      : {os.path.getsize(db_path)/(1024*1024):.2f} MB")
    print(f"  - Total Elapsed Time                 : {time.time() - t0:.2f} seconds")
    return total_docs


if __name__ == "__main__":
    download_dump()
    build_corpus_database()

