"""
Rule-based parsers that turn the raw extracted PDF text of each legal
document into structured Document chunks (Part / Chapter / Section
metadata + clause text).
"""

import re

from langchain_core.documents import Document


def parse_civil_code(civil_code_raw_text):
    # Split the raw document into lines
    civil_code_lines = civil_code_raw_text.split('\n')

    civil_code_chunks = []
    current_civil_part = "Unknown Part"
    current_civil_chapter = "Unknown Chapter"
    current_civil_section_title = ""
    current_civil_section_content = []

    # Regex patterns isolated for Civil Code formatting rules
    civil_part_pattern = re.compile(r'^Part\s*-\s*\d+', re.IGNORECASE)
    civil_chapter_pattern = re.compile(r'^Chapter\s*-\s*\d+', re.IGNORECASE)
    civil_section_pattern = re.compile(r'^(\d+)\.\s+(.*)')

    # Page footer regex: Matches lines that contain ONLY digits (e.g., "69", " 70 ")
    civil_page_footer_pattern = re.compile(r'^\d+$')

    def save_civil_chunk():
        if current_civil_section_content:
            civil_chunk_text = " ".join(current_civil_section_content).strip()
            # Clean up double spaces caused by joins and line breaks
            civil_chunk_text = re.sub(r'\s+', ' ', civil_chunk_text)

            civil_code_chunks.append(
                Document(
                    page_content=civil_chunk_text,
                    metadata={
                        "title": "The National Civil (Code) Act, 2017 (2074)",
                        "doc_type": "Civil Code",
                        "part": current_civil_part,
                        "chapter": current_civil_chapter,
                        "section_title": current_civil_section_title,
                    }
                )
            )

    for i, line in enumerate(civil_code_lines):
        line_clean = line.strip()

        # 1. Skip empty lines
        if not line_clean:
            continue

        # 2. PRE-PROCESSING FILTER: Strip bare page-footer numbers (e.g., "69")
        # This prevents mid-clause text from getting broken by page breaks
        if civil_page_footer_pattern.match(line_clean):
            continue

        # 3. Detect and update the current Part
        if civil_part_pattern.match(line_clean):
            save_civil_chunk()
            current_civil_part = line_clean
            if i + 1 < len(civil_code_lines) and not re.match(r'^(Chapter|Part)', civil_code_lines[i+1].strip(), re.IGNORECASE):
                current_civil_part += " - " + civil_code_lines[i+1].strip()
            continue

        # 4. Detect and update the current Chapter
        if civil_chapter_pattern.match(line_clean):
            save_civil_chunk()
            current_civil_chapter = line_clean
            if i + 1 < len(civil_code_lines) and not re.match(r'^(\d+\.|Chapter|Part)', civil_code_lines[i+1].strip(), re.IGNORECASE):
                current_civil_chapter += " - " + civil_code_lines[i+1].strip()
            continue

        # 5. Detect a new Section
        section_match = civil_section_pattern.match(line_clean)
        if section_match:
            save_civil_chunk()
            current_civil_section_title = section_match.group(0)
            current_civil_section_content = [current_civil_section_title]
            continue

        # 6. Append text to the current section (if we are inside one)
        if current_civil_section_title:
            current_civil_section_content.append(line_clean)

    # Save the final chunk in the document
    save_civil_chunk()
    return civil_code_chunks


def parse_penal_code(penal_code_raw_text):
    # Split the raw document into lines
    lines = penal_code_raw_text.split('\n')

    chunks = []

    # Initialize defaults for preamble/introductory text
    current_part = "Preamble"
    current_chapter = "Preliminary"
    current_section_number = "0"
    current_section_title = "Introduction"
    current_section_content = []

    # Regex patterns optimized for Penal Code formatting
    part_pattern = re.compile(r'^Part\s*-\s*\d+', re.IGNORECASE)
    chapter_pattern = re.compile(r'^Chapter\s*-\s*\d+', re.IGNORECASE)
    section_pattern = re.compile(r'^(\d+)\.\s+(.*)')

    # Page extraction artifacts: Isolates lines that are just numbers OR trailing numbers
    page_isolated_pattern = re.compile(r'^\s*\d+\s*$')
    page_trailing_pattern = re.compile(r'\s+\d+\s*$')

    def save_chunk():
        if current_section_content:
            chunk_text = " ".join(current_section_content).strip()
            # Clean up extra spaces
            chunk_text = re.sub(r'\s+', ' ', chunk_text)

            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "title": "The National Penal (Code) Act, 2017",
                            "doc_type": "Penal Code",
                            "part": current_part,
                            "chapter": current_chapter,
                            "section_number": current_section_number,
                            "section_title": current_section_title,
                        }
                    )
                )

    for i, line in enumerate(lines):
        line_clean = line.strip()

        # 1. Skip completely empty lines or lines that are entirely just a page number
        if not line_clean or page_isolated_pattern.match(line_clean):
            continue

        # 2. Strip trailing page numbers appended to valid text lines
        line_clean = page_trailing_pattern.sub('', line_clean)

        # 3. Detect and update the current Part
        if part_pattern.match(line_clean):
            save_chunk()
            current_part = line_clean
            # Look ahead to capture the actual Part Title (e.g., "General Provisions")
            if i + 1 < len(lines):
                next_line = page_trailing_pattern.sub('', lines[i+1].strip())
                if not re.match(r'^(Chapter|Part|\d+\.)', next_line, re.IGNORECASE):
                    current_part += " - " + next_line

            # Reset section tracking for the new part
            current_section_title = ""
            current_section_content = []
            continue

        # 4. Detect and update the current Chapter
        if chapter_pattern.match(line_clean):
            save_chunk()
            current_chapter = line_clean
            # Look ahead to capture the actual Chapter Title
            if i + 1 < len(lines):
                next_line = page_trailing_pattern.sub('', lines[i+1].strip())
                if not re.match(r'^(Chapter|Part|\d+\.)', next_line, re.IGNORECASE):
                    current_chapter += " - " + next_line

            # Reset section tracking for the new chapter
            current_section_title = ""
            current_section_content = []
            continue

        # 5. Detect a new Section
        section_match = section_pattern.match(line_clean)
        if section_match:
            save_chunk()
            current_section_number = section_match.group(1)
            current_section_title = line_clean
            current_section_content = [line_clean]
            continue

        # 6. Prevent Part/Chapter titles from being duplicated into the content body
        if current_part.endswith(line_clean) or current_chapter.endswith(line_clean):
            continue

        # 7. Append text to the current section
        current_section_content.append(line_clean)

    # Save the final section when the loop finishes
    save_chunk()
    return chunks


def parse_constitution(constitution_raw_text):
    # 1. PRE-PROCESSING: Split into lines and strip bare page-footer numbers before parsing
    raw_lines = constitution_raw_text.split('\n')
    page_footer_pattern = re.compile(r'^\s*\d+\s*$')

    # Keep only lines that are not standalone page numbers
    constitution_lines = [line for line in raw_lines if not page_footer_pattern.match(line)]

    constitution_chunks = []
    current_constitution_part = "Preamble"  # Defaults to Preamble for the document opening
    current_constitution_chapter = ""       # Kept empty unless explicitly found inside a Part
    current_constitution_article_title = ""
    current_constitution_article_content = []

    # Regex patterns adapted for Constitution formatting rules
    constitution_part_pattern = re.compile(r'^Part\s*-?\s*\d+', re.IGNORECASE)
    constitution_chapter_pattern = re.compile(r'^Chapter\s*-?\s*\d+', re.IGNORECASE)

    # Matches "Article 1. Title" OR just "1. Title"
    constitution_article_pattern = re.compile(r'^(?:Article\s+)?(\d+)\.\s+(.*)', re.IGNORECASE)

    # Matches "Schedule - 1" or "Schedule 1"
    constitution_schedule_pattern = re.compile(r'^Schedule\s*-?\s*\d+', re.IGNORECASE)

    def save_constitution_chunk():
        if current_constitution_article_content:
            constitution_chunk_text = " ".join(current_constitution_article_content).strip()
            # Clean up double spaces caused by joins and line breaks
            constitution_chunk_text = re.sub(r'\s+', ' ', constitution_chunk_text)

            constitution_chunks.append(
                Document(
                    page_content=constitution_chunk_text,
                    metadata={
                        "title": "THE CONSTITUTION OF NEPAL",
                        "doc_type": "Constitution",
                        "part": current_constitution_part,
                        "chapter": current_constitution_chapter,
                        "article_title": current_constitution_article_title,
                    }
                )
            )

    # 2. PARSING: Iterate through the cleaned lines
    for i, line in enumerate(constitution_lines):
        line_clean = line.strip()

        # Skip empty lines
        if not line_clean:
            continue

        # Detect and update Schedules (Treat them as unique Parts at the end)
        if constitution_schedule_pattern.match(line_clean):
            save_constitution_chunk()
            current_constitution_part = line_clean
            current_constitution_chapter = ""
            current_constitution_article_title = line_clean
            current_constitution_article_content = [line_clean]

            # Look ahead for the Schedule's title
            if i + 1 < len(constitution_lines) and not re.match(r'^(Schedule|Part|Article)', constitution_lines[i+1].strip(), re.IGNORECASE):
                schedule_title = constitution_lines[i+1].strip()
                current_constitution_part += " - " + schedule_title
                current_constitution_article_content.append(schedule_title)
            continue

        # Detect and update the current Part
        if constitution_part_pattern.match(line_clean):
            save_constitution_chunk()
            current_constitution_part = line_clean
            current_constitution_chapter = ""  # Reset chapter when a new part begins

            # Look ahead for the Part's title
            if i + 1 < len(constitution_lines) and not re.match(r'^(Chapter|Part|Article|\d+\.)', constitution_lines[i+1].strip(), re.IGNORECASE):
                current_constitution_part += " - " + constitution_lines[i+1].strip()
            continue

        # Detect and update the current Chapter (If applicable)
        if constitution_chapter_pattern.match(line_clean):
            # We don't save chunk here because chapters wrap articles, we just update the metadata
            current_constitution_chapter = line_clean
            if i + 1 < len(constitution_lines) and not re.match(r'^(\d+\.|Article|Chapter|Part)', constitution_lines[i+1].strip(), re.IGNORECASE):
                current_constitution_chapter += " - " + constitution_lines[i+1].strip()
            continue

        # Detect a new Article
        article_match = constitution_article_pattern.match(line_clean)
        if article_match:
            save_constitution_chunk()
            current_constitution_article_title = line_clean
            current_constitution_article_content = [current_constitution_article_title]
            continue

        # Initial Preamble edge-case handling (Before any article is explicitly declared)
        if not current_constitution_article_title and current_constitution_part == "Preamble":
            if line_clean.lower() == "preamble":
                current_constitution_article_title = "Preamble"
                current_constitution_article_content = [line_clean]
            else:
                if not current_constitution_article_content:
                    current_constitution_article_title = "Preamble"
                current_constitution_article_content.append(line_clean)
            continue

        # Append text to the current article (if we are inside one)
        if current_constitution_article_title:
            current_constitution_article_content.append(line_clean)

    # Save the final chunk in the document
    save_constitution_chunk()

    return constitution_chunks
