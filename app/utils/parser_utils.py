from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
import json

# Build a reusable StructuredOutputParser instance
schemas = [
    ResponseSchema(name="answer", description="Concise, direct answer."),
    ResponseSchema(name="found", description="Boolean: true if answer is found in the docs."),
    ResponseSchema(name="citations", description="List of citations: [{source, page, excerpt, score(optional)}]."),
    ResponseSchema(name="follow_up", description="Optional follow-up question or 'None'.")
]

PARSER = StructuredOutputParser.from_response_schemas(schemas)
FORMAT_INSTRUCTIONS = PARSER.get_format_instructions()


def try_parse_structured(raw_text: str):
    """Attempt to parse model output using StructuredOutputParser. Return dict on success.
    Raises on failure.
    """
    return PARSER.parse(raw_text)


def fallback_extract(raw_text: str):
    """Fallback parser: extract bracketed citations and first paragraph as answer.

    Returns dict with keys: answer, found (bool), citations (list of dict), follow_up (None)
    """
    # 1) Try explicit bracketed citations like: [paper.pdf, p.3]
    citation_pattern = r"\[([^,\]]+),\s*p\.?\s*(\d+)\]"
    matches = re.findall(citation_pattern, raw_text)
    citations = []
    for m in matches:
        citations.append({"source": m[0].strip(), "page": int(m[1]), "excerpt": ""})

    # 2) If no bracketed matches, try parsing JSON that might contain citations
    if not citations:
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict) and "citations" in obj:
                raw_cits = obj.get("citations") or []
                if isinstance(raw_cits, dict):
                    raw_cits = [raw_cits]
                for it in raw_cits:
                    if isinstance(it, dict):
                        citations.append({
                            "source": it.get("source") or it.get("file") or it.get("filename"),
                            "page": it.get("page") or it.get("p"),
                            "excerpt": it.get("excerpt", "")
                        })
        except Exception:
            pass

    # 3) If still no citations, try to heuristically extract patterns or treat lines as filenames
    if not citations:
        # simple heuristic for lines that look like 'filename.pdf p.3' or 'filename.pdf:3'
        line_pattern = r"([^\s,:]+\.pdf)[\s,:]*p\.?\s*(\d+)"
        line_matches = re.findall(line_pattern, raw_text, flags=re.IGNORECASE)
        for lm in line_matches:
            citations.append({"source": lm[0].strip(), "page": int(lm[1]), "excerpt": ""})

    # 4) As a last resort, if we still have nothing, and raw_text contains bracket-like substrings, try to salvage
    if not citations:
        # find any token that ends with .pdf
        pdf_tokens = re.findall(r"([A-Za-z0-9_\-./]+\.pdf)", raw_text)
        for t in pdf_tokens:
            citations.append({"source": t.strip(), "page": None, "excerpt": ""})

    # get first non-empty paragraph as the answer
    parts = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    answer = parts[0] if parts else raw_text.strip()

    return {"answer": answer, "found": bool(citations), "citations": citations, "follow_up": None}