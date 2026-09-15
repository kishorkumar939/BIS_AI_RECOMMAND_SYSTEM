"""
FastAPI application for the BIS Standards Recommendation Engine.

Endpoints:
  POST /api/v1/recommend  — Used by the Chrome extension & dashboard.
  POST /api/v1/audit-pdf  — Used by the React dashboard for reverse auditing.
  GET  /api/v1/health     — Health check.
"""

import re
import io
import sys
from pathlib import Path
from typing import List, Optional

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# SQLAlchemy
from sqlalchemy.orm import Session
from sqlalchemy import select

# PyMuPDF for PDF text extraction
import pymupdf as fitz  # PyMuPDF

from database import get_db, Base, engine
from database.models import Standard, QCOAlert, SimplifiedProcedure754
from pipeline.rag_engine import get_rag_engine

# Create tables on startup (for SQLite dev; use Alembic migrations for production)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="BIS Standards Recommendation Engine",
    description="AI-powered mapping of procurement requirements to Indian Standards.",
    version="1.0.0",
)

# CORS: allow the React dashboard and the Chrome extension origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(..., description="Natural-language procurement requirement.")
    top_k: int = Field(5, description="Number of recommendations to return.")
    generate_clause: bool = Field(
        True, description="Whether to synthesize a legal compliance clause."
    )


class NormativeRef(BaseModel):
    is_code: str
    title: str


class RecommendItem(BaseModel):
    is_code: str
    title: str
    score: float
    scope: str
    category: str
    qco_mandatory: bool
    crs_applicable: bool
    simplified_procedure: bool
    fast_track_days: Optional[int] = None
    normative_refs: List[NormativeRef] = []


class LegalCitation(BaseModel):
    source_pdf: str
    act_or_regulation: str
    provision: str
    excerpt: str
    applicability: str


class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem]
    compliance_clause: Optional[str] = None
    legal_framework: List[LegalCitation] = []


class AuditPDFResponse(BaseModel):
    extracted_is_codes: List[str]
    flagged_outdated: List[dict]
    missing_qco: List[dict]
    summary: str


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "service": "bis-recommendation-engine"}


@app.post("/api/v1/recommend", response_model=RecommendResponse)
async def recommend(
    req: RecommendRequest,
    db: Session = Depends(get_db),
):
    """
    Semantic search → merge relational flags (QCO, Simplified Procedure) →
    retrieve legal framework from BIS Acts / Regulations PDFs →
    synthesize a statutory tender compliance clause.
    """
    rag = get_rag_engine()
    recs = rag.recommend(req.query)[: req.top_k]

    items: List[RecommendItem] = []
    for rec in recs:
        # Enrich with relational data from PostgreSQL/SQLite
        stmt = select(Standard).where(Standard.is_code == rec.is_code)
        std = db.execute(stmt).scalar_one_or_none()

        qco_mandatory = rec.qco_mandatory
        crs_applicable = rec.crs_applicable
        fast_track_days: Optional[int] = None
        simplified = rec.simplified_procedure

        if std:
            if std.qco_alert:
                qco_mandatory = std.qco_alert.is_mandatory
                crs_applicable = std.qco_alert.crs_applicable
            if std.simplified_procedure and std.simplified_procedure.option2_eligible:
                simplified = True
                fast_track_days = std.simplified_procedure.fast_track_days

            # Fetch normative references from relational M2M
            refs = [
                NormativeRef(is_code=r.is_code, title=r.title)
                for r in std.normative_refs
            ]
        else:
            refs = [NormativeRef(**r) for r in rec.normative_refs]

        items.append(
            RecommendItem(
                is_code=rec.is_code,
                title=rec.title,
                score=round(rec.score, 4),
                scope=rec.scope,
                category=rec.category,
                qco_mandatory=qco_mandatory,
                crs_applicable=crs_applicable,
                simplified_procedure=simplified,
                fast_track_days=fast_track_days,
                normative_refs=refs,
            )
        )

    clause = None
    if req.generate_clause and items:
        clause = rag.synthesize_compliance_clause(req.query, recs[: req.top_k])

    # Retrieve applicable statutory rules from the 25 BIS PDFs
    raw_legal = rag.get_legal_framework(req.query, recs[: req.top_k])
    legal_citations = [LegalCitation(**l) for l in raw_legal]

    return RecommendResponse(
        recommendations=items,
        compliance_clause=clause,
        legal_framework=legal_citations,
    )


@app.post("/api/v1/audit-pdf", response_model=AuditPDFResponse)
async def audit_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Reverse Audit: extract text from an uploaded legacy tender PDF,
    identify all IS code references, and flag outdated / superseded
    standards or missing QCO flags.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Regex to find full IS code patterns: IS 1239, IS 1239 (Part 1), IS:2062, etc.
    is_pattern = re.compile(r"\bIS[:\s\-]*\d{2,6}(?:\s*\(?Part\s*\d+\)?)?", re.IGNORECASE)
    raw_matches = [m.group(0).strip() for m in is_pattern.finditer(full_text)]
    
    # Normalize: "IS: 1239" -> "IS 1239"
    found_codes = list(dict.fromkeys(
        re.sub(r"^IS[:\s\-]*", "IS ", m, flags=re.IGNORECASE) for m in raw_matches
    ))

    flagged_outdated: List[dict] = []
    missing_qco: List[dict] = []

    for code in found_codes:
        stmt = select(Standard).where(Standard.is_code == code)
        std = db.execute(stmt).scalar_one_or_none()

        if std:
            if std.status in ("Withdrawn", "Under Revision"):
                flagged_outdated.append(
                    {
                        "is_code": std.is_code,
                        "title": std.title,
                        "status": std.status,
                        "superseded_by": std.superseded_by,
                    }
                )
            if std.qco_alert and std.qco_alert.is_mandatory:
                pass  # QCO is flagged — good
            else:
                missing_qco.append(
                    {
                        "is_code": std.is_code,
                        "title": std.title,
                        "note": "No mandatory QCO flag found in catalog.",
                    }
                )
        else:
            missing_qco.append(
                {
                    "is_code": code,
                    "title": "Unknown / not in catalog",
                    "note": "Standard not found in BIS catalog — verify manually.",
                }
            )

    summary = (
        f"Extracted {len(found_codes)} unique IS code(s) from the tender PDF. "
        f"{len(flagged_outdated)} outdated/superseded, {len(missing_qco)} without QCO coverage."
    )

    return AuditPDFResponse(
        extracted_is_codes=found_codes,
        flagged_outdated=flagged_outdated,
        missing_qco=missing_qco,
        summary=summary,
    )
  app = FastAPI()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "BIS AI Recommendation API is running"
    }
