# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Minimal FastAPI server exposing the pipeline.

NOT shipped in the base deps (optional `pip install py-idp[api]`).

Run:
    uvicorn examples.api:app --reload

Routes:
    POST /extract     submit a file for extraction (sync)
    POST /jobs        async submit a file
    GET  /jobs/{id}   job status
    GET  /results/{id}  fetch a result
    POST /results/{id}/review  record a human review
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from idp.auth.keys import require_api_key, verify
from idp.core.document import Document
from idp.core.schemas import SCHEMA_REGISTRY
from idp.pipeline.pipeline import Pipeline
from idp.queue.jobs import InProcessQueue, Job
from idp.storage.store import JsonFileStorage, StoredResult

app = FastAPI(title="py-idp API", version="0.1.0")

STORAGE_DIR = Path(os.environ.get("IDP_STORAGE_DIR", "./idp_data"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
storage = JsonFileStorage(STORAGE_DIR / "results.jsonl")
_uploads = STORAGE_DIR / "uploads"
_uploads.mkdir(parents=True, exist_ok=True)


async def _job_runner(job: Job) -> None:
    """Sync pipeline call (CPU-light on small docs); wrap in to_thread for real CPU."""
    pipeline = Pipeline(
        backend=job.backend_name,
        schema=SCHEMA_REGISTRY[job.schema_name],
    )
    result = pipeline.run(Document.from_path(job.doc_path))
    stored = StoredResult(
        id="",
        doc_id=result.document.doc_id,
        schema_name=result.schema_name,
        backend_name=result.backend_name,
        mode=result.mode,
        classification=result.classification,
        extraction=result.document.extraction or {},
        confidence=result.confidence,
        validation=result.document.validation,
        source_path=result.document.source_path,
        created_at=0.0,
    )
    rid = storage.put(stored)
    job.result_id = rid


queue = InProcessQueue(_job_runner)


async def auth(x_api_key: str | None = None) -> None:
    expected = require_api_key()
    if expected is None:
        return  # dev mode: no auth required
    if x_api_key is None or not verify(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid API key")


class ExtractRequest(BaseModel):
    schema_name: str = "Invoice"
    backend: str = "auto"


class ExtractResponse(BaseModel):
    schema: str
    backend: str
    mode: str | None
    classification: str | None
    extraction: dict
    confidence: dict | None
    validation: dict | None


class JobResponse(BaseModel):
    id: str
    status: str
    result_id: str | None = None
    error: str | None = None


@app.post("/extract", response_model=ExtractResponse, dependencies=[Depends(auth)])
async def extract_sync(file: UploadFile, req: ExtractRequest) -> ExtractResponse:
    dst = _uploads / file.filename or "upload"
    dst.write_bytes(await file.read())
    pipeline = Pipeline(backend=req.backend, schema=req.schema_name)
    doc = Document.from_path(dst)
    res = pipeline.run(doc)
    return ExtractResponse(
        schema=res.schema_name,
        backend=res.backend_name,
        mode=res.mode,
        classification=res.classification,
        extraction=res.document.extraction or {},
        confidence=res.confidence,
        validation=res.document.validation,
    )


@app.post("/jobs", response_model=JobResponse, dependencies=[Depends(auth)])
async def submit_job(file: UploadFile, req: ExtractRequest) -> JobResponse:
    dst = _uploads / (file.filename or "upload")
    dst.write_bytes(await file.read())
    job = await queue.submit(str(dst), req.schema_name, req.backend)
    return JobResponse(id=job.id, status=job.status.value, error=job.error)


@app.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(auth)])
async def job_status(job_id: str) -> JobResponse:
    j = await queue.status(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="no such job")
    return JobResponse(id=j.id, status=j.status.value, result_id=j.result_id, error=j.error)


@app.get("/results/{result_id}", dependencies=[Depends(auth)])
async def get_result(result_id: str) -> dict:
    r = storage.get(result_id)
    if r is None:
        raise HTTPException(status_code=404, detail="no such result")
    return {
        "id": r.id,
        "doc_id": r.doc_id,
        "schema": r.schema_name,
        "backend": r.backend_name,
        "mode": r.mode,
        "extraction": r.extraction,
        "confidence": r.confidence,
        "validation": r.validation,
        "reviewed": r.reviewed,
    }


class ReviewRequest(BaseModel):
    edited: dict
    reviewer: str


@app.post("/results/{result_id}/review", dependencies=[Depends(auth)])
async def post_review(result_id: str, req: ReviewRequest) -> dict:
    storage.mark_reviewed(result_id, req.edited, req.reviewer)
    return {"id": result_id, "reviewed": True}
