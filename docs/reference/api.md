# API reference

This page is auto-generated from the source code's docstrings via [`mkdocstrings`](https://mkdocstrings.github.io/). If you're reading the rendered site, the symbols below expand on click.

## Top-level

::: idp
    options:
      members:
        - Document
        - Pipeline
        - discover_schema
      show_source: true

## Pipeline

::: idp.pipeline.pipeline
    options:
      members:
        - Pipeline
        - PipelineResult
        - run
        - save_result

## Backends

::: idp.llm.backend
    options:
      members:
        - get_backend
        - Backend

## Schemas

::: idp.core.schemas
    options:
      members:
        - Invoice
        - Contract
        - BankStatement
        - BankTransaction
        - Receipt
        - LineItem
        - ReceiptLineItem
        - SCHEMA_REGISTRY

## HITL

::: idp.storage.store
    options:
      members:
        - JsonFileStorage
        - StoredResult