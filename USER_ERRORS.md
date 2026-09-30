# py-idp API: error codes for end users

Every error the py-idp API returns uses the same envelope:

```json
{
  "error": {
    "code": "IDP-RATE-001",
    "message": "rate limit exceeded",
    "type": "RateLimitedError",
    "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "hint": "Too many requests. Slow down and retry after the time indicated by the Retry-After header.",
    "docs_url": "https://py-idp.dev/errors#IDP-RATE-001"
  }
}
```

| Field | What it means |
|-------|---------------|
| `code` | Stable machine-readable identifier. Switch on this in your client. |
| `message` | Free-form, technical message. Don't parse it -- use `code` instead. |
| `type` | Exception class name. Useful for debugging with Python clients. |
| `request_id` | UUID you can quote when contacting support. |
| `hint` | One-line, user-actionable message. **Show this to end users.** |
| `docs_url` | Anchor link to the matching section below. |

The `hint` and `docs_url` fields are present whenever the code is recognized. If your client can't reach the network for docs, ignore `docs_url` and use the `hint` directly.

## Quick reference

| Code | HTTP | When it fires | What the user should do |
|------|------|---------------|-------------------------|
| `IDP-PARSE-001` | 422 | Document couldn't be parsed (wrong format, corrupt, encrypted) | Use a supported file type (PDF, image, plain text) and ensure it's not password-protected. |
| `IDP-SCHEMA-001` | 422 | Extracted data didn't match the expected schema | Try a clearer document or a different schema. |
| `IDP-BACKEND-001` | 502 | The language model is unreachable | Retry in a few seconds; check the backend status page if it persists. |
| `IDP-OCR-001` | 502 | OCR couldn't read the document | Try a higher-resolution scan or a different document. |
| `IDP-STORE-001` | 500 | Result couldn't be saved | Check server disk space and file permissions, then retry. |
| `IDP-CONF-001` | 503 | Server is misconfigured | Contact the operator. Not a client-side fix. |
| `IDP-RATE-001` | 429 | Rate limit exceeded | Slow down; respect the `Retry-After` header. |
| `IDP-TIME-001` | 504 | Operation took too long | Retry with a smaller document or fewer pages. |
| `IDP-TMPL-404` | 404 | Template not found | Check `/templates` for the list of available templates. |
| `IDP-TMPL-001` | 500 | Template file is malformed | Contact the operator to fix the template. |
| `IDP-UPLOAD-413` | 413 | Upload exceeds server cap | Reduce the file size or raise `IDP_MAX_UPLOAD_BYTES`. |
| `IDP-INT-001` | 500 | Unexpected internal error | Retry shortly; contact support with the `request_id` if it keeps failing. |
| `IDP-CLIENT-400` | 400 | Malformed request | Check the API docs for the expected fields. |
| `IDP-CLIENT-401` | 401 | Missing or invalid authentication | Provide a valid `X-API-Key` header. |
| `IDP-CLIENT-403` | 403 | API key not authorized | Verify the key is correct and has the right permissions. |
| `IDP-CLIENT-404` | 404 | Endpoint or resource doesn't exist | Check the URL. |
| `IDP-CLIENT-405` | 405 | HTTP method not supported | Check the API docs for allowed methods. |
| `IDP-CLIENT-413` | 413 | Request body too large | Reduce the payload size. |
| `IDP-CLIENT-415` | 415 | Unsupported media type | Use a `Content-Type` matching what the endpoint expects. |
| `IDP-CLIENT-422` | 422 | Request body has validation errors | Check `error.details.errors` for the specific field errors. |
| `IDP-CLIENT-429` | 429 | Too many requests | Slow down; respect `Retry-After`. |
| `IDP-SERVER-500` | 500 | Internal server error | Retry shortly; contact support with `request_id`. |
| `IDP-SERVER-502` | 502 | Upstream service unavailable | Retry shortly; check the status page. |
| `IDP-SERVER-503` | 503 | Service starting up or temporarily unavailable | Retry shortly. |
| `IDP-SERVER-504` | 504 | Upstream service timed out | Retry shortly. |

## Detailed entries

### `IDP-PARSE-001` — Document couldn't be parsed

**When it fires:** The parser stage couldn't extract any text from the uploaded document. Common causes:
- File extension isn't supported (only PDF, images, and plain text work)
- File is corrupt or truncated
- File is password-protected
- The PDF uses a format that requires an extra dependency (e.g. Docling) that isn't installed

**What the user should do:** Make sure the file is a supported type and isn't password-protected. If you're uploading a complex PDF, the operator may need to install the Docling extra (`pip install py-idp[docling]`).

**Example response:**
```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/json

{
  "error": {
    "code": "IDP-PARSE-001",
    "message": "file not found: /uploads/invoice.pdf",
    "type": "DocumentParseError",
    "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "hint": "We couldn't read your document. Make sure it's a supported file type (PDF, image, plain text) and isn't password-protected.",
    "docs_url": "https://py-idp.dev/errors#IDP-PARSE-001"
  }
}
```

### `IDP-SCHEMA-001` — Schema validation failed

**When it fires:** Strict mode is on and the extracted data doesn't match the requested schema. The LLM returned fields with the wrong type, or missing required fields, that strict-mode Pydantic validation caught.

**What the user should do:** Try a clearer document (more typed fields, less handwriting) or use a more permissive schema. Or turn off strict mode if you don't want a hard failure here.

### `IDP-BACKEND-001` — LLM backend unavailable

**When it fires:** The configured language model couldn't be reached. Wraps HTTP 5xx, network timeouts, and auth failures from the upstream provider.

**What the user should do:** Retry in a few seconds. If it persists, check:
- OpenAI status: https://status.openai.com
- Anthropic status: https://status.anthropic.com
- Your own LLM server (if self-hosted)

### `IDP-OCR-001` — OCR failed

**When it fires:** The OCR stage produced no text or failed internally (e.g. Tesseract returned an error, PaddleOCR model load failed, etc.). Distinct from `IDP-PARSE-001`: this means the document *parsed* but OCR *couldn't read it*.

**What the user should do:** Try a higher-resolution scan (300+ DPI recommended) or a different document. If the issue is on the operator side, check that the OCR engine is installed and the model weights are downloaded.

### `IDP-STORE-001` — Storage failure

**When it fires:** Disk full, permission denied on the storage path, or the schema migration failed. Result couldn't be persisted.

**What the user should do:** This is an operator-side issue. Free disk space, fix file permissions, or fix the migration script.

### `IDP-CONF-001` — Server misconfigured

**When it fires:** A required environment variable is missing or invalid (e.g. `DATABASE_URL` set but unparseable). The server fails fast at startup, so this usually means the server isn't running at all.

**What the user should do:** Not a client-side fix. Contact the operator.

### `IDP-RATE-001` — Rate limit exceeded

**When it fires:** The configured per-minute request cap was hit.

**What the user should do:** Slow down. The response carries a `Retry-After: 60` header indicating how long to wait.

**Example response:**
```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 60

{
  "error": {
    "code": "IDP-RATE-001",
    "message": "rate limit exceeded",
    "type": "RateLimitedError",
    "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "hint": "Too many requests. Slow down and retry after the time indicated by the Retry-After header.",
    "docs_url": "https://py-idp.dev/errors#IDP-RATE-001"
  }
}
```

### `IDP-TIME-001` — Operation timed out

**When it fires:** An internal operation (LLM call, OCR inference, etc.) exceeded its time budget. Default budget is `IDP_LLM_TIMEOUT` seconds (60s by default).

**What the user should do:** Retry with a smaller document or fewer pages. If your documents are intrinsically large, ask the operator to raise the timeout.

### `IDP-TMPL-404` — Template not found

**When it fires:** You asked for a template that isn't registered (e.g. `/templates/invoice-old` when only `invoice` exists).

**What the user should do:** List available templates with `GET /templates` and use one of those.

**Example response:**
```http
HTTP/1.1 404 Not Found
Content-Type: application/json

{
  "error": {
    "code": "IDP-TMPL-404",
    "message": "template 'invoice-old' not found in registry",
    "type": "TemplateNotFoundError",
    "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "hint": "That template doesn't exist. Check the /templates endpoint for the list of available templates.",
    "docs_url": "https://py-idp.dev/errors#IDP-TMPL-404"
  }
}
```

### `IDP-TMPL-001` — Template file malformed

**When it fires:** A template `.md` file has bad YAML frontmatter, a missing required field, or an invalid schema reference.

**What the user should do:** Operator-side fix. Check the template file in the configured `IDP_TEMPLATE_DIR`.

### `IDP-UPLOAD-413` — Upload too large

**When it fires:** The uploaded file exceeded the configured `IDP_MAX_UPLOAD_BYTES` (25 MB by default).

**What the user should do:** Reduce the file size (compress images, drop unnecessary pages), or ask the operator to raise the server's cap.

**Example response:**
```http
HTTP/1.1 413 Payload Too Large
Content-Type: application/json

{
  "error": {
    "code": "IDP-UPLOAD-413",
    "message": "payload too large: 31457280 bytes > max 26214400",
    "type": "FileTooLargeError",
    "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "hint": "Your upload is too large. Reduce the file size or raise the server's max_upload_bytes setting.",
    "docs_url": "https://py-idp.dev/errors#IDP-UPLOAD-413"
  }
}
```

### `IDP-INT-001` — Unexpected error

**When it fires:** An unhandled exception bubbled up. Should be rare -- this is the catch-all for things we didn't classify.

**What the user should do:** Retry in a few seconds. If it persists, contact the operator with the `request_id`.

### `IDP-CLIENT-4xx` and `IDP-SERVER-5xx` — HTTP-derived codes

Codes prefixed `IDP-CLIENT-` and `IDP-SERVER-` are generated automatically from HTTP status codes (see `idp.errors.classify_status`). They cover:
- 400, 401, 403, 404, 405, 413, 415, 422, 429 (client-side)
- 500, 502, 503, 504 (server-side)

The `hint` field for each tells the user what to do. See the [Quick reference](#quick-reference) table for the full list.

## Versioning

`code`, `hint`, and `docs_url` are stable. New codes may be added; existing codes may receive a refined hint message but the meaning won't change without a major version bump. Client code should:
- Branch on `code`, not `message` or `type`
- Treat unknown codes as `IDP-INT-001` (show the generic "something went wrong" message)
- Always surface `request_id` in support tickets

