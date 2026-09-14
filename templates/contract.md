---
name: contract
schema: Contract
version: 1
mime_types: [application/pdf]
filename_patterns: ["*contract*", "*agreement*", "*msa*", "*nda*"]
classification_hints:
  - "Long legal document with clauses and signatures"
  - "Contains 'Agreement', 'Party', 'Term', 'Termination' keywords"
  - "Often has signature blocks at the end"
---

# Contract

## Fields

* **contract_title** — the document's title, often at the top in large
  bold text. e.g. "Master Services Agreement", "Non-Disclosure Agreement".
* **parties** — list of `{name, role}` for each contracting party.
  Parties are usually introduced with "between X and Y" near the top.
* **effective_date** — when the contract starts. Look for phrases like
  "effective as of", "dated", "entered into on".
* **term_length** — duration, e.g. "12 months", "2 years", "perpetual".
* **jurisdiction** — governing law, e.g. "State of Delaware", "Hong
  Kong SAR". Often in a "Governing Law" or "Choice of Law" clause.
* **total_value** — monetary value of the contract, if mentioned.
  Often absent in NDAs.
* **signature_blocks** — list of `{signatory_name, title, party,
  date_signed}`. Found at the end of the document.

## Common mistakes

* **parties vs signatories**: `parties` are the entities bound by the
  contract (e.g. "Acme Inc."). `signature_blocks` are the humans who
  signed it. Don't conflate.
* **effective_date vs signature_date**: the effective date is when the
  contract starts; signature dates are when each party signed (often
  different days). Use `effective_date` for the main field, leave
  individual signature dates in `signature_blocks`.
* **total_value**: only set if the contract explicitly states a value.
  Most NDAs have no value; leave `null` rather than guessing.
* **jurisdiction**: copy verbatim from the document. Don't normalize
  ("Delaware" vs "State of Delaware" both valid).

## Notes

Contracts are the hardest document type to extract reliably. The
language is dense, formatting varies wildly, and important clauses
may be in tiny footnotes. Use a high-quality LLM (GPT-4o or Claude
Sonnet) and a multimodal backend if the document is a scan.
