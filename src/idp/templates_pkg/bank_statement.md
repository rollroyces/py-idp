---
name: bank_statement
schema: BankStatement
version: 1
mime_types: [application/pdf]
filename_patterns: ["*statement*", "*bank*", "*account*"]
classification_hints:
  - "Has account number, opening/closing balance, transaction list"
  - "Often has bank name + logo at top"
  - "Contains 'Deposit', 'Withdrawal', 'Balance' columns"
---

# Bank Statement

## Fields

* **bank_name** — name of the issuing bank, often with logo.
* **account_holder** — name of the account owner.
* **account_number** — usually partially masked (e.g. "****1234").
  Capture the masked form verbatim.
* **statement_period** — `{start_date, end_date}`. Often labelled
  "Statement Period" or "From ... To ...".
* **opening_balance** — balance at the start of the period.
* **closing_balance** — balance at the end. **Should equal** opening +
  sum(transactions). If not, leave a note in `validation_errors` —
  this is a real data integrity signal.
* **currency** — three-letter ISO 4217 code.
* **transactions** — list of `{date, description, amount, balance}`.
  `amount` is signed: positive for deposits, negative for withdrawals.

## Common mistakes

* **amount sign**: in some statements, debits are shown as positive
  numbers in a "Withdrawals" column and credits in a separate
  "Deposits" column. Map both to a single signed `amount` field.
* **account_number**: copy the masked form ("****1234") — don't try
  to recover the full number.
* **balance continuity**: opening + sum(amounts) should equal closing
  within rounding. If it doesn't, set
  `validation_errors=["balance_mismatch"]` and continue.
* **description**: copy the description verbatim, but normalize
  whitespace (collapse multiple spaces, trim).

## Notes

Statements often span 1-5 pages. The transactions table may be split
across pages; the chunker handles this automatically. The
`closing_balance` cross-check is a strong signal for downstream
validation — if it fails, route the result to HITL review even if all
individual fields look correct.
