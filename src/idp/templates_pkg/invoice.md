---
name: invoice
schema: Invoice
version: 1
mime_types: [application/pdf, image/jpeg, image/png]
filename_patterns: ["*invoice*", "*inv-*", "*receipt*", "*bill*"]
classification_hints:
  - "Contains a total amount, vendor name, invoice number"
  - "Often has line items with quantity, unit price, subtotal, tax"
  - "Currency symbols ($ € £ ¥) usually present"
field_overrides:
  invoice_number:
    regex: "(?i)inv[\\-_ ]?\\d+"
    examples: [INV-001, inv_2024_123, INV 2024-001]
  total_amount:
    description: "Final amount due, including tax. NOT the subtotal."
  date_due:
    description: "Date by which the invoice must be paid. Usually later than date_issued."
---

# Invoice

## Fields

* **vendor_name** — the name of the company issuing the invoice. Look for
  the largest bold text near the top, or text from the company logo.
  Common aliases: "From", "Supplier", "Vendor", "Bill from".
* **invoice_number** — usually near the top, format varies wildly
  (`INV-2024-001`, `2024-09-15-001`, `ACME/09/001`).
* **date_issued** — date the invoice was created.
* **date_due** — payment due date. Usually later than `date_issued`.
  Often labelled "Due Date" or "Payment Due".
* **subtotal** — sum of line items, before tax. Often labelled
  "Subtotal" or "Net Amount".
* **tax_amount** — total tax. Often labelled "Tax", "VAT", or "GST".
* **total_amount** — final amount due including tax. **NOT** the
  subtotal. Often labelled "Total", "Total Due", "Amount Due".
* **currency** — three-letter ISO 4217 code (USD, EUR, GBP, CNY, JPY…).
  If only a symbol is present, infer from context (e.g. ¥ on a
  Japanese receipt → JPY).
* **line_items** — list of {description, quantity, unit_price, amount}.
  Each line typically appears in a table.

## Common mistakes

* **subtotal vs total_amount**: subtract `tax_amount` from `total_amount`
  to get `subtotal`. If the invoice only shows one of them, leave the
  other as `null` rather than guessing.
* **date_due vs date_issued**: due date is usually **later** than the
  issued date. If you see only one date, set the other to `null`.
* **currency**: the symbol `¥` is used for both JPY and CNY. Use the
  document's language and the issuer's country to disambiguate.
* **line items**: when the table is split across pages, the last row
  on page N may continue on page N+1. Read the description column
  carefully — don't drop a row just because the amount cell looks
  incomplete.

## Worked example

```text
ACME SUPPLIES LTD.
123 Industrial Park, HK
Invoice #: INV-2024-09-001
Date: 2024-09-15     Due: 2024-10-15

Bill To: Acme Customer Co.

Item                      Qty    Unit      Amount
Widget A                   10    $5.00     $50.00
Widget B                    2   $25.00     $50.00
                                    Subtotal: $100.00
                                    Tax (10%): $10.00
                                    TOTAL:    $110.00
```

→ Extraction:

```json
{
  "vendor_name": "ACME SUPPLIES LTD.",
  "invoice_number": "INV-2024-09-001",
  "date_issued": "2024-09-15",
  "date_due": "2024-10-15",
  "subtotal": 100.0,
  "tax_amount": 10.0,
  "total_amount": 110.0,
  "currency": "USD",
  "line_items": [
    {"description": "Widget A", "quantity": 10, "unit_price": 5.0, "amount": 50.0},
    {"description": "Widget B", "quantity": 2, "unit_price": 25.0, "amount": 50.0}
  ]
}
```
