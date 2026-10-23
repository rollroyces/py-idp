---
name: medical_claim
schema: MedicalClaim
version: 1
mime_types: [application/pdf, image/jpeg, image/png]
filename_patterns: ["*claim*", "*eob*", "*cms*1500*", "*ub-04*", "*medical*"]
classification_hints:
  - "Insurance claim with CPT/HCPCS service codes"
  - "Contains 'Claim', 'Patient', 'Provider', 'Allowed', 'Paid'"
  - "Often has ICD-10 diagnosis codes"
---

# Medical Insurance Claim

## Fields

* **claim_number** — insurer's claim tracking number.
* **patient_name** — patient (could be the subscriber or a dependent).
* **patient_dob** — patient date of birth.
* **patient_member_id** — patient insurance member ID.
* **insurer_name** — the insurance company paying the claim.
* **provider_name** — the healthcare provider who rendered services.
* **provider_npi** — National Provider Identifier, 10 digits.
* **date_of_service** — when the service was rendered.
* **diagnosis_codes** — list of ICD-10 codes (e.g. "J45.909").
  Capture verbatim including the dot.
* **line_items** — list of `{service_code, description, service_date,
  charged_amount, allowed_amount, paid_amount, patient_responsibility}`.
  service_code is CPT (5 digits) or HCPCS (letter + 4 digits).
* **total_charged** — sum of `charged_amount` over line items.
* **total_allowed** — sum of `allowed_amount` (insurer-allowed).
* **total_paid** — sum of `paid_amount` (insurer payment).
* **total_patient_responsibility** — sum of `patient_responsibility`
  (copay + deductible + coinsurance).

## Common mistakes

* **charged_amount ≠ allowed_amount ≠ paid_amount**. These are
  three different numbers:
  - charged: what the provider billed
  - allowed: what the insurer says is reasonable
  - paid: what the insurer actually paid
  Do not collapse them.
* **CPT vs ICD-10**: CPT codes describe WHAT was done; ICD-10
  codes describe WHY. They live in different fields.
* **patient_responsibility** is `charged - paid` only when
  `allowed == charged` (in-network). For out-of-network, it's
  `charged - paid` plus the difference between charged and
  allowed. Compute it; don't conflate.
