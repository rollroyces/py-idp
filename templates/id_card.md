---
name: id_card
schema: IDCard
version: 1
mime_types: [image/jpeg, image/png, application/pdf]
filename_patterns: ["*id*", "*license*", "*passport*", "*dl*", "*idcard*"]
classification_hints:
  - "Photo identification document with name, photo, date of birth"
  - "Has 'License', 'Passport', 'ID', 'DOB' text"
  - "Often has signature, hologram, or security features"
---

# ID Card / Driver's License

## Fields

* **document_type** — what kind of ID this is. Look for explicit
  labels like "DRIVER'S LICENSE", "PASSPORT", "STATE ID". If unclear,
  describe the visual layout ("Photo card with name + DOB").
* **full_name** — the cardholder's full legal name. Often in larger
  text near the photo.
* **date_of_birth** — date of birth, in the format shown on the card
  (MM/DD/YYYY, DD MMM YYYY, etc.). Do NOT normalize.
* **sex** — "M" / "F" / "X" — capture exactly as printed.
* **id_number** — license / passport / state ID number. Capture
  verbatim including any hyphens or spaces.
* **issuing_authority** — who issued the ID, e.g. "California DMV",
  "US Department of State". Often near the top or on a seal.
* **issue_date** — when the ID was issued.
* **expiration_date** — when the ID expires. **Critical** for
  downstream validation (an expired ID is usually invalid).
* **address** — current residential address on the card. Often on
  the back (driver's license) or on a separate page (passport).
* **nationality** — passport-specific. Often "USA" or country name.

## Common mistakes

* **NEVER truncate or partially mask id_number**. The full ID
  number IS the value being extracted; only the document holder
  decides what to do with it downstream.
* **expiration_date parsing**: be careful with US vs European date
  formats. MM/DD/YYYY in the US; DD/MM/YYYY in the EU. Use the
  document's own context to disambiguate.
* **issue_date vs date_of_birth**: issue dates are 18+ years AFTER
  the DOB. If you see two dates close together, they're probably
  DOB and issue_date.
