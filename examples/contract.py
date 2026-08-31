"""Example: extract a Contract using the mock backend."""
from pathlib import Path

from idp.core.document import Document
from idp.core.schemas import Contract
from idp._util import pretty_print_result
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline
from idp.validate.validator import required_fields_rule

SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/contracts/docs/svc-001.txt"


def main() -> None:
    backend = get_backend("mock")
    pipeline = Pipeline(
        backend=backend,
        schema=Contract,
        business_rules=[required_fields_rule("title", "effective_date", "expiration_date")],
    )
    doc = Document.from_path(SAMPLE)
    result = pipeline.run(doc)
    pretty_print_result(result)


if __name__ == "__main__":
    main()
