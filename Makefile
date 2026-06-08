# Lightweight dev shortcuts. CI is the source of truth — see
# .github/workflows/ci.yml — but these wrappers mirror what CI runs so
# contributors get the same feedback locally as they would after pushing.

.PHONY: lint fmt lint-fix test test-agent test-rec install-lint

install-lint:
	pip install -r requirements-lint.txt

lint:
	ruff check .
	black --check .

# Auto-fix what's safely auto-fixable, then reformat. Run before pushing.
fmt:
	ruff check --fix .
	black .

# Alias for `fmt` for muscle memory.
lint-fix: fmt

test: test-agent test-rec

test-agent:
	cd agent_dag_sandbox && python -m pytest -q

test-rec:
	cd llm_rec_svc && python -m pytest -q
