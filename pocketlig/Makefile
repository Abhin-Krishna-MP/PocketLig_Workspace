.PHONY: test web-install web-build api validate
test:
	python -m pytest -q
web-install:
	cd web && npm install
web-build:
	cd web && npm run build
api:
	POCKETLIG_ARTIFACTS=artifacts uvicorn pocketlig.api.main:app --port 8008
validate:
	python -m pocketlig.precompute.validate_artifacts artifacts
