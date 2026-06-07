.PHONY: validate mermaid sync-readme import all clean

validate:
	python scripts/validate_entries.py

mermaid:
	python scripts/render_mermaid.py

sync-readme: mermaid
	python scripts/sync_readme.py

import:
	python scripts/import_from_readme.py

all: validate mermaid sync-readme

clean:
	rm -f figures/*.mmd
