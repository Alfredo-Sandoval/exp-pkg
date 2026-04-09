.PHONY: env setup bootstrap env-macos env-linux env-windows loc lint typecheck test qa ci-local build package-check docs-build docs-serve clean

ENV_ARGS ?=
PYTHON ?= python
SOURCE_DIRS ?= src
DOCS_ADDR ?= 127.0.0.1:8123
RUN_IN_ENV := bash environment/run-in-env.sh
EXCLUDE_PATHS ?= .git .venv venv env node_modules .next .turbo out __pycache__ *.egg-info .eggs .pytest_cache .ruff_cache .mypy_cache .cache build dist htmlcov coverage .coverage site results data external
LOC_PRUNE_NAMES := $(foreach p,$(EXCLUDE_PATHS),-name '$(p)' -o ) -false
LOC_GROUP_DEPTH ?= 2

env:
	@test -f environment/setup.sh || { echo "Missing setup script: environment/setup.sh"; exit 1; }
	bash environment/setup.sh $(ENV_ARGS)

setup: env

bootstrap: env

env-macos:
	@test -f environment/macos/setup.sh || { echo "Missing setup script: environment/macos/setup.sh"; exit 1; }
	bash environment/macos/setup.sh $(ENV_ARGS)

env-linux:
	@test -f environment/linux/setup.sh || { echo "Missing setup script: environment/linux/setup.sh"; exit 1; }
	bash environment/linux/setup.sh $(ENV_ARGS)

# Optional target. Keep only if Windows support is explicitly enabled.
env-windows:
	@echo "Windows setup is optional. Run from PowerShell:"
	@echo "powershell -ExecutionPolicy Bypass -File environment/windows/setup.ps1"

lint:
	$(RUN_IN_ENV) ruff check .

typecheck:
	$(RUN_IN_ENV) ty check

test:
	$(RUN_IN_ENV) pytest

qa: lint typecheck test

ci-local: lint typecheck test package-check docs-build

build:
	$(RUN_IN_ENV) env UV_CACHE_DIR="$${UV_CACHE_DIR:-/tmp/uv-cache}" uv build --out-dir dist --clear

package-check:
	@tmpdir="$$(mktemp -d)"; \
	$(RUN_IN_ENV) env UV_CACHE_DIR="$${UV_CACHE_DIR:-/tmp/uv-cache}" uv build --out-dir "$$tmpdir" --clear; \
	$(RUN_IN_ENV) python -m twine check "$$tmpdir"/*; \
	rm -rf "$$tmpdir"

docs-build:
	$(RUN_IN_ENV) python -m mkdocs build --strict

docs-serve:
	$(RUN_IN_ENV) python -m mkdocs serve -a $(DOCS_ADDR)

# LOC summary: module breakdown (by depth) + language breakdown + file count.
loc:
	@paths=""; \
	for d in $(SOURCE_DIRS); do \
		if [ -d "$$d" ]; then \
			if [ -z "$$paths" ]; then paths="$$d"; else paths="$$paths $$d"; fi; \
		fi; \
	done; \
	if [ -z "$$paths" ]; then \
		echo "No source directories found in SOURCE_DIRS=$(SOURCE_DIRS)"; \
		exit 1; \
	fi; \
	{ \
		for d in $$paths; do \
			find "$$d" -type d \( $(LOC_PRUNE_NAMES) \) -prune -o -type f \( -name '*.py' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' \) -print0; \
		done; \
	} \
	| xargs -0 wc -l \
	| awk -v depth="$(LOC_GROUP_DEPTH)" ' \
		NF==2 && $$2!="total"{ \
			count=$$1; path=$$2; files++; \
			ne=split(path,ep,"."); ext=ep[ne]; lang[ext]+=count; \
			n=split(path,parts,"/"); if(n<2) next; \
			gd=depth; if(gd<1) gd=1; lim=gd; if(n-1<lim) lim=n-1; \
			key=parts[1]; for(i=2;i<=lim;i++) key=key"/"parts[i]; \
			mod[key]+=count; fc[key]++; total+=count \
		} \
		END{ \
			nm=0; for(k in mod){nm++;mk[nm]=k;mv[nm]=mod[k]} \
			for(i=2;i<=nm;i++){j=i;while(j>1&&mv[j]>mv[j-1]){ \
				t=mv[j];mv[j]=mv[j-1];mv[j-1]=t; \
				t=mk[j];mk[j]=mk[j-1];mk[j-1]=t;j--}} \
			printf "\n  %-34s %7s  %5s  %5s\n","Module","Lines","Files","%"; \
			printf "  %-34s %7s  %5s  %5s\n","------","-----","-----","--"; \
			for(i=1;i<=nm;i++) \
				printf "  %-34s %7d  %5d  %5.1f\n",mk[i],mv[i],fc[mk[i]],(total>0?mv[i]*100/total:0); \
			printf "  %-34s %7d  %5d\n","Total",total,files; \
			lm["Python"]=lang["py"]+0; \
			lm["TypeScript"]=(lang["ts"]+0)+(lang["tsx"]+0); \
			lm["JavaScript"]=(lang["js"]+0)+(lang["jsx"]+0); \
			lm["Rust"]=lang["rs"]+0; \
			nl=0; for(k in lm) if(lm[k]>0){nl++;lk[nl]=k;lv[nl]=lm[k]} \
			for(i=2;i<=nl;i++){j=i;while(j>1&&lv[j]>lv[j-1]){ \
				t=lv[j];lv[j]=lv[j-1];lv[j-1]=t; \
				t=lk[j];lk[j]=lk[j-1];lk[j-1]=t;j--}} \
			printf "\n  %-34s %7s  %5s\n","Language","Lines","%"; \
			printf "  %-34s %7s  %5s\n","--------","-----","--"; \
			for(i=1;i<=nl;i++) \
				printf "  %-34s %7d  %5.1f\n",lk[i],lv[i],(total>0?lv[i]*100/total:0); \
			printf "\n" \
		}'

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -name "*.tsbuildinfo" -type f -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
