DIST_DIR ?= dist

# OS Specific command
ifeq ($(OS),Windows_NT)
	VENV_DIR := .venv
	VENV_BIN := .venv/Scripts
	RMRF := rm -rf
	MVF := move /Y
else
	VENV_DIR := .venv
	VENV_BIN := .venv/bin
	RMRF := rm -rf
	MVF := mv -f
endif

# Cross-compilation targets
MACOS_AA_TARGETS := aarch64-apple-darwin
MACOS_X8_TARGETS := x86_64-apple-darwin
LINUX_AA_TARGETS := aarch64-unknown-linux-gnu
LINUX_X8_TARGETS := x86_64-unknown-linux-gnu
WINDOWS_TARGETS := x86_64-pc-windows-msvc
PYTHON_VERSIONS := 3.9 3.10 3.11 3.12 3.13

RUSTFLAGS_NATIVE := -C target-cpu=native -C opt-level=3
RUSTFLAGS_X86_64_V3 := -C target-cpu=x86-64-v3 -C opt-level=3
RUSTFLAGS_APPLE_M1 := -C target-cpu=apple-m1 -C opt-level=3

# Create a virtual environment
define create_venv
	$(RMRF) $(VENV_DIR)
	uv venv $(VENV_DIR)
	uv pip install --upgrade --compile-bytecode --no-build -r requirements-python.txt
endef

# Create a virtual environment for a specific platform
define create_venv_py
	$(RMRF) $(VENV_DIR)
	uv venv $(VENV_DIR) --python $(1)
	uv pip install --upgrade --compile-bytecode --no-build -r requirements-python.txt
endef

.PHONY: ngram_rs_release
ngram_rs_release:	clippy
	cargo build -p ngram_rs -r

.PHONY: ngram_polars_dev 
ngram_polars_dev: ngram_rs_release
	$(call create_venv)
	$(VENV_BIN)/maturin develop -m ngram_polars/Cargo.toml --release

.PHONY: test
test: ngram_rs_release ngram_polars_dev
	RUSTFLAGS="$(RUSTFLAGS_NATIVE)" cargo test
	uv run pytest ngram_polars/tests/test_ngram_polars_reg.py

.PHONY: ngram_polars_release
ngram_polars_release: clippy test
ifeq ($(OS),Windows_NT)
	powershell -Command "Remove-Item -Path ngram_polars\*.pyd -Force"
endif
	$(call create_venv)
	$(VENV_BIN)/maturin sdist -m ngram_polars/Cargo.toml --out $(DIST_DIR)
	$(foreach target,$(MACOS_AA_TARGETS),\
		$(foreach pyver,$(PYTHON_VERSIONS),\
			$(call create_venv_py, $(pyver)) ;\
			RUSTFLAGS="$(RUSTFLAGS_APPLE_M1)" $(VENV_BIN)/uv run --python $(pyver) python -m maturin build -m ngram_polars/Cargo.toml --release --strip --target $(target) --out $(DIST_DIR) ;\
		)\
	)
	$(foreach target,$(MACOS_X8_TARGETS),\
		$(foreach pyver,$(PYTHON_VERSIONS),\
			$(call create_venv_py, $(pyver)) ;\
			RUSTFLAGS="$(RUSTFLAGS_X86_64_V3)" $(VENV_BIN)/uv run --python $(pyver) python -m maturin build -m ngram_polars/Cargo.toml  --release --strip --target $(target) --out $(DIST_DIR) ;\
		)\
	)
	$(foreach target,$(LINUX_X8_TARGETS),\
		$(foreach pyver,$(PYTHON_VERSIONS),\
			$(call create_venv_py, $(pyver)) ;\
			RUSTFLAGS="$(RUSTFLAGS_X86_64_V3)" $(VENV_BIN)/uv run --python $(pyver) python -m maturin build -m ngram_polars/Cargo.toml --release -i python$(pyver) --strip --target $(target) --manylinux 2014 --zig --out $(DIST_DIR) ;\
		)\
	)
	$(foreach target,$(LINUX_AA_TARGETS),\
		$(foreach pyver,$(PYTHON_VERSIONS),\
			$(call create_venv_py, $(pyver)) ;\
			$(VENV_BIN)/uv run --python $(pyver) python -m maturin build -m ngram_polars/Cargo.toml --release -i python$(pyver) --strip --target $(target) --manylinux 2014 --zig --out $(DIST_DIR) ;\
		)\
	)
	$(foreach target,$(WINDOWS_TARGETS),\
		$(foreach pyver,$(PYTHON_VERSIONS),\
			$(call create_venv_py, $(pyver)) ;\
			RUSTFLAGS="$(RUSTFLAGS_X86_64_V3)" $(VENV_BIN)/uv run --python $(pyver) python -m maturin build -m ngram_polars/Cargo.toml --release -i python$(pyver) --strip --target $(target) --out $(DIST_DIR) ;\
		)\
	)

.PHONY: publish_ngram_rs
publish_iban_validation_rs: test
	cargo doc
	cargo publish -p ngram_rs 

.PHONY: publishing_pypi
publishing_pypi:
	$(VENV_BIN)/python -m twine upload $(DIST_DIR)/* --verbose

.PHONY: publishing_testpypi
publishing_testpypi:
	$(VENV_BIN)/python -m twine upload --repository-url https://test.pypi.org/legacy/ $(DIST_DIR)/* --verbose


.PHONY: clean
clean:
	rustup update
	rustup component add llvm-tools-preview
	cargo clean
	$(RMRF) .pytest_cache
	$(RMRF) .venv
	$(RMRF) target
	$(RMRF) $(DIST_DIR)

.PHONY: clippy
clippy:
	cargo update
	cargo fmt -p ngram_rs
	cargo fmt -p ngram_polars
	cargo clippy -p ngram_rs
	cargo clippy -p ngram_polars
