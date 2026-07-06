VSAGYM_DIR := vsa-gym-wrapper
VSAGYM_URL := https://github.com/ctn-waterloo/vsa-gym-wrapper

.PHONY: setup vsagym sync clean-vsagym

# Clone vsa-gym-wrapper if it isn't already present, then sync the env.
setup: vsagym sync

# Clone the local editable dependency only if it's missing.
vsagym:
	@if [ -d "$(VSAGYM_DIR)" ]; then \
		echo "$(VSAGYM_DIR)/ already present, skipping clone."; \
	else \
		echo "Cloning $(VSAGYM_URL) into $(VSAGYM_DIR)/ ..."; \
		git clone "$(VSAGYM_URL)" "$(VSAGYM_DIR)"; \
	fi

sync:
	uv sync

clean-vsagym:
	rm -rf "$(VSAGYM_DIR)"
