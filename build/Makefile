LIBS := $(shell for pkg in `grep -Po '"(\.\./)+libs/hermes/hermes\..*"' pyproject.toml | sed "s/\.\.\///g" | sed -e 's/"//g'`; do echo $$pkg; done)
WHEELS := $(LIBS:%=/opt/gw-iaas/%/dist/*.whl)

.PHONY: all

all: $(BUILD_DIR)/*.whl

$(BUILD_DIR)/deps: 
	mkdir -p $(BUILD_DIR)/deps

$(WHEELS): $(BUILD_DIR)/deps 
	mkdir -p $(@D)
	cd $(@D)/../ && poetry build
	cp $@ ${BUILD_DIR}/deps/.

$(BUILD_DIR)/%.whl: $(BUILD_DIR)/deps/*.whl
	poetry build
	cp dist/*.whl $(BUILD_DIR)

$(BUILD_DIR)/deps/%.whl: $(WHEELS)
	@echo $@ > /dev/null
