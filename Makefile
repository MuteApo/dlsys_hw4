.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@if not exist build mkdir build
	@cd build && cmake ..
	@cd build && $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	@rmdir /s /q build
	@cd python/needle/backend_ndarray
	@del /q ndarray_backend*.pyd ndarray_backend*.manifest
