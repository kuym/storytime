BUILD_DIR := .build/release
LIB_OUT := lib
HEADER_OUT := include

.PHONY: all clean

all: $(LIB_OUT)/libKokoroMLX.a

$(LIB_OUT)/libKokoroMLX.a: Sources/KokoroMLX/*.swift Package.swift
	swift build -c release 2>&1
	mkdir -p $(LIB_OUT)
	# SPM produces the static library under the build artifacts
	cp $(BUILD_DIR)/libKokoroMLX.a $(LIB_OUT)/ 2>/dev/null || \
		cp $(BUILD_DIR)/KokoroMLX.o $(LIB_OUT)/ 2>/dev/null || \
		echo "Note: library may be at a different path; check .build/release/"

clean:
	swift package clean
	rm -rf $(LIB_OUT)/libKokoroMLX.a
