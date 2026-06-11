#!/bin/bash
set -e
cd /Users/kuy/Projects/storytime/spike/interp
cargo build --release 2>&1 | grep -E '^error' && exit 1
cd /Users/kuy/Projects/storytime
exec ./spike/interp/target/release/interp "$@"
