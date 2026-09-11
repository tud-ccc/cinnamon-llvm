#!/usr/bin/env bash
set -euo pipefail

# Builds the prebuilt LLVM that Cinnamon (https://github.com/tud-ccc/Cinnamon)
# downloads instead of building LLVM itself: configures an LLVM checkout with
# llvm-config.cmake, builds and installs it, and packages the install tree so
# that it works wherever it is unpacked. Runs inside the pixi environment next
# to this file; .github/workflows/cinnamon-prebuilt.yml is its main user.
#
# Usage: build-prebuilt.sh SOURCE_DIR BUILD_DIR OUTPUT_DIR
#
# Cinnamon's build-llvm.sh relies on what this produces, so keep the two in
# sync. With <name> being cinnamon-llvm-<first 12 digits of the commit>-linux-x86_64:
#   OUTPUT_DIR/<name>.tar.zst         the install tree, in a directory <name>
#   OUTPUT_DIR/<name>.tar.zst.sha256
#   <name>/cinnamon-llvm-revision     the full commit it was built from
#   <name>/cinnamon-llvm-python       the Python version of the MLIR bindings
# The workflow publishes them in a release tagged cinnamon-<first 12 digits>.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 SOURCE_DIR BUILD_DIR OUTPUT_DIR" >&2
  exit 1
fi
script_dir="$(cd -- "$(dirname "$0")" && pwd -P)"
: "${CONDA_PREFIX:?Run this inside the pixi environment in $script_dir}"
source_dir="$(cd -- "$1" && pwd -P)"
mkdir -p "$2" "$3"
build_dir="$(cd -- "$2" && pwd -P)"
out_dir="$(cd -- "$3" && pwd -P)"

revision="$(git -C "$source_dir" rev-parse HEAD)"
name="cinnamon-llvm-${revision:0:12}-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
prefix="$out_dir/$name"
python="$(command -v python3)"

# The configuration of the commit being built, if it has one. Commits from
# before it existed get the current one.
config="$source_dir/cinnamon/llvm-config.cmake"
[[ -f "$config" ]] || config="$script_dir/llvm-config.cmake"

# The tools built along the way (the tablegens above all) need the pixi
# toolchain's libstdc++, which is newer than the system's. Once installed, they
# use the copy bundled below instead.
env_rpath="-Wl,-rpath,$CONDA_PREFIX/lib"
cmake -S "$source_dir/llvm" -B "$build_dir" -G Ninja -Wno-dev \
  -C "$config" \
  -DLLVM_CCACHE_BUILD=ON \
  -DPython3_EXECUTABLE="$python" \
  -DCMAKE_EXE_LINKER_FLAGS="$env_rpath" \
  -DCMAKE_SHARED_LINKER_FLAGS="$env_rpath" \
  -DCMAKE_MODULE_LINKER_FLAGS="$env_rpath"
cmake --build "$build_dir"

rm -rf "$prefix"
cmake --install "$build_dir" --prefix "$prefix" >/dev/null

# With LLVM_INSTALL_UTILS, an installed LLVM tells the projects that test
# against it to run lit as bin/llvm-lit, but nothing installs that. Ship lit
# itself, and a launcher for it.
mkdir -p "$prefix/share/llvm-lit"
cp -r "$source_dir/llvm/utils/lit/lit" "$prefix/share/llvm-lit/"
find "$prefix/share/llvm-lit" -name __pycache__ -prune -exec rm -rf {} +
cat > "$prefix/bin/llvm-lit" <<'EOF'
#!/usr/bin/env python3
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "share", "llvm-lit")
)

from lit.main import main

if __name__ == "__main__":
    main()
EOF
chmod +x "$prefix/bin/llvm-lit"

# Bundle every library LLVM needs from the pixi environment (the toolchain's
# libstdc++ and libgcc_s, zlib) next to LLVM's own libraries, where the
# installed binaries find them through their $ORIGIN/../lib RPATH. Repeat until
# nothing is added, as those libraries have dependencies of their own.
needed_libs() {
  # Not every file is an ELF file; readelf skips the others with a complaint.
  find "$prefix" -type f \( -name '*.so*' -o -perm -u+x \) -exec readelf -d {} + 2>/dev/null \
    | sed -n 's/.*(NEEDED).*\[\(.*\)\]$/\1/p' \
    | sort -u
}
added=1
while [[ "$added" -eq 1 ]]; do
  added=0
  while IFS= read -r lib; do
    if [[ -e "$CONDA_PREFIX/lib/$lib" && ! -e "$prefix/lib/$lib" ]]; then
      echo "Bundling $lib"
      cp -L "$CONDA_PREFIX/lib/$lib" "$prefix/lib/"
      added=1
    fi
  done < <(needed_libs)
done

echo "$revision" > "$prefix/cinnamon-llvm-revision"
"$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' > "$prefix/cinnamon-llvm-python"

echo "Install tree: $(du -sh "$prefix" | cut -f1)"
tar -C "$out_dir" -cf - "$name" | zstd -q -T0 -19 -f -o "$out_dir/$name.tar.zst"
(cd "$out_dir" && sha256sum "$name.tar.zst" > "$name.tar.zst.sha256")
rm -rf "$prefix"
echo "Package: $(du -h "$out_dir/$name.tar.zst" | cut -f1)"
