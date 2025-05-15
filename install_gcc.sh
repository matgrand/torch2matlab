# 1. Set variables
export GCC_VERSION=13.2.0
export PREFIX="$HOME/gcc-local"
export TMP_DIR="$HOME/tmp-gcc-build"
export PATH="$PREFIX/bin:$PATH"

# 2. Create build and install dirs
mkdir -p "$TMP_DIR" "$PREFIX"

# 3. Download GCC source
cd "$TMP_DIR"
wget https://ftp.gnu.org/gnu/gcc/gcc-$GCC_VERSION/gcc-$GCC_VERSION.tar.gz

# 4. Extract it
tar -xf gcc-$GCC_VERSION.tar.gz
cd gcc-$GCC_VERSION

# 5. Download prerequisites (GMP, MPFR, MPC, ISL)
./contrib/download_prerequisites

# 6. Create a separate build directory
mkdir build && cd build

# 7. Configure the build
../configure --prefix="$PREFIX" --disable-multilib --enable-languages=c,c++

# 8. Compile (this takes time)
make -j$(nproc)

# 9. Install to your home directory
make install

# 10. Add to your environment
echo "export PATH=$PREFIX/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$PREFIX/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# 11. Verify
gcc --version
