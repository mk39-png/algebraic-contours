FROM ubuntu:22.04

# Set image CMake version
ARG CMAKE_VERSION=3.29.9

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    gnupg \
    software-properties-common \
    pkg-config \
    libeigen3-dev \
    libspdlog-dev \
    libsuitesparse-dev \
    libgl1-mesa-dev \
    libglfw3-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    xvfb \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*


# Install CMake 3.29.9 manually
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.9/cmake-3.29.9-linux-x86_64.sh 
RUN sh cmake-3.29.9-linux-x86_64.sh --skip-license --prefix=/usr/local 
RUN rm cmake-3.29.9-linux-x86_64.sh

# Verify CMake version
RUN cmake --version

# Set working directory and copy files over to Docker image
WORKDIR /opt/algebraic-contours
COPY . .

# Create build directory
RUN mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j $(nproc)

# Add the built executables to PATH
ENV PATH="/opt/algebraic-contours/build/bin:${PATH}"

# Keep container alive indefinitely
CMD ["tail", "-f", "/dev/null"]
