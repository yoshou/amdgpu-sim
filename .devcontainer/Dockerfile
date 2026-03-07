FROM mcr.microsoft.com/devcontainers/rust:2-1-trixie

ARG LLVM_VERSION=20

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gpg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/apt-llvm-org.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/apt-llvm-org.gpg] http://apt.llvm.org/trixie/ llvm-toolchain-trixie-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-${LLVM_VERSION} \
        lld-${LLVM_VERSION} \
        llvm-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_SYS_201_PREFIX=/usr/lib/llvm-20
ENV PATH=/usr/lib/llvm-20/bin:${PATH}
