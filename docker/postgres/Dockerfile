# Same base as before (postgres:16-alpine), so the text collation of an existing
# data volume does not change. pgvector is compiled in.
FROM postgres:16-alpine

ARG PGVECTOR_VERSION=v0.8.0

RUN apk add --no-cache --virtual .build-deps git build-base \
    && git clone --depth 1 --branch ${PGVECTOR_VERSION} https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make with_llvm=no \
    && make with_llvm=no install \
    && cd / && rm -rf /tmp/pgvector \
    && apk del .build-deps
