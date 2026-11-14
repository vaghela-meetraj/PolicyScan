# Cloud Vector Database Migration: FAISS to Qdrant

This document specifies the architecture, data structures, protocol trade-offs, and operational procedures for migrating the vector retrieval subsystem from an embedded local index (FAISS) to a managed cloud vector database (Qdrant Cloud).

---

## 1. System Overview and Rationale

The vector storage layer provides semantic indexing and similarity retrieval over chunked insurance policy documents. 

### Architectural Comparison: Embedded Index vs. Cloud Vector Database

| Dimension | Embedded Local Index (FAISS) | Managed Cloud Vector Database (Qdrant Cloud) |
| :--- | :--- | :--- |
| **Execution Context** | In-process C++ library (`faiss-cpu`) linked via Python bindings | Independent distributed service accessed over network APIs |
| **Persistence Boundary** | Local file system (`data/faiss_index/index.faiss`, `index.pkl`) | Cloud-hosted multi-tenant storage engine with write-ahead logs |
| **Scalability Model** | Vertical scaling constrained by local host memory | Elastic horizontal clustering with distributed sharding |
| **Metadata Filtering** | Limited; requires auxiliary serialization (Pickle) | Native JSON payload filtering via structured indexing |
| **Concurrency & Multi-Worker** | File contention and deserialization overhead across WSGI workers | Concurrent read/write with atomic updates and ACID semantics |
| **Deployment Portability** | Requires committing/mounting large binary files into containers | Fully decoupled; application container remains stateless |
| **Operational Visibility** | Opaque binary files with no visual inspection tooling | Web-based management console for collection and point telemetry |

---

## 2. Core Architecture and Network Protocols

Qdrant exposes two distinct transport protocols for communication:
1. **gRPC Protocol (Port 6334):** High-throughput binary serialization using Protocol Buffers over HTTP/2.
2. **REST HTTP/JSON Protocol (Port 6333 / HTTPS 443):** Standard RESTful endpoints supporting JSON payloads.

### Protocol Selection: REST vs. gRPC

In the PolicyScan architecture, communication uses standard HTTPS REST:

```python
QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
)
```

**Technical Rationale:**
- **Zero-Dependency Transport:** Standard HTTP REST uses pure HTTPS sockets, avoiding platform-specific C-extensions (`grpc._cython`) that can trigger binary execution blocks under enterprise OS security controls or Python pre-releases.
- **Firewall & Proxy Compatibility:** Cloud deployments behind standard corporate HTTP proxies or serverless platforms communicate seamlessly over HTTPS port 443 without requiring custom gRPC streaming configurations.
- **Latency Profile:** For document chunk counts typical in policy analysis (<10,000 vectors), query latency differences between REST and gRPC are sub-millisecond, while operational reliability is substantially higher.

---

## 3. Implementation Specification

### Configuration Layer (`insurance_assistant/config.py`)

The cloud connection parameters are sourced from environment variables:

```python
## Cloud Vector DB (Qdrant Cloud)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "Insurance_Assistant")
```

The validation routine `check_api_keys()` enforces that both `QDRANT_API_KEY` and `QDRANT_URL` are populated at application boot, failing fast before vector operations commence.

### Vector Store Module (`insurance_assistant/vector_store.py`)

The module provides four primary interfaces:

1. **`vector_store_exists(collection_name)`:**
   Invokes `client.collection_exists(collection_name)` against Qdrant Cloud. Returns a boolean without downloading index metadata.

2. **`build_vector_store(chunks, collection_name)`:**
   Executes document embedding via Jina AI (`jina-embeddings-v2-base-en`), initializes the remote collection with matching vector dimensions (768), and uploads points with metadata payloads in batched HTTP PUT operations.

3. **`load_vector_store(collection_name)`:**
   Binds to the existing cloud collection using `QdrantVectorStore.from_existing_collection()`. No vector re-indexing occurs.

4. **`get_retriever(vector_store, k)`:**
   Configures a top-$k$ similarity retriever returning standard `langchain_core.documents.Document` instances.

---

## 4. Migration Strategy and Backward Compatibility

To preserve operational rollback capability, legacy FAISS routines were preserved in-source as documented comments:

```python
# ==========================================
# Legacy FAISS Vector Store (Deprecated)
# ==========================================
# import os
# from langchain_community.vectorstores import FAISS
# ...
```

### Persistence Decoupling
Under FAISS, the pipeline required an explicit disk write step:
```python
# Legacy FAISS:
# vector_store = build_vector_store(chunks)
# save_vector_store(vector_store)
```

Under Qdrant Cloud, points are committed immediately upon ingestion with synchronous confirmation (`wait=true`). The `save_vector_store` function is maintained as a no-op stub for pipeline interface compatibility.

---

## 5. Security and Operational Standards

1. **API Key Isolation:** Cloud credentials must reside exclusively in `.env` and `.streamlit/secrets.toml`. They must never be hardcoded or checked into version control.
2. **Collection Lifecycle Management:**
   - Collection creation is idempotent; `vector_store_exists` prevents accidental point duplication.
   - For re-indexing updated policy documents, the collection can be purged via the Qdrant Cloud console or the administrative client method `client.delete_collection(collection_name)`.
3. **Stateless Containers:** Migrating off local FAISS files (`data/faiss_index/`) enables complete container ephemerality. Docker containers can scale horizontally across multiple instances without needing shared persistent volume mounts.

