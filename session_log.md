# CacheFlow Self-Analysis
**Started:** 2026-06-03 16:36:58  

---

## Q1

**Walk me through the full architecture of this codebase. What are the key modules and how do they connect?**

**Tokens used:** 2117 &nbsp;|&nbsp; **Saved:** 0 &nbsp;|&nbsp; **Time:** 61s &nbsp;|&nbsp; *first session — full codebase ingested across chunks*

I'll walk you through the key modules and how they connect in this codebase:

### Key Modules

1. **CacheFlowConfig**: Configuration management for the CacheFlow project. It handles settings like model paths, context sizes, and GPU layers.
2. **CodeItem**: Represents a code unit such as a function, class, module, or pattern. It includes metadata like type, name, signature, location, and body.
3. **CodeIndexer**: Extracts metadata from the codebase. It walks through the code files and extracts function and class information.
4. **CodeItem**: Represents a code unit such as a function, class, module, or pattern. It includes metadata like type, name, signature, location, and body.
5. **LlamaServer**: Manages a llama server subprocess, providing an API to interact with the server. It handles starting the server, sending completion requests, and saving KV cache snapshots.
6. **cli**: The main command-line interface function for CacheFlow.
7. **_ingest_codebase_progressively**: A method in the agent module that processes the codebase by breaking it into chunks, sending them to the llama server, and accumulating knowledge in the KV cache.

### Connections Between Modules

1. **CacheFlowConfig**: The configuration is used by other modules to set up and manage the CacheFlow project.
2. **CodeIndexer**: The indexer is responsible for extracting code metadata, which is then used by other modules.
3. **LlamaServer**: The llama server is managed by the LlamaServer class, and its completion API is used by the agent to process the codebase.
4. **cli**: The command-line interface initializes and configures the CacheFlow project using CacheFlowConfig and interacts with other modules as needed.
5. **_ingest_codebase_progressively**: This method orchestrates the ingestion process, using the CodeIndexer to extract metadata and the LlamaServer to process it.

### Key Processes

1. **Code Ingestion**:
   - **CodeIndexer**: Extracts code metadata from the codebase.
   - **_ingest_codebase_progressively**: Processes the extracted metadata, breaking it into chunks, and sending them to the llama server.
   - **LlamaServer**: Receives chunks, processes them, and accumulates knowledge in the KV cache.

2. **Configuration Management**:
   - **CacheFlowConfig**: Manages and validates configuration settings used throughout the application.

3. **Completion API**:
   - **LlamaServer**: Provides the completion API to interact with the llama server subprocess.

### Summary

The codebase is structured around a set of modules that work together to manage a CacheFlow project. The main components include configuration management, code indexing, and interaction with a llama server. The `CacheFlowConfig` class handles project settings, `CodeIndexer` extracts code metadata, and `_ingest_codebase_progress

---

## Q2

**How does KV cache persistence work end to end? Trace a single `cf run` from CLI call to snapshot saved on disk.**

**Tokens used:** 1835 &nbsp;|&nbsp; **Saved:** 73 &nbsp;|&nbsp; **Time:** 59s

To trace a single `cf run` from the CLI call to snapshot saved on disk, we can follow the call graph and understand the sequence of function calls and data flow.

1. **CLI Call**: The user runs `cf run`, which triggers the `cli()` function in `cacheflow/cli.py`. This function initializes the CacheFlow application.

2. **Agent Initialization**: The `Agent` class is instantiated, and its methods are called to gather and process the codebase.

3. **File Collection**: The `_collect_source_files()` method in `agent.py` is called to gather all source files from the project, excluding generated or vendor directories.

4. **File Chunking**: The `_chunk_files_for_ingestion()` method is called to chunk the collected files into smaller pieces that fit within the context window.

5. **Ingestion Loop**: The `_ingest_codebase_progressively()` method is called. It feeds the codebase into the model across multiple passes, accumulating knowledge in the KV cache between each pass. Each pass restores the previous KV state, ensuring all files receive full attention.

6. **Model Completion**: The `completion()` method in `llama_server_custom.py` is called to generate a response based on the current state of the KV cache and the current chunk of code.

7. **Save KV Cache**: After each chunk, the `save_slot()` method in `llama_server_custom.py` is called to save the current state of the KV cache to disk. This ensures that the next pass builds on the previously accumulated knowledge.

8. **Snapshot Saved**: The saved KV cache state is serialized and saved to disk with a unique filename.

In summary, the end-to-end process of KV cache persistence for a single `cf run` involves collecting source files, chunking them, and sequentially feeding them into the model while saving the KV cache state after each chunk to ensure persistence across multiple passes.

---

## Q3

**What does the SlotPool do and what happens when all 8 slots are occupied? Be specific about the LRU eviction code.**

**Tokens used:** 1978 &nbsp;|&nbsp; **Saved:** 139 &nbsp;|&nbsp; **Time:** 65s

The `SlotPool` class manages multiple KV cache slots for concurrent agent execution. It ensures that agents have exclusive access to slots while also implementing an LRU eviction policy to manage the limited number of slots. When all 8 slots are occupied, the system must evict the least recently used (LRU) slot to allocate a new one.

### Key Functions and Behavior:

1. **Initialization (`__init__`)**:
   - Initializes the slot pool with a specified number of slots (default is 8).
   - Creates a dictionary to track `SlotState` objects for each slot.
   - Stores a mapping between agent IDs and their assigned slot IDs.
   - Initializes a lock to ensure thread safety during concurrent access.

2. **Acquiring a Slot (`acquire_slot`)**:
   - If an agent already has a slot, it returns the existing slot.
   - If there is an available slot, it allocates a new one.
   - If all slots are in use, it finds the LRU slot, evicts it, and assigns it to the new agent.

3. **LRU Eviction**:
   - The `_find_lru_slot` method is called when all slots are occupied.
   - It iterates through the slots to find the one with the oldest `access_time` and marks it as LRU.
   - The LRU slot is then evicted, and its state is updated to indicate it is available for reuse.

### Relevant Code Snippets:

#### `_find_lru_slot` Method:
```python
def _find_lru_slot(self):
    """Find the least recently used slot."""
    min_access_time = float('inf')
    lru_slot_id = None
    for slot_id, slot_state in self.slots.items():
        if slot_state.access_time < min_access_time:
            min_access_time = slot_state.access_time
            lru_slot_id = slot_id
    return lru_slot_id
```

#### Slot Eviction in `acquire_slot`:
```python
def acquire_slot(self, agent_id: UUID) -> SlotLease:
    with self._lock:
        if agent_id in self.agent_slot_map:
            slot_id = self.agent_slot_map[agent_id]
            slot_state = self.slots[slot_id]
            slot_state.access_time = time.time()
            return SlotLease(slot_id=slot_id, pool=self)

        if len(self.agent_slot_map) < self.max_slots:
            for slot_id, slot_state in self.slots.items():
                if slot_id not in self.agent_slot_map:
                    slot_state.access_time = time.time()
                    self.agent_slot_map[agent_id] = slot_id
                    return SlotLease(slot_id=slot_id, pool=self)

        lru_slot_id = self._find_lru_slot()
        self.agent_slot_map.pop(next(iter(self.agent_slot_map.keys())))
        self.agent_slot_map[agent_id

---

## Q4

**How does the compressor decide when to consolidate? Walk through the exact threshold logic and what it does.**

**Tokens used:** 2059 &nbsp;|&nbsp; **Saved:** 0 &nbsp;|&nbsp; **Time:** 52s

To determine when to consolidate, the `Compressor` class checks if the agent's context exceeds a certain threshold. Here's the detailed breakdown of the process:

1. **Threshold Calculation**:
   - The threshold is calculated as 70% of the agent's context size (`ctx_size`). This is done in the `needs_compaction` method:
     ```python
     threshold = int(0.7 * agent.ctx_size)
     ```

2. **Token Counting**:
   - The compressor retrieves the commit history for the given agent using the `store.get_commit_history(agent)` method.
   - It then iterates over the commits to count the total number of tokens from the last consolidation commit forward. This is to ensure that the count resets after each consolidation.

3. **Threshold Comparison**:
   - The total tokens counted are compared with the threshold. If the total tokens exceed the threshold, `needs_compaction` returns `True`, indicating that compaction is needed. Otherwise, it returns `False`.

This logic ensures that the context is only consolidated when the agent's context size is significantly larger than the configured threshold, preventing unnecessary compaction.

### What I Learned About the Codebase

1. **Threshold Configuration**:
   - The threshold for compaction is hardcoded at 70% of the context size, which is a specific configuration that might need to be adjusted based on usage patterns or performance requirements.

2. **Commit History**:
   - The compressor uses the commit history to track the number of tokens processed since the last consolidation. This history is crucial for determining when to trigger consolidation.

3. **Asynchronous Executor**:
   - The `Compressor` class uses a `ThreadPoolExecutor` for asynchronous operations, ensuring that the consolidation process does not block the main thread.

4. **Resource Management**:
   - The `__del__` method ensures that the thread executor is properly shut down when the `Compressor` instance is destroyed, preventing resource leaks.

This understanding helps in comprehending how the system manages agent contexts and when to perform consolidations, which is essential for maintaining efficient resource usage and performance.

---

## Q5

**How does semantic search work across snapshots? What does the retriever do and how are embeddings stored?**

**Tokens used:** 1981 &nbsp;|&nbsp; **Saved:** 136 &nbsp;|&nbsp; **Time:** 67s

### Summary of Task Completion

**Task:** Understand how semantic search works across snapshots, including the roles of the retriever and how embeddings are stored.

**Solution:**

1. **API Query Function (`api_query` in `cacheflow/dashboard.py`)**:
   - The `api_query` function is the entry point for the semantic search API. It retrieves the query text and agent name from the request parameters.
   - It initializes the `CacheFlowStore` and `SnapshotQueryEngine` to perform the search.
   - The `SnapshotQueryEngine`'s `query` method is called, which processes the query text and returns a list of `SnapshotMatch` objects.

2. **Snapshot Query Engine (`SnapshotQueryEngine` in `cacheflow/snapshot_query.py`)**:
   - The `SnapshotQueryEngine` class contains the logic for performing semantic search.
   - The `query` method first encodes the query text into an embedding using an embedding model.
   - It then searches the database for snapshots that match the query embedding, returning the top `k` results based on semantic similarity.

3. **Snapshot Match (`SnapshotMatch` in `cacheflow/snapshot_query.py`)**:
   - This class represents a single match in the semantic search, containing details such as commit ID, agent name, task, short summary, and score.

4. **Snapshot Embedding (`SnapshotEmbedding` in `cacheflow/store.py`)**:
   - This class represents the semantic embedding and knowledge facets for a snapshot, stored in the database.
   - The `save_snapshot_embedding` function saves a new `SnapshotEmbedding` to the database.

5. **CacheFlow Store (`CacheFlowStore` in `cacheflow/store.py`)**:
   - This class manages the SQLite database for storing snapshot embeddings and other related data.
   - It provides methods for accessing and manipulating the database.

### Key Learnings

1. **Semantic Search Workflow**:
   - The semantic search workflow involves encoding the query text into an embedding.
   - The embedding is then used to search the database for matching snapshots.
   - The results are returned as a list of `SnapshotMatch` objects, each containing relevant details about the matching snapshot.

2. **Embedding Storage**:
   - Snapshots are stored with their embeddings and associated metadata in the SQLite database.
   - The `SnapshotEmbedding` class defines the structure of these stored embeddings.
   - The `save_snapshot_embedding` function handles the process of saving new embeddings to the database.

3. **Database Management**:
   - The `CacheFlowStore` class manages the SQLite database, providing methods for querying and manipulating the data.
   - This ensures that the search functionality can efficiently access and retrieve the necessary data.

4. **Error Handling**:
   - The `api_query` function includes basic error handling, returning a JSON response with an error

---


**Done:** 16:42:05
