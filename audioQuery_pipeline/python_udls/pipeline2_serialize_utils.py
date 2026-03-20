import numpy as np
import warnings
warnings.filterwarnings("ignore")


def utf8_length(s: str) -> int:
    """Computes the length of a UTF-8 encoded string without actually encoding it."""
    return sum(1 + (ord(c) >= 0x80) + (ord(c) >= 0x800) + (ord(c) >= 0x10000) for c in s)

# ------------------------    STEP A (speech2text) UDL batcher  -------------------------

class AudioBatcher:
    def __init__(self):
        self.audio_data = [] # List of np.ndarray float64
        self.question_ids = []
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        
        # variable used by serialization
        self.audio_pos = {}  # qid -> (offset, length)

    def serialize(self) -> np.ndarray:
        """
        Serializes the batch into a contiguous buffer:
        - Header: 8 bytes for batch_size (int64).
        - Metadata: batch_size × (offset, length) int64 pairs.
        - Question IDs: batch_size × int64.
        - Audio data: concatenated raw float64 arrays.
        """
        assert len(self.audio_data) == len(self.question_ids), "Mismatch in audio and question_ids"
        batch_size = len(self.audio_data)

        # Sanitize and standardize input
        sanitized_audio = []
        sanitized_qids = []
        for qid, audio in zip(self.question_ids, self.audio_data):
            if not isinstance(audio, np.ndarray):
                continue
            if audio.dtype != np.float64:
                audio = audio.astype(np.float64)
            if not audio.flags['C_CONTIGUOUS']:
                audio = np.ascontiguousarray(audio)
            if audio.nbytes == 0:
                continue
            sanitized_audio.append(audio)
            sanitized_qids.append(qid)

        self.audio_data = sanitized_audio
        self.question_ids = sanitized_qids
        batch_size = len(self.audio_data)

        # Sizes and layout
        header_size = np.dtype(np.int64).itemsize  # 8 bytes
        metadata_dtype = np.dtype([("offset", np.int64), ("length", np.int64)])
        metadata_size = batch_size * metadata_dtype.itemsize  # 16 * N
        qid_size = batch_size * np.dtype(np.int64).itemsize   # 8 * N
        fixed_size = header_size + metadata_size + qid_size

        variable_data_offset = fixed_size
        total_audio_bytes = sum(audio.nbytes for audio in self.audio_data)
        total_size = fixed_size + total_audio_bytes

        # Allocate the final buffer
        buffer = np.zeros(total_size, dtype=np.uint8)
        assert buffer.ctypes.data % 8 == 0, "Buffer not 8-byte aligned"

        # Header (int64 batch size)
        buffer[:8] = np.array([batch_size], dtype=np.int64).view(np.uint8)

        # Metadata view
        metadata_view = buffer[8 : 8 + metadata_size].view(metadata_dtype)
        qid_view = buffer[8 + metadata_size : fixed_size].view(np.int64)
        qid_view[:] = self.question_ids

        # Audio data segment
        write_ptr = fixed_size
        for i, audio_array in enumerate(self.audio_data):
            audio_bytes = audio_array.nbytes
            assert audio_bytes % 8 == 0
            assert write_ptr % 8 == 0

            metadata_view[i] = (write_ptr - variable_data_offset, audio_bytes)
            buffer[write_ptr : write_ptr + audio_bytes] = audio_array.view(np.uint8)
            write_ptr += audio_bytes

        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Reconstructs the audio batch from a serialized buffer.
        """
        self._bytes = data

        header_size = np.dtype(np.int64).itemsize  # 8 bytes
        metadata_dtype = np.dtype([("offset", np.int64), ("length", np.int64)])
        batch_size = np.frombuffer(data[:8], dtype=np.int64)[0]

        metadata_size = batch_size * metadata_dtype.itemsize
        qid_size = batch_size * np.dtype(np.int64).itemsize
        fixed_size = header_size + metadata_size + qid_size
        variable_data_offset = fixed_size

        metadata_view = np.frombuffer(data[8 : 8 + metadata_size], dtype=metadata_dtype)
        qid_view = np.frombuffer(data[8 + metadata_size : fixed_size], dtype=np.int64)

        self.question_ids = qid_view.tolist()
        self.audio_data = []

        for i in range(batch_size):
            offset = metadata_view[i]["offset"]
            length = metadata_view[i]["length"]
            start = variable_data_offset + offset
            end = start + length
            audio_array = np.frombuffer(data[start:end], dtype=np.float64)
            self.audio_data.append(audio_array)

    def print_info(self):
        print(f"AudioBatcher: {len(self.audio_data)} audio samples")
        for i, audio in enumerate(self.audio_data):
            print(f"  audio[{i}]: shape={audio.shape}, dtype={audio.dtype}, size={audio.size}, bytes={audio.nbytes}")
        print(f"  question IDs: {self.question_ids}")


class PendingAudioRecDataBatcher:
    '''
    Super batch of AudioBatcher, used for model runtime batch process.
    '''
    def __init__(self, batch_size: int,len_limit_for_batching: int = 200000):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []  # List[int]
        self.audio_data = []   # List[np.ndarray]
        self.len_limit_for_batching = len_limit_for_batching
        self.contain_large_audio = False
    
    def space_left(self):
        return self.max_batch_size - self.num_pending
        
    # def space_left_large(self, cur_arr_len):
    #     # for large array audio, it cannot be processed in batch and needs to have a batch to it self
    #     if cur_arr_len > self.len_limit_for_batching:
    #         # only add to pending if this batch is empty
    #         if self.num_pending == 0 :
    #             self.contain_large_audio = True
    #             return 1
    #         else:
    #             return 0
    #     if self.contain_large_audio:
    #         # If the batch contains large audio, we need to ensure that we have enough space for it
    #         return 0
    #     return self.max_batch_size - self.num_pending
    
    
    def add_data(self, audioBatcher, start_pos):
        num_to_add = min(self.space_left(), len(audioBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(audioBatcher.question_ids[start_pos:end_pos])
        # Make deep copies of each audio array as it's added, 
        #  because memory of list[np array] is managed by numpy, leading to memory corruption if not copy out
        for i in range(start_pos, end_pos):
            # Create a new array with its own memory
            new_array = np.array(audioBatcher.audio_data[i], copy=True)
            
        self.audio_data.append(new_array)
        self.num_pending += num_to_add
        return end_pos
    
    def add_single_data(self, audioBatcher, start_pos):
        num_to_add = min(self.space_left(), 1, len(audioBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(audioBatcher.question_ids[start_pos:end_pos])
        # Make deep copies of each audio array as it's added, 
        #  because memory of list[np array] is managed by numpy, leading to memory corruption if not copy out
        for i in range(start_pos, end_pos):
            # Create a new array with its own memory
            new_array = np.array(audioBatcher.audio_data[i], copy=True)
            
        self.audio_data.append(new_array)
        self.num_pending += num_to_add
        return end_pos
    
    def reset(self):
        self.question_ids = []
        self.audio_data = []
        self.num_pending = 0


class QueryBatcherManager:
    """
    Batches of text queries to pass results from generated by audio recognition to text encoder UDL
    """
    def __init__(self):
        self.question_ids = []   # List[int] of length batch size
        self.queries = []        # List[str] of length batch size
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        self.num_queries = 0
        # variable used by serialization
        self.text_pos = {}  # qid -> (offset, length)
        
    def add_result(self, question_id: int, query: str):
        """
        Add one result to the batch.
        It is assumed that each query is a string.
        """
        self.question_ids.append(question_id)
        self.queries.append(query)
        self.num_queries += 1
        
    def serialize(self, start_pos: int, end_pos: int) -> np.ndarray:
        """
        Serializes a slice of the queries from start_pos to end_pos into a contiguous buffer:
        - Header: 4 bytes for batch_size (uint32).
        - Metadata: For each text_sequence element, store two int64 values (offset and length).
        - Fixed segments:
            * question_ids: (batch_size,) int64.
        - Variable segment:
            * text_sequence: concatenated UTF-8 encoded bytes.
        """
        selected_queries = self.queries[start_pos:end_pos]
        selected_question_ids = self.question_ids[start_pos:end_pos]
        num_queries = len(selected_queries)

        header_size = np.dtype(np.uint32).itemsize

        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = num_queries * metadata_dtype.itemsize
        question_id_size = num_queries * np.dtype(np.int64).itemsize
        text_offset = header_size + metadata_size + question_id_size

        text_pos = {}
        total_utf8_size = 0
        for idx, q in enumerate(selected_queries):
            utf8_len = utf8_length(q)
            text_pos[selected_question_ids[idx]] = (text_offset, utf8_len)
            text_offset += utf8_len
            total_utf8_size += utf8_len

        total_size = header_size + metadata_size + question_id_size + total_utf8_size

        buffer = np.zeros(total_size, dtype=np.uint8)

        # Header
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = num_queries

        # Metadata
        metadata_view = np.frombuffer(buffer[header_size:header_size + metadata_size], dtype=metadata_dtype)

        # Question IDs
        question_ids_view = np.frombuffer(
            buffer[header_size + metadata_size:header_size + metadata_size + question_id_size], dtype=np.int64)
        question_ids_view[:] = selected_question_ids

        # Text sequence
        text_start = header_size + metadata_size + question_id_size
        for idx, q in enumerate(selected_queries):
            offset, length = text_pos[selected_question_ids[idx]]
            metadata_view[idx] = (offset, length)
            encoded = q.encode("utf-8")  # one copy here, still required to actually fill the buffer
            buffer[text_start:text_start + length] = np.frombuffer(encoded, dtype=np.uint8)
            text_start += length

        self._bytes = buffer
        return buffer
    
    def deserialize(self, data: np.ndarray):
        self._bytes = data
        header_size = np.dtype(np.uint32).itemsize
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        num_queries = np.frombuffer(data[:header_size], dtype=np.uint32)[0]
        metadata_size = num_queries * metadata_dtype.itemsize
        question_id_size = num_queries * np.dtype(np.int64).itemsize
        text_start = header_size + metadata_size + question_id_size 
        
        metadata_view = np.frombuffer(data[header_size:header_size + metadata_size], dtype=metadata_dtype)
        question_ids_view = np.frombuffer(data[header_size + metadata_size:header_size + metadata_size + question_id_size], dtype=np.int64)

        self.question_ids = question_ids_view.tolist()
        self.queries = []
        for idx in range(num_queries):
            text_offset, text_length = metadata_view[idx]
            self.queries.append(memoryview(data)[text_offset:text_offset + text_length].tobytes().decode("utf-8"))
            
    def print_info(self):
        print(f"QueryBatcherManager: {len(self.queries)} queries")
        for i, q in enumerate(self.queries):
            print(f" query {i}: {q}")
        print(f" question IDs: {self.question_ids}")

class PendingEncodeDataBatcher:
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []  
        self.queries = []
    
    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, queryBatcher, start_pos):
        num_to_add = min(self.space_left(), len(queryBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(queryBatcher.question_ids[start_pos:end_pos])
        self.queries.extend(queryBatcher.queries[start_pos:end_pos])
        self.num_pending += num_to_add
        return end_pos 


    def reset(self):
        self.question_ids = []
        self.queries = []
        self.num_pending = 0

# ------------------------    STEP B (Encoder) UDL batcher  -------------------------


class EncodeResultBatchManager:
    
    def __init__(self):
        self.question_ids = []  # list[int]
        self.queries = []       # list[str]
        self.embeddings_list = []  # list[np.ndarray]
        self.emb_dim = 0
        self.num_queries = 0
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        self.emb_dtype = np.float16  

    def add_result(self, question_id: int, query: str, embeddings: np.ndarray):
        self.question_ids.append(question_id)
        self.queries.append(query)
        self.embeddings_list.append(embeddings)
        self.num_queries += 1
    
    def serialize(self, start_pos, end_pos) -> np.ndarray:
        batch_size = end_pos - start_pos
        header_type = np.dtype([
            ('count', np.uint32),
            ('embeddings_start', np.uint32)
        ])
        
        metadata_type = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('embeddings_position', np.uint32),
            ('embeddings_dim', np.uint32),
        ])
        
        count = batch_size
        header_size = header_type.itemsize
        metadata_size = count * metadata_type.itemsize
        
        # Preprocess query texts
        text_bytes_list = [q.encode('utf-8') for q in self.queries[start_pos:end_pos]]
        text_lengths = [len(b) for b in text_bytes_list]
        total_text_size = sum(text_lengths)
        
        # Preprocess embeddings
        first_emb = self.embeddings_list[0]
        if len(first_emb.shape) == 1:
            self.emb_dim = first_emb.shape[0]
        else:
            self.emb_dim = first_emb.shape[1]
        emb_itemsize = np.dtype(self.emb_dtype).itemsize  
        embedding_bytes_per_query = self.emb_dim * emb_itemsize
        total_embeddings_size = count * embedding_bytes_per_query
        
        embeddings_start = header_size + metadata_size + total_text_size
        total_size = header_size + metadata_size + total_text_size + total_embeddings_size
        buffer = np.zeros(total_size, dtype=np.uint8)
        
        header_array = np.frombuffer(buffer[:header_size], dtype=header_type)
        header_array[0] = (count, embeddings_start)
        
        metadata_array = np.frombuffer(buffer[header_size:header_size + metadata_size], dtype=metadata_type)
        text_block_start = header_size + metadata_size
        embeddings_block_start = embeddings_start
        current_text_offset = 0
        
        for i in range(count):
            text_pos = text_block_start + current_text_offset
            text_len = text_lengths[i]
            emb_pos = embeddings_block_start + i * embedding_bytes_per_query
            
            metadata_array[i]['question_id'] = self.question_ids[start_pos + i]
            metadata_array[i]['text_position'] = text_pos
            metadata_array[i]['text_length'] = text_len
            metadata_array[i]['embeddings_position'] = emb_pos
            metadata_array[i]['embeddings_dim'] = self.emb_dim
            
            current_text_offset += text_len
        
        current_text_offset = 0
        for b in text_bytes_list:
            start = text_block_start + current_text_offset
            end = start + len(b)
            buffer[start:end] = np.frombuffer(b, dtype=np.uint8)
            current_text_offset += len(b)
        
        current_embedding_offset = 0
        for emb in self.embeddings_list[start_pos:end_pos]:
            start = embeddings_block_start + current_embedding_offset
            if emb.dtype != self.emb_dtype:
                emb = emb.astype(self.emb_dtype) 
            emb_bytes = emb.flatten().view(np.uint8)
            end = start + emb.nbytes
            buffer[start:end] = np.frombuffer(emb_bytes, dtype=np.uint8)
            current_embedding_offset += emb.nbytes
        
        return buffer

    def deserialize(self, buffer: np.ndarray):
        mv = memoryview(buffer)
        
        header_dtype = np.dtype([
            ('count', np.uint32),
            ('embeddings_start', np.uint32)
        ])
        metadata_dtype = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('embeddings_position', np.uint32),
            ('embeddings_dim', np.uint32),
        ])
        
        header = np.frombuffer(mv, dtype=header_dtype, count=1)[0]
        count = int(header['count'])
        self.num_queries = count
        embeddings_start = int(header['embeddings_start'])
        
        header_size = header_dtype.itemsize
        metadata_size = count * metadata_dtype.itemsize
        
        metadata = np.frombuffer(mv, dtype=metadata_dtype, count=count, offset=header_size)
      
        for rec in metadata:
            self.question_ids.append(int(rec['question_id']))
            text_pos = int(rec['text_position'])
            text_len = int(rec['text_length'])
            text_bytes = mv[text_pos:text_pos + text_len].tobytes()
            self.queries.append(text_bytes.decode('utf-8'))
            
            emb_pos = int(rec['embeddings_position'])
            self.emb_dim = int(rec['embeddings_dim'])
            embedding = np.frombuffer(mv, dtype=self.emb_dtype, count=self.emb_dim, offset=emb_pos)
            self.embeddings_list.append(embedding)
    
    def print_info(self):
        print(f"EncodeResultBatchManager: {self.num_queries} queries")
        for i, q in enumerate(self.queries):
            print(f" query {i}: {q}")
        print(f" question IDs: {self.question_ids}")
        print(f" embs:")
        for i, emb in enumerate(self.embeddings_list):
            print(f"  {i}: {emb[-5:]}")
            print(f"  type: {emb.dtype}")


# ------------------------    STEP C (Searcher) UDL batcher  -------------------------


class PendingSearchDataBatcher:
    def __init__(self, batch_size: int, emb_dim: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []
        self.queries = []
        self.embeddings = np.empty((self.max_batch_size, emb_dim), dtype=np.float16)  # CHANGED to float16

    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, embedding_batcher, start_pos):
        num_to_add = min(self.space_left(), len(embedding_batcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(embedding_batcher.question_ids[start_pos:end_pos])
        self.queries.extend(embedding_batcher.queries[start_pos:end_pos])
        
        # Ensure compatibility if embedding_batcher data isn't already float16
        for i in range(num_to_add):
            emb = embedding_batcher.embeddings_list[start_pos + i]
            if emb.dtype != np.float16:
                emb = emb.astype(np.float16)
            self.embeddings[self.num_pending + i] = emb

        self.num_pending += num_to_add
        return end_pos

    def reset(self):
        self.question_ids = []
        self.queries = []
        self.embeddings = np.empty((self.max_batch_size, self.embeddings.shape[1]), dtype=np.float16)  # CHANGED to float16
        self.num_pending = 0


class SearchResultBatchManager:
    def __init__(self):
        self.question_ids = []  # list[int]
        self.queries = []       # list[str]
        self.doc_ids = []       # list[list[int]]
        self.num_queries = 0
        self.text_pos = {}  # qid -> (offset, length)
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
    
    def add_result(self, question_id, query, doc_ids):
        self.question_ids.append(question_id)
        self.queries.append(query)
        self.doc_ids.append(doc_ids)
        self.num_queries += 1
    
    def serialize(self, start_pos, end_pos) -> np.ndarray:
        """
        Serializes the batch with the following layout:
          [Header][Metadata records][Concatenated query texts][Flattened doc_ids block]

        Header (dtype header_type):
          - count: uint32, the number of queries
          - doc_ids_start: uint32, byte offset to the start of the flattened doc_ids block

        Metadata (dtype metadata_type) for each query:
          - question_id: uint64
          - text_position: uint32 (absolute offset to the query text)
          - text_length: uint32   (length in bytes of the query text)
          - doc_ids_position: uint32 (absolute offset to the doc_ids for this query)
          - doc_ids_count: uint32    (number of doc_ids for this query)
        """
        batch_size = end_pos - start_pos
        header_dtype = np.dtype([
            ('count', np.uint32),
            ('doc_ids_start', np.uint32)
        ])
        metadata_dtype = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('doc_ids_position', np.uint32),
            ('doc_ids_count', np.uint32)
        ])
        
        count = batch_size
        header_size = header_dtype.itemsize
        metadata_size = count * metadata_dtype.itemsize

        # Preprocess query texts for the selected range.
        text_bytes_list = [q.encode('utf-8') for q in self.queries[start_pos:end_pos]]
        text_lengths = [len(b) for b in text_bytes_list]
        total_text_size = sum(text_lengths)

        # Preprocess and flatten doc_ids for the selected range.
        flattened_doc_ids = []
        doc_ids_counts = []
        for docs in self.doc_ids[start_pos:end_pos]:
            doc_ids_counts.append(len(docs))
            flattened_doc_ids.extend(docs)
        total_doc_ids_count = len(flattened_doc_ids)
        doc_ids_dtype = np.uint64
        doc_ids_itemsize = np.dtype(doc_ids_dtype).itemsize
        total_doc_ids_size = total_doc_ids_count * doc_ids_itemsize

        # Calculate the offsets.
        doc_ids_start = header_size + metadata_size + total_text_size
        total_size = header_size + metadata_size + total_text_size + total_doc_ids_size

        # Allocate the output buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)

        # Fill header.
        header_array = np.frombuffer(buffer[:header_size], dtype=header_dtype)
        header_array[0] = (count, doc_ids_start)

        # Fill metadata records.
        metadata_array = np.frombuffer(buffer[header_size:header_size+metadata_size], dtype=metadata_dtype)
        text_block_start = header_size + metadata_size
        current_text_offset = 0
        current_doc_ids_offset = 0
        for i in range(count):
            text_pos = text_block_start + current_text_offset
            text_len = text_lengths[i]
            doc_ids_pos = doc_ids_start + current_doc_ids_offset
            doc_count = doc_ids_counts[i]
            metadata_array[i] = (
                self.question_ids[start_pos + i],
                text_pos,
                text_len,
                doc_ids_pos,
                doc_count
            )
            current_text_offset += text_len
            current_doc_ids_offset += doc_count * doc_ids_itemsize

        # Write the concatenated text block.
        current_text_offset = 0
        for b in text_bytes_list:
            start = text_block_start + current_text_offset
            end = start + len(b)
            buffer[start:end] = np.frombuffer(b, dtype=np.uint8)
            current_text_offset += len(b)

        # Write the flattened doc_ids block.
        if total_doc_ids_count > 0:
            doc_ids_array = np.array(flattened_doc_ids, dtype=doc_ids_dtype)
            doc_ids_bytes = doc_ids_array.view(np.uint8)
            start = doc_ids_start
            end = start + total_doc_ids_size
            buffer[start:end] = np.frombuffer(doc_ids_bytes, dtype=np.uint8)

        return buffer


    def deserialize(self, buffer: np.ndarray):
        """
        Deserializes the given buffer and directly assigns the resulting data
        to the instance variables of self. The layout is:

          [Header][Metadata records][Concatenated query texts][Flattened doc_ids block]

        Uses memory views and np.frombuffer to avoid unnecessary data copies.
        """
        mv = memoryview(buffer)
        header_dtype = np.dtype([
            ('count', np.uint32),
            ('doc_ids_start', np.uint32)
        ])
        metadata_dtype = np.dtype([
            ('question_id', np.uint64),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('doc_ids_position', np.uint32),
            ('doc_ids_count', np.uint32)
        ])
        # Read header.
        header = np.frombuffer(mv, dtype=header_dtype, count=1)[0]
        count = int(header['count'])
        doc_ids_start = int(header['doc_ids_start'])
        header_size = header_dtype.itemsize
        metadata_size = count * metadata_dtype.itemsize

        # Read metadata as a view.
        metadata = np.frombuffer(mv, dtype=metadata_dtype, count=count, offset=header_size)

        # Directly assign deserialized values to self.
        text_block_start = header_size + metadata_size
        doc_ids_dtype = np.uint64  # as stored

        for rec in metadata:
            # Append question id.
            self.question_ids.append(int(rec['question_id']))
            # Decode and append the query text.
            text_pos = int(rec['text_position'])
            text_len = int(rec['text_length'])
            text_bytes = mv[text_pos:text_pos + text_len].tobytes()
            self.queries.append(text_bytes.decode('utf-8'))
            # Create a zero-copy view for the doc_ids.
            doc_pos = int(rec['doc_ids_position'])
            doc_count = int(rec['doc_ids_count'])
            if doc_count > 0:
                doc_ids_view = np.frombuffer(mv, dtype=doc_ids_dtype, count=doc_count, offset=doc_pos)
                self.doc_ids.append(doc_ids_view)
            else:
                self.doc_ids.append(np.array([], dtype=doc_ids_dtype))
        self.num_queries = count
        
    def print_info(self):
        print(f"SearchResultBatchManager: {self.num_queries} queries")
        for i, q in enumerate(self.queries):
            print(f" query {i}: {q}")
        print(f" question IDs: {self.question_ids}")
        print(f" doc IDs: {self.doc_ids}")
        
# ------------------------    STEP E (Loader) UDL batcher  -------------------------

class PendingLoaderDataBatcher:
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []
        self.queries = []
        self.doc_ids = []
    
    def space_left(self):
        return self.max_batch_size - self.num_pending
    
    def add_data(self, searchBatcher, start_pos):
        num_to_add = min(self.space_left(), len(searchBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        self.question_ids.extend(searchBatcher.question_ids[start_pos:end_pos])
        self.queries.extend(searchBatcher.queries[start_pos:end_pos])
        self.doc_ids.extend(searchBatcher.doc_ids[start_pos:end_pos])
        self.num_pending += num_to_add
        return end_pos

    def reset(self):
        self.question_ids = []
        self.queries = []
        self.doc_ids = []
        self.num_pending = 0

class DocResultBatchManager:
    def __init__(self):
        self.question_ids = []  # list[int]
        self.queries = []       # list[str]
        self.doc_list = []       # list[list[str]]
        self.num_queries = 0
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
    

    def add_result(self, question_id: int, query: str, doc_list: list[str]):
        self.question_ids.append(question_id)
        self.queries.append(query)
        self.doc_list.append(doc_list)
        self.num_queries += 1

    def serialize(self, start_pos: int, end_pos: int) -> np.ndarray:
        """
        Serializes a slice of results (from start_pos to end_pos) into a contiguous buffer.
        Assumes every query has the same number of documents.

        Layout:
          [Header][Query metadata][Document metadata][Fixed segment: question_ids]
          [Variable segment: concatenated query texts][Variable segment: concatenated doc texts]

        Header (8 bytes total):
          - query_count: uint32 (number of queries)
          - doc_count: uint32 (number of docs per query)

        Query metadata (per query):
          - query_text_offset: int64 (absolute offset in the buffer)
          - query_text_length: int64

        Document metadata (per document per query):
          - doc_offset: int64 (absolute offset in the buffer)
          - doc_length: int64

        Fixed segment:
          - question_ids as int64 (one per query)

        The variable segment holds the actual query and document text data.
        """
        # ---- Select slice of data ----
        selected_queries = self.queries[start_pos:end_pos]
        selected_question_ids = self.question_ids[start_pos:end_pos]
        selected_doc_lists = self.doc_list[start_pos:end_pos]
        num_queries = len(selected_queries)
        # Assume all queries have the same number of docs.
        doc_count = len(selected_doc_lists[0]) if num_queries > 0 else 0

        # ---- Define metadata types ----
        header_dtype = np.dtype([("query_count", np.uint32), ("doc_count", np.uint32)])
        header_size = header_dtype.itemsize  # 8 bytes

        query_meta_dtype = np.dtype([("query_text_offset", np.int64), ("query_text_length", np.int64)])
        query_meta_size = num_queries * query_meta_dtype.itemsize

        doc_meta_dtype = np.dtype([("doc_offset", np.int64), ("doc_length", np.int64)])
        total_doc_meta = num_queries * doc_count
        doc_meta_size = total_doc_meta * doc_meta_dtype.itemsize

        question_ids_size = num_queries * np.dtype(np.int64).itemsize

        # ---- Process variable-length data (encode each string only once) ----
        # For queries:
        query_encodings = []
        query_text_lengths = []
        for q in selected_queries:
            encoded = q.encode("utf-8")
            # Optionally verify: assert len(encoded) == utf8_length(q)
            query_encodings.append(encoded)
            query_text_lengths.append(len(encoded))
        total_query_text_size = sum(query_text_lengths)

        # For documents:
        doc_encodings_list = []   # List (per query) of lists (per document) of encoded bytes.
        doc_lengths_list = []     # Corresponding lengths.
        total_docs_size = 0
        for docs in selected_doc_lists:
            curr_encodings = []
            curr_lengths = []
            for doc in docs:
                encoded = doc.encode("utf-8")
                # Optionally verify: assert len(encoded) == utf8_length(doc)
                curr_encodings.append(encoded)
                curr_lengths.append(len(encoded))
                total_docs_size += len(encoded)
            doc_encodings_list.append(curr_encodings)
            doc_lengths_list.append(curr_lengths)

        # ---- Compute segment offsets ----
        # Fixed segment: header, query metadata, doc metadata, question_ids.
        fixed_segment_size = header_size + query_meta_size + doc_meta_size + question_ids_size
        # Variable segments follow:
        query_text_block_start = fixed_segment_size
        docs_block_start = query_text_block_start + total_query_text_size

        total_size = fixed_segment_size + total_query_text_size + total_docs_size
        buffer = np.zeros(total_size, dtype=np.uint8)

        # ---- Write Metadata First ----
        offset = 0

        # Write header.
        header_array = np.frombuffer(buffer[offset:offset + header_size], dtype=header_dtype)
        header_array[0] = (num_queries, doc_count)
        offset += header_size

        # Write query metadata.
        query_meta_view = np.frombuffer(buffer[offset: offset + query_meta_size], dtype=query_meta_dtype)
        current_q_offset = 0
        for i in range(num_queries):
            query_meta_view[i] = (query_text_block_start + current_q_offset, query_text_lengths[i])
            current_q_offset += query_text_lengths[i]
        offset += query_meta_size

        # Write document metadata.
        doc_meta_view = np.frombuffer(buffer[offset: offset + doc_meta_size], dtype=doc_meta_dtype)
        meta_idx = 0
        current_d_offset = 0
        for i in range(num_queries):
            for j in range(doc_count):
                doc_meta_view[meta_idx] = (docs_block_start + current_d_offset, doc_lengths_list[i][j])
                current_d_offset += doc_lengths_list[i][j]
                meta_idx += 1
        offset += doc_meta_size

        # Write fixed segment: question_ids.
        qids_view = np.frombuffer(buffer[offset: offset + question_ids_size], dtype=np.int64)
        qids_view[:] = selected_question_ids
        offset += question_ids_size
        # All metadata is now written.

        # ---- Write Variable Data (Actual String Data) ----
        # Write concatenated query texts.
        var_ptr = query_text_block_start
        for encoded in query_encodings:
            end_ptr = var_ptr + len(encoded)
            buffer[var_ptr:end_ptr] = np.frombuffer(encoded, dtype=np.uint8)
            var_ptr = end_ptr

        # Write concatenated document texts.
        var_ptr = docs_block_start
        for encodings in doc_encodings_list:
            for encoded in encodings:
                end_ptr = var_ptr + len(encoded)
                buffer[var_ptr:end_ptr] = np.frombuffer(encoded, dtype=np.uint8)
                var_ptr = end_ptr

        self._bytes = buffer
        return buffer


    def deserialize(self, data: np.ndarray):
        """
        Deserializes the given buffer and populates self.question_ids, self.queries, and self.doc_list.
        Expects the layout as produced by serialize.
        """
        self._bytes = data
        header_dtype = np.dtype([("query_count", np.uint32), ("doc_count", np.uint32)])
        header_size = header_dtype.itemsize
        header = np.frombuffer(data[:header_size], dtype=header_dtype)[0]
        num_queries = int(header["query_count"])
        doc_count = int(header["doc_count"])

        query_meta_dtype = np.dtype([("query_text_offset", np.int64), ("query_text_length", np.int64)])
        query_meta_size = num_queries * query_meta_dtype.itemsize

        doc_meta_dtype = np.dtype([("doc_offset", np.int64), ("doc_length", np.int64)])
        total_doc_meta = num_queries * doc_count
        doc_meta_size = total_doc_meta * doc_meta_dtype.itemsize

        question_ids_size = num_queries * np.dtype(np.int64).itemsize
        fixed_segment_size = header_size + query_meta_size + doc_meta_size + question_ids_size

        # Read metadata.
        query_meta_view = np.frombuffer(data[header_size: header_size + query_meta_size], dtype=query_meta_dtype)
        doc_meta_view = np.frombuffer(data[header_size + query_meta_size: header_size + query_meta_size + doc_meta_size], dtype=doc_meta_dtype)
        qids_start = header_size + query_meta_size + doc_meta_size
        question_ids_view = np.frombuffer(data[qids_start: qids_start + question_ids_size], dtype=np.int64)
        self.question_ids = question_ids_view.tolist()

        # Variable segments: query texts.
        query_text_block_start = fixed_segment_size
        self.queries = []
        for i in range(num_queries):
            offset = int(query_meta_view[i]["query_text_offset"])
            length = int(query_meta_view[i]["query_text_length"])
            q_bytes = memoryview(data)[offset: offset + length].tobytes()
            self.queries.append(q_bytes.decode("utf-8"))

        total_query_text_size = sum(int(query_meta_view[i]["query_text_length"]) for i in range(num_queries))
        docs_block_start = query_text_block_start + total_query_text_size

        # Variable segments: document texts.
        self.doc_list = []
        for i in range(num_queries):
            curr_docs = []
            for j in range(doc_count):
                idx = i * doc_count + j
                offset = int(doc_meta_view[idx]["doc_offset"])
                length = int(doc_meta_view[idx]["doc_length"])
                d_bytes = memoryview(data)[offset: offset + length].tobytes()
                curr_docs.append(d_bytes.decode("utf-8"))
            self.doc_list.append(curr_docs)
        
        self.num_queries = num_queries

    def print_info(self):
        print(f"SearchResultBatchManager: {self.num_queries} queries")
        for i in range(self.num_queries):
            print(f" query {i} (ID {self.question_ids[i]}): {self.queries[i]}")
            print(f"  docs ({len(self.doc_list[i])}): {self.doc_list[i]}")
            

# ------------------------    STEP F (Text Checker) UDL batcher  -------------------------
class PendingCheckDataBatcher:
    def __init__(self, batch_size: int):
        self.max_batch_size = batch_size
        self.num_pending = 0
        self.question_ids = []
        self.queries = []
        self.doc_list = []  # list[list[str]]
        self.doc_per_query = 0
        

    def space_left(self):
        return self.max_batch_size - self.num_pending

    def add_data(self, docBatcher, start_pos):
        num_to_add = min(self.space_left(), len(docBatcher.question_ids) - start_pos)
        end_pos = start_pos + num_to_add
        # Assume all queries have the same number of docs.
        if len(self.doc_list) == 0:
            self.doc_per_query = len(docBatcher.doc_list[start_pos])
        self.question_ids.extend(docBatcher.question_ids[start_pos:end_pos])
        self.queries.extend(docBatcher.queries[start_pos:end_pos])
        # self.doc_list.extend(docBatcher.doc_list[start_pos:end_pos])
        self.doc_list.extend([list(docs) for docs in docBatcher.doc_list[start_pos:end_pos]])
        self.num_pending += num_to_add
        return end_pos

    def reset(self):
        self.question_ids = []
        # self.queries = []
        self.doc_list = []
        self.num_pending = 0
    
class TextCheckResultBatchManager:
    """
    Manages the results of the text checking process.
    The output layout is:
        [Header][question ids][Concatenated list of doc_types that the corresponding doc is harmful]
    
    Header (dtype header_type):
        - num_queries: uint32, the number of queries
        - doc_count: uint32, the number of documents per query
    
    Metadata (dtype metadata_type) for each query:
        - question_id: uint64
    
    Fixed segment:
        - doc_types: uint32 (one per document)
        0 hate speech, 1 offensive language, 2 neither
    """
    def __init__(self):
        self.question_ids = []
        self.doc_types = [] # list[list[int]] for each question_id there are several doc_type for the docs related to it
        self.num_queries = 0
        self.doc_count = 0
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        
    def add_result(self, question_id: int, doc_types: list[int]):
        """
        Add one result to the batch.
        It is assumed that each embedding is a numpy array of type float32
        with either shape (d,) or (1, d) and that all embeddings share the same dimension.
        """
        self.question_ids.append(question_id)
        self.doc_types.append(doc_types)
        self.num_queries += 1
        self.doc_count = len(doc_types)
        # Check if all doc counts are the same
        if self.doc_count != 0 and len(doc_types) != self.doc_count:
            raise ValueError("All doc counts must be the same")
    
    def serialize(self, start_pos, end_pos) -> np.ndarray:
        """
        Serializes the following fields into a contiguous byte buffer:
          - Header: 4 bytes for num_queries (uint32) and 4 bytes for doc_count (uint32).
          - Fixed segments:
            * question_ids: (num_queries,) int64.
            * doc_types: (num_queries * doc_count,) float32.
        """
        # Select the desired slice.
        selected_question_ids = self.question_ids[start_pos:end_pos]
        selected_doc_probs = self.doc_types[start_pos:end_pos]
        num_q = len(selected_question_ids)
        # doc_count is assumed uniform.
        d_count = self.doc_count
        
        # Define header.
        header_dtype = np.dtype([("num_queries", np.uint32), ("doc_count", np.uint32)])
        header_size = header_dtype.itemsize  # 8 bytes
        
        # Question IDs fixed segment.
        qids_dtype = np.dtype(np.int64)
        question_ids_size = num_q * qids_dtype.itemsize
        
        # Document doc_types fixed segment.
        doc_probs_dtype = np.dtype(np.uint32)
        doc_probs_size = num_q * d_count * doc_probs_dtype.itemsize
        
        # Total size of the serialized buffer.
        total_size = header_size + question_ids_size + doc_probs_size
        buffer = np.zeros(total_size, dtype=np.uint8)
        
        offset = 0
        # Write Header.
        header_arr = np.frombuffer(buffer[offset:offset+header_size], dtype=header_dtype)
        header_arr[0] = (num_q, d_count)
        offset += header_size
        
        # Write Question IDs.
        qids_arr = np.frombuffer(buffer[offset:offset+question_ids_size], dtype=qids_dtype)
        qids_arr[:] = np.array(selected_question_ids, dtype=np.int64)
        offset += question_ids_size
        
        # Write Document doc_types.
        # Flatten the list of lists into a single list in row-major order.
        flat_doc_probs = [p for probs in selected_doc_probs for p in probs]
        doc_probs_arr = np.frombuffer(buffer[offset:offset+doc_probs_size], dtype=doc_probs_dtype)
        doc_probs_arr[:] = np.array(flat_doc_probs, dtype=np.float32)
        
        self._bytes = buffer
        return buffer
    
    def deserialize(self, data: np.ndarray):
        """
        Deserializes the given buffer and populates self.question_ids and self.doc_types.
        Expects the layout as produced by serialize.
        """
        self._bytes = data
        header_dtype = np.dtype([("num_queries", np.uint32), ("doc_count", np.uint32)])
        header_size = header_dtype.itemsize
        header = np.frombuffer(data[:header_size], dtype=header_dtype)[0]
        num_q = int(header["num_queries"])
        d_count = int(header["doc_count"])
        
        # Read question IDs.
        qids_dtype = np.dtype(np.int64)
        qids_size = num_q * qids_dtype.itemsize
        qids_start = header_size
        qids_arr = np.frombuffer(data[qids_start:qids_start+qids_size], dtype=qids_dtype)
        self.question_ids = qids_arr.tolist()
        
        # Read document probabilities.
        doc_types_dtype = np.dtype(np.uint32)
        doc_types_size = num_q * d_count * doc_types_dtype.itemsize
        doc_types_start = qids_start + qids_size
        doc_types_arr = np.frombuffer(data[doc_types_start:doc_types_start+doc_types_size], dtype=doc_types_dtype)
        flat_types = doc_types_arr.tolist()
        
        # Reshape into list of lists.
        self.doc_types = []
        for i in range(num_q):
            start_idx = i * d_count
            self.doc_types.append(flat_types[start_idx:start_idx + d_count])
        
        self.num_queries = num_q
        self.doc_count = d_count
        
    def print_info(self):
        print(f"TextCheckResultBatchManager: {self.num_queries} queries, {self.doc_count} docs per query")
        for i in range(self.num_queries):
            print(f" query ID {self.question_ids[i]}: doc_types = {self.doc_types[i]}")

# --------------------------    STEP G (Language Detection) UDL batcher  -------------------------
class LangDetResultBatchManager:
    def __init__(self):
        self.question_ids = []
        self.lang_codes = [] # list[list[str]] for each question_id there are several languages related to it
        self.num_queries = 0
        self.lang_count = 0

    def add_result(self, question_id: int, lang_codes: list[str]):
        """
        Add one result to the batch.
        It is assumed that each embedding is a numpy array of type float32
        with either shape (d,) or (1, d) and that all embeddings share the same dimension.
        """
        self.question_ids.append(question_id)
        self.lang_codes.append(lang_codes)
        self.num_queries += 1
        self.lang_count = len(lang_codes)
        # Check if all lang counts are the same
        if self.lang_count != 0 and len(lang_codes) != self.lang_count:
            raise ValueError("All lang counts must be the same")
    

    def serialize(self, start_pos: int, end_pos: int) -> np.ndarray:
        """
        Serializes a slice of the batch into a buffer:
        - Header: num_queries (uint32), lang_count (uint32)
        - Metadata: for each language code, two int64 values (offset, length)
        - Fixed segment: question_ids (int64)
        - Variable segment: UTF-8 encoded language code strings
        """
        batch_size = end_pos - start_pos
        if batch_size == 0:
            return np.array([], dtype=np.uint8)
        
        # Types and sizes
        header_dtype = np.dtype([("num_queries", np.uint32), ("lang_count", np.uint32)])
        meta_dtype = np.dtype([("offset", np.int64), ("length", np.int64)])
        header_size = header_dtype.itemsize
        meta_size = batch_size * self.lang_count * meta_dtype.itemsize
        qid_size = batch_size * np.dtype(np.int64).itemsize

        # First pass: compute lengths and total variable segment size
        code_lengths = []
        total_text_bytes = 0
        for query_langs in self.lang_codes[start_pos:end_pos]:
            for lang_code in query_langs:
                l = utf8_length(lang_code)
                code_lengths.append(l)
                total_text_bytes += l

        total_size = header_size + meta_size + qid_size + total_text_bytes
        buffer = np.empty(total_size, dtype=np.uint8)

        # Segment positions
        meta_pos = header_size
        qid_pos = meta_pos + meta_size
        text_pos = qid_pos + qid_size

        # Write header
        np.frombuffer(buffer[:header_size], dtype=header_dtype)[0] = (batch_size, self.lang_count)

        # Prepare views
        meta_view = np.frombuffer(buffer[meta_pos:meta_pos + meta_size], dtype=meta_dtype)
        qid_view = np.frombuffer(buffer[qid_pos:qid_pos + qid_size], dtype=np.int64)
        qid_view[:] = self.question_ids[start_pos:end_pos]

        # Second pass: write UTF-8 encoded strings and record offsets
        write_ptr = text_pos
        meta_index = 0
        for query_langs in self.lang_codes[start_pos:end_pos]:
            for lang_code in query_langs:
                encoded = lang_code.encode("utf-8")
                length = len(encoded)
                meta_view[meta_index] = (write_ptr, length)
                buffer[write_ptr:write_ptr + length] = np.frombuffer(encoded, dtype=np.uint8)
                write_ptr += length
                meta_index += 1

        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the buffer created by `serialize()`
        """
        header_dtype = np.dtype([("num_queries", np.uint32), ("lang_count", np.uint32)])
        meta_dtype = np.dtype([("offset", np.int64), ("length", np.int64)])
        qid_dtype = np.dtype(np.int64)

        # Read header
        header = np.frombuffer(data[:header_dtype.itemsize], dtype=header_dtype)[0]
        num_queries = int(header["num_queries"])
        lang_count = int(header["lang_count"])

        # Compute segment offsets
        header_size = header_dtype.itemsize
        meta_size = num_queries * lang_count * meta_dtype.itemsize
        qid_size = num_queries * qid_dtype.itemsize
        meta_start = header_size
        qid_start = meta_start + meta_size
        text_start = qid_start + qid_size

        # Views
        meta_view = np.frombuffer(data[meta_start:meta_start + meta_size], dtype=meta_dtype)
        qid_view = np.frombuffer(data[qid_start:qid_start + qid_size], dtype=qid_dtype)

        # Restore values
        self.question_ids = qid_view.tolist()
        self.lang_codes = []
        for i in range(num_queries):
            query_langs = []
            for j in range(lang_count):
                meta_index = i * lang_count + j
                offset = int(meta_view[meta_index]["offset"])
                length = int(meta_view[meta_index]["length"])
                s = memoryview(data)[offset:offset + length].tobytes().decode("utf-8")
                query_langs.append(s)
            self.lang_codes.append(query_langs)
        
        self.num_queries = num_queries
        self.lang_count = lang_count
        
    def print_info(self):
        print(f"LangDetResultBatchManager: {self.num_queries} queries, {self.lang_count} languages per query")
        for i in range(self.num_queries):
            print(f" query ID {self.question_ids[i]}: languages = {self.lang_codes[i]}")
    
