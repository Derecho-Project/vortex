import numpy as np

class MonoDataBatcher:
    def __init__(self):
        # All fields must have the same batch size.
        self.pixel_values = None         # np.ndarray of shape (batch_size, 1, 224, 224) and dtype=np.float32
        self.text_sequence = []          # List of strings (one per query)
        self.question_ids = []           # List of ints (one per query)
        self.questions = []              # List of strings (one per query)
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
    def utf8_length(self, s: str) -> int:
        return len(s.encode("utf-8"))  # More efficient way to compute UTF-8 byte length
    
    def serialize(self) -> np.ndarray:

        batch_size = len(self.question_ids)
        if not (len(self.questions) == len(self.text_sequence) == batch_size):
            raise ValueError("All input lists must have the same length")
        if self.pixel_values is None or self.pixel_values.shape[0] != batch_size:
            raise ValueError("pixel_values must be provided and its first dimension must equal batch_size")
        # Prepare variable-length segments: questions and text_sequence.
        question_encodings = [q.encode("utf-8") for q in self.questions]
        text_seq_encodings = [t.encode("utf-8") for t in self.text_sequence]
        # Compute offsets for questions segment.
        question_offsets = []
        offset = 0
        for enc in question_encodings:
            question_offsets.append(offset)
            offset += len(enc)
        total_questions_size = offset
        # Compute offsets for text_sequence segment.
        text_seq_offsets = []
        offset = 0
        for enc in text_seq_encodings:
            text_seq_offsets.append(offset)
            offset += len(enc)
        total_text_seq_size = offset
        # Fixed-length parts.
        header_size = 4  # 4 bytes for batch_size (uint32).
        metadata_dtype = np.dtype([
            ("question_offset", np.int64),
            ("question_length", np.int64),
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64),
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        qids_size = batch_size * np.dtype(np.int64).itemsize
        pixel_values_size = self.pixel_values.nbytes  # e.g., batch_size * 1 * 224 * 224 * 4
        total_size = (header_size + metadata_size + qids_size +
                      total_questions_size + total_text_seq_size + pixel_values_size)
        # Allocate one contiguous buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        # --- Write header ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        # --- Write metadata ---
        metadata_start = header_size
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size],
                                       dtype=metadata_dtype)
        for i in range(batch_size):
            metadata_array[i]["question_offset"] = question_offsets[i]
            metadata_array[i]["question_length"] = len(question_encodings[i])
            metadata_array[i]["text_sequence_offset"] = text_seq_offsets[i]
            metadata_array[i]["text_sequence_length"] = len(text_seq_encodings[i])
        # --- Write question_ids ---
        qids_start = metadata_start + metadata_size
        qids_array = np.frombuffer(buffer[qids_start:qids_start+qids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        # --- Write questions segment ---
        questions_start = qids_start + qids_size
        pos = questions_start
        for enc in question_encodings:
            n = len(enc)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        # --- Write text_sequence segment ---
        text_seq_start = questions_start + total_questions_size
        pos = text_seq_start
        for enc in text_seq_encodings:
            n = len(enc)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        # --- Write pixel_values segment ---
        pixel_values_start = text_seq_start + total_text_seq_size
        pixel_bytes = self.pixel_values.tobytes()  # Get contiguous raw bytes.
        buffer[pixel_values_start:pixel_values_start+pixel_values_size] = np.frombuffer(pixel_bytes, dtype=np.uint8)
        self._bytes = buffer
        return buffer
    
    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous byte array back into the original fields.
        Assumes the same layout as produced by serialize().
        """
        self._bytes = data
        buffer = data
        offset = 0
        # --- Read header ---
        batch_size = int(np.frombuffer(buffer[offset:offset+4], dtype=np.uint32)[0])
        offset += 4
        # --- Read metadata ---
        metadata_dtype = np.dtype([
            ("question_offset", np.int64),
            ("question_length", np.int64),
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64),
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer[offset:offset+metadata_size], dtype=metadata_dtype)
        offset += metadata_size
        # --- Read question_ids ---
        qids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer[offset:offset+qids_size], dtype=np.int64).tolist()
        offset += qids_size
        # --- Read questions segment ---
        total_questions_size = sum(int(x) for x in metadata_array["question_length"])
        questions_bytes = buffer[offset:offset+total_questions_size]
        offset += total_questions_size
        questions = []
        for m in metadata_array:
            start = int(m["question_offset"])
            length = int(m["question_length"])
            questions.append(questions_bytes[start:start+length].tobytes().decode("utf-8"))
            
        # --- Read text_sequence segment ---
        total_text_seq_size = sum(int(x) for x in metadata_array["text_sequence_length"])
        text_seq_bytes = buffer[offset:offset+total_text_seq_size]
        offset += total_text_seq_size
        text_sequence = []
        for m in metadata_array:
            start = int(m["text_sequence_offset"])
            length = int(m["text_sequence_length"])
            text_sequence.append(text_seq_bytes[start:start+length].tobytes().decode("utf-8"))
            
        # --- Read pixel_values segment ---
        pixel_values_size = batch_size * 1 * 3 * 224 * 224 * np.dtype(np.float32).itemsize
        pixel_values_bytes = buffer[offset:offset+pixel_values_size]
        offset += pixel_values_size
        pixel_values = np.frombuffer(pixel_values_bytes, dtype=np.float32).reshape((batch_size, 1, 3,224, 224))
        # Restore fields.
        
        self.question_ids = qids
        self.questions = questions
        self.pixel_values = pixel_values
        self.text_sequence = text_sequence
        
    def get_data(self):
        return {
            "question_ids": self.question_ids,
            "questions": self.questions,
            "pixel_values": self.pixel_values,
            "text_sequence": self.text_sequence
        }
        
        
class PixelValueBatcher:
    def __init__(self):
        # Expected shape: (batch_size, 1, 3, 224, 224) and dtype=np.float32.
        self.pixel_values: np.ndarray = None
        # Query IDs as a NumPy array of shape (batch_size,) and dtype=np.int64.
        self.question_ids: np.ndarray = None
        # Internal buffer (serialized data)
        self._bytes: np.ndarray = None

    def serialize(self) -> np.ndarray:
        """
        Serialize the pixel_values and question_ids into a contiguous buffer.
        Layout:
        [4 bytes: batch_size (uint32)] | [question_ids (int64)] | [pixel_values (float32)]
        """
        if self.pixel_values is None or self.question_ids is None:
            raise ValueError("Both pixel_values and question_ids must be provided.")
        
        # Ensure question_ids is a NumPy array.
        question_ids_array = np.asarray(self.question_ids, dtype=np.int64)
        batch_size = question_ids_array.shape[0]
        
        if self.pixel_values.shape[0] != batch_size:
            raise ValueError("The first dimension of pixel_values must match the number of question_ids.")
        
        # Calculate sizes.
        header_size = np.dtype(np.uint32).itemsize  # 4 bytes
        question_ids_size = question_ids_array.nbytes       # batch_size * 8 bytes
        pixel_values_size = self.pixel_values.nbytes  # batch_size * 1 * 3 * 224 * 224 * 4
        
        total_size = header_size + question_ids_size + pixel_values_size
        # Allocate one contiguous buffer.
        buffer = np.empty(total_size, dtype=np.uint8)
        offset = 0

        # --- Write header: batch_size ---
        header_arr = np.array([batch_size], dtype=np.uint32)
        # Create a view into header_arr as uint8 (zero-copy view) and copy into the buffer slice.
        buffer[offset:offset + header_size] = header_arr.view(np.uint8)
        offset += header_size

        # --- Write question_ids ---
        # Use a view of the question_ids array as uint8.
        buffer[offset:offset + question_ids_size] = question_ids_array.view(np.uint8)
        offset += question_ids_size

        # --- Write pixel_values ---
        buffer[offset:offset + pixel_values_size] = self.pixel_values.view(np.uint8).reshape(-1)
        offset += pixel_values_size

        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserialize a contiguous uint8 buffer into question_ids and pixel_values with zero-copy.
        The layout of the buffer is:
        [4 bytes: batch_size (uint32)] | [question_ids (int64)] | [pixel_values (float32)]
        where pixel_values is expected to have shape (batch_size, 1, 3, 224, 224).
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header: batch_size ---
        header_size = np.dtype(np.uint32).itemsize  # 4 bytes
        batch_size = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0]
        offset += header_size

        # --- Read question_ids ---
        question_ids_count = batch_size  # one int64 per example
        question_ids = np.frombuffer(buffer, dtype=np.int64, count=question_ids_count, offset=offset)
        offset += question_ids.nbytes  # or batch_size * np.dtype(np.int64).itemsize

        # --- Read pixel_values ---
        pixel_shape = (batch_size, 1, 3, 224, 224)
        num_pixels = np.prod(pixel_shape)
        # Create a zero-copy view and then reshape it
        pixel_values = np.frombuffer(buffer, dtype=np.float32, count=num_pixels, offset=offset).reshape(pixel_shape)
        offset += num_pixels * np.dtype(np.float32).itemsize

        # Assign fields (these are zero-copy views into the original buffer)
        self.question_ids = question_ids
        self.pixel_values = pixel_values

    def get_data(self):
        return {
            "question_ids": self.question_ids,
            "pixel_values": self.pixel_values
        }


class TextDataBatcher:
    def __init__(self):
        # All fields must have the same batch size.
        self.question_ids = []      # List[int] of length batch_size.
        self.text_sequence = []     # List[str] of length batch_size.
        self.input_ids = None       # np.ndarray of shape (batch_size, 32), dtype=np.int64.
        self.attention_mask = None  # np.ndarray of shape (batch_size, 32), dtype=np.int64.
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)

    def utf8_length(self, s: str) -> int:
        """Return the byte-length of s when encoded in UTF-8."""
        return len(s.encode("utf-8"))

    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous buffer:
          - Header: 4 bytes for batch_size (uint32).
          - Metadata: For each text_sequence element, store two int64 values (offset and length).
          - Fixed segments:
              * question_ids: (batch_size,) int64.
              * input_ids: (batch_size, 32) int64.
              * attention_mask: (batch_size, 32) int64.
          - Variable segment:
              * text_sequence: concatenated UTF-8 encoded bytes.
        """
        batch_size = len(self.question_ids)
        if len(self.text_sequence) != batch_size:
            raise ValueError("text_sequence and question_ids must have the same length")
        if self.input_ids is None or self.input_ids.shape[0] != batch_size:
            raise ValueError("input_ids must be provided and its first dimension must equal batch_size")
        if self.attention_mask is None or self.attention_mask.shape[0] != batch_size:
            raise ValueError("attention_mask must be provided and its first dimension must equal batch_size")
        
        # Compute offsets and total size for text_sequence without storing encoded values.
        text_seq_offsets = []
        offset_temp = 0
        for t in self.text_sequence:
            text_seq_offsets.append(offset_temp)
            offset_temp += self.utf8_length(t)
        total_text_seq_size = offset_temp

        header_size = np.dtype(np.uint32).itemsize   # 4 bytes
        
        # Define metadata: two int64 values per example.
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        
        # Fixed segments sizes.
        question_ids_size = batch_size * np.dtype(np.int64).itemsize
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        attention_mask_size = batch_size * 32 * np.dtype(np.int64).itemsize
        
        total_size = (header_size + metadata_size + question_ids_size +
                      input_ids_size + attention_mask_size + total_text_seq_size)
        
        # Allocate one contiguous buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        offset = 0

        # --- Write header: batch_size (uint32) ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        offset += header_size

        # --- Write metadata for text_sequence ---
        metadata_start = offset
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size],
                                       dtype=metadata_dtype)
        for i, t in enumerate(self.text_sequence):
            metadata_array[i]["text_sequence_offset"] = text_seq_offsets[i]
            metadata_array[i]["text_sequence_length"] = self.utf8_length(t)
        offset += metadata_size

        # --- Write question_ids ---
        qids_start = offset
        qids_array = np.frombuffer(buffer[qids_start:qids_start+question_ids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        offset += question_ids_size

        # --- Write input_ids ---
        input_ids_start = offset
        buffer[input_ids_start:input_ids_start+input_ids_size] = self.input_ids.view(np.uint8).reshape(-1)
        offset += input_ids_size

        # --- Write attention_mask ---
        attention_mask_start = offset
        buffer[attention_mask_start:attention_mask_start+attention_mask_size] = self.attention_mask.view(np.uint8).reshape(-1)
        offset += attention_mask_size

        # --- Write text_sequence segment ---
        # Instead of pre-encoding, encode each string inline.
        text_seq_start = offset
        pos = text_seq_start
        for t in self.text_sequence:
            enc = t.encode("utf-8")  # Encode inline.
            n = len(enc)             # Alternatively, self.utf8_length(t)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        offset += total_text_seq_size

        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous buffer back into the fields.
        Layout:
          - Header: 4 bytes (batch_size, uint32)
          - Metadata: per example text_sequence offset and length (int64)
          - question_ids: array of int64, shape (batch_size,)
          - input_ids: array of int64, shape (batch_size, 32)
          - attention_mask: array of int64, shape (batch_size, 32)
          - text_sequence: concatenated UTF-8 bytes.
        After deserialization, question_ids is converted to a Python list.
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header: batch_size ---
        header_size = np.dtype(np.uint32).itemsize
        batch_size = int(np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0])
        offset += header_size

        # --- Read metadata for text_sequence ---
        metadata_dtype = np.dtype([
            ("text_sequence_offset", np.int64),
            ("text_sequence_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer, dtype=metadata_dtype, count=batch_size, offset=offset)
        offset += metadata_size

        # --- Read question_ids ---
        question_ids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset).tolist()
        offset += question_ids_size

        # --- Read input_ids ---
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        num_input_ids = batch_size * 32
        input_ids = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += input_ids_size

        # --- Read attention_mask ---
        attention_mask_size = batch_size * 32 * np.dtype(np.int64).itemsize
        attention_mask = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += attention_mask_size

        # --- Read text_sequence segment ---
        total_text_seq_size = sum(int(x) for x in metadata_array["text_sequence_length"])
        text_seq_bytes = buffer[offset:offset+total_text_seq_size]
        offset += total_text_seq_size
        text_sequence = []
        for m in metadata_array:
            start = int(m["text_sequence_offset"])
            length = int(m["text_sequence_length"])
            text_sequence.append(text_seq_bytes[start:start+length].tobytes().decode("utf-8"))

        # Restore fields.
        self.question_ids = qids           # as a Python list.
        self.text_sequence = text_sequence
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def get_data(self):
        return {
            "question_ids": self.question_ids,
            "text_sequence": self.text_sequence,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask
        }
        
        
        
class StepAMessageDataBatcher:
    def __init__(self):
        # All fields must have the same batch size.
        self.question_ids = []      # List[int] of length batch_size.
        self.input_ids = None       # np.ndarray of shape (batch_size, 32), dtype=np.int64.
        self.text_embeds = None     # np.ndarray of shape (batch_size, 32, 128), dtype=np.float32.
        self.text_encoder_hidden_states = None  # np.ndarray of shape (batch_size, 32, 768), dtype=np.float32.
        self.queries = []           # List[str] of length batch_size.
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)

    def utf8_length(self, s: str) -> int:
        """Return the byte-length of s when encoded in UTF-8."""
        return len(s.encode("utf-8"))

    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous byte buffer:
          - Header: 4 bytes (batch_size, as uint32).
          - Metadata for queries: For each query, two int64 values:
                * query_offset: starting offset (in bytes) within the variable segment.
                * query_length: byte-length of the query.
          - Fixed segments:
                * question_ids: (batch_size,) int64.
                * input_ids: (batch_size, 32) int64.
                * text_embeds: (batch_size, 32, 128) float32.
                * text_encoder_hidden_states: (batch_size, 32, 768) float32.
          - Variable segment:
                * queries: concatenated UTF-8 encoded bytes.
        """
        batch_size = len(self.question_ids)
        if len(self.queries) != batch_size:
            raise ValueError("Length of queries must equal length of question_ids")
        if self.input_ids is None or self.input_ids.shape[0] != batch_size:
            raise ValueError("input_ids must have first dimension equal to batch_size")
        if self.text_embeds is None or self.text_embeds.shape[0] != batch_size:
            raise ValueError("text_embeds must have first dimension equal to batch_size")
        if self.text_encoder_hidden_states is None or self.text_encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("text_encoder_hidden_states must have first dimension equal to batch_size")

        # --- Compute offsets for queries variable segment using utf8_length() ---
        query_offsets = []
        offset_temp = 0
        for q in self.queries:
            query_offsets.append(offset_temp)
            offset_temp += self.utf8_length(q)
        total_queries_size = offset_temp

        # --- Determine sizes for fixed parts ---
        header_size = np.dtype(np.uint32).itemsize  # 4 bytes.
        # Metadata: for each query, two int64 fields.
        metadata_dtype = np.dtype([
            ("query_offset", np.int64),
            ("query_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize

        qids_size = batch_size * np.dtype(np.int64).itemsize
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        text_embeds_size = batch_size * 32 * 128 * np.dtype(np.float32).itemsize
        hidden_states_size = batch_size * 32 * 768 * np.dtype(np.float32).itemsize

        total_size = (header_size + metadata_size + qids_size + input_ids_size +
                      text_embeds_size + hidden_states_size + total_queries_size)

        # Allocate one contiguous buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        offset = 0

        # --- Write header: batch_size (uint32) ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        offset += header_size

        # --- Write metadata for queries ---
        metadata_start = offset
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size],
                                       dtype=metadata_dtype)
        for i, q in enumerate(self.queries):
            metadata_array[i]["query_offset"] = query_offsets[i]
            metadata_array[i]["query_length"] = self.utf8_length(q)
        offset += metadata_size

        # --- Write question_ids ---
        qids_start = offset
        qids_array = np.frombuffer(buffer[qids_start:qids_start+qids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        offset += qids_size

        # --- Write input_ids ---
        input_ids_start = offset
        buffer[input_ids_start:input_ids_start+input_ids_size] = self.input_ids.view(np.uint8).reshape(-1)
        offset += input_ids_size

        # --- Write text_embeds ---
        text_embeds_start = offset
        buffer[text_embeds_start:text_embeds_start+text_embeds_size] = self.text_embeds.view(np.uint8).reshape(-1)
        offset += text_embeds_size

        # --- Write text_encoder_hidden_states ---
        hidden_states_start = offset
        buffer[hidden_states_start:hidden_states_start+hidden_states_size] = self.text_encoder_hidden_states.view(np.uint8).reshape(-1)
        offset += hidden_states_size

        # --- Write queries variable segment ---
        queries_start = offset
        pos = queries_start
        for q in self.queries:
            enc = q.encode("utf-8")  # Encode inline.
            n = len(enc)             # or self.utf8_length(q)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        offset += total_queries_size

        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous buffer back into the original fields.
        Expected layout (in order):
          - Header: 4 bytes (uint32: batch_size)
          - Metadata for queries: per query (query_offset and query_length as int64)
          - question_ids: array of int64, shape (batch_size,)
          - input_ids: array of int64, shape (batch_size, 32)
          - text_embeds: array of float32, shape (batch_size, 32, 128)
          - text_encoder_hidden_states: array of float32, shape (batch_size, 32, 768)
          - queries: concatenated UTF-8 bytes.
        After deserialization, question_ids is converted to a Python list.
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header ---
        header_size = np.dtype(np.uint32).itemsize
        batch_size = int(np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0])
        offset += header_size

        # --- Read metadata for queries ---
        metadata_dtype = np.dtype([
            ("query_offset", np.int64),
            ("query_length", np.int64)
        ])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer, dtype=metadata_dtype, count=batch_size, offset=offset)
        offset += metadata_size

        # --- Read question_ids ---
        qids_size = batch_size * np.dtype(np.int64).itemsize
        qids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset).tolist()
        offset += qids_size

        # --- Read input_ids ---
        input_ids_size = batch_size * 32 * np.dtype(np.int64).itemsize
        num_input_ids = batch_size * 32
        input_ids = np.frombuffer(buffer, dtype=np.int64, count=num_input_ids, offset=offset).reshape((batch_size, 32))
        offset += input_ids_size

        # --- Read text_embeds ---
        text_embeds_size = batch_size * 32 * 128 * np.dtype(np.float32).itemsize
        num_text_embeds = batch_size * 32 * 128
        text_embeds = np.frombuffer(buffer, dtype=np.float32, count=num_text_embeds, offset=offset).reshape((batch_size, 32, 128))
        offset += text_embeds_size

        # --- Read text_encoder_hidden_states ---
        hidden_states_size = batch_size * 32 * 768 * np.dtype(np.float32).itemsize
        num_hidden_states = batch_size * 32 * 768
        text_encoder_hidden_states = np.frombuffer(buffer, dtype=np.float32, count=num_hidden_states, offset=offset).reshape((batch_size, 32, 768))
        offset += hidden_states_size

        # --- Read queries variable segment ---
        total_queries_size = sum(int(x) for x in metadata_array["query_length"])
        queries_bytes = buffer[offset:offset+total_queries_size]
        offset += total_queries_size
        queries = []
        for m in metadata_array:
            start = int(m["query_offset"])
            length = int(m["query_length"])
            queries.append(queries_bytes[start:start+length].tobytes().decode("utf-8"))

        # Restore fields.
        self.question_ids = qids
        self.input_ids = input_ids
        self.text_embeds = text_embeds
        self.text_encoder_hidden_states = text_encoder_hidden_states
        self.queries = queries

    def get_data(self):
        return {
            "question_ids": self.question_ids,
            "input_ids": self.input_ids,
            "text_embeds": self.text_embeds,
            "text_encoder_hidden_states": self.text_encoder_hidden_states,
            "queries": self.queries
        }


class StepDMessageBatcher:
    def __init__(self):
        # All fields must have the same batch size.
        self.question_ids = []       # List[int] of length batch_size.
        self.queries = []            # List[str] of length batch_size.
        self.query_embeddings = None # np.ndarray of shape (batch_size, 320, 128), dtype=np.float32.
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)

    def utf8_length(self, s: str) -> int:
        """Return the byte-length of s when encoded in UTF-8."""
        return len(s.encode("utf-8"))

    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous byte buffer:
          - Header: 4 bytes for batch_size (uint32).
          - Metadata for queries: for each query, two int64 values:
                * query_offset: starting offset (in bytes) within the variable segment.
                * query_length: byte-length of the query.
          - Fixed segments:
                * question_ids: (batch_size,) int64.
                * query_embeddings: (batch_size, 320, 128) float32.
          - Variable segment:
                * queries: concatenated UTF-8 encoded bytes.
        """
        batch_size = len(self.question_ids)
        if len(self.queries) != batch_size:
            raise ValueError("Length of queries must equal length of question_ids")
        if self.query_embeddings is None or self.query_embeddings.shape[0] != batch_size:
            raise ValueError("query_embeddings must be provided and its first dimension must equal batch_size")

        # --- Compute offsets for queries using utf8_length() inline ---
        query_offsets = []
        offset_temp = 0
        for q in self.queries:
            query_offsets.append(offset_temp)
            offset_temp += self.utf8_length(q)
        total_queries_size = offset_temp

        # --- Compute sizes for fixed parts ---
        header_size = np.dtype(np.uint32).itemsize  # 4 bytes.

        # Metadata: for each query, store (query_offset, query_length) as int64.
        metadata_dtype = np.dtype([("query_offset", np.int64), ("query_length", np.int64)])
        metadata_size = batch_size * metadata_dtype.itemsize

        # question_ids: one int64 per example.
        qids_size = batch_size * np.dtype(np.int64).itemsize

        # query_embeddings: its total size in bytes.
        query_embeddings_size = self.query_embeddings.nbytes

        total_size = header_size + metadata_size + qids_size + query_embeddings_size + total_queries_size

        # --- Allocate one contiguous buffer ---
        buffer = np.zeros(total_size, dtype=np.uint8)
        offset = 0

        # --- Write header: batch_size (uint32) ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        offset += header_size

        # --- Write metadata for queries ---
        metadata_start = offset
        metadata_array = np.frombuffer(buffer[metadata_start:metadata_start+metadata_size], dtype=metadata_dtype)
        for i, q in enumerate(self.queries):
            metadata_array[i]["query_offset"] = query_offsets[i]
            metadata_array[i]["query_length"] = self.utf8_length(q)
        offset += metadata_size

        # --- Write question_ids ---
        qids_start = offset
        qids_array = np.frombuffer(buffer[qids_start:qids_start+qids_size], dtype=np.int64)
        qids_array[:] = np.array(self.question_ids, dtype=np.int64)
        offset += qids_size

        # --- Write query_embeddings ---
        embeddings_start = offset
        # Write a flat view (zero-copy) of the query_embeddings.
        buffer[embeddings_start:embeddings_start+query_embeddings_size] = self.query_embeddings.view(np.uint8).reshape(-1)
        offset += query_embeddings_size

        # --- Write queries variable segment ---
        queries_start = offset
        pos = queries_start
        for q in self.queries:
            enc = q.encode("utf-8")  # Inline encoding.
            n = len(enc)             # Alternatively, self.utf8_length(q)
            buffer[pos:pos+n] = np.frombuffer(enc, dtype=np.uint8)
            pos += n
        offset += total_queries_size

        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous byte buffer back into the fields.
        Expected layout (in order):
          - Header: 4 bytes (uint32: batch_size)
          - Metadata: for each query, (query_offset, query_length) as int64.
          - question_ids: array of int64, shape (batch_size,)
          - query_embeddings: array of float32, shape (batch_size, 320, 128)
          - queries: concatenated UTF-8 encoded bytes.
        After deserialization, question_ids is converted to a Python list.
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header ---
        header_size = np.dtype(np.uint32).itemsize
        batch_size = int(np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0])
        offset += header_size

        # --- Read metadata for queries ---
        metadata_dtype = np.dtype([("query_offset", np.int64), ("query_length", np.int64)])
        metadata_size = batch_size * metadata_dtype.itemsize
        metadata_array = np.frombuffer(buffer, dtype=metadata_dtype, count=batch_size, offset=offset)
        offset += metadata_size

        # --- Read question_ids ---
        qids_size = batch_size * np.dtype(np.int64).itemsize
        question_ids = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset).tolist()
        offset += qids_size

        # --- Read query_embeddings ---
        query_embeddings_size = batch_size * 320 * 128 * np.dtype(np.float32).itemsize
        num_embeddings = batch_size * 320 * 128
        query_embeddings = np.frombuffer(buffer, dtype=np.float32, count=num_embeddings, offset=offset).reshape((batch_size, 320, 128))
        offset += query_embeddings_size

        # --- Read queries variable segment ---
        total_queries_size = sum(int(x) for x in metadata_array["query_length"])
        queries_bytes = buffer[offset:offset+total_queries_size]
        offset += total_queries_size
        queries = []
        for m in metadata_array:
            start = int(m["query_offset"])
            length = int(m["query_length"])
            queries.append(queries_bytes[start:start+length].tobytes().decode("utf-8"))

        self.question_ids = question_ids
        self.query_embeddings = query_embeddings
        self.queries = queries

    def get_data(self):
        return {
            "question_ids": self.question_ids,
            "query_embeddings": self.query_embeddings,
            "queries": self.queries
        }


class VisionDataBatcher:
    def __init__(self):
        # Fields must have the same batch size.
        self.question_id = []           # List[int] of length batch_size.
        self.vision_embedding = None    # np.ndarray of shape (batch_size, 32, 128), dtype=np.float32.
        self.vision_hidden_states = None  # np.ndarray of shape (batch_size, 256, 768), dtype=np.float32.
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)

    def serialize(self) -> np.ndarray:
        """
        Serializes the following fields into a contiguous byte buffer:
          - Header: 4 bytes for batch_size (uint32).
          - question_id: fixed-size array of int64 (batch_size,).
          - vision_embedding: raw bytes of the array (batch_size, 32, 128), dtype np.float32.
          - vision_hidden_states: raw bytes of the array (batch_size, 256, 768), dtype np.float32.
        """
        if self.vision_embedding is None or self.vision_hidden_states is None:
            raise ValueError("Both vision_embedding and vision_hidden_states must be provided.")
        
        batch_size = self.vision_embedding.shape[0]
        if self.vision_hidden_states.shape[0] != batch_size:
            raise ValueError("The first dimension (batch_size) must match for both arrays.")
        if not self.question_id or len(self.question_id) != batch_size:
            raise ValueError("question_id must be provided as a list of ints with length equal to batch_size.")
        
        header_size = np.dtype(np.uint32).itemsize  # 4 bytes.
        qid_size = batch_size * np.dtype(np.int64).itemsize  # Fixed field: question_id.
        embedding_size = self.vision_embedding.nbytes
        hidden_states_size = self.vision_hidden_states.nbytes

        total_size = header_size + qid_size + embedding_size + hidden_states_size
        
        # Allocate one contiguous buffer.
        buffer = np.zeros(total_size, dtype=np.uint8)
        offset = 0

        # --- Write header: batch_size (uint32) ---
        np.frombuffer(buffer[:header_size], dtype=np.uint32)[0] = batch_size
        offset += header_size

        # --- Write question_id field ---
        # Convert question_id list to np.array of int64 and write via zero-copy view.
        qid_array = np.array(self.question_id, dtype=np.int64)
        buffer[offset:offset+qid_size] = qid_array.view(np.uint8).reshape(-1)
        offset += qid_size

        # --- Write vision_embedding ---
        buffer[offset:offset+embedding_size] = self.vision_embedding.view(np.uint8).reshape(-1)
        offset += embedding_size

        # --- Write vision_hidden_states ---
        buffer[offset:offset+hidden_states_size] = self.vision_hidden_states.view(np.uint8).reshape(-1)
        offset += hidden_states_size

        self._bytes = buffer
        return buffer

    def deserialize(self, data: np.ndarray):
        """
        Deserializes the contiguous byte buffer back into the fields.
        Expected layout (in order):
          - Header: 4 bytes (uint32: batch_size).
          - question_id: array of int64, shape (batch_size,).
          - vision_embedding: array of float32 with shape (batch_size, 32, 128).
          - vision_hidden_states: array of float32 with shape (batch_size, 256, 768).
        After deserialization, question_id is converted to a Python list.
        """
        self._bytes = data
        buffer = data
        offset = 0

        # --- Read header ---
        header_size = np.dtype(np.uint32).itemsize
        batch_size = int(np.frombuffer(buffer, dtype=np.uint32, count=1, offset=offset)[0])
        offset += header_size

        # --- Read question_id field ---
        qid_size = batch_size * np.dtype(np.int64).itemsize
        qid_array = np.frombuffer(buffer, dtype=np.int64, count=batch_size, offset=offset)
        question_id = qid_array.tolist()  # Convert to Python list.
        offset += qid_size

        # --- Read vision_embedding ---
        embedding_size = batch_size * 32 * 128 * np.dtype(np.float32).itemsize
        vision_embedding = np.frombuffer(buffer, dtype=np.float32, count=batch_size * 32 * 128, offset=offset)
        vision_embedding = vision_embedding.reshape((batch_size, 32, 128))
        offset += embedding_size

        # --- Read vision_hidden_states ---
        hidden_states_size = batch_size * 256 * 768 * np.dtype(np.float32).itemsize
        vision_hidden_states = np.frombuffer(buffer, dtype=np.float32, count=batch_size * 256 * 768, offset=offset)
        vision_hidden_states = vision_hidden_states.reshape((batch_size, 256, 768))
        offset += hidden_states_size

        self.question_id = question_id
        self.vision_embedding = vision_embedding
        self.vision_hidden_states = vision_hidden_states

    def get_data(self):
        return {
            "question_id": self.question_id,
            "vision_embedding": self.vision_embedding,
            "vision_hidden_states": self.vision_hidden_states
        }

# === Example usage ===
if __name__ == "__main__":
    batch_size = 2
    # Create dummy arrays.
    dummy_vision_embedding = np.random.rand(batch_size, 32, 128).astype(np.float32)
    dummy_vision_hidden_states = np.random.rand(batch_size, 256, 768).astype(np.float32)
    
    batcher = VisionDataBatcher()
    batcher.vision_embedding = dummy_vision_embedding
    batcher.vision_hidden_states = dummy_vision_hidden_states
    
    serialized = batcher.serialize()
    print("Serialized buffer size:", serialized.nbytes)
    
    new_batcher = VisionDataBatcher()
    new_batcher.deserialize(serialized)
    data = new_batcher.get_data()
    print("Deserialized vision_embedding shape:", data["vision_embedding"].shape)
    print("Deserialized vision_hidden_states shape:", data["vision_hidden_states"].shape)

