import threading

from abc import ABC, abstractmethod

class ExecWorker(ABC):
    '''
    This is a batch executor for UDL
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        
        # Fields need to be set by child classes
        self.max_exe_batch_size = None
        self.batch_time_us = None
        self.initial_pending_batch_num = None
        
        self.pending_batches = None
        self.current_batch = -1    # current batch idx that main is executing
        self.next_batch = 0        # next batch idx to add new data
        self.next_to_process = 0 
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = False
        self.new_space_available = True
        

    @abstractmethod
    def create_pending_manager(self):
        """
        Factory method that must be implemented by child classes to
        return a new instance of the batch manager appropriate for that worker.
        """
        pass
    
    def start(self):
        self.running = True
        self.pending_batches = [self.create_pending_manager() for _ in range(self.initial_pending_batch_num)]
        self.thread = threading.Thread(target=self.main_loop)
        self.thread.start()
    
    def join(self):
        if self.thread is not None:
            self.thread.join()
    
    def signal_stop(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()


    def push_to_pending_batches(self, queryBatcher):
        num_questions = len(queryBatcher.question_ids)
        question_added = 0
        while question_added < num_questions:
            with self.cv:
                while True:
                    if not self.new_space_available:
                        self.cv.wait(timeout=3)
                    if not self.running:
                        break
                    if self.new_space_available:
                        break
                free_batch = self.next_batch
                space_left = self.pending_batches[free_batch].space_left()
                initial_batch = free_batch
                # Find the idx in the pending_batches to add the data
                while space_left == 0:
                    free_batch = (free_batch + 1) % len(self.pending_batches)
                    if free_batch == self.current_batch:
                        free_batch = (free_batch + 1) % len(self.pending_batches)
                    if free_batch == initial_batch:
                        break
                    space_left = self.pending_batches[free_batch].space_left()
                if space_left != 0:
                    # add as many questions as possible to the pending batch
                    self.next_batch = free_batch
                    question_start_idx = question_added
                    end_idx = self.pending_batches[free_batch].add_data(queryBatcher, question_start_idx)
                    question_added = end_idx
                    #  if we complete filled the buffer, cycle to the next
                    if self.pending_batches[free_batch].space_left() == 0:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                        if self.next_batch == self.current_batch:
                            self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    self.cv.notify()
                else:
                    self.new_space_available = False


    @abstractmethod
    def main_loop(self):
        """
        Child classes must implement their own main_loop.
        """
        pass
    


class EmitWorker(ABC):
    '''
    This is a batcher for encoderUDL to emit to centroids search UDL
    '''
    def __init__(self, parent, thread_id):
        self.thread = None
        self.parent = parent
        self.my_thread_id = thread_id
        self.send_buffer = None
        
        # Fields need to be set by child classes
        self.max_emit_batch_size = None
        self.next_udl_shards = None
        self.next_udl_subgroup_type = None
        self.next_udl_subgroup_index = None
        self.emit_log_flag = None
        self.next_udl_prefix = None
        
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = False
        
        self.sent_batch_counter = 0        
        
    
    @abstractmethod
    def create_batch_manager(self):
        """
        Factory method that must be implemented by child classes to
        return a new instance of the batch manager appropriate for that worker.
        """
        pass


    def start(self):
        self.running = True
        self.send_buffer = [self.create_batch_manager()  for _ in range(len(self.next_udl_shards))]
        self.thread = threading.Thread(target=self.main_loop)
        self.thread.start()
    
    def join(self):
        if self.thread is not None:
            self.thread.join()
    
    def signal_stop(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()
    
    @abstractmethod
    def add_to_buffer(self, batch, result, num_queries):
        """
        Child classes must implement their own add_to_buffer."
        """
        pass
            
    def process_and_emit_results(self, to_send):
        for idx, batch_manager in enumerate(to_send):
            if batch_manager.num_queries == 0:
                continue
            num_sent = 0
            cur_shard_id = self.next_udl_shards[idx]
            while num_sent < batch_manager.num_queries:
                serialize_batch_size = min(self.max_emit_batch_size, batch_manager.num_queries - num_sent)
                start_pos = num_sent
                end_pos = num_sent + serialize_batch_size
                serialized_batch = batch_manager.serialize(start_pos, end_pos)  

                new_key = self.next_udl_prefix + str(self.parent.sent_msg_count) + "_" + str(cur_shard_id)
                shard_idx = self.next_udl_shards[idx]
                
                for qid in batch_manager.question_ids[start_pos:end_pos]:
                    self.parent.tl.log(self.emit_log_flag, qid, 0, end_pos - start_pos)
                
                self.parent.capi.put_nparray(new_key, serialized_batch, 
                                subgroup_type=self.next_udl_subgroup_type,
                                subgroup_index=self.next_udl_subgroup_index, 
                                shard_index=shard_idx, 
                                message_id=1, as_trigger=True, blocking=False)
                self.parent.sent_msg_count += 1
                num_sent += serialize_batch_size
                self.sent_batch_counter += 1
                # print(f"sent {new_key} to next UDL")
                # batch_manager.print_info()
                
    
    def main_loop(self):
        batch_wait_time = self.parent.batch_time_us/1000000
        while self.running:
            to_send = []
            empty = True
            with self.cv:
                for i in range(len(self.send_buffer)):
                    if self.send_buffer[i].num_queries > 0:
                        empty = False
                        break

                if empty:
                    self.cv.wait(timeout=batch_wait_time)
                if not self.running:
                    break
                
                if not empty:
                    to_send = self.send_buffer
                    # Below is shallow copy, to avoid deep copy of the data
                    self.send_buffer = [self.create_batch_manager()  for _ in range(len(self.next_udl_shards))]
                    
            self.process_and_emit_results(to_send)
