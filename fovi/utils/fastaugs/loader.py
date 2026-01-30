import torch
from typing import Any, Dict, Mapping, Optional, Sequence, Type, Union
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.fields.base import Field
from ffcv.loader.epoch_iterator import EpochIterator
from ffcv.loader.loader import Loader, ORDER_TYPE, DEFAULT_OS_CACHE

IS_CUDA = torch.cuda.is_available()

__all__ = ['FlashLoader']

class CustomEpochIterator(EpochIterator):
    """Extended EpochIterator with support for after-batch transforms.
    
    Applies additional transformations to batches after they are loaded
    from the FFCV pipeline.
    
    Args:
        loader (FlashLoader): The parent loader.
        order: Sample ordering for the epoch.
        after_batch_pipelines (dict, optional): Dictionary mapping field names
            to transform pipelines to apply after batch loading. Defaults to None.
    """
    def __init__(self, loader, order, after_batch_pipelines=None):
        super().__init__(loader, order)
        self.after_batch_pipelines = after_batch_pipelines
        self.fields_order = loader.pipelines.keys()

    def __next__(self):
        result = self.output_queue.get()
        if result is None:
            self.close()
            raise StopIteration()
        slot, result = result
        if IS_CUDA:
            stream = self.cuda_streams[slot]
            self.current_stream.wait_stream(stream)
        
        if self.after_batch_pipelines is None:
            return result
    
        # Apply the after_batch_transforms here
        result = list(result)  # Convert the result tuple to a list to modify it.
        for i, key in enumerate(self.fields_order):
            if key in self.after_batch_pipelines:
                pipeline = self.after_batch_pipelines[key]
                if isinstance(pipeline, (list, tuple)) and len(result[i].shape)==5:
                    # result is B x numSamples x C x H x W
                    # apply different pipeline per sample
                    result[i] = tuple([pipe(result[i][:,pipe_num]) for pipe_num,pipe in enumerate(pipeline)])
                elif isinstance(pipeline, (list, tuple)):
                    # apply multiple pipelines to the same sample
                    result[i] = tuple([pipe(result[i]) for pipe_num,pipe in enumerate(pipeline)])
                else:
                    result[i] = pipeline(result[i])
        return tuple(result)  # Convert it back to a tuple.

class FlashLoader(Loader):
    """Extended FFCV Loader with support for after-batch transforms.
    
    Extends the base FFCV Loader to apply additional PyTorch transformations
    to batches after they have been processed by the FFCV pipeline.
    
    Args:
        path (str): Path to the FFCV dataset file (.beton).
        batch_size (int): Number of samples per batch.
        order (ORDER_TYPE): Sample ordering strategy.
        num_workers (int, optional): Number of data loading workers. Defaults to -1.
        os_cache (bool, optional): Whether to use OS page cache. Defaults to DEFAULT_OS_CACHE.
        distributed (bool, optional): Whether to use distributed sampling. Defaults to False.
        seed (int, optional): Random seed for sample ordering. Defaults to None.
        indices (Sequence[int], optional): Subset of indices to use. Defaults to None.
        pipelines (Mapping, optional): FFCV processing pipelines per field. Defaults to {}.
        custom_fields (Mapping, optional): Custom field type mappings. Defaults to {}.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        batches_ahead (int, optional): Number of batches to prefetch. Defaults to 3.
        recompile (bool, optional): Whether to recompile pipelines each epoch. Defaults to False.
        custom_field_mapper (int, optional): Custom field mapper. Defaults to None.
        after_batch_pipelines (dict, optional): Dictionary mapping field names to
            transforms applied after batch loading. Can be a single transform or
            list of transforms. Defaults to None.
    """
    def __init__(
        self,
        path: str,
        batch_size: int,
        order: ORDER_TYPE,
        num_workers: int = -1,
        os_cache: bool = DEFAULT_OS_CACHE,
        distributed: bool = False,
        seed: int = None,  # For ordering of samples
        indices: Sequence[int] = None,  # For subset selection
        pipelines: Mapping[str, Sequence[Union[Operation, torch.nn.Module]]] = {},
        custom_fields: Mapping[str, Type[Field]] = {},
        drop_last: bool = True,
        batches_ahead: int = 3,
        recompile: bool = False,  # Recompile at every epoch
        custom_field_mapper: int = None,
        after_batch_pipelines: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            path=path, 
            batch_size=batch_size, 
            order=order, 
            num_workers=num_workers, 
            os_cache=os_cache, 
            distributed=distributed, 
            seed=seed,
            indices=indices,
            pipelines=pipelines, 
            custom_fields=custom_fields, 
            drop_last=drop_last, 
            batches_ahead=batches_ahead,
            recompile=recompile,
            custom_field_mapper=custom_field_mapper
        )
        self.after_batch_pipelines = after_batch_pipelines
        self.order = order
        self.os_cache = os_cache
        
    def __iter__(self):
        Compiler.set_num_threads(self.num_workers)
        order = self.next_traversal_order()
        selected_order = order[: len(self) * self.batch_size]
        self.next_epoch += 1

        # Compile at the first epoch
        if self.code_per_stage is None or self.recompile:
            self.generate_code()

        return CustomEpochIterator(self, selected_order, self.after_batch_pipelines)
        
    def __repr__(self):
        repr_str = (f"FlashLoader(\n"
                    f"\tData Path: {self.path}\n"
                    f"\tBatch Size: {self.batch_size}\n"
                    f"\tOrder: {self.order}\n"
                    f"\tNumber of Workers: {self.num_workers}\n"
                    f"\tOS Cache: {self.os_cache}\n"
                    f"\tDistributed: {self.distributed}\n"
                    f"\tDrop Last: {self.drop_last}\n"
                    f"\tRecompile: {self.recompile}\n"
                    f"\tAfter Batch Pipelines:\n {self.after_batch_pipelines}\n"
                    f")")
        return repr_str