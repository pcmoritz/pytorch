torch.mps
===================================
.. automodule:: torch.mps
.. currentmodule:: torch.mps

.. autosummary::
    :toctree: generated
    :nosignatures:

    device_count
    synchronize
    get_rng_state
    set_rng_state
    manual_seed
    seed
    empty_cache
    set_per_process_memory_fraction
    current_allocated_memory
    driver_allocated_memory
    recommended_max_memory
    compile_shader

MPS Profiler
------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.start
    profiler.stop
    profiler.profile

    profiler.is_capturing_metal
    profiler.is_metal_capture_enabled
    profiler.metal_capture

MPS Event
------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    event.Event


.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.mps.event
.. py:module:: torch.mps.profiler
