# Background traceback watchdog blocked test completion

The native process sample of Python 3.11 worker 67334, taken at 01:52:32 EDT on September 25, identifies a diagnostic failure after the GeoJAX test had already printed `PASSED`.

- All 709 samples of the main thread end in `faulthandler_cancel_dump_traceback_later_py`, then `cancel_dump_traceback_later`, then a condition-variable wait.
- All 709 samples of the traceback watchdog end in `faulthandler_thread`, `_Py_DumpTracebackThreads`, `dump_traceback`, and `dump_frame`.
- The JAX worker threads are waiting for work. The sample does not show an ongoing GeoJAX computation.

The installed pytest 9.0.3 hook starts `faulthandler.dump_traceback_later` before a test and unconditionally cancels it in the protocol's `finally` block. Its zero-timeout branch avoids both calls. The CPython 3.11.15 implementation cancels by waiting for the watchdog's running lock; that lock is released only after the traceback dump completes. These implementations explain the observed blocked test teardown. [pytest hook](https://raw.githubusercontent.com/pytest-dev/pytest/9.0.3/src/_pytest/faulthandler.py), [CPython implementation](https://raw.githubusercontent.com/python/cpython/v3.11.15/Modules/faulthandler.c).

CPython has documented invalid/freed-frame failures in the background traceback dumper, including a report whose stack traverses the same watchdog and traceback functions. This supports a runtime diagnostic failure as the likely underlying cause; the precise invalid frame or loop causing this observed hang has not been proven. [CPython issue 140815](https://github.com/python/cpython/issues/140815).

The proposed correction is to force pytest's background traceback timeout to zero. Retain the independent parent-process wall deadline and request the already-supported signal snapshot only once that deadline has expired, immediately before terminating the failed worker. A periodic signal during a still-valid run is not an equivalent safe replacement: a separate CPython issue documents frame-reading failures in an all-thread signal dump on Python 3.11. [CPython issue 116008](https://github.com/python/cpython/issues/116008), [Python 3.11 documentation](https://docs.python.org/3.11/library/faulthandler.html).

No numerical implementation, test selection, assertion, tolerance, collection check, coverage requirement, or package artifact needs to change. Completed environments remain evidence for their recorded package/test hashes. The failed and incomplete environments must retain their original logs and be rerun under the corrected diagnostic control; old and new harness hashes must be distinguished in the aggregate evidence.
