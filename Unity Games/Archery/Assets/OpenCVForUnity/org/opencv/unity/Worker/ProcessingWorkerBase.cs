using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Worker
{
    /// <summary>
    /// Base class for processing workers that manage input and output Mats.
    /// </summary>
    /// <remarks>
    /// This class provides both synchronous and asynchronous execution modes for processing.
    /// All public methods in this class are thread-safe.
    ///
    /// IMPORTANT:
    /// 1. Execution Modes:
    ///    - Synchronous: Execute() - Blocks until completion
    ///    - Asynchronous: ExecuteTaskAsync() - Returns immediately, runs in background
    ///    - Only one operation (sync or async) can run at a time
    ///
    /// 2. Accessing Results:
    ///    - PeekOutput(): Returns a REFERENCE (submat) to internal buffer
    ///      - Fast but unsafe across executions
    ///      - Reference becomes invalid after next execution
    ///      - Should only be used for immediate, read-only access
    ///    - CopyOutput(): Returns a NEW copy of the result
    ///      - Thread-safe and reliable
    ///      - Recommended for most use cases
    ///      - Required when storing results or sharing across threads
    ///
    /// 3. Resource Management:
    ///    - Implements IDisposable and IAsyncDisposable for proper cleanup
    ///    - Dispose() / DisposeAsync() safely cancels running operations
    ///    - Waits for completion with a default timeout when disposing
    ///    - Automatically cleans up internal buffers
    ///
    /// 4. Thread Safety:
    ///    - All public methods are thread-safe
    ///    - Internal buffers are protected by locks
    ///    - Async operations use proper synchronization
    ///
    /// Example Usage:
    /// ```csharp
    /// using (var worker = new ConcreteWorker())
    /// {
    ///     void OnCompleted(ProcessingCompletion completion)
    ///     {
    ///         switch (completion.Kind)
    ///         {
    ///             case ProcessingCompletionKind.Succeeded:
    ///                 break;
    ///             case ProcessingCompletionKind.Canceled:
    ///                 break;
    ///             case ProcessingCompletionKind.Faulted:
    ///                 if (completion.Error != null)
    ///                     Debug.LogException(completion.Error);
    ///                 break;
    ///         }
    ///     }
    ///
    ///     worker.ProcessingCompleted += OnCompleted;
    ///
    ///     // ProcessingCompleted is raised when each Execute / ExecuteTaskAsync finishes (handler: ProcessingCompletion only).
    ///     // Synchronous execution
    ///     worker.Execute(input);
    ///     using (var result = worker.PeekOutput())
    ///     {
    ///         // Process result safely
    ///     }
    ///     // or
    ///     using (var result = worker.CopyOutput())
    ///     {
    ///         // Process result safely
    ///     }
    ///
    ///     // Asynchronous execution
    ///     await worker.ExecuteTaskAsync(input);
    ///     using (var result = worker.CopyOutput())
    ///     {
    ///         // Process result safely
    ///     }
    ///
    ///     worker.ProcessingCompleted -= OnCompleted;
    /// }
    /// ```
    /// </remarks>
    public abstract class ProcessingWorkerBase : IProcessingWorker, IAsyncDisposable
    {
        /// <summary>
        /// Lock object for thread synchronization.
        /// Protected to allow derived classes to implement thread-safe operations
        /// while maintaining the same locking mechanism as the base class.
        /// </summary>
        protected readonly object _lockObject = new object();

        protected Mat[] _inputs;
        protected Mat[] _outputs;
        protected bool _disposed = false;

        protected bool _isRunning = false;
        protected CancellationTokenSource _cts;
        protected TaskCompletionSource<bool> _executionCompletion;

        /// <summary>Maximum time <see cref="Dispose"/> and <see cref="DisposeAsync"/> allow for an in-progress operation to finish before continuing cleanup and logging a timeout.</summary>
        private static readonly TimeSpan DefaultDisposeOperationWaitTimeout = TimeSpan.FromMilliseconds(500);

        /// <summary>
        /// Completion status for <see cref="ProcessingCompleted"/>.
        /// </summary>
        public enum ProcessingCompletionKind
        {
            Succeeded,
            Canceled,
            Faulted
        }

        /// <summary>
        /// Occurs when a processing operation completes.
        /// </summary>
        /// <remarks>
        /// This event is raised after the processing operation completes (success, cancellation, or fault).
        /// The event is raised on the thread that completed the operation.
        /// Handlers receive only the <see cref="ProcessingCompletion"/> value (no sender argument).
        /// <see cref="ProcessingCompletion.Kind"/> uses the nested <see cref="ProcessingCompletionKind"/> on this class (same member names as <see cref="OpenCVForUnity.UnityIntegration.Runner.WorkCompletionKind"/> in the Runner namespace, but a distinct CLR type). Cancellation is reported when the
        /// </remarks>
        public event Action<ProcessingCompletion> ProcessingCompleted;

        /// <summary>
        /// Payload for <see cref="ProcessingCompleted"/>; same Kind / Error shape as <see cref="WorkCompletion{TResult}"/> without a results buffer (use <see cref="PeekOutput"/> / <see cref="CopyOutput"/> after success).
        /// </summary>
        public readonly struct ProcessingCompletion
        {
            /// <summary>
            /// Gets the completion status of the processing operation.
            /// </summary>
            public ProcessingCompletionKind Kind { get; }

            /// <summary>
            /// Gets the exception for <see cref="ProcessingCompletionKind.Canceled"/> or <see cref="ProcessingCompletionKind.Faulted"/>, if any.
            /// </summary>
            public Exception Error { get; }

            private ProcessingCompletion(ProcessingCompletionKind kind, Exception error)
            {
                Kind = kind;
                Error = error;
            }

            /// <summary>
            /// Successful completion.
            /// </summary>
            public static ProcessingCompletion ForSucceeded() =>
                new ProcessingCompletion(ProcessingCompletionKind.Succeeded, null);

            /// <summary>
            /// Canceled completion (e.g. <see cref="OperationCanceledException"/> from the async path).
            /// </summary>
            public static ProcessingCompletion ForCanceled(Exception error) =>
                new ProcessingCompletion(ProcessingCompletionKind.Canceled, error);

            /// <summary>
            /// Failed completion.
            /// </summary>
            public static ProcessingCompletion ForFaulted(Exception error) =>
                new ProcessingCompletion(ProcessingCompletionKind.Faulted, error);
        }

        /// <summary>
        /// Gets a value indicating whether a processing operation is currently running.
        /// </summary>
        /// <remarks>
        /// This property is thread-safe and can be safely accessed from any thread.
        /// The value is guaranteed to be consistent with the actual execution state.
        /// Use this property to check operation status before calling ExecuteTaskAsync
        /// or before accessing results.
        /// </remarks>
        public bool IsRunning
        {
            get
            {
                lock (_lockObject)
                {
                    return _isRunning;
                }
            }
        }

        /// <summary>
        /// Finalizer to ensure unmanaged resources are released.
        /// </summary>
        ~ProcessingWorkerBase()
        {
            Dispose(false);
        }

        /// <summary>
        /// Sets the input Mat at the specified index.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// The input Mat can be null to clear the input at the specified index.
        /// If not null, the input Mat is not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mat.
        /// </remarks>
        /// <param name="input">The input Mat. Can be null to clear the input.</param>
        /// <param name="index">The index at which to set the input.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        public virtual void SetInput(Mat input, int index = 0)
        {
            ThrowIfDisposed();

            lock (_lockObject)
            {
                EnsureInputsCapacity(index + 1);
                _inputs[index] = input;
            }
        }

        private void EnsureInputsCapacity(int capacity)
        {
            if (_inputs == null || _inputs.Length < capacity)
            {
                Array.Resize(ref _inputs, capacity);
            }
        }

        /// <summary>
        /// Sets multiple input Mats (optimized for 2 inputs).
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// The input Mats can be null to clear the respective inputs.
        /// If not null, the input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// </remarks>
        /// <param name="input1">The first input Mat. Can be null to clear the input.</param>
        /// <param name="input2">The second input Mat. Can be null to clear the input.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        public virtual void SetInputs(Mat input1, Mat input2)
        {
            ThrowIfDisposed();

            lock (_lockObject)
            {
                Array.Resize(ref _inputs, 2);
                _inputs[0] = input1;
                _inputs[1] = input2;
            }
        }

        /// <summary>
        /// Sets multiple input Mats (optimized for 3 inputs).
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// The input Mats can be null to clear the respective inputs.
        /// If not null, the input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// </remarks>
        /// <param name="input1">The first input Mat. Can be null to clear the input.</param>
        /// <param name="input2">The second input Mat. Can be null to clear the input.</param>
        /// <param name="input3">The third input Mat. Can be null to clear the input.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        public virtual void SetInputs(Mat input1, Mat input2, Mat input3)
        {
            ThrowIfDisposed();

            lock (_lockObject)
            {
                Array.Resize(ref _inputs, 3);
                _inputs[0] = input1;
                _inputs[1] = input2;
                _inputs[2] = input3;
            }
        }

        /// <summary>
        /// Sets multiple input Mats.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// The inputs array itself can be null to clear all inputs.
        /// Individual elements in the array can also be null to clear specific inputs.
        /// If not null, the input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// For better performance with 2 or 3 inputs, consider using the specialized SetInputs overloads instead.
        /// </remarks>
        /// <param name="inputs">The input Mats. Can be null to clear all inputs. Individual elements can also be null.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        public virtual void SetInputs(params Mat[] inputs)
        {
            ThrowIfDisposed();

            lock (_lockObject)
            {
                if (inputs != null)
                {
                    // Shrink as well as grow: otherwise a shorter call leaves stale Mat refs in the tail,
                    // and RunCoreProcessing may iterate _inputs.Length and hit disposed objects.
                    Array.Resize(ref _inputs, inputs.Length);
                    Array.Copy(inputs, _inputs, inputs.Length);
                }
                else
                {
                    _inputs = null;
                }
            }
        }

        /// <summary>
        /// Returns a reference (submat) to the output Mat at the specified index.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe but has important usage constraints:
        ///
        /// IMPORTANT:
        /// 1. The returned Mat is a REFERENCE (submat) to the internal buffer:
        ///    - Any modifications to the returned Mat will affect future results
        ///    - Disposing the returned Mat is safe and does not dispose the internal buffer itself; only the submat handle becomes invalid
        ///    - The reference becomes invalid after the next Execute/ExecuteTaskAsync call
        ///    - The reference becomes invalid if the worker is disposed
        ///
        /// 2. Thread Safety Considerations:
        ///    - DO NOT use with ExecuteTaskAsync() as the internal buffer may be modified during async operation
        ///    - Even with Execute(), the returned reference is only valid until the next execution
        ///    - For thread-safe access to results, always use CopyOutput() instead
        ///
        /// Best Practices:
        /// - Use CopyOutput() if you need to:
        ///   - Store the result for later use
        ///   - Modify the result
        ///   - Share the result across threads
        /// - Only use PeekOutput() when:
        ///   - You need immediate read-only access to the result
        ///   - You will complete all operations before the next Execute call
        ///   - Performance is critical and you understand the risks
        /// </remarks>
        /// <param name="index">The output index.</param>
        /// <returns>A reference to the internal output Mat. DO NOT store this reference.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is out of range or no outputs are available.</exception>
        public virtual Mat PeekOutput(int index = 0)
        {
            ThrowIfDisposed();

            lock (_lockObject)
            {
                if (_outputs == null || index >= _outputs.Length)
                    throw new ArgumentOutOfRangeException(nameof(index));

                return _outputs[index];
            }
        }

        /// <summary>
        /// Returns a copy of the output Mat at the specified index.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and can be safely used with both synchronous and asynchronous execution.
        /// When working with ExecuteTaskAsync(), always use this method instead of PeekOutput()
        /// to ensure thread safety and data consistency.
        /// </remarks>
        /// <param name="index">The output index.</param>
        /// <returns>A new Mat containing a copy of the output.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is out of range or no outputs are available.</exception>
        public virtual Mat CopyOutput(int index = 0)
        {
            ThrowIfDisposed();

            lock (_lockObject)
            {
                var output = PeekOutput(index);
                return output.clone();
            }
        }

        /// <summary>
        /// Copies the output Mat at the specified index to a destination Mat.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and can be safely used with both synchronous and asynchronous execution.
        /// When working with ExecuteTaskAsync(), always use this method instead of PeekOutput()
        /// to ensure thread safety and data consistency.
        ///
        /// The destination Mat will be automatically resized if its dimensions
        /// or type don't match the output Mat.
        /// </remarks>
        /// <param name="dst">The destination Mat.</param>
        /// <param name="index">The output index.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="ArgumentNullException">Thrown when the destination Mat is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is out of range or no outputs are available.</exception>
        public virtual void CopyOutputTo(Mat dst, int index = 0)
        {
            ThrowIfDisposed();

            if (dst != null) dst.ThrowIfDisposed();

            lock (_lockObject)
            {
                var output = PeekOutput(index);
                output.copyTo(dst);
            }
        }

        /// <summary>
        /// Visualizes the processing result on the given image.
        /// Should be overridden in derived classes.
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="result">The result Mat returned by processing.</param>
        /// <param name="printResult">Whether to print result to console.</param>
        /// <param name="isRGB">Whether the image is in RGB format (vs BGR).</param>
        public virtual void Visualize(Mat image, Mat result, bool printResult = false, bool isRGB = false)
        {
            throw new NotImplementedException("Visualize() must be overridden in a subclass to handle result rendering.");
        }

        /// <summary>
        /// Executes processing with pre-set inputs.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and will block until the processing is complete.
        /// Only one processing operation (sync or async) can run at a time.
        /// Attempting to execute while another operation is running will throw InvalidOperationException.
        /// </remarks>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        public virtual void Execute()
        {
            ThrowIfDisposed();
            lock (_lockObject)
            {
                if (_isRunning)
                    throw new InvalidOperationException("Another processing operation is already running.");

                _isRunning = true;
                TaskCompletionSource<bool> completionSource;
                try
                {
                    completionSource = new TaskCompletionSource<bool>();
                    _executionCompletion = completionSource;
                }
                catch
                {
                    _isRunning = false;
                    throw;
                }
                try
                {
                    if (_outputs != null)
                    {
                        foreach (var output in _outputs)
                            output?.Dispose();
                        _outputs = null;
                    }
                    _outputs = RunCoreProcessing(_inputs);
                    OnProcessingCompleted(true);
                }
                catch (Exception ex)
                {
                    OnProcessingCompleted(false, ex);
                    throw;
                }
                finally
                {
                    _isRunning = false;
                    completionSource.TrySetResult(true);
                    _executionCompletion = null;
                }
            }
        }

        /// <summary>
        /// Executes processing using single input Mat.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and will block until the processing is complete.
        /// Only one processing operation (sync or async) can run at a time.
        /// Attempting to execute while another operation is running will throw InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// </remarks>
        /// <param name="input">The input Mat.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        public virtual void Execute(Mat input)
        {
            SetInput(input);
            Execute();
        }

        /// <summary>
        /// Executes processing using multiple input Mats (optimized for 2 inputs).
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and will block until the processing is complete.
        /// Only one processing operation (sync or async) can run at a time.
        /// Attempting to execute while another operation is running will throw InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// </remarks>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        public virtual void Execute(Mat input1, Mat input2)
        {
            SetInputs(input1, input2);
            Execute();
        }

        /// <summary>
        /// Executes processing using multiple input Mats (optimized for 3 inputs).
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and will block until the processing is complete.
        /// Only one processing operation (sync or async) can run at a time.
        /// Attempting to execute while another operation is running will throw InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// </remarks>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        /// <param name="input3">The third input Mat.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        public virtual void Execute(Mat input1, Mat input2, Mat input3)
        {
            SetInputs(input1, input2, input3);
            Execute();
        }

        /// <summary>
        /// Executes processing using multiple inputs Mats.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and will block until the processing is complete.
        /// Only one processing operation (sync or async) can run at a time.
        /// Attempting to execute while another operation is running will throw InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// For better performance with 2 or 3 inputs, consider using the specialized SetInputs overloads instead.
        /// </remarks>
        /// <param name="inputs">The input Mats.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        public virtual void Execute(params Mat[] inputs)
        {
            SetInputs(inputs);
            Execute();
        }

        /// <summary>
        /// Executes processing asynchronously with pre-set inputs.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// Only one processing operation (sync or async) can run at a time - calling this method while another
        /// operation is in progress will throw an InvalidOperationException.
        ///
        /// IMPORTANT: When using async execution, you must:
        /// 1. Use CopyOutput() instead of PeekOutput() to retrieve results
        /// 2. Check IsRunning property to verify the operation status
        /// 3. Handle the cancellation token appropriately if needed
        ///
        /// The method creates a snapshot of inputs by storing references (not deep copies) to the input Mats,
        /// and the referenced Mats must not be modified or disposed until the async operation completes to ensure consistency.
        /// The internal output buffer may be modified during async operation,
        /// so direct references obtained via PeekOutput() are not safe to use.
        /// </remarks>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
        public virtual async Task ExecuteTaskAsync(CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            CancellationTokenSource cts = null;
            TaskCompletionSource<bool> completionSource = null;

            lock (_lockObject)
            {
                if (_isRunning)
                    throw new InvalidOperationException("Another processing operation is already running.");

                _isRunning = true;
                try
                {
                    // Create new CancellationTokenSource
                    cts = new CancellationTokenSource();
                    var previousCts = _cts;
                    _cts = cts;
                    previousCts?.Dispose();

                    completionSource = new TaskCompletionSource<bool>();
                    _executionCompletion = completionSource;
                }
                catch
                {
                    _isRunning = false;
                    throw;
                }
            }

            try
            {
                using (var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, cts.Token))
                {
                    linkedCts.Token.ThrowIfCancellationRequested();

                    Mat[] inputsSnapshot;
                    lock (_lockObject)
                    {
                        inputsSnapshot = (Mat[])_inputs?.Clone();
                    }

                    var results = await RunCoreProcessingTaskAsync(inputsSnapshot, linkedCts.Token);

                    lock (_lockObject)
                    {
                        if (_outputs != null)
                        {
                            foreach (var output in _outputs)
                                output?.Dispose();
                        }
                        _outputs = results;
                    }

                    OnProcessingCompleted(true);
                }
            }
            catch (Exception ex)
            {
                OnProcessingCompleted(false, ex);
                throw;
            }
            finally
            {
                lock (_lockObject)
                {
                    _isRunning = false;
                    completionSource?.TrySetResult(true);
                    _executionCompletion = null;

                    // Cleanup CTS
                    if (_cts == cts)
                    {
                        _cts.Dispose();
                        _cts = null;
                    }
                }
            }
        }

        /// <summary>
        /// Executes processing asynchronously using single input Mat.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// Only one processing operation (sync or async) can run at a time - calling this method while another
        /// operation is in progress will throw an InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        ///
        /// IMPORTANT: When using async execution, you must:
        /// 1. Use CopyOutput() instead of PeekOutput() to retrieve results
        /// 2. Check IsRunning property to verify the operation status
        /// 3. Handle the cancellation token appropriately if needed
        ///
        /// The method creates a snapshot of inputs by storing references (not deep copies) to the input Mats,
        /// and the referenced Mats must not be modified or disposed until the async operation completes to ensure consistency.
        /// The internal output buffer may be modified during async operation,
        /// so direct references obtained via PeekOutput() are not safe to use.
        /// </remarks>
        /// <param name="input">The input Mat.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
        public virtual Task ExecuteTaskAsync(Mat input, CancellationToken cancellationToken = default)
        {
            SetInput(input);
            return ExecuteTaskAsync(cancellationToken);
        }

        /// <summary>
        /// Executes processing asynchronously using multiple input Mats (optimized for 2 inputs).
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// Only one processing operation (sync or async) can run at a time - calling this method while another
        /// operation is in progress will throw an InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        ///
        /// IMPORTANT: When using async execution, you must:
        /// 1. Use CopyOutput() instead of PeekOutput() to retrieve results
        /// 2. Check IsRunning property to verify the operation status
        /// 3. Handle the cancellation token appropriately if needed
        ///
        /// The method creates a snapshot of inputs by storing references (not deep copies) to the input Mats,
        /// and the referenced Mats must not be modified or disposed until the async operation completes to ensure consistency.
        /// The internal output buffer may be modified during async operation,
        /// so direct references obtained via PeekOutput() are not safe to use.
        /// </remarks>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
        public virtual Task ExecuteTaskAsync(Mat input1, Mat input2, CancellationToken cancellationToken = default)
        {
            SetInputs(input1, input2);
            return ExecuteTaskAsync(cancellationToken);
        }

        /// <summary>
        /// Executes processing asynchronously using multiple input Mats (optimized for 3 inputs).
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// Only one processing operation (sync or async) can run at a time - calling this method while another
        /// operation is in progress will throw an InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        ///
        /// IMPORTANT: When using async execution, you must:
        /// 1. Use CopyOutput() instead of PeekOutput() to retrieve results
        /// 2. Check IsRunning property to verify the operation status
        /// 3. Handle the cancellation token appropriately if needed
        ///
        /// The method creates a snapshot of inputs by storing references (not deep copies) to the input Mats,
        /// and the referenced Mats must not be modified or disposed until the async operation completes to ensure consistency.
        /// The internal output buffer may be modified during async operation,
        /// so direct references obtained via PeekOutput() are not safe to use.
        /// </remarks>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        /// <param name="input3">The third input Mat.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
        public virtual Task ExecuteTaskAsync(Mat input1, Mat input2, Mat input3, CancellationToken cancellationToken = default)
        {
            SetInputs(input1, input2, input3);
            return ExecuteTaskAsync(cancellationToken);
        }

        /// <summary>
        /// Executes processing asynchronously using multiple input Mats.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe.
        /// Only one processing operation (sync or async) can run at a time - calling this method while another
        /// operation is in progress will throw an InvalidOperationException.
        /// The input Mats are not modified or disposed by this class.
        /// The caller maintains ownership and is responsible for managing the lifecycle of the input Mats.
        /// For better performance with 2 or 3 inputs, consider using the specialized SetInputs overloads instead.
        ///
        /// IMPORTANT: When using async execution, you must:
        /// 1. Use CopyOutput() instead of PeekOutput() to retrieve results
        /// 2. Check IsRunning property to verify the operation status
        /// 3. Handle the cancellation token appropriately if needed
        ///
        /// The method creates a snapshot of inputs by storing references (not deep copies) to the input Mats,
        /// and the referenced Mats must not be modified or disposed until the async operation completes to ensure consistency.
        /// The internal output buffer may be modified during async operation,
        /// so direct references obtained via PeekOutput() are not safe to use.
        /// </remarks>
        /// <param name="inputs">The input Mats.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the worker has been disposed.</exception>
        /// <exception cref="InvalidOperationException">Thrown when another processing operation is already running.</exception>
        /// <exception cref="OperationCanceledException">Thrown when the operation is cancelled.</exception>
        public virtual Task ExecuteTaskAsync(Mat[] inputs, CancellationToken cancellationToken = default)
        {
            SetInputs(inputs);
            return ExecuteTaskAsync(cancellationToken);
        }

        /// <inheritdoc cref="IProcessingWorker.ExecuteAsync(CancellationToken)"/>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="ExecuteTaskAsync(CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task"/>.
        /// </remarks>
        [Obsolete("Use ExecuteTaskAsync(). ExecuteAsync() will return Awaitable in a future version.")]
        public virtual Task ExecuteAsync(CancellationToken cancellationToken = default) =>
            ExecuteTaskAsync(cancellationToken);

        /// <inheritdoc cref="IProcessingWorker.ExecuteAsync(Mat, CancellationToken)"/>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="ExecuteTaskAsync(Mat, CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task"/>.
        /// </remarks>
        [Obsolete("Use ExecuteTaskAsync(). ExecuteAsync() will return Awaitable in a future version.")]
        public virtual Task ExecuteAsync(Mat input, CancellationToken cancellationToken = default) =>
            ExecuteTaskAsync(input, cancellationToken);

        /// <inheritdoc cref="IProcessingWorker.ExecuteAsync(Mat, Mat, CancellationToken)"/>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="ExecuteTaskAsync(Mat, Mat, CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task"/>.
        /// </remarks>
        [Obsolete("Use ExecuteTaskAsync(). ExecuteAsync() will return Awaitable in a future version.")]
        public virtual Task ExecuteAsync(Mat input1, Mat input2, CancellationToken cancellationToken = default) =>
            ExecuteTaskAsync(input1, input2, cancellationToken);

        /// <inheritdoc cref="IProcessingWorker.ExecuteAsync(Mat, Mat, Mat, CancellationToken)"/>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="ExecuteTaskAsync(Mat, Mat, Mat, CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task"/>.
        /// </remarks>
        [Obsolete("Use ExecuteTaskAsync(). ExecuteAsync() will return Awaitable in a future version.")]
        public virtual Task ExecuteAsync(Mat input1, Mat input2, Mat input3, CancellationToken cancellationToken = default) =>
            ExecuteTaskAsync(input1, input2, input3, cancellationToken);

        /// <inheritdoc cref="IProcessingWorker.ExecuteAsync(Mat[], CancellationToken)"/>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="ExecuteTaskAsync(Mat[], CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task"/>.
        /// </remarks>
        [Obsolete("Use ExecuteTaskAsync(). ExecuteAsync() will return Awaitable in a future version.")]
        public virtual Task ExecuteAsync(Mat[] inputs, CancellationToken cancellationToken = default) =>
            ExecuteTaskAsync(inputs, cancellationToken);

        /// <summary>
        /// Cancels the current processing operation if running.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and can be safely called from any thread.
        /// This method provides a convenient way to cancel the current operation without managing a CancellationToken.
        /// It has the same effect as canceling the token passed to ExecuteTaskAsync.
        /// </remarks>
        public virtual void Cancel()
        {
            lock (_lockObject)
            {
                try
                {
                    _cts?.Cancel();
                }
                catch (ObjectDisposedException)
                {
                    // Ignore already disposed
                }
            }
        }

        /// <summary>
        /// Releases resources used by the worker.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and ensures proper cleanup of resources:
        /// - Cancels any running processing operation
        /// - Waits for operation completion with a timeout (<see cref="DefaultDisposeOperationWaitTimeout"/>)
        /// - Disposes all managed resources (Mats, CancellationTokenSource)
        ///
        /// Multiple calls to Dispose are allowed but only the first call will release the resources.
        /// If a running operation doesn't complete within the wait period (typical processing takes less than 200ms),
        /// it will be forcefully terminated.
        /// <para>
        /// <b>WebGL:</b> This method performs synchronous blocking waits, which can cause deadlocks or hangs on WebGL.
        /// On WebGL, prefer <see cref="DisposeAsync"/>.
        /// </para>
        /// </remarks>
        /// <exception cref="ObjectDisposedException">Thrown when trying to access a disposed instance.</exception>
        public void Dispose()
        {
            if (_disposed) return;

            TaskCompletionSource<bool> completionSource = null;
            lock (_lockObject)
            {
                if (_isRunning)
                {
                    try
                    {
                        _cts?.Cancel();
                        completionSource = _executionCompletion;
                    }
                    catch (ObjectDisposedException)
                    {
                        // Ignore already disposed
                    }
                }
            }

            if (completionSource != null)
            {
                if (!TryWaitForTaskUntilCompleteOrTimeout(completionSource.Task, DefaultDisposeOperationWaitTimeout))
                {
                    Debug.LogWarning($"[{GetType().Name}] Dispose timeout: Operation did not complete within {DefaultDisposeOperationWaitTimeout.TotalMilliseconds:0}ms. This may indicate a performance issue or deadlock.");
                }
                else if (completionSource.Task.IsFaulted)
                {
                    _ = completionSource.Task.Exception;
                }
            }

            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Asynchronously releases resources: cancels a running operation, waits up to <see cref="DefaultDisposeOperationWaitTimeout"/> for it to complete,
        /// then disposes internal buffers. Prefer over <see cref="Dispose()"/> when you can <see langword="await"/>
        /// to avoid blocking a thread on the completion wait.
        /// </summary>
        public virtual async ValueTask DisposeAsync()
        {
            if (_disposed) return;

            TaskCompletionSource<bool> completionSource = null;
            lock (_lockObject)
            {
                if (_isRunning)
                {
                    try
                    {
                        _cts?.Cancel();
                        completionSource = _executionCompletion;
                    }
                    catch (ObjectDisposedException)
                    {
                        // Ignore already disposed
                    }
                }
            }

            if (completionSource != null)
            {
                if (!await TryWaitForTaskUntilCompleteOrTimeoutAsync(completionSource.Task, DefaultDisposeOperationWaitTimeout))
                {
                    Debug.LogWarning($"[{GetType().Name}] DisposeAsync timeout: Operation did not complete within {DefaultDisposeOperationWaitTimeout.TotalMilliseconds:0}ms. This may indicate a performance issue or deadlock.");
                }
                else if (completionSource.Task.IsFaulted)
                {
                    _ = completionSource.Task.Exception;
                }
            }

            if (_disposed) return;

            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the object, optionally releasing managed resources.
        /// </summary>
        /// <param name="disposing">True to release managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                lock (_lockObject)
                {
                    _inputs = null;

                    if (_outputs != null)
                    {
                        foreach (var output in _outputs)
                            output?.Dispose();
                        _outputs = null;
                    }

                    _cts?.Dispose();
                    _cts = null;
                }
            }

            _disposed = true;
        }

        /// <summary>
        /// Throws an exception if the worker has been disposed.
        /// </summary>
        protected virtual void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(GetType().FullName);
        }

        /// <summary>
        /// Subclasses implement the synchronous processing.
        /// </summary>
        /// <remarks>
        /// This method is called by Execute. The implementation should:
        /// - Handle resources properly
        /// - Not modify the input Mats
        /// - Return new output Mats. Typically return extracted submats instead of direct Mats to avoid buffer dispose
        /// - Never return null elements in the output array. Use empty Mats to indicate no output
        /// </remarks>
        /// <param name="inputs">The input Mats.</param>
        /// <returns>Array of output Mats containing processing results. No element should be null.</returns>
        protected abstract Mat[] RunCoreProcessing(Mat[] inputs);

        /// <summary>
        /// Subclasses implement the asynchronous processing.
        /// </summary>
        /// <remarks>
        /// This method is called by ExecuteTaskAsync with a snapshot of the inputs
        /// to ensure thread safety. The implementation should:
        /// - Respect the cancellation token by calling ThrowIfCancellationRequested periodically
        /// - Handle resources properly
        /// - Not modify the input Mats
        /// - Return new output Mats. Typically return extracted submats instead of direct Mats to avoid buffer dispose
        /// - Never return null elements in the output array. Use empty Mats to indicate no output
        /// </remarks>
        /// <param name="inputs">Snapshot of the input Mats.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        protected abstract Task<Mat[]> RunCoreProcessingTaskAsync(Mat[] inputs, CancellationToken cancellationToken);

        /// <summary>
        /// Subclasses implement the asynchronous processing.
        /// </summary>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="RunCoreProcessingTaskAsync(Mat[], CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task{TResult}"/>.
        /// </remarks>
        /// <param name="inputs">Snapshot of the input Mats.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        [Obsolete("Use RunCoreProcessingTaskAsync(). RunCoreProcessingAsync() will return Awaitable in a future version.")]
        protected Task<Mat[]> RunCoreProcessingAsync(Mat[] inputs, CancellationToken cancellationToken) =>
            RunCoreProcessingTaskAsync(inputs, cancellationToken);

        /// <summary>
        /// Waits for the current processing operation to complete with a timeout.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and can be safely called from any thread.
        /// If no processing operation is running, returns immediately.
        /// If the operation doesn't complete within the specified timeout, throws TimeoutException.
        /// This method works for both synchronous (Execute) and asynchronous (ExecuteTaskAsync) operations.
        /// </remarks>
        /// <param name="timeout">The timeout duration. If null, waits indefinitely.</param>
        /// <returns>A task that completes when the processing operation completes or times out.</returns>
        /// <exception cref="TimeoutException">Thrown when the operation doesn't complete within the specified timeout.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the worker is in an invalid internal state (execution without a completion source).</exception>
        public virtual async Task WaitForCompletionTaskAsync(TimeSpan? timeout = null)
        {
            TaskCompletionSource<bool> completionSource;
            lock (_lockObject)
            {
                if (!_isRunning)
                    return;

                completionSource = _executionCompletion;
            }

            if (completionSource == null)
            {
                throw new InvalidOperationException("Processing is running but the completion source is not available. This is an internal error in ProcessingWorkerBase.");
            }

            if (!await TryWaitForTaskUntilCompleteOrTimeoutAsync(completionSource.Task, timeout))
            {
                throw new TimeoutException($"Processing operation did not complete within {timeout.Value.TotalMilliseconds}ms.");
            }

            await completionSource.Task;
        }

        /// <summary>
        /// For synchronous APIs. Polls on each iteration while yielding via <see cref="Task.Yield"/>.
        /// </summary>
        /// <param name="timeout">When <see langword="null"/>, waits indefinitely.</param>
        /// <returns><see langword="true"/> when the task completed; <see langword="false"/> on timeout.</returns>
        private static bool TryWaitForTaskUntilCompleteOrTimeout(Task task, TimeSpan? timeout) =>
            PollUntilTaskCompleteOrTimeout(task, timeout, () => Task.Yield().GetAwaiter().GetResult());

        /// <summary>
        /// For asynchronous APIs. Polls on each iteration with <see cref="Task.Yield"/> so the Unity player loop can advance (WebGL-safe).
        /// </summary>
        /// <param name="timeout">When <see langword="null"/>, waits indefinitely.</param>
        /// <returns><see langword="true"/> when the task completed; <see langword="false"/> on timeout.</returns>
        private static Task<bool> TryWaitForTaskUntilCompleteOrTimeoutAsync(Task task, TimeSpan? timeout) =>
            PollUntilTaskCompleteOrTimeoutAsync(task, timeout);

        /// <summary>
        /// Polls until <paramref name="task"/> completes or <paramref name="timeout"/> elapses (synchronous yield).
        /// </summary>
        private static bool PollUntilTaskCompleteOrTimeout(Task task, TimeSpan? timeout, Action yieldStep)
        {
            double? deadline = GetWaitDeadline(timeout);

            while (!task.IsCompleted)
            {
                if (IsWaitDeadlineExceeded(deadline))
                    return false;

                yieldStep();
            }

            return true;
        }

        /// <summary>
        /// Polls until <paramref name="task"/> completes or <paramref name="timeout"/> elapses (asynchronous yield).
        /// </summary>
        private static async Task<bool> PollUntilTaskCompleteOrTimeoutAsync(Task task, TimeSpan? timeout)
        {
            double? deadline = GetWaitDeadline(timeout);

            while (!task.IsCompleted)
            {
                if (IsWaitDeadlineExceeded(deadline))
                    return false;

                await Task.Yield();
            }

            return true;
        }

        private static double? GetWaitDeadline(TimeSpan? timeout) =>
            timeout.HasValue ? Time.realtimeSinceStartupAsDouble + timeout.Value.TotalSeconds : null;

        private static bool IsWaitDeadlineExceeded(double? deadline) =>
            deadline.HasValue && Time.realtimeSinceStartupAsDouble >= deadline.Value;

        /// <summary>
        /// Waits for the current processing operation to complete with a timeout.
        /// </summary>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="WaitForCompletionTaskAsync(TimeSpan?)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task"/>.
        /// </remarks>
        /// <param name="timeout">The timeout duration. If null, waits indefinitely.</param>
        /// <returns>A task that completes when the processing operation completes or times out.</returns>
        /// <exception cref="TimeoutException">Thrown when the operation doesn't complete within the specified timeout.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the worker is in an invalid internal state (execution without a completion source).</exception>
        [Obsolete("Use WaitForCompletionTaskAsync(). WaitForCompletionAsync() will return Awaitable in a future version.")]
        public virtual Task WaitForCompletionAsync(TimeSpan? timeout = null) =>
            WaitForCompletionTaskAsync(timeout);

        /// <summary>
        /// Waits for the current processing operation to complete with a timeout.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and can be safely called from any thread.
        /// If no processing operation is running, returns immediately.
        /// If the operation doesn't complete within the specified timeout, throws TimeoutException.
        /// This method works for both synchronous (Execute) and asynchronous (ExecuteTaskAsync) operations.
        /// <para>
        /// <b>WebGL:</b> This method performs synchronous blocking waits, which can cause deadlocks or hangs on WebGL.
        /// On WebGL, prefer <see cref="WaitForCompletionTaskAsync(TimeSpan?)"/>.
        /// </para>
        /// </remarks>
        /// <param name="timeout">The timeout duration. If null, waits indefinitely.</param>
        /// <exception cref="TimeoutException">Thrown when the operation doesn't complete within the specified timeout.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the worker is in an invalid internal state (execution without a completion source).</exception>
        public virtual void WaitForCompletion(TimeSpan? timeout = null)
        {
            TaskCompletionSource<bool> completionSource;
            lock (_lockObject)
            {
                if (!_isRunning)
                    return;

                completionSource = _executionCompletion;
            }

            if (completionSource == null)
            {
                throw new InvalidOperationException("Processing is running but the completion source is not available. This is an internal error in ProcessingWorkerBase.");
            }

            if (!TryWaitForTaskUntilCompleteOrTimeout(completionSource.Task, timeout))
            {
                throw new TimeoutException($"Processing operation did not complete within {timeout.Value.TotalMilliseconds}ms.");
            }

            completionSource.Task.GetAwaiter().GetResult();
        }

        /// <summary>
        /// Raises the ProcessingCompleted event.
        /// </summary>
        /// <param name="isSuccess">Whether the operation completed successfully.</param>
        /// <param name="error">The exception that occurred when <paramref name="isSuccess"/> is <see langword="false"/>.</param>
        protected virtual void OnProcessingCompleted(bool isSuccess, Exception error = null)
        {
            ProcessingCompletion completion;
            if (isSuccess)
                completion = ProcessingCompletion.ForSucceeded();
            else if (error is OperationCanceledException)
                completion = ProcessingCompletion.ForCanceled(error);
            else
                completion = ProcessingCompletion.ForFaulted(error);

            ProcessingCompleted?.Invoke(completion);
        }
    }
}
