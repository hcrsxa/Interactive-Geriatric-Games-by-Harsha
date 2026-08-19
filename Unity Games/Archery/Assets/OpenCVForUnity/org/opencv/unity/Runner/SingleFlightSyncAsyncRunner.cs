using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Runner
{
    /// <summary>
    /// Completion status for <see cref="SingleFlightSyncAsyncRunner{TInput,TResult}.WorkCompleted"/>.
    /// </summary>
    public enum WorkCompletionKind
    {
        Succeeded,
        Canceled,
        Faulted
    }

    /// <summary>
    /// Payload for <see cref="SingleFlightSyncAsyncRunner{TInput,TResult}.WorkCompleted"/>.
    /// </summary>
    /// <typeparam name="TResult">Reference result element type.</typeparam>
    public readonly struct WorkCompletion<TResult> where TResult : class
    {
        /// <summary>
        /// Gets the completion kind.
        /// </summary>
        public WorkCompletionKind Kind { get; }

        /// <summary>
        /// Gets the exception for <see cref="WorkCompletionKind.Canceled"/> or <see cref="WorkCompletionKind.Faulted"/>, if any.
        /// </summary>
        public Exception Error { get; }

        /// <summary>
        /// Gets the latest result array when <see cref="Kind"/> is <see cref="WorkCompletionKind.Succeeded"/>; otherwise <see langword="null"/>.
        /// Same live buffer as <see cref="SingleFlightSyncAsyncRunner{TInput,TResult}.TryGetLatestResult"/>; runner-owned until replaced or disposed.
        /// </summary>
        public TResult[] Results { get; }

        private WorkCompletion(WorkCompletionKind kind, Exception error, TResult[] results)
        {
            Kind = kind;
            Error = error;
            Results = results;
        }

        /// <summary>
        /// Successful completion with the runner's current latest result buffer.
        /// </summary>
        public static WorkCompletion<TResult> ForSucceeded(TResult[] results) =>
            new WorkCompletion<TResult>(WorkCompletionKind.Succeeded, null, results);

        /// <summary>
        /// Canceled completion (e.g. <see cref="OperationCanceledException"/> on the async path).
        /// </summary>
        public static WorkCompletion<TResult> ForCanceled(Exception error) =>
            new WorkCompletion<TResult>(WorkCompletionKind.Canceled, error, null);

        /// <summary>
        /// Failed completion.
        /// </summary>
        public static WorkCompletion<TResult> ForFaulted(Exception error) =>
            new WorkCompletion<TResult>(WorkCompletionKind.Faulted, error, null);
    }

    /// <summary>
    /// Orchestrates at most one in-flight asynchronous <see cref="Task"/> versus synchronous <c>syncWork</c>,
    /// optional host cancellation, and a latest-result buffer for reference-type pipelines.
    /// </summary>
    /// <remarks>
    /// <typeparamref name="TInput"/> and <typeparamref name="TResult"/> are constrained to reference types.
    /// The async path does not copy inputs by itself; override <see cref="CopyInputsForAsync"/> to snapshot inputs
    /// before <c>asyncWork</c> runs (for example OpenCV <c>Mat</c> via <c>copyTo</c> in a derived type).
    ///
    /// IMPORTANT:
    /// 1. Execution modes (<see cref="UseAsyncWork"/>):
    ///    - <see langword="false"/>: <see cref="SubmitWork"/> runs <c>syncWork</c> on the caller thread (after any
    ///      still-running async task from a prior async mode has finished).
    ///    - <see langword="true"/>: <see cref="SubmitWork"/> starts at most one <c>asyncWork</c> <see cref="Task"/>; while
    ///      <see cref="IsWorkInFlight"/> is <see langword="true"/>, further <see cref="SubmitWork"/> calls are ignored.
    ///
    /// 2. Single in-flight rule:
    ///    - Keeps one async job in flight and avoids overlapping <c>syncWork</c> with a still-running <c>asyncWork</c>
    ///      (for example shared DNN <c>Net</c> or buffers).
    ///    - <see cref="TryGetLatestResult"/> still returns the last completed results when a submission is skipped.
    ///
    /// 3. Cancellation:
    ///    - Pass a host <see cref="CancellationToken"/> to the constructor (<c>asyncWorkCancellationToken</c>); it is
    ///      linked with a per-runner local token for each async submission.
    ///    - Inside <c>asyncWork</c>, pass <see cref="InFlightAsyncWorkCancellationToken"/> to APIs that accept a
    ///      <see cref="CancellationToken"/> (it is only non-default while <c>asyncWork</c> is executing). Do not rely on
    ///      it from <c>syncWork</c>.
    ///    - <see cref="Cancel"/> cancels the in-flight scope via the local token; changing <see cref="UseAsyncWork"/> also cancels.
    ///
    /// 4. Accessing results:
    ///    - Use <see cref="TryGetLatestResult"/> for the live buffer owned by the runner; do not <c>Dispose</c> returned
    ///      elements unless you have copied ownership off—see <see cref="TryGetLatestResult"/> remarks.
    ///
    /// 5. Resource management:
    ///    - Implements <see cref="IDisposable"/> and <see cref="IAsyncDisposable"/>; prefer <see cref="DisposeAsync"/> when
    ///      you can <see langword="await"/> so the optional <c>disposeAsyncAfterWorkTask</c> runs after the in-flight task.
    ///    - <see cref="Dispose"/> does not await that hook.
    ///
    /// 6. Thread safety:
    ///    - Public surface is safe to call from multiple threads; internal coordination uses locks.
    ///
    /// 7. Work completion event:
    ///    - <see cref="WorkCompleted"/> is raised on the thread that completed the work (same as <see cref="ProcessingWorkerBase.ProcessingCompleted"/>); handlers receive only <see cref="WorkCompletion{TResult}"/> (no sender argument). <see cref="WorkCompletionKind"/> is defined in this namespace; <see cref="ProcessingWorkerBase"/> uses its own nested <see cref="ProcessingWorkerBase.ProcessingCompletionKind"/> for <see cref="ProcessingCompletion"/>.
    ///      Synchronous path: <see cref="SubmitWork"/> caller thread. Asynchronous path: the <c>await</c> continuation thread;
    ///      when <see cref="SubmitWork"/> is called from the Unity main thread, <c>await asyncWork(...).ConfigureAwait(true)</c> typically
    ///      resumes on the captured <see cref="System.Threading.SynchronizationContext"/> (often the main thread), but this is not guaranteed.
    ///      Do not use Unity main-thread-only APIs in the handler unless you know the completion thread; marshal if needed.
    ///    - Do not dispose <c>Results</c> elements in the handler; ownership matches <see cref="TryGetLatestResult"/>.
    ///    - No event when <see cref="SubmitWork"/> is skipped (in-flight) or when work returns <c>null</c> outputs.
    ///    - Prefer unsubscribing before <see cref="Dispose"/> when possible.
    ///
    /// Example Usage:
    /// ```csharp
    /// // TInput/TResult are reference types (e.g. Mat[] in a derived runner).
    /// var runner = new MyRunner(
    ///     useAsyncWork: true,
    ///     asyncWorkCancellationToken: hostCts.Token,
    ///     disposeAsyncAfterWorkTask: async () => { await worker.WaitForCompletionTaskAsync(); });
    ///
    /// runner.SubmitWork(
    ///     myInputs,
    ///     syncWork: inputs => ComputeSync(inputs),
    ///     asyncWork: async inputs =>
    ///     {
    ///         CancellationToken ct = runner.InFlightAsyncWorkCancellationToken;
    ///         return await ComputeAsync(inputs, ct);
    ///     });
    ///
    /// void OnWorkCompleted(WorkCompletion&lt;TResult&gt; completion)
    /// {
    ///     switch (completion.Kind)
    ///     {
    ///         case WorkCompletionKind.Succeeded:
    ///             // completion.Results is the same live buffer as TryGetLatestResult; do not Dispose elements.
    ///             break;
    ///         case WorkCompletionKind.Canceled:
    ///             break;
    ///         case WorkCompletionKind.Faulted:
    ///             Debug.LogException(completion.Error);
    ///             break;
    ///     }
    /// }
    /// runner.WorkCompleted += OnWorkCompleted;
    ///
    /// if (runner.TryGetLatestResult(out TResult[] results))
    /// {
    ///     // Consume results; do not Dispose owned elements unless documented otherwise.
    /// }
    ///
    /// runner.WorkCompleted -= OnWorkCompleted;
    /// await runner.DisposeAsync();
    /// ```
    /// </remarks>
    public abstract class SingleFlightSyncAsyncRunner<TInput, TResult> : IDisposable, IAsyncDisposable
        where TInput : class
        where TResult : class
    {
        private readonly CancellationToken _asyncWorkCancellationToken;
        private readonly Func<Task> _disposeAsyncAfterWorkTask;
        private readonly object _ctsLock = new object();
        private CancellationTokenSource _localCts;
        private CancellationTokenSource _linkedCts;

        private bool _useAsyncWork;
        private bool _disposed;

        private CancellationToken _inFlightAsyncWorkCancellationToken;
        private TResult[] _latestResults;
        private TInput[] _asyncInputBuffer;
        private Task _inFlightWorkTask;
        private readonly object _inFlightWorkTaskLock = new object();

        private readonly object _workCompletedGate = new object();
        private CancellationTokenSource _workCompletedDeliveryCts;

        /// <summary>
        /// Occurs when a submitted work unit completes (success, cancellation, or fault), on the thread that completed the operation.
        /// </summary>
        /// <remarks>
        /// For <see cref="WorkCompletionKind.Succeeded"/>, <see cref="WorkCompletion{TResult}.Results"/> is the same live buffer as
        /// <see cref="TryGetLatestResult"/>; do not <c>Dispose</c> elements unless ownership has been copied off.
        /// Unity objects and UGUI should only be touched from the main thread; marshal from the handler when the completion thread may differ.
        /// Handlers receive only the <see cref="WorkCompletion{TResult}"/> value (no sender argument).
        /// </remarks>
        public event Action<WorkCompletion<TResult>> WorkCompleted;

        /// <summary>
        /// Initializes a new runner. Only <see cref="DisposeAsync"/> runs <paramref name="disposeAsyncAfterWorkTask"/>;
        /// <see cref="Dispose"/> does not.
        /// </summary>
        /// <param name="useAsyncWork">Initial async (non-in-flight) mode.</param>
        /// <param name="asyncWorkCancellationToken">Token that cancels the linked work scope; combined with
        /// the local token used by <see cref="Cancel"/> and <see cref="UseAsyncWork"/> changes.</param>
        /// <param name="disposeAsyncAfterWorkTask">Optional async cleanup after the in-flight <see cref="Task"/>
        /// (e.g. <c>async () => { await a.WaitForCompletionTaskAsync(); await b.WaitForCompletionTaskAsync(); }</c>).</param>
        protected SingleFlightSyncAsyncRunner(
            bool useAsyncWork,
            CancellationToken asyncWorkCancellationToken = default,
            Func<Task> disposeAsyncAfterWorkTask = null)
        {
            _useAsyncWork = useAsyncWork;
            _asyncWorkCancellationToken = asyncWorkCancellationToken;
            _disposeAsyncAfterWorkTask = disposeAsyncAfterWorkTask;
            InitializeLocalAndLinkedCts();
            _workCompletedDeliveryCts = new CancellationTokenSource();
        }

        /// <summary>
        /// Cancellation token produced by linking the constructor cancellation token with the runner local cancellation token
        /// while an asynchronous <c>asyncWork</c> delegate is executing. Outside that window (including the synchronous path),
        /// this is <see langword="default"/>.
        /// </summary>
        public CancellationToken InFlightAsyncWorkCancellationToken => _inFlightAsyncWorkCancellationToken;

        /// <summary>
        /// <see langword="true"/> while a previously started asynchronous <see cref="Task"/> is still
        /// in progress (not yet completed, faulted, or canceled). When <see langword="true"/>, <see cref="SubmitWork"/>
        /// will not start new work.
        /// </summary>
        public bool IsWorkInFlight
        {
            get
            {
                lock (_inFlightWorkTaskLock)
                {
                    return _inFlightWorkTask != null && !_inFlightWorkTask.IsCompleted;
                }
            }
        }

        /// <summary>
        /// When <see langword="true"/>, at most one async task runs; a new one starts only when the previous
        /// task has completed. When <see langword="false"/>, <see cref="SubmitWork"/> runs <c>syncWork</c> on
        /// the caller thread (after any still-running asynchronous task from a prior async mode has finished;
        /// see <see cref="IsWorkInFlight"/> and <see cref="SubmitWork"/> remarks).
        /// </summary>
        public bool UseAsyncWork
        {
            get => _useAsyncWork;
            set
            {
                if (_useAsyncWork == value) return;
                CancelInFlightThroughLocalToken();
                _useAsyncWork = value;
            }
        }

        /// <summary>
        /// Submits <paramref name="inputs"/> for work: <paramref name="syncWork"/> on the caller when
        /// <see cref="UseAsyncWork"/> is <see langword="false"/>, or a single in-flight
        /// <paramref name="asyncWork"/> when <see langword="true"/>. When <see langword="true"/>,
        /// <see cref="CopyInputsForAsync"/> is used before <c>asyncWork</c>. <c>syncWork</c> receives
        /// <paramref name="inputs"/> as passed in.
        /// </summary>
        /// <remarks>
        /// If a previous asynchronous <see cref="Task"/> is still in progress
        /// (<see cref="IsWorkInFlight"/> is <see langword="true"/>), this method does nothing: neither
        /// <paramref name="syncWork"/> nor a new <paramref name="asyncWork"/> is started. That keeps at most
        /// one async job in flight and avoids overlapping synchronous work with a still-running async job
        /// (e.g. shared <c>Net</c> or buffers).
        /// <see cref="TryGetLatestResult"/> still reflects the last completed work when a submission is skipped.
        /// </remarks>
        /// <param name="inputs">Input batch (not null, non-empty).</param>
        /// <param name="syncWork">Synchronous work (not null when <see cref="UseAsyncWork"/> is false).</param>
        /// <param name="asyncWork">Asynchronous work (not null when <see cref="UseAsyncWork"/> is true);
        /// no <see cref="CancellationToken"/> parameter—use <see cref="InFlightAsyncWorkCancellationToken"/>.</param>
        public void SubmitWork(
            TInput[] inputs,
            Func<TInput[], TResult[]> syncWork,
            Func<TInput[], Task<TResult[]>> asyncWork)
        {
            ThrowIfDisposed();
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));
            if (inputs.Length == 0) throw new ArgumentException("At least one input is required.", nameof(inputs));
            if (syncWork == null) throw new ArgumentNullException(nameof(syncWork));
            if (asyncWork == null) throw new ArgumentNullException(nameof(asyncWork));

            lock (_inFlightWorkTaskLock)
            {
                if (_inFlightWorkTask != null && !_inFlightWorkTask.IsCompleted)
                    return;
            }

            if (_useAsyncWork)
            {
                EnsureAsyncInputBuffer(inputs.Length);
                CopyInputsForAsync(inputs, _asyncInputBuffer);

                CancellationToken asyncWorkToken;
                lock (_ctsLock)
                {
                    if (_localCts != null && _localCts.IsCancellationRequested)
                    {
                        _linkedCts?.Dispose();
                        _localCts.Dispose();
                        _localCts = new CancellationTokenSource();
                        _linkedCts = CancellationTokenSource.CreateLinkedTokenSource(_asyncWorkCancellationToken, _localCts.Token);
                    }
                    if (_linkedCts == null)
                    {
                        if (_localCts != null)
                        {
                            _localCts.Dispose();
                            _localCts = null;
                        }
                        InitializeLocalAndLinkedCts();
                    }
                    asyncWorkToken = _linkedCts.Token;
                }

                Task runTask = RunAsyncWork(asyncWork, _asyncInputBuffer, asyncWorkToken);
                lock (_inFlightWorkTaskLock)
                {
                    _inFlightWorkTask = runTask;
                }

                if (runTask.IsCompleted)
                {
                    lock (_inFlightWorkTaskLock)
                    {
                        if (ReferenceEquals(_inFlightWorkTask, runTask))
                            _inFlightWorkTask = null;
                    }
                }
                else
                {
                    _ = runTask.ContinueWith(
                        t =>
                        {
                            lock (_inFlightWorkTaskLock)
                            {
                                if (ReferenceEquals(_inFlightWorkTask, t))
                                    _inFlightWorkTask = null;
                            }
                        },
                        TaskScheduler.Default);
                }
            }
            else
            {
                try
                {
                    TResult[] outputs = syncWork(inputs);
                    if (outputs != null)
                    {
                        ReplaceLatest(outputs);
                        RaiseWorkCompleted(WorkCompletion<TResult>.ForSucceeded(_latestResults));
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"{GetType().Name} (sync) error: {ex.Message}");
                    RaiseWorkCompleted(WorkCompletion<TResult>.ForFaulted(ex));
                }
            }
        }

        /// <summary>
        /// Tries to return the latest successful result array. Deriving types can tighten validation
        /// via <see cref="ValidateLatestForTryGet"/>.
        /// </summary>
        /// <remarks>
        /// Disposal of the latest result buffer (including each element when <see cref="IDisposable"/>) is this
        /// runner's responsibility: it is released when newer results replace it or when this instance is disposed.
        /// Do not call <c>Dispose</c> on <paramref name="results"/> or its elements; the same instances remain
        /// owned by the runner until then.
        /// <paramref name="results"/> is the runner's live buffer: mutating an element in place changes the stored
        /// result. If you need to modify it, clone (or copy) the element (or the array) first, then work on the copy.
        /// </remarks>
        public bool TryGetLatestResult(out TResult[] results)
        {
            ThrowIfDisposed();
            results = null;
            if (_latestResults == null || _latestResults.Length == 0) return false;
            if (!ValidateLatestForTryGet(_latestResults)) return false;
            results = _latestResults;
            return true;
        }

        /// <summary>
        /// Optional extra checks for <see cref="TryGetLatestResult"/>. The default is <see langword="true"/>.
        /// </summary>
        protected virtual bool ValidateLatestForTryGet(TResult[] latest) => true;

        /// <summary>
        /// Snapshots <paramref name="source"/> into <paramref name="destination"/> (async path only).
        /// The arrays are the same length; buffer allocation is handled by
        /// <see cref="EnsureAsyncInputBuffer"/> / overrides.
        /// </summary>
        protected abstract void CopyInputsForAsync(TInput[] source, TInput[] destination);

        /// <summary>
        /// Ensures the internal buffer used for async input snapshots has length
        /// <paramref name="requiredLength"/>.
        /// </summary>
        protected virtual void EnsureAsyncInputBuffer(int requiredLength)
        {
            if (requiredLength < 0) throw new ArgumentOutOfRangeException(nameof(requiredLength));
            if (_asyncInputBuffer == null || _asyncInputBuffer.Length != requiredLength)
            {
                ReleaseAsyncInputBuffers();
                _asyncInputBuffer = new TInput[requiredLength];
            }
        }

        public void Cancel()
        {
            ThrowIfDisposed();
            CancelInFlightThroughLocalToken();
        }

        /// <summary>
        /// Cancels in-flight work, clears the task reference without awaiting, and releases latest
        /// and async buffers. Does not call <c>disposeAsyncAfterWorkTask</c>.
        /// </summary>
        public void Dispose()
        {
            lock (_workCompletedGate)
            {
                if (_disposed) return;
                _disposed = true;
                _workCompletedDeliveryCts?.Cancel();
            }

            CancelInFlightThroughLocalToken();
            lock (_inFlightWorkTaskLock)
            {
                _inFlightWorkTask = null;
            }
            ReleaseAllResourcesExceptDisposeCallback();
        }

        /// <summary>
        /// Cancels, awaits the in-flight <see cref="Task"/>, then awaits the optional
        /// <c>disposeAsyncAfterWorkTask</c> callback, then releases state.
        /// </summary>
        public async ValueTask DisposeAsync()
        {
            lock (_workCompletedGate)
            {
                if (_disposed) return;
                _disposed = true;
                _workCompletedDeliveryCts?.Cancel();
            }

            CancelInFlightThroughLocalToken();

            Task task;
            lock (_inFlightWorkTaskLock)
            {
                task = _inFlightWorkTask;
                _inFlightWorkTask = null;
            }
            if (task != null && !task.IsCompleted)
            {
                try
                {
                    await task;
                    Debug.Log($"{GetType().Name} in-flight work task completed (DisposeAsync)");
                }
                catch (OperationCanceledException)
                {
                    Debug.Log($"{GetType().Name} in-flight work task canceled (DisposeAsync)");
                }
                catch (Exception ex)
                {
                    Debug.LogException(ex);
                }
            }

            if (_disposeAsyncAfterWorkTask != null)
            {
                try
                {
                    await _disposeAsyncAfterWorkTask();
                }
                catch (OperationCanceledException)
                {
                    Debug.Log($"{GetType().Name} dispose async hook canceled (DisposeAsync)");
                }
                catch (Exception ex)
                {
                    Debug.LogException(ex);
                }
            }

            ReleaseAllResourcesExceptDisposeCallback();
        }

        private async Task RunAsyncWork(
            Func<TInput[], Task<TResult[]>> asyncWork,
            TInput[] buffer,
            CancellationToken asyncWorkToken)
        {
            _inFlightAsyncWorkCancellationToken = asyncWorkToken;
            try
            {
                TResult[] outputs = await asyncWork(buffer).ConfigureAwait(true);
                if (outputs != null)
                {
                    ReplaceLatest(outputs);
                    RaiseWorkCompleted(WorkCompletion<TResult>.ForSucceeded(_latestResults));
                }
            }
            catch (OperationCanceledException ex)
            {
                Debug.Log($"{GetType().Name} async work canceled: {ex.Message}");
                RaiseWorkCompleted(WorkCompletion<TResult>.ForCanceled(ex));
            }
            catch (Exception ex)
            {
                Debug.LogError($"{GetType().Name} async error: {ex}");
                RaiseWorkCompleted(WorkCompletion<TResult>.ForFaulted(ex));
            }
            finally
            {
                _inFlightAsyncWorkCancellationToken = default;
            }
        }

        private void ReplaceLatest(TResult[] newResults)
        {
            if (newResults == null) return;
            ReleaseLatestResults();
            _latestResults = newResults;
        }

        private void ReleaseLatestResults()
        {
            if (_latestResults == null) return;
            for (int i = 0; i < _latestResults.Length; i++)
                ReleaseSingleResult(_latestResults[i]);
            _latestResults = null;
        }

        /// <summary>
        /// Disposes a single result element. Default: <c>IDisposable</c> when implemented.
        /// </summary>
        protected virtual void ReleaseSingleResult(TResult result)
        {
            if (result is IDisposable d)
                d.Dispose();
        }

        private void RaiseWorkCompleted(WorkCompletion<TResult> completion)
        {
            Action<WorkCompletion<TResult>> handler;
            lock (_workCompletedGate)
            {
                if (_disposed) return;
                if (_workCompletedDeliveryCts == null) return;
                if (_workCompletedDeliveryCts.Token.IsCancellationRequested) return;
                handler = WorkCompleted;
            }

            if (handler == null) return;
            try
            {
                handler(completion);
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
            }
        }

        private void ReleaseAllResourcesExceptDisposeCallback()
        {
            lock (_workCompletedGate)
            {
                _workCompletedDeliveryCts?.Dispose();
                _workCompletedDeliveryCts = null;
            }

            ReleaseLatestResults();
            ReleaseAsyncInputBuffers();
            lock (_ctsLock)
            {
                _linkedCts?.Dispose();
                _linkedCts = null;
                _localCts?.Dispose();
                _localCts = null;
            }
        }

        /// <summary>
        /// Releases per-element resources held in <see cref="EnsureAsyncInputBuffer"/> (e.g. OpenCV <c>Mat</c> slots in a derived type).
        /// </summary>
        protected virtual void ReleaseAsyncInputBuffers()
        {
            if (_asyncInputBuffer == null) return;
            for (int i = 0; i < _asyncInputBuffer.Length; i++)
            {
                if (_asyncInputBuffer[i] is IDisposable d)
                    d.Dispose();
                _asyncInputBuffer[i] = null;
            }
            _asyncInputBuffer = null;
        }

        private void CancelInFlightThroughLocalToken()
        {
            lock (_ctsLock)
            {
                if (_localCts != null) _localCts.Cancel();
            }
        }

        private void InitializeLocalAndLinkedCts()
        {
            _localCts = new CancellationTokenSource();
            _linkedCts = CancellationTokenSource.CreateLinkedTokenSource(_asyncWorkCancellationToken, _localCts.Token);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(GetType().Name);
        }
    }
}
