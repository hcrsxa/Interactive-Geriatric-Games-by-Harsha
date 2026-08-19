using System;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;

namespace OpenCVForUnity.UnityIntegration.Runner
{
    /// <summary>
    /// <see cref="SingleFlightSyncAsyncRunner{TInput,TResult}"/> specialized for OpenCV <see cref="Mat"/> inputs and
    /// outputs. The generic type parameters of the base are both <see cref="Mat"/>. Async input snapshots use
    /// <c>copyTo</c> into an internal buffer before <c>asyncWork</c> runs.
    /// </summary>
    public sealed class MatSingleFlightSyncAsyncRunner : SingleFlightSyncAsyncRunner<Mat, Mat>
    {
        private readonly Mat[] _singleInputScratch = new Mat[1];

        /// <summary>
        /// Initializes a new runner. Only <see cref="SingleFlightSyncAsyncRunner{Mat,Mat}.DisposeAsync"/> runs
        /// <paramref name="disposeAsyncAfterWorkTask"/>; <see cref="SingleFlightSyncAsyncRunner{Mat,Mat}.Dispose"/> does not.
        /// </summary>
        /// <param name="useAsyncWork">Initial async (single in-flight) mode.</param>
        /// <param name="asyncWorkCancellationToken">Host lifetime token combined with the runner local token for async work.</param>
        /// <param name="disposeAsyncAfterWorkTask">Optional hook after the in-flight task (e.g. worker <c>WaitForCompletionTaskAsync</c>).</param>
        public MatSingleFlightSyncAsyncRunner(
            bool useAsyncWork,
            CancellationToken asyncWorkCancellationToken = default,
            Func<Task> disposeAsyncAfterWorkTask = null)
            : base(useAsyncWork, asyncWorkCancellationToken, disposeAsyncAfterWorkTask)
        {
        }

        /// <summary>
        /// Submits a single image by delegating to the <see cref="Mat"/>[] overload (length 1). Reuses an internal
        /// one-element scratch to avoid per-frame array allocation.
        /// </summary>
        public void SubmitWork(
            Mat input,
            Func<Mat, Mat[]> syncWork,
            Func<Mat, Task<Mat[]>> asyncWork)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (syncWork == null) throw new ArgumentNullException(nameof(syncWork));
            if (asyncWork == null) throw new ArgumentNullException(nameof(asyncWork));

            _singleInputScratch[0] = input;
            SubmitWork(
                _singleInputScratch,
                ins => syncWork(ins[0]),
                ins => asyncWork(ins[0]));
        }

        /// <summary>
        /// Submits a single image; wraps <see cref="Mat"/> results as a one-element array for the base runner.
        /// </summary>
        public void SubmitWork(
            Mat input,
            Func<Mat, Mat> syncWork,
            Func<Mat, Task<Mat>> asyncWork)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (syncWork == null) throw new ArgumentNullException(nameof(syncWork));
            if (asyncWork == null) throw new ArgumentNullException(nameof(asyncWork));

            SubmitWork(
                input,
                m => new[] { syncWork(m) },
                async m => new[] { await asyncWork(m) });
        }

        /// <summary>
        /// Returns the single output <see cref="Mat"/> only when the latest buffer has length exactly one and
        /// that element is non-null. Whether the <see cref="Mat"/> is empty is for the caller to decide.
        /// If the latest results have two or more mats, returns <see langword="false"/>.
        /// </summary>
        /// <remarks>
        /// Disposal of the latest <see cref="Mat"/> (and the underlying buffer) is this runner's responsibility:
        /// it is released when newer results replace it or when this instance is disposed. Do not call
        /// <c>Dispose</c> on <paramref name="result"/>; the same instance remains owned by the runner until then.
        /// <paramref name="result"/> is the runner's live buffer: in-place <see cref="Mat"/> operations change the
        /// stored result. If you need to modify it, call <c>clone()</c> (or <c>copyTo</c> to your own
        /// <see cref="Mat"/>) first, then work on the copy.
        /// </remarks>
        public bool TryGetLatestResult(out Mat result)
        {
            result = null;
            if (!TryGetLatestResult(out Mat[] results))
                return false;
            if (results.Length != 1)
                return false;
            result = results[0];
            return true;
        }

        /// <inheritdoc />
        protected override void CopyInputsForAsync(Mat[] source, Mat[] destination)
        {
            for (int i = 0; i < source.Length; i++)
            {
                if (source[i] == null)
                    continue;

                if (destination[i] == null)
                    destination[i] = new Mat();

                source[i].copyTo(destination[i]);
            }
        }

        /// <summary>
        /// Rejects <paramref name="latest"/> when any element is <see langword="null"/>. Empty <see cref="Mat"/>
        /// values are not filtered here; callers use <c>empty()</c> or other checks as needed.
        /// </summary>
        protected override bool ValidateLatestForTryGet(Mat[] latest)
        {
            for (int i = 0; i < latest.Length; i++)
            {
                if (latest[i] == null)
                    return false;
            }

            return true;
        }
    }
}
