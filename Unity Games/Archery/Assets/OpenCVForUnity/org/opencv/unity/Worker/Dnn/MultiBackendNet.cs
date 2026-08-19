#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
#if !UNITY_WSA_10_0
using OpenCVForUnity.DnnModule;
#endif
using UnityEngine;
using UnityEngine.Rendering;
#if OPENCV_SENTIS_AVAILABLE
using Unity.InferenceEngine;
using SentisModel = Unity.InferenceEngine.Model;
using SentisWorker = Unity.InferenceEngine.Worker;
#endif

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule
{
    /// <summary>
    /// Subset of OpenCV <see cref="Net"/> for inference with either OpenCV DNN or Sentis, with deferred
    /// initialization after <see cref="MultiBackendDnn.readNet"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Initialization order (required):</b>
    /// <see cref="MultiBackendDnn.readNet"/> → <see cref="setPreferableBackend"/> → <see cref="setPreferableTarget"/> →
    /// <see cref="getUnconnectedOutLayersNames"/> (optional cache) → <see cref="setInput"/> / <see cref="forward"/>.
    /// Do not call <see cref="getUnconnectedOutLayersNames"/>, <see cref="setInput"/>, or <see cref="forward"/> before
    /// both <see cref="setPreferableBackend"/> and <see cref="setPreferableTarget"/>; doing so throws.
    /// </para>
    /// <para>
    /// <b>Deferred loading:</b> the factory does not load OpenCV <see cref="Net"/> or build a Sentis
    /// <c>Worker</c> until the backend and target are set. The Sentis path uses the model’s static input
    /// <c>TensorShape</c> from <c>ModelLoader</c> metadata (dynamic input shapes are not supported).
    /// </para>
    /// <para>
    /// <b>Output layer names:</b> <see cref="getUnconnectedOutLayersNames"/> returns names sorted with
    /// <see cref="StringComparer.Ordinal"/> so OpenCV DNN and Sentis call sites share one ordering. Pass the same list to
    /// <see cref="forward"/> and <see cref="forwardTaskAsync"/> as <c>outBlobNames</c>.
    /// </para>
    /// <para>
    /// <b>Backend id:</b> <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/> and the equivalent alias
    /// <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_INFERENCE_ENGINE"/> (same integer discriminator) select the Sentis path only.
    /// Neither is OpenCV’s Intel backend id (<see cref="Dnn.DNN_BACKEND_INFERENCE_ENGINE"/>); those constants differ in value and meaning.
    /// </para>
    /// <para>
    /// <b>OpenCV mode</b> (<see cref="forward"/>): the wrapper disposes <see cref="Mat"/> instances it appended on the
    /// previous <see cref="forward"/> when the next <see cref="forward"/> runs, and in <see cref="Dispose"/>. Do not
    /// dispose those outputs in application code. Do not pre-fill <c>outputBlobs</c> with foreign <see cref="Mat"/> and
    /// expect the wrapper to own them; only outputs produced by the internal OpenCV <see cref="Net.forward"/> call are managed.
    /// </para>
    /// <para>
    /// <b>Sentis mode</b>: internal output <see cref="Mat"/> buffers are reused; the same instances are
    /// appended each time to the provided list (after <see cref="List{T}.Clear"/> inside <see cref="forward"/> / <see cref="forwardTaskAsync"/>).
    /// </para>
    /// </remarks>
    public sealed class MultiBackendNet : IDisposable
    {
        static readonly StringComparer LayerNameComparer = StringComparer.Ordinal;

        readonly string _modelPath;

        bool _disposed;
        bool _backendSet;
        bool _targetSet;
        int _preferredBackend;
        int _preferredTarget;
        bool _sentisPath;

#if !UNITY_WSA_10_0
        Net _opencvNet;
        List<Mat> _ownedOpenCvForwardMats;
#endif
        List<string> _sortedUnconnectedOutLayerNames;

#if OPENCV_SENTIS_AVAILABLE
        SentisModel _sentisModel;
        SentisWorker _sentisWorker;
        Tensor<float> _sentisInputTensor;
        float[] _sentisUploadScratch;
        Mat[] _sentisOutputMats;
#endif

        internal MultiBackendNet(string modelPath)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("Model path is null or empty.", nameof(modelPath));
            _modelPath = modelPath;
        }

        /// <summary>
        /// True when <see cref="setPreferableBackend"/> was called with <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/> or
        /// <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_INFERENCE_ENGINE"/> (same value; Sentis path).
        /// </summary>
        /// <exception cref="InvalidOperationException">If <see cref="setPreferableBackend"/> has not been called yet.</exception>
        public bool UsesSentis
        {
            get
            {
                ThrowIfDisposed();
                if (!_backendSet)
                    throw new InvalidOperationException("Call setPreferableBackend first.");
                return _sentisPath;
            }
        }

        /// <summary>Last <see cref="setPreferableBackend"/> argument: an OpenCV <c>DNN_BACKEND_*</c> value, <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, or the equivalent alias <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_INFERENCE_ENGINE"/>.</summary>
        /// <exception cref="InvalidOperationException">If <see cref="setPreferableBackend"/> has not been called yet.</exception>
        public int PreferredBackend
        {
            get
            {
                ThrowIfDisposed();
                if (!_backendSet)
                    throw new InvalidOperationException("Call setPreferableBackend first.");
                return _preferredBackend;
            }
        }

        /// <summary>Last <see cref="setPreferableTarget"/> argument: OpenCV <c>DNN_TARGET_*</c> or, for Sentis, an <see cref="int"/> cast from the Sentis runtime <c>BackendType</c> enum (API namespace <c>Unity.InferenceEngine</c>).</summary>
        /// <exception cref="InvalidOperationException">If <see cref="setPreferableTarget"/> has not been called yet.</exception>
        public int PreferredTarget
        {
            get
            {
                ThrowIfDisposed();
                if (!_targetSet)
                    throw new InvalidOperationException("Call setPreferableTarget first.");
                return _preferredTarget;
            }
        }

        /// <summary>Commits the DNN backend; triggers lazy creation of the OpenCV <see cref="Net"/> (non-Sentis) or prepares the Sentis path.</summary>
        /// <remarks>Must be called before <see cref="setPreferableTarget"/>, <see cref="getUnconnectedOutLayersNames"/>, and inference. You cannot switch between OpenCV DNN and Sentis on the same instance after the first call.</remarks>
        /// <param name="backendId">OpenCV <c>DNN_BACKEND_*</c>, <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, or the equivalent alias <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_INFERENCE_ENGINE"/> (same value as <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>). Sentis is <b>not</b> OpenCV/Intel <see cref="Dnn.DNN_BACKEND_INFERENCE_ENGINE"/> (different id).</param>
        public void setPreferableBackend(int backendId)
        {
            ThrowIfDisposed();
            if (backendId == MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS)
            {
#if !OPENCV_SENTIS_AVAILABLE
                throw new NotSupportedException(
                    "DNN_BACKEND_UNITY_SENTIS requires Sentis (com.unity.ai.inference) 2.6.1 or newer in the project and the OPENCV_SENTIS_AVAILABLE define.");
#else
                if (_backendSet && !_sentisPath)
                    throw new InvalidOperationException("Cannot switch this MultiBackendNet from OpenCV to Sentis after initialization.");
                ReleaseOpenCv();
                _sentisPath = true;
                _preferredBackend = backendId;
                _backendSet = true;
                InvalidateMetadataCache();
                TryInitializeSentis();
#endif
            }
            else
            {
#if OPENCV_SENTIS_AVAILABLE
                if (_backendSet && _sentisPath)
                    throw new InvalidOperationException("Cannot switch this MultiBackendNet from Sentis to OpenCV after initialization.");
                ReleaseSentis();
#endif
#if !UNITY_WSA_10_0
                _sentisPath = false;
                _preferredBackend = backendId;
                _backendSet = true;
                EnsureOpenCvNet();
                _opencvNet.setPreferableBackend(backendId);
                InvalidateMetadataCache();
                if (_targetSet)
                    _opencvNet.setPreferableTarget(_preferredTarget);
#else
                throw new NotSupportedException(
                    "OpenCV DnnModule is not included for Universal Windows Platform (UNITY_WSA_10_0). Use DNN_BACKEND_UNITY_SENTIS with Sentis (com.unity.ai.inference) and OPENCV_SENTIS_AVAILABLE.");
#endif
            }
        }

        /// <summary>Commits the DNN target (CPU, GPU, etc.); for Sentis, together with <see cref="setPreferableBackend"/>, finishes lazy allocation of the Sentis <c>Worker</c> and I/O buffers.</summary>
        /// <remarks>Requires <see cref="setPreferableBackend"/> to have been called first.</remarks>
        /// <param name="targetId">OpenCV <c>DNN_TARGET_*</c> or a Sentis <c>BackendType</c> value as <see cref="int"/>.</param>
        public void setPreferableTarget(int targetId)
        {
            ThrowIfDisposed();
            if (!_backendSet)
                throw new InvalidOperationException("Call setPreferableBackend before setPreferableTarget.");
            _preferredTarget = targetId;
            _targetSet = true;
            if (_sentisPath)
            {
#if OPENCV_SENTIS_AVAILABLE
                TryInitializeSentis();
#endif
            }
            else
            {
#if !UNITY_WSA_10_0
                EnsureOpenCvNet();
                _opencvNet.setPreferableTarget(targetId);
                InvalidateMetadataCache();
#else
                throw new NotSupportedException(
                    "OpenCV DnnModule is not included for Universal Windows Platform (UNITY_WSA_10_0).");
#endif
            }
        }

        /// <summary>
        /// Returns the model’s unconnected output layer (tensor) names, sorted with <see cref="StringComparer.Ordinal"/>
        /// so OpenCV DNN and Sentis agree on a single order for <see cref="forward"/>.
        /// </summary>
        public List<string> getUnconnectedOutLayersNames()
        {
            ThrowIfDisposed();
            EnsureBackendAndTargetConfigured();
            if (_sortedUnconnectedOutLayerNames != null)
                return new List<string>(_sortedUnconnectedOutLayerNames);

#if !UNITY_WSA_10_0 || OPENCV_SENTIS_AVAILABLE
            List<string> raw;
#endif
#if OPENCV_SENTIS_AVAILABLE
            if (_sentisPath)
            {
                EnsureSentisInitialized();
                raw = GetUnconnectedOutLayersNamesFromSentisModel(_sentisModel);
            }
            else
#endif
#if !UNITY_WSA_10_0
            {
                raw = _opencvNet.getUnconnectedOutLayersNames();
            }
#else
            {
                throw new NotSupportedException(
                    "OpenCV DnnModule is not included for Universal Windows Platform (UNITY_WSA_10_0).");
            }
#endif

#if !UNITY_WSA_10_0 || OPENCV_SENTIS_AVAILABLE
            _sortedUnconnectedOutLayerNames = SortLayerNamesCopy(raw);
            return new List<string>(_sortedUnconnectedOutLayerNames);
#endif
        }

        /// <summary>Sets the network input. OpenCV: delegates to <see cref="Net.setInput(Mat)"/>. Sentis: copies <paramref name="blob"/> into the preallocated input <c>Tensor&lt;float&gt;</c> (element count must match the model; dynamic shapes are not supported).</summary>
        public void setInput(Mat blob)
        {
            ThrowIfDisposed();
            EnsureBackendAndTargetConfigured();
            if (blob != null)
                blob.ThrowIfDisposed();
            if (blob == null)
                throw new ArgumentNullException(nameof(blob));
#if OPENCV_SENTIS_AVAILABLE
            if (_sentisPath)
            {
                EnsureSentisInitialized();
                ValidateBlobMatchesInputTensor(blob);
                UploadMatToTensorFloat(blob, _sentisInputTensor, _sentisUploadScratch);
                return;
            }
#endif
#if !UNITY_WSA_10_0
            _opencvNet.setInput(blob);
#else
            throw new NotSupportedException(
                "OpenCV DnnModule is not included for Universal Windows Platform (UNITY_WSA_10_0).");
#endif
        }

        /// <summary>Runs a forward pass. <paramref name="outBlobNames"/> must equal the list from <see cref="getUnconnectedOutLayersNames"/> in order and count.</summary>
        /// <param name="outputBlobs">Cleared and filled with outputs; see class remarks for OpenCV vs Sentis ownership of <see cref="Mat"/> instances.</param>
        /// <param name="outBlobNames">Output names in <see cref="StringComparer.Ordinal"/> order, matching <see cref="getUnconnectedOutLayersNames"/>.</param>
        public void forward(List<Mat> outputBlobs, List<string> outBlobNames)
        {
            ThrowIfDisposed();
            EnsureBackendAndTargetConfigured();
            if (outputBlobs == null)
                throw new ArgumentNullException(nameof(outputBlobs));
            if (outBlobNames == null)
                throw new ArgumentNullException(nameof(outBlobNames));
            EnsureSortedOutNamesCached();
            AssertOutBlobNamesMatchCached(outBlobNames);

#if OPENCV_SENTIS_AVAILABLE
            if (_sentisPath)
            {
                EnsureSentisInitialized();
                outputBlobs.Clear();
                RunSentisForwardIntoListMat(_sentisWorker, _sentisInputTensor, _sortedUnconnectedOutLayerNames, _sentisOutputMats, outputBlobs);
                return;
            }
#endif
#if !UNITY_WSA_10_0
            DisposeOwnedOpenCvForwardMats();
            _opencvNet.forward(outputBlobs, outBlobNames);
            RegisterOwnedOpenCvOutputs(outputBlobs);
#else
            throw new NotSupportedException(
                "OpenCV DnnModule is not included for Universal Windows Platform (UNITY_WSA_10_0).");
#endif
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>Asynchronous forward pass for the Sentis path only, using <c>ReadbackAndCloneAsync</c> for outputs.</summary>
        /// <remarks>Throws if <see cref="UsesSentis"/> is false. <paramref name="outBlobNames"/> must match <see cref="getUnconnectedOutLayersNames"/> (same as <see cref="forward"/>).</remarks>
        public async Task forwardTaskAsync(List<Mat> outputBlobs, List<string> outBlobNames, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            EnsureBackendAndTargetConfigured();
            if (!_sentisPath)
                throw new InvalidOperationException("forwardTaskAsync is only supported when UsesSentis is true.");
            if (outputBlobs == null)
                throw new ArgumentNullException(nameof(outputBlobs));
            if (outBlobNames == null)
                throw new ArgumentNullException(nameof(outBlobNames));
            EnsureSentisInitialized();
            EnsureSortedOutNamesCached();
            AssertOutBlobNamesMatchCached(outBlobNames);

            outputBlobs.Clear();
            await RunSentisForwardIntoListMatAsync(
                _sentisWorker,
                _sentisInputTensor,
                _sortedUnconnectedOutLayerNames,
                _sentisOutputMats,
                outputBlobs,
                cancellationToken);
        }

        /// <summary>Asynchronous forward pass for the Sentis path only, using <c>ReadbackAndCloneAsync</c> for outputs.</summary>
        /// <remarks>
        /// <para><c>@deprecated</c> Use <see cref="forwardTaskAsync"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task"/>.</para>
        /// <para>Throws if <see cref="UsesSentis"/> is false. <paramref name="outBlobNames"/> must match <see cref="getUnconnectedOutLayersNames"/> (same as <see cref="forward"/>).</para>
        /// </remarks>
        [Obsolete("Use forwardTaskAsync(). forwardAsync() will return Awaitable in a future version.")]
        public Task forwardAsync(List<Mat> outputBlobs, List<string> outBlobNames, CancellationToken cancellationToken = default) =>
            forwardTaskAsync(outputBlobs, outBlobNames, cancellationToken);
#endif

        /// <summary>Releases the OpenCV <see cref="Net"/>, Sentis resources, and any OpenCV <see cref="Mat"/> instances owned from the last <see cref="forward"/> call.</summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            DisposeOwnedOpenCvForwardMats();
            ReleaseOpenCv();
#if OPENCV_SENTIS_AVAILABLE
            ReleaseSentis();
#endif
        }

        void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MultiBackendNet));
        }

        void EnsureBackendAndTargetConfigured()
        {
            if (!_backendSet || !_targetSet)
                throw new InvalidOperationException("Call setPreferableBackend and setPreferableTarget before this operation.");
        }

        void EnsureOpenCvNet()
        {
#if !UNITY_WSA_10_0
            if (_opencvNet != null)
                return;
            _opencvNet = Dnn.readNet(_modelPath);
#endif
        }

        void ReleaseOpenCv()
        {
#if !UNITY_WSA_10_0
            DisposeOwnedOpenCvForwardMats();
            _opencvNet?.Dispose();
            _opencvNet = null;
#endif
        }

        void InvalidateMetadataCache()
        {
            _sortedUnconnectedOutLayerNames = null;
        }

        void EnsureSortedOutNamesCached()
        {
            if (_sortedUnconnectedOutLayerNames != null)
                return;
            _ = getUnconnectedOutLayersNames();
        }

        void AssertOutBlobNamesMatchCached(List<string> outBlobNames)
        {
            if (_sortedUnconnectedOutLayerNames == null)
                throw new InvalidOperationException("Internal error: sorted output names not cached.");
            if (outBlobNames.Count != _sortedUnconnectedOutLayerNames.Count)
                throw new ArgumentException("outBlobNames count does not match getUnconnectedOutLayersNames().", nameof(outBlobNames));
            for (int i = 0; i < outBlobNames.Count; i++)
            {
                if (!LayerNameComparer.Equals(outBlobNames[i], _sortedUnconnectedOutLayerNames[i]))
                    throw new ArgumentException(
                        "outBlobNames must match the list returned by getUnconnectedOutLayersNames() in order (StringComparer.Ordinal).",
                        nameof(outBlobNames));
            }
        }

        void DisposeOwnedOpenCvForwardMats()
        {
#if !UNITY_WSA_10_0
            if (_ownedOpenCvForwardMats == null)
                return;
            for (int i = 0; i < _ownedOpenCvForwardMats.Count; i++)
                _ownedOpenCvForwardMats[i]?.Dispose();
            _ownedOpenCvForwardMats.Clear();
#endif
        }

        void RegisterOwnedOpenCvOutputs(List<Mat> outputBlobs)
        {
#if !UNITY_WSA_10_0
            _ownedOpenCvForwardMats ??= new List<Mat>();
            _ownedOpenCvForwardMats.Clear();
            _ownedOpenCvForwardMats.AddRange(outputBlobs);
#endif
        }

#if OPENCV_SENTIS_AVAILABLE
        void TryInitializeSentis()
        {
            if (!_sentisPath || !_backendSet || !_targetSet)
                return;
            if (_sentisWorker != null)
                return;

            _sentisModel = ModelLoader.Load(_modelPath);
            if (_sentisModel.inputs == null || _sentisModel.inputs.Count == 0)
                throw new InvalidOperationException("Model has no inputs; cannot create input tensor.");
            DynamicTensorShape dynamicInShape = _sentisModel.inputs[0].shape;
            if (!dynamicInShape.IsStatic())
                throw new NotSupportedException("Dynamic or non-concrete input shapes are not supported by MultiBackendNet.");
            TensorShape inShape = dynamicInShape.ToTensorShape();

            var backendType = (BackendType)_preferredTarget;
            _sentisWorker = new SentisWorker(_sentisModel, backendType);
            _sentisInputTensor = new Tensor<float>(inShape);
            _sentisUploadScratch = new float[TensorShapeElementCount(inShape)];

            var rawNames = GetUnconnectedOutLayersNamesFromSentisModel(_sentisModel);
            _sortedUnconnectedOutLayerNames = SortLayerNamesCopy(rawNames);
            _sentisOutputMats = CreateMatBuffersForSentisOutputs(_sentisWorker, _sortedUnconnectedOutLayerNames, _sentisInputTensor);
        }

        void EnsureSentisInitialized()
        {
            if (_sentisWorker == null)
                throw new InvalidOperationException("Sentis worker is not initialized; ensure setPreferableBackend(SENTIS) and setPreferableTarget were called.");
        }

        void ReleaseSentis()
        {
            if (_sentisOutputMats != null)
            {
                for (int i = 0; i < _sentisOutputMats.Length; i++)
                    _sentisOutputMats[i]?.Dispose();
                _sentisOutputMats = null;
            }
            _sentisInputTensor?.Dispose();
            _sentisInputTensor = null;
            _sentisUploadScratch = null;
            _sentisWorker?.Dispose();
            _sentisWorker = null;
            _sentisModel = null;
        }

        static int TensorShapeElementCount(TensorShape shape)
        {
            int p = 1;
            for (int i = 0; i < shape.rank; i++)
                p *= shape[i];
            return p;
        }

        void ValidateBlobMatchesInputTensor(Mat blob)
        {
            long total = blob.total();
            if (total != (long)_sentisUploadScratch.Length)
                throw new ArgumentException(
                    $"Input Mat element count ({total}) does not match model input tensor ({_sentisUploadScratch.Length} elements).",
                    nameof(blob));
        }
#endif

        static List<string> SortLayerNamesCopy(List<string> names)
        {
            var copy = new List<string>(names);
            copy.Sort(LayerNameComparer);
            return copy;
        }

#if OPENCV_SENTIS_AVAILABLE
        internal static void UploadMatToTensorFloat(Mat blob, Tensor<float> tensor, float[] scratch)
        {
            blob.AsSpan<float>().CopyTo(scratch);
            tensor.Upload(scratch);
        }

        internal static void TensorFloatToMatFromReadback(Tensor<float> cpuTensor, Mat destination)
        {
            ReadOnlySpan<float> src = cpuTensor.AsReadOnlySpan();
            src.CopyTo(destination.AsSpan<float>());
        }

        internal static List<string> GetUnconnectedOutLayersNamesFromSentisModel(SentisModel model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (model.outputs == null || model.outputs.Count == 0)
                return new List<string>();

            var list = new List<string>(model.outputs.Count);
            foreach (var output in model.outputs)
                list.Add(output.name);
            return list;
        }

        internal static Mat[] CreateMatBuffersForSentisOutputs(SentisWorker worker, List<string> outLayersNames, Tensor<float> input)
        {
            if (worker == null)
                throw new ArgumentNullException(nameof(worker));
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (outLayersNames == null)
                throw new ArgumentNullException(nameof(outLayersNames));
            if (outLayersNames.Count == 0)
                throw new ArgumentException("outLayersNames must contain at least one output name.", nameof(outLayersNames));

            worker.Schedule(input);
            int n = outLayersNames.Count;
            var mats = new Mat[n];
            for (int i = 0; i < n; i++)
            {
                var t = (Tensor<float>)worker.PeekOutput(outLayersNames[i]);
                TensorShape shape = t.shape;
                int rank = shape.rank;
                var dims = new int[rank];
                for (int j = 0; j < rank; j++)
                    dims[j] = shape[j];
                mats[i] = new Mat(dims, CvType.CV_32FC1);
            }

            return mats;
        }

        internal static void RunSentisForwardIntoListMat(
            SentisWorker worker,
            Tensor<float> input,
            List<string> outLayersNames,
            Mat[] outputMats,
            List<Mat> destinationList)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.WebGPU)
                Debug.LogWarning(nameof(MultiBackendNet) + ": On WebGL with WebGPU, Sentis may be unavailable due to ReadbackAndClone limitations.");
#endif
            worker.Schedule(input);
            for (int i = 0; i < outLayersNames.Count; i++)
            {
                var outputTensor = (Tensor<float>)worker.PeekOutput(outLayersNames[i]);
                using (Tensor<float> cpu = (Tensor<float>)outputTensor.ReadbackAndClone())
                {
                    TensorFloatToMatFromReadback(cpu, outputMats[i]);
                    destinationList.Add(outputMats[i]);
                }
            }
        }

        internal static async Task RunSentisForwardIntoListMatAsync(
            SentisWorker worker,
            Tensor<float> input,
            List<string> outLayersNames,
            Mat[] outputMats,
            List<Mat> destinationList,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            worker.Schedule(input);
            cancellationToken.ThrowIfCancellationRequested();
            int n = outLayersNames.Count;
            var readbackAwaitables = new Awaitable<Tensor<float>>[n];
            for (int i = 0; i < n; i++)
            {
                var outputTensor = (Tensor<float>)worker.PeekOutput(outLayersNames[i]);
                readbackAwaitables[i] = outputTensor.ReadbackAndCloneAsync();
            }

            for (int i = 0; i < n; i++)
            {
                Tensor<float> cpuTensor = await readbackAwaitables[i];
                try
                {
                    TensorFloatToMatFromReadback(cpuTensor, outputMats[i]);
                    destinationList.Add(outputMats[i]);
                }
                finally
                {
                    cpuTensor?.Dispose();
                }
            }

            cancellationToken.ThrowIfCancellationRequested();
        }
#endif
    }
}

#endif
