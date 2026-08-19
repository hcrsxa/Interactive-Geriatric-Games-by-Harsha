#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityIntegration.Worker;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule
{
    /// <summary>
    /// Base class for <see cref="ProcessingWorkerBase"/> derivatives that record DNN runtime options
    /// (<see cref="DnnBackend"/> / <see cref="DnnTarget"/>), support Unity Inference Engine
    /// (<c>com.unity.ai.inference</c>) as an alternative to OpenCV DNN, and share the same backend
    /// discriminator constants as <see cref="MultiBackendDnn"/>.
    /// </summary>
    public abstract class DnnInferenceWorkerBase : ProcessingWorkerBase
    {
        private readonly int _dnnBackend;
        private readonly int _dnnTarget;

        /// <summary>
        /// OpenCV DNN <c>DNN_BACKEND_*</c> constant or <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/> supplied at construction.
        /// </summary>
        public int DnnBackend => _dnnBackend;

        /// <summary>
        /// OpenCV DNN <c>DNN_TARGET_*</c> constant, or when <see cref="DnnBackend"/> is <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>,
        /// an integer interpreted as Unity Inference Engine <c>BackendType</c>.
        /// </summary>
        public int DnnTarget => _dnnTarget;

        /// <summary>
        /// Initializes backend and target values exposed by <see cref="DnnBackend"/> and <see cref="DnnTarget"/>.
        /// </summary>
        /// <param name="dnnBackend">Inference backend discriminator.</param>
        /// <param name="dnnTarget">Inference target discriminator.</param>
        protected DnnInferenceWorkerBase(int dnnBackend, int dnnTarget)
        {
            _dnnBackend = dnnBackend;
            _dnnTarget = dnnTarget;
        }
    }
}

#endif
