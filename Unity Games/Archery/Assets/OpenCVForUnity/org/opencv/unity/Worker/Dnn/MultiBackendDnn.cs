#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
#if !UNITY_WSA_10_0
using OpenCVForUnity.DnnModule;
#endif
#if OPENCV_SENTIS_AVAILABLE
using Unity.InferenceEngine;
#endif

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule
{
    /// <summary>
    /// Subset of OpenCV <see cref="OpenCVForUnity.DnnModule.Dnn"/> for loading models that may run on either
    /// OpenCV DNN or Sentis (<c>com.unity.ai.inference</c>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Backend id <see cref="DNN_BACKEND_UNITY_SENTIS"/> / <see cref="DNN_BACKEND_UNITY_INFERENCE_ENGINE"/></b> is reserved for Sentis.
    /// It is <b>not</b> the same value as OpenCV’s Intel OpenVINO backend constant
    /// <see cref="OpenCVForUnity.DnnModule.Dnn.DNN_BACKEND_INFERENCE_ENGINE"/> (different numeric id and vendor).
    /// </para>
    /// <para>
    /// <b>Call order</b> must follow classic DNN usage:
    /// <see cref="readNet"/> → <see cref="MultiBackendNet.setPreferableBackend"/> → <see cref="MultiBackendNet.setPreferableTarget"/> →
    /// <see cref="MultiBackendNet.getUnconnectedOutLayersNames"/> before inference
    /// (<see cref="MultiBackendNet.setInput"/> / <see cref="MultiBackendNet.forward"/>).
    /// Calling <see cref="MultiBackendNet.getUnconnectedOutLayersNames"/> before both <c>setPreferable*</c> calls will fail.
    /// </para>
    /// <para>
    /// <see cref="readNet"/> only stores the model path; it does not load an OpenCV <see cref="OpenCVForUnity.DnnModule.Net"/>
    /// or construct a Sentis <c>Worker</c> until the backend and target are set on <see cref="MultiBackendNet"/>
    /// (see that type for deferred initialization details).
    /// </para>
    /// </remarks>
    public static class MultiBackendDnn
    {
        /// <summary>
        /// Backend id for Sentis inference. Use with <c>com.unity.ai.inference</c> 2.6.1 or newer
        /// and the <c>OPENCV_SENTIS_AVAILABLE</c> script define. Not the OpenCV/Intel
        /// <see cref="OpenCVForUnity.DnnModule.Dnn.DNN_BACKEND_INFERENCE_ENGINE"/> id.
        /// </summary>
        public const int DNN_BACKEND_UNITY_SENTIS = 100;

        /// <summary>
        /// Alias of <see cref="DNN_BACKEND_UNITY_SENTIS"/> for the Sentis path. This is the same <b>discriminator
        /// value</b> as <see cref="DNN_BACKEND_UNITY_SENTIS"/>; it is unrelated to OpenCV’s
        /// <see cref="OpenCVForUnity.DnnModule.Dnn.DNN_BACKEND_INFERENCE_ENGINE"/> (Intel OpenVINO).
        /// </summary>
        public const int DNN_BACKEND_UNITY_INFERENCE_ENGINE = DNN_BACKEND_UNITY_SENTIS;

        /// <summary>
        /// Factory with the same signature as <see cref="OpenCVForUnity.DnnModule.Dnn.readNet(string)"/>; returns
        /// <see cref="MultiBackendNet"/>, which does <b>not</b> construct an OpenCV <see cref="OpenCVForUnity.DnnModule.Net"/>
        /// or a Sentis <c>Worker</c> at this point—initialization is deferred until
        /// <see cref="MultiBackendNet.setPreferableBackend"/> and <see cref="MultiBackendNet.setPreferableTarget"/> (see
        /// <see cref="MultiBackendNet"/> remarks).
        /// </summary>
        /// <param name="model">Filesystem path to the model (e.g. ONNX for OpenCV, serialized model for Sentis when that path is used).</param>
        public static MultiBackendNet readNet(string model)
        {
            return new MultiBackendNet(model);
        }

        /// <summary>
        /// Returns a short human-readable label for a DNN backend id (OpenCV <c>DNN_BACKEND_*</c> or
        /// <see cref="DNN_BACKEND_UNITY_SENTIS"/> / <see cref="DNN_BACKEND_UNITY_INFERENCE_ENGINE"/>; same Sentis discriminator), e.g. for UI overlays or logs.
        /// </summary>
        /// <param name="dnnBackend">Value from <see cref="DnnInferenceWorkerBase.DnnBackend"/> or equivalent.</param>
        public static string GetBackendDisplayString(int dnnBackend)
        {
            if (dnnBackend == DNN_BACKEND_UNITY_SENTIS)
                return "SENTIS";
#if !UNITY_WSA_10_0
            if (dnnBackend == Dnn.DNN_BACKEND_OPENCV)
                return "OPENCV";
            if (dnnBackend == Dnn.DNN_BACKEND_DEFAULT)
                return "DEFAULT";
            if (dnnBackend == Dnn.DNN_BACKEND_CUDA)
                return "CUDA";
#endif
            return dnnBackend.ToString();
        }

        /// <summary>
        /// Returns a short human-readable label for a target id. Maps OpenCV <c>DNN_TARGET_*</c> values first; if none
        /// match and <c>OPENCV_SENTIS_AVAILABLE</c> is set, maps a defined Sentis <c>BackendType</c> (API namespace <c>Unity.InferenceEngine</c>);
        /// otherwise returns the integer as a string. (OpenCV ids are checked first so small integer overlap with
        /// <c>BackendType</c> is resolved in favor of OpenCV labels when both apply.)
        /// </summary>
        /// <param name="dnnTarget">Value from <see cref="DnnInferenceWorkerBase.DnnTarget"/> or equivalent.</param>
        public static string GetTargetDisplayString(int dnnTarget)
        {
#if !UNITY_WSA_10_0
            if (dnnTarget == Dnn.DNN_TARGET_CPU)
                return "CPU";
            if (dnnTarget == Dnn.DNN_TARGET_CPU_FP16)
                return "CPU_FP16";
            if (dnnTarget == Dnn.DNN_TARGET_OPENCL)
                return "OPENCL";
            if (dnnTarget == Dnn.DNN_TARGET_OPENCL_FP16)
                return "OPENCL_FP16";
            if (dnnTarget == Dnn.DNN_TARGET_CUDA)
                return "CUDA";
            if (dnnTarget == Dnn.DNN_TARGET_CUDA_FP16)
                return "CUDA_FP16";
            if (dnnTarget == Dnn.DNN_TARGET_VULKAN)
                return "VULKAN";
#endif
#if OPENCV_SENTIS_AVAILABLE
            if (Enum.IsDefined(typeof(BackendType), dnnTarget))
                return ((BackendType)dnnTarget).ToString();
#endif
            return dnnTarget.ToString();
        }
    }
}

#endif
