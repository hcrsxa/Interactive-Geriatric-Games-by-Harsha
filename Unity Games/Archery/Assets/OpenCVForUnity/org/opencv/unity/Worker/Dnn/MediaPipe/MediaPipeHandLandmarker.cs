#if !UNITY_WSA_10_0
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Worker;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using OpenCVForUnity.UnityIntegration.Worker.Utils;
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe
{
    /// <summary>
    /// Processing worker that reproduces the hand landmarking graph logic of
    /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker
    /// on top of the OpenCV for Unity Dnn module.
    /// </summary>
    public class MediaPipeHandLandmarker : DnnInferenceWorkerBase
    {
        /// <summary>
        /// Execution modes compatible with the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker task.
        /// This enum corresponds to the task running mode that switches between
        /// per-image processing and stateful video processing.
        /// </summary>
        public enum MediaPipeHandRunningMode : byte
        {
            /// <summary>
            /// IMAGE mode.
            /// Runs hand detection and hand landmarking for each input image without
            /// reusing loopback tracking state from previous frames.
            /// </summary>
            IMAGE = 0,

            /// <summary>
            /// VIDEO mode.
            /// Assumes a frame sequence and reuses hand rectangles from the previous
            /// frame so the detector can be skipped on frames where tracking remains valid.
            /// </summary>
            VIDEO = 1,
        }

        public enum KeyPoint : byte
        {
            Wrist,
            Thumb1, Thumb2, Thumb3, Thumb4,
            Index1, Index2, Index3, Index4,
            Middle1, Middle2, Middle3, Middle4,
            Ring1, Ring2, Ring3, Ring4,
            Pinky1, Pinky2, Pinky3, Pinky4
        }

        const float kLandmarksNormalizeZ = 0.4f;

        /// <summary>
        /// Returns a signed scalar equivalent to the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>tensors_to_classification_calculator.cc</c> behavior with
        /// <c>binary_classification</c> and <c>top_k=1</c>.
        /// From the two scores <c>(s, 1-s)</c>, the larger score becomes the top-1 result.
        /// If label index 0 (Right) wins, the method returns positive <paramref name="rawScoreClass0"/>.
        /// If label index 1 (Left) wins, it returns <c>-(1-s)</c>.
        /// Ties prefer index 0 to match the first surviving element after the original partial sort behavior.
        /// </summary>
        /// <param name="rawScoreClass0">
        /// First model output element.
        /// Corresponds to label index 0 = Right in the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>HandLandmarksDetectorGraph</c> handedness classification output.
        /// </param>
        internal static float PackHandednessBinaryTop1(float rawScoreClass0)
        {
            float s0 = rawScoreClass0;
            float s1 = 1f - rawScoreClass0;
            if (s0 >= s1)
                return s0;
            return -s1;
        }

        /// <summary>
        /// Scale factor used when reflecting <c>|z|</c> from normalized landmarks
        /// in the circle radius of <see cref="DrawHandSkeleton"/> joint markers.
        /// Because z is normalized relative to the ROI width, adding
        /// <c>|z| * Min(w, h)</c> directly would make the radius almost always saturate,
        /// so the value is attenuated.
        /// </summary>
        const float kHandSkeletonCircleZPixelScale = 0.15f;

        /// <summary>
        /// Upper cap for <c>|z|</c> used in visualization radius computation,
        /// preventing outliers from producing oversized circles.
        /// </summary>
        const float kHandSkeletonCircleZAbsCap = 0.5f;

        /// <summary>
        /// Number of float elements per palm detection row:
        /// 4 bbox values + 14 keypoint values + 1 score value.
        /// </summary>
        const int PalmDetectionRowElementCount = 19;

        /// <summary>
        /// Rotation target angle in radians for the palm
        /// <c>DetectionsToRectsCalculator</c>.
        /// This matches <c>rotation_vector_target_angle_degrees: 90</c> from
        /// <c>palm_detection_detection_to_roi.pbtxt</c>, i.e. π/2.
        /// Because the proto field <c>rotation_vector_target_angle</c> is specified in radians,
        /// a raw value of <c>90</c> in <c>hand_detector_graph.cc</c> would mean 90 radians,
        /// which does not match 90 degrees.
        /// </summary>
        const float kDetectionPalmRotationTargetAngleRadians = Mathf.PI * 0.5f;

        /// <summary>BGRA visualization color without heap allocation. OpenCV uses BGR by default.</summary>
        private static readonly Vec4d VizWhiteBgra = new Vec4d(255, 255, 255, 255);
        private static readonly Vec4d VizRedBgra = new Vec4d(0, 0, 255, 255);
        private static readonly Vec4d VizBlueBgra = new Vec4d(255, 0, 0, 255);

        /// <summary>
        /// Connections for the 21 hand landmarks used for visualization.
        /// Each element is an index pair <c>(from, to)</c>.
        /// </summary>
        private static readonly (KeyPoint from, KeyPoint to)[] HAND_LANDMARK_CONNECTIONS = new (KeyPoint, KeyPoint)[]
        {
            (KeyPoint.Wrist, KeyPoint.Thumb1), (KeyPoint.Thumb1, KeyPoint.Thumb2), (KeyPoint.Thumb2, KeyPoint.Thumb3), (KeyPoint.Thumb3, KeyPoint.Thumb4), // Thumb
            (KeyPoint.Wrist, KeyPoint.Index1), (KeyPoint.Index1, KeyPoint.Index2), (KeyPoint.Index2, KeyPoint.Index3), (KeyPoint.Index3, KeyPoint.Index4), // Index finger
            (KeyPoint.Wrist, KeyPoint.Middle1), (KeyPoint.Middle1, KeyPoint.Middle2), (KeyPoint.Middle2, KeyPoint.Middle3), (KeyPoint.Middle3, KeyPoint.Middle4), // Middle finger
            (KeyPoint.Wrist, KeyPoint.Ring1), (KeyPoint.Ring1, KeyPoint.Ring2), (KeyPoint.Ring2, KeyPoint.Ring3), (KeyPoint.Ring3, KeyPoint.Ring4), // Ring finger
            (KeyPoint.Wrist, KeyPoint.Pinky1), (KeyPoint.Pinky1, KeyPoint.Pinky2), (KeyPoint.Pinky2, KeyPoint.Pinky3), (KeyPoint.Pinky3, KeyPoint.Pinky4), // Pinky
            (KeyPoint.Index1, KeyPoint.Middle1), (KeyPoint.Middle1, KeyPoint.Ring1), (KeyPoint.Ring1, KeyPoint.Pinky1) // Palm
        };

        readonly MediaPipeHandRunningMode _runningMode;
        readonly int _maxNumHands;
        readonly float _minHandDetectionConfidence;
        readonly float _minHandPresenceConfidence;
        readonly float _minHandTrackingConfidence;

        readonly MultiBackendNet _palmNet;
        /// <summary>Output layer names for palm detection inference. Cached to avoid calling <c>getUnconnectedOutLayersNames()</c> every frame.</summary>
        readonly List<string> _palmNetOutLayerNames;
        readonly MultiBackendNet _handLandmarksNet;
        /// <summary>Output layer names for hand landmark inference. Cached to avoid calling <c>getUnconnectedOutLayersNames()</c> every frame.</summary>
        readonly List<string> _handLandmarksNetOutLayerNames;

        /// <summary>Reusable list of <see cref="Mat"/> outputs for palm detection <c>forward</c>, following the same reuse strategy as <see cref="MediaPipeFaceLandmarker._faceDetectorForwardOutputList"/>.</summary>
        readonly List<Mat> _palmForwardOutputList = new List<Mat>();

        /// <summary>Reusable list of hand landmark <c>forward</c> outputs.</summary>
        readonly List<Mat> _handLandmarksForwardOutputList = new List<Mat>();

        /// <summary>Temporary merged bbox and 18-value row list for <see cref="NonMaxSuppressionCalculator"/>.</summary>
        readonly List<float[]> _handNmsMergedBoxScratch = new List<float[]>();

        readonly List<float[]> _handNmsMergedLmScratch = new List<float[]>();
        readonly List<float> _handNmsMergedScScratch = new List<float>();

        readonly Stack<float[]> _poolHandPalmNmsBox4 = new Stack<float[]>();
        readonly Stack<float[]> _poolHandPalmNmsLm18 = new Stack<float[]>();
        readonly Stack<float[]> _poolHandDetectionProjRow19 = new Stack<float[]>();

        float[] _palmWnmsKpAccumulator14;
        float[] _palmNmsRowBuf18;

        float[] _handTensorsToLmNorm;
        float[] _handTensorsToLmWorld;
        readonly float[] _handLetterboxRemovedNormScratch = new float[HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT];
        readonly float[] _handPackRowScratch = new float[HandLandmarkerEstimationData.ELEMENT_COUNT];

        float[] _handDedupBaselineScratch;
        DedupRectF[] _handDedupBoundsScratch;

        float[] RentHandDetectionProjRow19()
        {
            return _poolHandDetectionProjRow19.Count > 0
                ? _poolHandDetectionProjRow19.Pop()
                : new float[PalmDetectionRowElementCount];
        }

        void ReleaseHandDetectionProjRow19(float[] row)
        {
            if (row != null && row.Length == PalmDetectionRowElementCount)
                _poolHandDetectionProjRow19.Push(row);
        }

        void ReleaseHandDetectionProjRowList(IList<float[]> rows)
        {
            if (rows == null)
                return;
            for (int i = 0; i < rows.Count; i++)
                ReleaseHandDetectionProjRow19(rows[i]);
        }

        float[] RentHandPalmNmsBox4()
        {
            return _poolHandPalmNmsBox4.Count > 0 ? _poolHandPalmNmsBox4.Pop() : new float[4];
        }

        void ReleaseHandPalmNmsBox4(float[] row)
        {
            if (row != null && row.Length == 4)
                _poolHandPalmNmsBox4.Push(row);
        }

        float[] RentHandPalmNmsLm18()
        {
            return _poolHandPalmNmsLm18.Count > 0 ? _poolHandPalmNmsLm18.Pop() : new float[18];
        }

        void ReleaseHandPalmNmsLm18(float[] row)
        {
            if (row != null && row.Length == 18)
                _poolHandPalmNmsLm18.Push(row);
        }

        void ReleaseHandPalmNmsMergedScratchLists()
        {
            for (int i = 0; i < _handNmsMergedBoxScratch.Count; i++)
                ReleaseHandPalmNmsBox4(_handNmsMergedBoxScratch[i]);
            for (int i = 0; i < _handNmsMergedLmScratch.Count; i++)
                ReleaseHandPalmNmsLm18(_handNmsMergedLmScratch[i]);
            _handNmsMergedBoxScratch.Clear();
            _handNmsMergedLmScratch.Clear();
            _handNmsMergedScScratch.Clear();
        }

        /// <summary>Indices 0..K-1 used by <see cref="DetectionProjectionCalculator"/> after weighted NMS.</summary>
        MatOfInt _nmsIndices;

        /// <summary>Bboxes after weighted <see cref="NonMaxSuppressionCalculator"/> (K x 4).</summary>
        Mat _handWnmsMergedBoxXywh;
        /// <summary>Tensor rows after weighted NMS (K x 18).</summary>
        Mat _handWnmsMergedLm18;
        /// <summary>Score column after weighted NMS (K x 1).</summary>
        Mat _handWnmsMergedScore;

        readonly List<(int idx, float sc)> _handWnmsIndexed = new List<(int, float)>();
        List<(int idx, float sc)> _handWnmsRemained = new List<(int, float)>();
        List<(int idx, float sc)> _handWnmsNextRemained = new List<(int, float)>();

        // Output buffer for inference results (rows = hand indices).
        // Reused to speed up PeekOutput(); only the required rows are returned via rowRange.
        Mat _outputBuffer;

        // Palm anchors cached (avoid rebuilding every frame).
        Mat _anchors;
        Mat _anchorsNx14;

        // 192x192 letterboxed image for HandDetectorGraph / ImagePreprocessingGraph. Reused instead of allocating every frame.
        Mat _handDetectorLetterbox192;

        /// <summary>
        /// Same layout as the MATRIX output of <c>ImageToTensorCalculator</c> (row-major 4x4).
        /// <c>DetectionProjectionCalculator</c> uses only the affine terms
        /// <c>[0,1,3]</c> and <c>[4,5,7]</c> to map tensor-normalized coordinates
        /// back to normalized input-image coordinates.
        /// </summary>
        readonly float[] _handDetectorProjectionMatrix16 = new float[16];

        /// <summary>
        /// Source points for palm preprocessing when <c>NORM_RECT</c> is provided.
        /// Uses the same 4-point to 192x192 perspective transform path as
        /// <c>image_to_tensor_converter_opencv.cc</c>.
        /// Allocated lazily on first use.
        /// </summary>
        Mat _handDetectorWarpSrcPts;

        Mat _handDetectorWarpDstPts;

        // Buffers for InferenceSubgraph_PalmDetection. The preprocessed image is always 192x192 BGR 3-channel.
        Mat _palmInferenceBlob;
        Mat _palmInferenceBlobHxW;
        Mat _palmInferenceInput8u;

        // boxXywh buffer for TensorsToDetectionsCalculator. Row count depends on the model output; column count is fixed to 4.
        Mat _tensorsToDetectionsBoxXywh;

        /// <summary>
        /// NMS input after applying
        /// <c>TensorsToDetectionsCalculatorOptions.min_score_thresh</c>,
        /// which corresponds to the task-level <c>min_detection_confidence</c>.
        /// The row count shrinks only when entries are removed by thresholding.
        /// </summary>
        Mat _palmScoreFilteredBoxXywh;
        Mat _palmScoreFilteredScore;
        Mat _palmScoreFilteredLm18;

        // Buffers for ImagePreprocessingGraph_SingleHandLandmarks (fixed 224x224).
        Mat _singleHandLandmarkSrcPts;
        Mat _singleHandLandmarkDstPts;
        Mat _singleHandLandmarkWarpedBgr;
        Mat _singleHandLandmarkWarpedRgb;
        Mat _singleHandLandmarkBlob;
        Mat _singleHandLandmarkBlobHxW;

        // Loopback state for stream (VIDEO) mode.
        readonly List<NormalizedRect> _prevHandRectsFromLandmarks = new List<NormalizedRect>();

        /// <summary>
        /// Starting value for sequential IDs assigned to rectangles whose <c>rect_id</c>
        /// is not set, following the same behavior as the original HandAssociationCalculator.
        /// The sequence continues for the lifetime of this worker instance.
        /// </summary>
        long _handAssociationNextRectId = 1L;

        /// <summary>
        /// Structure representing a hand ROI looped back from the previous frame.
        /// Equivalent to a MediaPipe-style normalized rectangle.
        /// Private nested type used only inside this worker.
        /// </summary>
        private struct NormalizedRect
        {
            public float XCenter;
            public float YCenter;
            public float Width;
            public float Height;
            public float Rotation;
            /// <summary>Corresponds to <c>NormalizedRect.rect_id</c>. Unset is represented as <c>null</c>, equivalent to <c>has_rect_id() == false</c>.</summary>
            public long? RectId;
        }

        /// <summary>
        /// Lightweight structure representing the result for one detected hand.
        /// <c>Landmarks</c> stores normalized image coordinates after
        /// <see cref="LandmarkProjectionCalculator"/>, matching the semantics of
        /// <c>NormalizedLandmark</c>: x, y, z are expected in the [0, 1] range and
        /// z corresponds to <c>landmark.z * NORM_RECT.width</c>.
        /// Private nested type used only inside this worker.
        /// </summary>
        private struct HandResult
        {
            /// <summary>PRESENCE output from the original SingleHandLandmarksDetectorGraph after thresholding.</summary>
            public bool HandPresence;
            public Vec3f[] Landmarks;
            public Vec3f[] WorldLandmarks;
            /// <summary>Signed handedness score using the same binary top-1 convention as the original <c>TensorsToClassificationCalculator</c> output. See <see cref="PackHandednessBinaryTop1"/>.</summary>
            public float Handedness;
            public float PresenceConfidence;
            public NormalizedRect NextFrameRect;
        }

        /// <summary>
        /// Creates a hand landmarker worker backed by a palm detector model and a hand landmark model.
        /// This public API maps to the model assets and runtime options used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand detector graph and
        /// hand landmarks detector graph.
        /// </summary>
        /// <param name="palmModelFilepath">
        /// File path to the palm detector model.
        /// Corresponds to the detector model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand detector graph.
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, pass the full path to a serialized model that <see cref="Unity.InferenceEngine.ModelLoader.Load(string)"/> can load (e.g. <c>.sentis</c>); the caller may rewrite the path from ONNX.
        /// </param>
        /// <param name="handLandmarksModelFilepath">
        /// File path to the hand landmarks model.
        /// Corresponds to the landmark model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) SingleHandLandmarksDetectorGraph path.
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, as for palm, pass the full path to the Inference Engine serialized model.
        /// </param>
        /// <param name="runningMode">
        /// Task running mode.
        /// Corresponds to whether the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) task behaves like single-image processing
        /// or stateful video processing with loopback tracking state.
        /// </param>
        /// <param name="numHands">
        /// Maximum number of hands to return.
        /// Corresponds to the max number of hands option used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker task.
        /// </param>
        /// <param name="minHandDetectionConfidence">
        /// Minimum confidence for palm detections to be kept before later stages.
        /// Corresponds to the hand detector minimum detection confidence used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) task configuration.
        /// </param>
        /// <param name="minHandPresenceConfidence">
        /// Minimum presence confidence required for landmark results to be treated as present.
        /// Corresponds to the hand presence threshold used after the landmark model in
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe).
        /// </param>
        /// <param name="minTrackingConfidence">
        /// Minimum tracking confidence required to reuse the previous-frame rectangle.
        /// Corresponds to the hand tracking confidence gate used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) video pipeline.
        /// </param>
        /// <param name="dnnBackend">
        /// Inference backend: an OpenCV <see cref="Dnn"/> <c>DNN_BACKEND_*</c> constant, or <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>.
#if OPENCV_SENTIS_AVAILABLE
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, both palm and hand use Unity Inference Engine; <paramref name="dnnTarget"/> is interpreted as an integer <see cref="Unity.InferenceEngine.BackendType"/> value. Assumes Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer.
#else
        /// <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/> is only usable when the project includes Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer.
#endif
        /// </param>
        /// <param name="dnnTarget">
#if OPENCV_SENTIS_AVAILABLE
        /// An OpenCV DNN <c>DNN_TARGET_*</c> constant, or if <paramref name="dnnBackend"/> is <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, an integer to cast to <see cref="Unity.InferenceEngine.BackendType"/>.
#else
        /// An OpenCV <see cref="Dnn"/> <c>DNN_TARGET_*</c> constant.
#endif
        /// </param>
        public MediaPipeHandLandmarker(
            string palmModelFilepath,
            string handLandmarksModelFilepath,
            MediaPipeHandRunningMode runningMode = MediaPipeHandRunningMode.IMAGE,
            int numHands = 1,
            float minHandDetectionConfidence = 0.5f,
            float minHandPresenceConfidence = 0.5f,
            float minTrackingConfidence = 0.5f,
            int dnnBackend = Dnn.DNN_BACKEND_OPENCV,
            int dnnTarget = Dnn.DNN_TARGET_CPU)
            : base(dnnBackend, dnnTarget)
        {
            if (string.IsNullOrEmpty(palmModelFilepath))
                throw new ArgumentException("The palm detection model file path is not specified.", nameof(palmModelFilepath));
            if (string.IsNullOrEmpty(handLandmarksModelFilepath))
                throw new ArgumentException("The Hand Landmarker model file path is not specified.", nameof(handLandmarksModelFilepath));
            if (numHands <= 0)
                throw new ArgumentOutOfRangeException(nameof(numHands), "numHands must be greater than or equal to 1.");

            _runningMode = runningMode;
            _maxNumHands = numHands;
            _minHandDetectionConfidence = Mathf.Clamp01(minHandDetectionConfidence);
            _minHandPresenceConfidence = Mathf.Clamp01(minHandPresenceConfidence);
            _minHandTrackingConfidence = Mathf.Clamp01(minTrackingConfidence);

#if !OPENCV_SENTIS_AVAILABLE
            if (DnnBackend == MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS)
            {
                throw new NotSupportedException(
                    "DNN_BACKEND_UNITY_SENTIS requires Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer in the project and the OPENCV_SENTIS_AVAILABLE define.");
            }
#endif

            try
            {
                _palmNet = MultiBackendDnn.readNet(palmModelFilepath);
                _palmNet.setPreferableBackend(DnnBackend);
                _palmNet.setPreferableTarget(DnnTarget);
                _palmNetOutLayerNames = _palmNet.getUnconnectedOutLayersNames();

                _handLandmarksNet = MultiBackendDnn.readNet(handLandmarksModelFilepath);
                _handLandmarksNet.setPreferableBackend(DnnBackend);
                _handLandmarksNet.setPreferableTarget(DnnTarget);
                _handLandmarksNetOutLayerNames = _handLandmarksNet.getUnconnectedOutLayersNames();
            }
            catch (Exception e)
            {
                throw new ArgumentException("Failed to initialize the DNN models for Hand Landmarker. Check the model paths and file contents.", e);
            }
        }

        /// <summary>
        /// High-level inference API equivalent to the synchronous detect entry points of
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker.
        /// Returns one packed output matrix where each row stores one hand result.
        /// </summary>
        /// <param name="image">
        /// Input image in BGR 3-channel format.
        /// Corresponds to the input image consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand landmarker graph.
        /// </param>
        /// <param name="useCopyOutput">
        /// If true, returns a copied output matrix.
        /// If false, returns a view backed by the worker's reusable output buffer.
        /// </param>
        /// <returns>
        /// Packed result matrix with one row per detected hand and
        /// <see cref="HandLandmarkerEstimationData.ELEMENT_COUNT"/> columns per row.
        /// The matrix is returned as <c>CV_32FC1</c> and each row matches the memory layout of
        /// <see cref="HandLandmarkerEstimationData"/>.
        /// Columns <c>[0 .. 62]</c> store 21 normalized hand landmarks as xyz triplets,
        /// columns <c>[63 .. 125]</c> store 21 world hand landmarks as xyz triplets,
        /// and column <c>[126]</c> stores the handedness top-1 classification score.
        /// </returns>
        public Mat Detect(Mat image, bool useCopyOutput = false)
        {
            if (image != null) image.ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            Execute(image);
            return useCopyOutput ? CopyOutput(0) : PeekOutput(0);
        }

        /// <summary>
        /// Asynchronous inference API equivalent to the async detect entry points of
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker.
        /// The effective behavior still depends on the configured <c>runningMode</c>.
        /// </summary>
        /// <param name="image">
        /// Input image in BGR 3-channel format.
        /// Corresponds to the input image consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand landmarker graph.
        /// </param>
        /// <param name="cancellationToken">
        /// Cancellation token for the C# async operation.
        /// </param>
        /// <returns>
        /// Packed result matrix with one row per detected hand and
        /// <see cref="HandLandmarkerEstimationData.ELEMENT_COUNT"/> columns per row.
        /// The matrix is returned as <c>CV_32FC1</c> and each row matches the memory layout of
        /// <see cref="HandLandmarkerEstimationData"/>.
        /// Columns <c>[0 .. 62]</c> store 21 normalized hand landmarks as xyz triplets,
        /// columns <c>[63 .. 125]</c> store 21 world hand landmarks as xyz triplets,
        /// and column <c>[126]</c> stores the handedness top-1 classification score.
        /// This method always returns a copied <see cref="Mat"/>, so the caller owns its lifetime.
        /// </returns>
        /// <remarks>
        /// For the OpenCV Dnn module, inference is scheduled on a background thread when thread-pool scheduling is available.
        /// Web builds cannot use thread pools; only then does the OpenCV Dnn path run synchronously on the caller thread.
        /// When <c>OPENCV_SENTIS_AVAILABLE</c> and Sentis is selected, inference uses Sentis forward APIs asynchronously on every platform, including Web.
        /// </remarks>
        public async Task<Mat> DetectTaskAsync(Mat image, CancellationToken cancellationToken = default)
        {
            if (image != null) image.ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            await ExecuteTaskAsync(new[] { image }, cancellationToken);
            return CopyOutput(0);
        }

        /// <summary>
        /// Asynchronous inference API equivalent to the async detect entry points of
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker.
        /// </summary>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task{TResult}"/>.
        /// See <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. Web synchronous fallback applies only to the OpenCV Dnn backend; Sentis remains asynchronous on every platform, including Web.
        /// </remarks>
        [Obsolete("Use DetectTaskAsync(). DetectAsync() will return Awaitable in a future version.")]
        public Task<Mat> DetectAsync(Mat image, CancellationToken cancellationToken = default) =>
            DetectTaskAsync(image, cancellationToken);

        /// <summary>
        /// Converts a packed result matrix into a managed array of <see cref="HandLandmarkerEstimationData"/>.
        /// Each returned element corresponds to one row from <see cref="Detect(Mat, bool)"/>.
        /// </summary>
        /// <param name="result">
        /// Packed output matrix returned by <see cref="Detect(Mat, bool)"/> or a compatible source.
        /// Each row corresponds to one hand and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand landmarks,
        /// hand world landmarks, and handedness outputs.
        /// </param>
        /// <returns>
        /// Managed array of hand estimation data.
        /// Returns an empty array when no hands are present.
        /// </returns>
        public virtual HandLandmarkerEstimationData[] ToStructuredData(Mat result)
        {
            ThrowIfDisposed();

            if (result != null)
                result.ThrowIfDisposed();
            if (result.empty())
                return Array.Empty<HandLandmarkerEstimationData>();

            int elementCount = HandLandmarkerEstimationData.ELEMENT_COUNT;
            if (result.cols() < elementCount)
                throw new ArgumentException("Invalid result matrix. It must have at least " + elementCount + " columns.");

            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            int handCount = result.rows();
            var dst = new HandLandmarkerEstimationData[handCount];
            // result is expected to be CV_32FC1 with shape (handCount, ELEMENT_COUNT),
            // and is copied 1:1 into the memory layout of HandLandmarkerEstimationData.
            OpenCVMatUtils.CopyFromMat(result, dst);

            return dst;
        }

        /// <summary>
        /// Views a packed result matrix as a zero-allocation <see cref="Span{T}"/> of
        /// <see cref="HandLandmarkerEstimationData"/>.
        /// </summary>
        /// <remarks>
        /// The returned span remains valid only while <paramref name="result"/> stays allocated
        /// and unchanged.
        /// If the matrix has more than <see cref="HandLandmarkerEstimationData.ELEMENT_COUNT"/> columns,
        /// interpreting the underlying memory as contiguous rows of
        /// <see cref="HandLandmarkerEstimationData"/> can cross row boundaries.
        /// The worker-generated packed matrices use the exact expected column count.
        /// </remarks>
        /// <param name="result">
        /// Packed output matrix: <see cref="Detect(Mat, bool)"/> at index <c>0</c>.
        /// Each row corresponds to one hand and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand landmarks,
        /// hand world landmarks, and handedness outputs.
        /// </param>
        /// <returns>
        /// Span whose elements correspond to hands in row order.
        /// Returns an empty span when the matrix is empty.
        /// </returns>
        public virtual Span<HandLandmarkerEstimationData> ToStructuredDataAsSpan(Mat result)
        {
            ThrowIfDisposed();

            if (result != null)
                result.ThrowIfDisposed();
            if (result.empty())
                return Span<HandLandmarkerEstimationData>.Empty;

            int elementCount = HandLandmarkerEstimationData.ELEMENT_COUNT;
            if (result.cols() < elementCount)
                throw new ArgumentException("Invalid result matrix. It must have at least " + elementCount + " columns.");

            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            return result.AsSpan<HandLandmarkerEstimationData>();
        }

        /// <summary>
        /// Draws hand landmarks from a <see cref="Mat"/> array whose layout matches
        /// <see cref="Detect(Mat, bool)"/>.
        /// Element <c>[0]</c> is required and contains one row per hand.
        /// Array input is supported for compatibility with <see cref="MediaPipeHolisticLandmarker"/>.
        /// </summary>
        /// <param name="image">Destination image for visualization.</param>
        /// <param name="results">
        /// Array of output matrices.
        /// <c>results[0]</c> corresponds to the packed hand output derived from the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand landmark outputs.
        /// </param>
        /// <param name="printResult">If true, prints the decoded result to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public void Visualize(Mat image, Mat[] results, bool printResult = false, bool isRGB = false)
        {
            ThrowIfDisposed();
            VisualizePackedHandOutputMat(image, (results == null || results.Length == 0) ? null : results[0], printResult, isRGB);
        }

        /// <summary>
        /// Visualizes the packed hand output returned by <see cref="Detect(Mat, bool)"/><c>[0]</c>.
        /// Each row is decoded as one <see cref="HandLandmarkerEstimationData"/> value.
        /// </summary>
        /// <param name="image">Destination image for visualization.</param>
        /// <param name="results">
        /// Packed result matrix with one row per hand.
        /// This matrix stores the public packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand landmarks,
        /// hand world landmarks, and handedness outputs.
        /// </param>
        /// <param name="printResult">If true, prints the decoded result to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public override void Visualize(Mat image, Mat results, bool printResult = false, bool isRGB = false)
        {
            Visualize(image, results == null ? null : new[] { results }, printResult, isRGB);
        }

        /// <summary>
        /// Draws a <see cref="HandLandmarkerEstimationData"/> matrix with one row per hand,
        /// using the same logic as <see cref="Visualize(Mat, Mat, bool, bool)"/> but without disposal checks.
        /// Intended for the left-hand and right-hand slots of <see cref="MediaPipeHolisticLandmarker"/>.
        /// </summary>
        internal static void VisualizePackedHandOutputMat(Mat image, Mat results, bool printResult, bool isRGB)
        {
            if (image != null)
                image.ThrowIfDisposed();
            if (results == null || results.empty() || results.rows() <= 0)
                return;

            if (results.cols() < HandLandmarkerEstimationData.ELEMENT_COUNT)
                throw new ArgumentException("Invalid result matrix. It must have at least " + HandLandmarkerEstimationData.ELEMENT_COUNT + " columns.");

            if (!results.isContinuous())
                throw new ArgumentException("result is not continuous.");

            Span<HandLandmarkerEstimationData> dataSpan = results.AsSpan<HandLandmarkerEstimationData>();
            for (int h = 0; h < dataSpan.Length; h++)
            {
                ref readonly HandLandmarkerEstimationData row = ref dataSpan[h];
                VisualizeHandLandmarkerEstimationData(image, in row, handIndex: h, printResult, isRGB);
            }
        }

        /// <summary>
        /// Packed result for one detected hand, without an explicit bounding box.
        /// The memory layout matches one row produced by <see cref="PackResultsToMats"/>.
        /// <see cref="NormLandmarks"/> corresponds to the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) Task API
        /// <c>hand_landmarks</c> output represented as 21 <see cref="Vec3f"/> values.
        /// <see cref="WorldLandmarks"/> corresponds to the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) Task API
        /// <c>hand_world_landmarks</c> output.
        /// </summary>
        [Serializable]
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public unsafe struct HandLandmarkerEstimationData
        {
            public const int LANDMARK_VEC3F_COUNT = 21;
            public const int LANDMARK_ELEMENT_COUNT = 3 * LANDMARK_VEC3F_COUNT;
            public const int ELEMENT_COUNT = LANDMARK_ELEMENT_COUNT + LANDMARK_ELEMENT_COUNT + 1;
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Packed normalized hand landmarks.
            /// Corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_landmarks</c> output flattened as 21 xyz triplets in row-major float order.
            /// </summary>
            public fixed float NormLandmarks[LANDMARK_ELEMENT_COUNT];
            /// <summary>
            /// Packed world hand landmarks.
            /// Corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_world_landmarks</c> output flattened as 21 xyz triplets in row-major float order.
            /// </summary>
            public fixed float WorldLandmarks[LANDMARK_ELEMENT_COUNT];
            /// <summary>
            /// Packed handedness top-1 classification.
            /// Corresponds to the handedness classification produced by
            /// [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// after the binary <c>TensorsToClassificationCalculator</c> style top-1 selection.
            /// A positive value means label index 0 (Right) won and stores its score.
            /// A negative value means label index 1 (Left) won and stores <c>-score</c>.
            /// Zero means no valid handedness result is available.
            /// </summary>
            public float Handedness;

            /// <summary>
            /// Returns true when the packed handedness value indicates that the
            /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) label index 0 (Right)
            /// was selected as the top-1 handedness class.
            /// </summary>
            public static bool IsRightHandDominant(float packedHandedness) => packedHandedness > 0f;

            /// <summary>
            /// Returns true when the packed handedness value indicates that the
            /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) label index 1 (Left)
            /// was selected as the top-1 handedness class.
            /// </summary>
            public static bool IsLeftHandDominant(float packedHandedness) => packedHandedness < 0f;

            /// <summary>
            /// Returns the absolute top-1 handedness score.
            /// Corresponds to the winning <c>Classification.score()</c> value from
            /// [MediaPipe](https://github.com/google-ai-edge/mediapipe).
            /// </summary>
            public static float HandednessScore(float packedHandedness) => Mathf.Abs(packedHandedness);

            /// <summary>
            /// Creates one packed hand result from decoded landmark arrays and a packed handedness score.
            /// </summary>
            /// <param name="normLandmarks">
            /// Normalized image-space landmarks.
            /// Corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_landmarks</c> output represented as 21 <see cref="Vec3f"/> values.
            /// </param>
            /// <param name="worldLandmarks">
            /// World-space landmarks in meters.
            /// Corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_world_landmarks</c> output represented as 21 <see cref="Vec3f"/> values.
            /// </param>
            /// <param name="handedness">
            /// Packed handedness value stored into <see cref="Handedness"/>.
            /// Corresponds to the winning handedness classification from
            /// [MediaPipe](https://github.com/google-ai-edge/mediapipe).
            /// When generated with the same binary top-1 rule as this worker, use the return value of
            /// <see cref="PackHandednessBinaryTop1"/>.
            /// </param>
            public HandLandmarkerEstimationData(Vec3f[] normLandmarks, Vec3f[] worldLandmarks,
                float handedness)
            {
                if (normLandmarks == null || normLandmarks.Length != LANDMARK_VEC3F_COUNT)
                    throw new ArgumentException("normLandmarks must be a Vec3f[" + LANDMARK_VEC3F_COUNT + "]");
                if (worldLandmarks == null || worldLandmarks.Length != LANDMARK_VEC3F_COUNT)
                    throw new ArgumentException("worldLandmarks must be a Vec3f[" + LANDMARK_VEC3F_COUNT + "]");

                Handedness = handedness;

                for (int i = 0; i < normLandmarks.Length; i++)
                {
                    int offset = i * 3;
                    ref readonly var s = ref normLandmarks[i];
                    NormLandmarks[offset + 0] = s.Item1;
                    NormLandmarks[offset + 1] = s.Item2;
                    NormLandmarks[offset + 2] = s.Item3;
                }
                for (int i = 0; i < worldLandmarks.Length; i++)
                {
                    int offset = i * 3;
                    ref readonly var w = ref worldLandmarks[i];
                    WorldLandmarks[offset + 0] = w.Item1;
                    WorldLandmarks[offset + 1] = w.Item2;
                    WorldLandmarks[offset + 2] = w.Item3;
                }
            }

            /// <summary>
            /// Returns <see cref="NormLandmarks"/> as a typed read-only span of 21 <see cref="Vec3f"/> values.
            /// The returned data corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_landmarks</c> output.
            /// </summary>
            public readonly ReadOnlySpan<Vec3f> GetNormLandmarks()
            {
                unsafe
                {
                    fixed (float* p = NormLandmarks)
                    {
                        return MemoryMarshal.Cast<float, Vec3f>(new ReadOnlySpan<float>(p, LANDMARK_ELEMENT_COUNT));
                    }
                }
            }

            /// <summary>
            /// Returns <see cref="WorldLandmarks"/> as a typed read-only span of 21 <see cref="Vec3f"/> values.
            /// The returned data corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_world_landmarks</c> output.
            /// </summary>
            public readonly ReadOnlySpan<Vec3f> GetWorldLandmarks()
            {
                unsafe
                {
                    fixed (float* p = WorldLandmarks)
                    {
                        return MemoryMarshal.Cast<float, Vec3f>(new ReadOnlySpan<float>(p, LANDMARK_ELEMENT_COUNT));
                    }
                }
            }

            /// <summary>
            /// Copies <see cref="NormLandmarks"/> into a managed array of 21 <see cref="Vec3f"/> values.
            /// The returned array corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_landmarks</c> output.
            /// </summary>
            public readonly Vec3f[] GetNormLandmarksArray()
            {
                Vec3f[] landmarks = new Vec3f[LANDMARK_VEC3F_COUNT];
                unsafe
                {
                    for (int i = 0; i < landmarks.Length; i++)
                    {
                        int offset = i * 3;
                        landmarks[i] = new Vec3f(NormLandmarks[offset + 0],
                            NormLandmarks[offset + 1],
                            NormLandmarks[offset + 2]);
                    }
                }
                return landmarks;
            }

            /// <summary>
            /// Copies <see cref="WorldLandmarks"/> into a managed array of 21 <see cref="Vec3f"/> values.
            /// The returned array corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>hand_world_landmarks</c> output.
            /// </summary>
            public readonly Vec3f[] GetWorldLandmarksArray()
            {
                Vec3f[] landmarks = new Vec3f[LANDMARK_VEC3F_COUNT];
                unsafe
                {
                    for (int i = 0; i < landmarks.Length; i++)
                    {
                        int offset = i * 3;
                        landmarks[i] = new Vec3f(WorldLandmarks[offset + 0],
                            WorldLandmarks[offset + 1],
                            WorldLandmarks[offset + 2]);
                    }
                }
                return landmarks;
            }

            /// <summary>
            /// Returns a diagnostic string for the packed hand result.
            /// Includes normalized landmarks, world landmarks, and the packed handedness classification.
            /// </summary>
            public readonly override string ToString()
            {
                StringBuilder sb = new StringBuilder(1536);

                sb.Append("HandLandmarkerEstimationData(");

                sb.Append("NormLandmarks:");
                ReadOnlySpan<Vec3f> landmarks = GetNormLandmarks();
                for (int i = 0; i < landmarks.Length; i++)
                {
                    ref readonly var p = ref landmarks[i];
                    sb.Append(p.ToString());
                }
                sb.Append(" ");

                sb.Append("WorldLandmarks:");
                ReadOnlySpan<Vec3f> landmarksWorld = GetWorldLandmarks();
                for (int i = 0; i < landmarksWorld.Length; i++)
                {
                    ref readonly var p = ref landmarksWorld[i];
                    sb.Append(p.ToString());
                }
                sb.Append(" ");

                sb.AppendFormat("Handedness:{0},({1}) ", Handedness,
                    Handedness == 0f ? "?" : (IsRightHandDominant(Handedness) ? "Right" : "Left"));
                sb.Append(")");
                return sb.ToString();
            }
        }

        protected override Mat[] RunCoreProcessing(Mat[] inputs)
        {
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("MediaPipeHandLandmarker accepts only a single input image at index 0.", nameof(inputs));

            var image = inputs[0];
            if (image != null) image.ThrowIfDisposed();
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            List<HandResult> hands = _runningMode == MediaPipeHandRunningMode.IMAGE
                ? DetectPipeline(image)
                : DetectForVideoPipeline(image);

            return PackResultsToMats(hands);
        }

        protected override async Task<Mat[]> RunCoreProcessingTaskAsync(Mat[] inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("MediaPipeHandLandmarker accepts only a single input image at index 0.", nameof(inputs));

            var image = inputs[0];
            if (image != null) image.ThrowIfDisposed();
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

#if OPENCV_SENTIS_AVAILABLE
            if (_handLandmarksNet.UsesSentis)
            {
                List<HandResult> hands = _runningMode == MediaPipeHandRunningMode.IMAGE
                    ? await ProcessImageDataAsync(image, cancellationToken)
                    : await ProcessVideoDataAsync(image, cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                return PackResultsToMats(hands);
            }
#endif

            cancellationToken.ThrowIfCancellationRequested();
#if UNITY_WEBGL && !UNITY_EDITOR
            return await Task.FromResult(RunCoreProcessing(inputs));
#else
            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return RunCoreProcessing(inputs);
            }, cancellationToken);
#endif
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                _palmNet?.Dispose();
                _handLandmarksNet?.Dispose();
                _nmsIndices?.Dispose();
                _nmsIndices = null;
                _handWnmsMergedBoxXywh?.Dispose();
                _handWnmsMergedBoxXywh = null;
                _handWnmsMergedLm18?.Dispose();
                _handWnmsMergedLm18 = null;
                _handWnmsMergedScore?.Dispose();
                _handWnmsMergedScore = null;
                _outputBuffer?.Dispose();
                _outputBuffer = null;
                _anchors?.Dispose();
                _anchors = null;
                _anchorsNx14?.Dispose();
                _anchorsNx14 = null;
                _handDetectorLetterbox192?.Dispose();
                _handDetectorLetterbox192 = null;
                _palmInferenceBlobHxW?.Dispose();
                _palmInferenceBlobHxW = null;
                _palmInferenceBlob?.Dispose();
                _palmInferenceBlob = null;
                _palmInferenceInput8u?.Dispose();
                _palmInferenceInput8u = null;
                _tensorsToDetectionsBoxXywh?.Dispose();
                _tensorsToDetectionsBoxXywh = null;
                _palmScoreFilteredBoxXywh?.Dispose();
                _palmScoreFilteredBoxXywh = null;
                _palmScoreFilteredScore?.Dispose();
                _palmScoreFilteredScore = null;
                _palmScoreFilteredLm18?.Dispose();
                _palmScoreFilteredLm18 = null;
                _singleHandLandmarkBlobHxW?.Dispose();
                _singleHandLandmarkBlobHxW = null;
                _singleHandLandmarkBlob?.Dispose();
                _singleHandLandmarkBlob = null;
                _singleHandLandmarkWarpedRgb?.Dispose();
                _singleHandLandmarkWarpedRgb = null;
                _singleHandLandmarkWarpedBgr?.Dispose();
                _singleHandLandmarkWarpedBgr = null;
                _singleHandLandmarkDstPts?.Dispose();
                _singleHandLandmarkDstPts = null;
                _singleHandLandmarkSrcPts?.Dispose();
                _singleHandLandmarkSrcPts = null;
                _handDetectorWarpSrcPts?.Dispose();
                _handDetectorWarpSrcPts = null;
                _handDetectorWarpDstPts?.Dispose();
                _handDetectorWarpDstPts = null;
                _palmForwardOutputList.Clear();
                _handLandmarksForwardOutputList.Clear();
            }

            base.Dispose(disposing);
        }


        /// <summary>
        /// Internal IMAGE-mode entry point.
        /// Equivalent to <c>HandLandmarker::Detect</c> and called from <c>RunCoreProcessing</c>.
        /// </summary>
        List<HandResult> DetectPipeline(Mat image)
        {
            return ProcessImageData(image);
        }

        /// <summary>
        /// Internal VIDEO-mode entry point.
        /// Equivalent to <c>HandLandmarker::DetectForVideo</c>.
        /// </summary>
        List<HandResult> DetectForVideoPipeline(Mat image)
        {
            return ProcessVideoData(image);
        }
        /// <summary>
        /// IMAGE-mode pipeline entry corresponding to the Task API <c>ProcessImageData</c>.
        /// Follows the flow of <c>HandLandmarkerGraph</c> and only invokes dedicated
        /// methods for the lower-level calculators and subgraphs in order.
        ///
        /// Mapping to the original <c>hand_landmarker_graph.cc</c> (<c>HandLandmarkerGraph</c>):
        /// - PreviousLoopbackCalculator → PreviousLoopbackCalculator
        /// - HandDetectorGraph → HandDetectorGraph
        /// - ClipNormalizedRectVectorSizeCalculator → ClipNormalizedRectVectorSizeCalculator
        /// - MultipleHandLandmarksDetectorGraph → MultipleHandLandmarksDetectorGraph
        /// - ImagePropertiesCalculator → ImagePropertiesCalculator
        /// - HandLandmarksDeduplicationCalculator → HandLandmarksDeduplicationCalculator
        /// </summary>
        List<HandResult> ProcessImageData(Mat image)
        {
            // In IMAGE mode, the LOOP input to PreviousLoopbackCalculator is always treated as empty.
            var prevHandRects = PreviousLoopbackCalculator(image, new List<NormalizedRect>());

            // In IMAGE mode, HandDetectorGraph always runs for every frame.
            var handDetectorOutputs = HandDetectorGraph(image, null);

            // ClipNormalizedRectVectorSizeCalculator: clamp HAND_RECTS to maxNumHands.
            var clippedHandRects = ClipNormalizedRectVectorSizeCalculator(handDetectorOutputs.HandRects);

            // MultipleHandLandmarksDetectorGraph: run landmark inference and produce next-frame ROIs.
            var multiOutputs = MultipleHandLandmarksDetectorGraph(image, clippedHandRects);

            // ImagePropertiesCalculator -> HandLandmarksDeduplicationCalculator, in the same order as hand_landmarker_graph.cc.
            var imageSize = ImagePropertiesCalculator(image);
            var hands = HandLandmarksDeduplicationCalculator(multiOutputs.HandResults, imageSize.Width, imageSize.Height);
            // The final Task API result keeps only hands whose presence is true; graph-level EndLoop vectors may still contain false placeholder slots.
            hands.RemoveAll(h => !h.HandPresence);

            // Back edge to PreviousLoopbackCalculator: update HAND_RECT_NEXT_FRAME for the next frame.
            _prevHandRectsFromLandmarks.Clear();
            foreach (var hand in hands)
            {
                _prevHandRectsFromLandmarks.Add(hand.NextFrameRect);
            }

            return hands;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="ProcessImageData"/> using the Sentis path with <see cref="ReadbackAndCloneAsync"/>.
        /// </summary>
        async Task<List<HandResult>> ProcessImageDataAsync(Mat image, CancellationToken cancellationToken)
        {
            var prevHandRects = PreviousLoopbackCalculator(image, new List<NormalizedRect>());
            var handDetectorOutputs = await HandDetectorGraphAsync(image, null, cancellationToken);
            var clippedHandRects = ClipNormalizedRectVectorSizeCalculator(handDetectorOutputs.HandRects);
            var multiOutputs = await MultipleHandLandmarksDetectorGraphAsync(image, clippedHandRects, cancellationToken);
            var imageSize = ImagePropertiesCalculator(image);
            var hands = HandLandmarksDeduplicationCalculator(multiOutputs.HandResults, imageSize.Width, imageSize.Height);
            hands.RemoveAll(h => !h.HandPresence);
            _prevHandRectsFromLandmarks.Clear();
            foreach (var hand in hands)
                _prevHandRectsFromLandmarks.Add(hand.NextFrameRect);
            return hands;
        }

#endif
        /// <summary>
        /// VIDEO-mode pipeline entry corresponding to the Task API <c>ProcessVideoData</c>.
        /// Matches the stream-mode <c>HandLandmarkerGraph</c> and only invokes dedicated
        /// methods for the lower-level calculators and subgraphs in order.
        ///
        /// Mapping to the original <c>hand_landmarker_graph.cc</c>:
        /// - PreviousLoopbackCalculator → PreviousLoopbackCalculator
        /// - NormalizedRectVectorHasMinSizeCalculator → NormalizedRectVectorHasMinSizeCalculator
        /// - DisallowIf + HandDetectorGraph → only the HandDetectorGraph branch is called explicitly
        /// - HandAssociationCalculator → HandAssociationCalculator
        /// - ClipNormalizedRectVectorSizeCalculator → ClipNormalizedRectVectorSizeCalculator
        /// - MultipleHandLandmarksDetectorGraph → MultipleHandLandmarksDetectorGraph
        /// - ImagePropertiesCalculator / HandLandmarksDeduplicationCalculator → methods with the same names
        /// </summary>
        List<HandResult> ProcessVideoData(Mat image)
        {
            // 1. PreviousLoopbackCalculator: get the previous frame's HAND_RECT_NEXT_FRAME values as PREV_LOOP.
            var prevHandRects = PreviousLoopbackCalculator(image, _prevHandRectsFromLandmarks);

            // 2. NormalizedRectVectorHasMinSizeCalculator: check whether the previous-frame rectangle count is at least maxNumHands.
            bool hasEnoughHands = NormalizedRectVectorHasMinSizeCalculator(prevHandRects, _maxNumHands);

            List<NormalizedRect> handRectsFromDetector = new List<NormalizedRect>();

            // 3. DisallowIf + HandDetectorGraph path:
            //    only run HandDetectorGraph when hasEnoughHands == false, yielding new HAND_RECTS.
            if (!hasEnoughHands)
            {
                var detectorOutputs = HandDetectorGraph(image, null);
                handRectsFromDetector = detectorOutputs.HandRects;
            }

            // 4. HandAssociationCalculator equivalent:
            //    first add all BASE_RECTS (prevHandRects) to the result, then evaluate
            //    RECTS (handRectsFromDetector) with DoesRectOverlap
            //    using IoU and min_hand_tracking_confidence, matching the order of GetNonOverlappingElements.
            var associatedHandRects = HandAssociationCalculator(prevHandRects, handRectsFromDetector);

            // 5. ClipNormalizedRectVectorSizeCalculator: clamp to maxNumHands.
            var clippedHandRects = ClipNormalizedRectVectorSizeCalculator(associatedHandRects);

            // 6. MultipleHandLandmarksDetectorGraph: run landmark inference and generate next-frame ROIs.
            var multiOutputs = MultipleHandLandmarksDetectorGraph(image, clippedHandRects);
            // 7. ImagePropertiesCalculator + HandLandmarksDeduplicationCalculator, shared post-landmark path in the graph.
            var imageSize = ImagePropertiesCalculator(image);
            var hands = HandLandmarksDeduplicationCalculator(multiOutputs.HandResults, imageSize.Width, imageSize.Height);
            hands.RemoveAll(h => !h.HandPresence);

            // 8. Update the back edge to PreviousLoopbackCalculator.
            _prevHandRectsFromLandmarks.Clear();
            foreach (var hand in hands)
            {
                _prevHandRectsFromLandmarks.Add(hand.NextFrameRect);
            }

            return hands;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="ProcessVideoData"/> using the Sentis path with <see cref="ReadbackAndCloneAsync"/>.
        /// </summary>
        async Task<List<HandResult>> ProcessVideoDataAsync(Mat image, CancellationToken cancellationToken)
        {
            var prevHandRects = PreviousLoopbackCalculator(image, _prevHandRectsFromLandmarks);
            bool hasEnoughHands = NormalizedRectVectorHasMinSizeCalculator(prevHandRects, _maxNumHands);
            List<NormalizedRect> handRectsFromDetector = new List<NormalizedRect>();
            if (!hasEnoughHands)
            {
                var detectorOutputs = await HandDetectorGraphAsync(image, null, cancellationToken);
                handRectsFromDetector = detectorOutputs.HandRects;
            }
            var associatedHandRects = HandAssociationCalculator(prevHandRects, handRectsFromDetector);
            var clippedHandRects = ClipNormalizedRectVectorSizeCalculator(associatedHandRects);
            var multiOutputs = await MultipleHandLandmarksDetectorGraphAsync(image, clippedHandRects, cancellationToken);
            var imageSize = ImagePropertiesCalculator(image);
            var hands = HandLandmarksDeduplicationCalculator(multiOutputs.HandResults, imageSize.Width, imageSize.Height);
            hands.RemoveAll(h => !h.HandPresence);
            _prevHandRectsFromLandmarks.Clear();
            foreach (var hand in hands)
                _prevHandRectsFromLandmarks.Add(hand.NextFrameRect);
            return hands;
        }

#endif
        /// <summary>
        /// Equivalent to <c>PreviousLoopbackCalculator</c>.
        /// Receives <c>MAIN = IMAGE</c> and <c>LOOP = previous-frame HAND_RECT_NEXT_FRAME</c>,
        /// and returns <c>PREV_LOOP</c>, i.e. the previous-frame rectangle vector.
        /// </summary>
        List<NormalizedRect> PreviousLoopbackCalculator(Mat image, List<NormalizedRect> loopHandRects)
        {
            // In this C# implementation, the image itself is not used; the method simply returns the looped-back rectangle vector.
            return loopHandRects != null ? new List<NormalizedRect>(loopHandRects) : new List<NormalizedRect>();
        }

        /// <summary>
        /// Equivalent to <c>NormalizedRectVectorHasMinSizeCalculator</c>.
        /// Returns true when <c>prev_hand_rects_from_landmarks</c> contains at least <paramref name="minSize"/> elements.
        /// </summary>
        bool NormalizedRectVectorHasMinSizeCalculator(List<NormalizedRect> rects, int minSize)
        {
            if (rects == null) return false;
            return rects.Count >= minSize;
        }

        /// <summary>
        /// Equivalent to <c>HandDetectorGraph</c> from <c>hand_detector_graph.cc</c>.
        /// This method only invokes the lower-level calculators and subgraphs in the original connection order,
        /// and returns <c>HAND_RECTS</c>, <c>PALM_RECTS</c>, and <c>PALM_DETECTIONS</c>.
        ///
        /// Mapping to the original <c>hand_detector_graph.cc</c>:
        /// - ImagePreprocessingGraph → ImagePreprocessingGraph (palm 192x192 path; stores the tensor image and MATRIX in <see cref="_handDetectorProjectionMatrix16"/>. When <c>NORM_RECT</c> is present, the warpPerspective path is used)
        /// - Inference subgraph (original <c>AddInference</c> / <c>mediapipe.tasks.core.InferenceSubgraph</c>) → <see cref="InferenceSubgraph_PalmDetection"/>
        /// - SsdAnchorsCalculator → SsdAnchorsCalculator
        /// - TensorsToDetectionsCalculator → TensorsToDetectionsCalculator
        /// - min_score_thresh (<c>min_detection_confidence</c>) → <see cref="PalmDetectionsFilterByMinScoreThresh"/> (removes entries before NMS in the same phase as the original <c>ConvertToDetection</c>)
        /// - NonMaxSuppressionCalculator → NonMaxSuppressionCalculator
        /// - DetectionLabelIdToTextCalculator → DetectionLabelIdToTextCalculator
        /// - DetectionProjectionCalculator → DetectionProjectionCalculator
        /// - DetectionsToRectsCalculator → DetectionsToRectsCalculator (<c>PALM_RECTS</c>)
        /// - RectTransformationCalculator → RectTransformationCalculator (<c>HAND_RECTS</c>)
        /// - ClipNormalizedRectVectorSizeCalculator → ClipNormalizedRectVectorSizeCalculator (<c>num_hands</c>)
        /// </summary>
        (List<NormalizedRect> HandRects, List<NormalizedRect> PalmRects, List<float[]> PalmDetections) HandDetectorGraph(Mat image, NormalizedRect? normRect)
        {
            var empty = (new List<NormalizedRect>(), new List<NormalizedRect>(), new List<float[]>());
            int origW = image.cols();
            int origH = image.rows();
            if (origW <= 0 || origH <= 0)
                return empty;

            if (_handDetectorLetterbox192 == null)
                _handDetectorLetterbox192 = new Mat(192, 192, image.type());

            Mat maxSizeImg = _handDetectorLetterbox192;
            ImagePreprocessingGraph(image, maxSizeImg, normRect);
            var outputBlobs = InferenceSubgraph_PalmDetection(maxSizeImg);
            if (outputBlobs == null || outputBlobs.Count < 2)
                return empty;

            Mat output0 = outputBlobs[0];
            Mat output1 = outputBlobs[1];
            int num = output0.size(1);
            Mat boxXywh = null;

            SsdAnchorsCalculator(num, out Mat anchors, out Mat anchorsNx14);
            TensorsToDetectionsCalculator(output0, output1, anchors, anchorsNx14, out boxXywh);
            DetectionLabelIdToTextCalculator();
            var palmDetections = new List<float[]>();
            MatOfInt indices;
            using (var scoreView = output1.reshape(1, num))
            using (var boxAndLandmark = output0.reshape(1, num))
            {
                PalmDetectionsFilterByMinScoreThresh(
                    boxXywh, scoreView, boxAndLandmark, _minHandDetectionConfidence,
                    out Mat nmsBoxXywh, out Mat nmsScore, out Mat nmsLm18);
                indices = NonMaxSuppressionCalculator(nmsBoxXywh, nmsScore, nmsLm18);
                DetectionProjectionCalculator(
                    _handWnmsMergedBoxXywh, _handWnmsMergedScore, _handWnmsMergedLm18, indices,
                    _handDetectorProjectionMatrix16, origW, origH, palmDetections);
            }

            List<NormalizedRect> palmRects = DetectionsToRectsCalculator(palmDetections, origW, origH);
            List<NormalizedRect> handRects = RectTransformationCalculator(palmRects, origW, origH);
            handRects = ClipNormalizedRectVectorSizeCalculator(handRects, _maxNumHands);
            // The caller does not use these rows, so return the detection row buffers to the pool before clearing the list.
            ReleaseHandDetectionProjRowList(palmDetections);
            palmDetections.Clear();
            return (handRects, palmRects, palmDetections);
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="HandDetectorGraph"/> using the Sentis path with <see cref="InferenceSubgraph_PalmDetectionAsync"/>.
        /// </summary>
        async Task<(List<NormalizedRect> HandRects, List<NormalizedRect> PalmRects, List<float[]> PalmDetections)> HandDetectorGraphAsync(
            Mat image, NormalizedRect? normRect, CancellationToken cancellationToken)
        {
            var empty = (new List<NormalizedRect>(), new List<NormalizedRect>(), new List<float[]>());
            int origW = image.cols();
            int origH = image.rows();
            if (origW <= 0 || origH <= 0)
                return empty;

            if (_handDetectorLetterbox192 == null)
                _handDetectorLetterbox192 = new Mat(192, 192, image.type());

            Mat maxSizeImg = _handDetectorLetterbox192;
            ImagePreprocessingGraph(image, maxSizeImg, normRect);
            var outputBlobs = await InferenceSubgraph_PalmDetectionAsync(maxSizeImg, cancellationToken);
            if (outputBlobs == null || outputBlobs.Count < 2)
                return empty;

            Mat output0 = outputBlobs[0];
            Mat output1 = outputBlobs[1];
            int num = output0.size(1);
            Mat boxXywh = null;

            SsdAnchorsCalculator(num, out Mat anchors, out Mat anchorsNx14);
            TensorsToDetectionsCalculator(output0, output1, anchors, anchorsNx14, out boxXywh);
            DetectionLabelIdToTextCalculator();
            var palmDetections = new List<float[]>();
            MatOfInt indices;
            using (var scoreView = output1.reshape(1, num))
            using (var boxAndLandmark = output0.reshape(1, num))
            {
                PalmDetectionsFilterByMinScoreThresh(
                    boxXywh, scoreView, boxAndLandmark, _minHandDetectionConfidence,
                    out Mat nmsBoxXywh, out Mat nmsScore, out Mat nmsLm18);
                indices = NonMaxSuppressionCalculator(nmsBoxXywh, nmsScore, nmsLm18);
                DetectionProjectionCalculator(
                    _handWnmsMergedBoxXywh, _handWnmsMergedScore, _handWnmsMergedLm18, indices,
                    _handDetectorProjectionMatrix16, origW, origH, palmDetections);
            }

            List<NormalizedRect> palmRects = DetectionsToRectsCalculator(palmDetections, origW, origH);
            List<NormalizedRect> handRects = RectTransformationCalculator(palmRects, origW, origH);
            handRects = ClipNormalizedRectVectorSizeCalculator(handRects, _maxNumHands);
            ReleaseHandDetectionProjRowList(palmDetections);
            palmDetections.Clear();
            return (handRects, palmRects, palmDetections);
        }

#endif
        /// <summary>
        /// Equivalent to <c>HandAssociationCalculator</c>, following the same procedure as
        /// <c>GetNonOverlappingElements</c> in <c>hand_association_calculator.cc</c>.
        /// <list type="number">
        /// <item><description>The input stream priority is higher for BASE_RECTS than for RECTS, so BASE is processed first just like the original implementation.</description></item>
        /// <item><description>Add all rectangles from <paramref name="baseRects"/> (<c>BASE_RECTS = prev_hand_rects_from_landmarks</c>) to the result without deduplication.</description></item>
        /// <item><description>Process <paramref name="rects"/> (<c>RECTS = HAND_RECTS</c> from HandDetector) in order. If any existing result rectangle forms a pair with IoU above <c>min_similarity_threshold</c> according to <see cref="HandAssociationCalculator_DoesRectOverlap"/> from <c>rectangle_util.cc</c>, the new detection is skipped; otherwise it is added.</description></item>
        /// </list>
        /// NormalizedRect dimensions are treated as axis-aligned rectangles that ignore rotation,
        /// matching the <c>rectangle_util</c> TODO behavior in the original code.
        /// <c>rect_id</c> values are assigned from <see cref="_handAssociationNextRectId"/>
        /// only for elements where the ID is not already set.
        /// </summary>
        /// <param name="baseRects">BASE_RECTS, i.e. the rectangle list derived from previous-frame landmarks.</param>
        /// <param name="rects">RECTS, i.e. the rectangle list from the palm detection path.</param>
        List<NormalizedRect> HandAssociationCalculator(List<NormalizedRect> baseRects, List<NormalizedRect> rects)
        {
            var result = new List<NormalizedRect>();

            if (baseRects != null && baseRects.Count > 0)
            {
                foreach (var r in baseRects)
                {
                    var copy = r;
                    if (!copy.RectId.HasValue)
                        copy.RectId = HandAssociationGetNextRectId();
                    result.Add(copy);
                }
            }

            if (rects == null || rects.Count == 0)
                return result;

            foreach (var rect in rects)
            {
                if (HandAssociationCalculator_DoesRectOverlap(rect, result, _minHandTrackingConfidence))
                    continue;
                var copy = rect;
                if (!copy.RectId.HasValue)
                    copy.RectId = HandAssociationGetNextRectId();
                result.Add(copy);
            }

            return result;
        }

        /// <summary>Equivalent to <c>GetNextRectId()</c> from <c>hand_association_calculator.cc</c>. Returns incrementing IDs starting from 1.</summary>
        long HandAssociationGetNextRectId() => _handAssociationNextRectId++;

        /// <summary>
        /// Equivalent to <c>DoesRectOverlap</c> from <c>rectangle_util.cc</c>.
        /// Returns true if the IoU between <paramref name="newRect"/> and any rectangle in
        /// <paramref name="existing"/> exceeds <paramref name="minSimilarityThreshold"/>.
        /// </summary>
        static bool HandAssociationCalculator_DoesRectOverlap(NormalizedRect newRect, List<NormalizedRect> existing, float minSimilarityThreshold)
        {
            for (int i = 0; i < existing.Count; i++)
            {
                if (HandAssociationCalculator_ComputeIoU(existing[i], newRect) > minSimilarityThreshold)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Equivalent to <c>CalculateIou</c> from <c>rectangle_util.cc</c>.
        /// Computes IoU for axis-aligned rectangles while ignoring rotation.
        /// </summary>
        static float HandAssociationCalculator_ComputeIoU(NormalizedRect a, NormalizedRect b)
        {
            float ax1 = a.XCenter - a.Width * 0.5f;
            float ay1 = a.YCenter - a.Height * 0.5f;
            float ax2 = a.XCenter + a.Width * 0.5f;
            float ay2 = a.YCenter + a.Height * 0.5f;

            float bx1 = b.XCenter - b.Width * 0.5f;
            float by1 = b.YCenter - b.Height * 0.5f;
            float bx2 = b.XCenter + b.Width * 0.5f;
            float by2 = b.YCenter + b.Height * 0.5f;

            float ix1 = Mathf.Max(ax1, bx1);
            float iy1 = Mathf.Max(ay1, by1);
            float ix2 = Mathf.Min(ax2, bx2);
            float iy2 = Mathf.Min(ay2, by2);

            if (ix2 <= ix1 || iy2 <= iy1)
                return 0f;

            float intersection = (ix2 - ix1) * (iy2 - iy1);
            float areaA = (ax2 - ax1) * (ay2 - ay1);
            float areaB = (bx2 - bx1) * (by2 - by1);
            float union = areaA + areaB - intersection;
            return union > 0f ? intersection / union : 0f;
        }

        /// <summary>
        /// Equivalent to <c>ClipNormalizedRectVectorSizeCalculator</c>.
        /// Clips the hand rectangle vector so its size does not exceed <paramref name="maxVecSize"/>.
        /// </summary>
        List<NormalizedRect> ClipNormalizedRectVectorSizeCalculator(List<NormalizedRect> rects, int maxVecSize)
        {
            if (rects == null) return new List<NormalizedRect>();
            if (rects.Count <= maxVecSize)
                return new List<NormalizedRect>(rects);

            var clipped = new List<NormalizedRect>(rects);
            clipped.RemoveRange(maxVecSize, clipped.Count - maxVecSize);
            return clipped;
        }

        /// <summary>
        /// Equivalent to <c>ClipNormalizedRectVectorSizeCalculator</c> using the
        /// <c>maxNumHands</c> option from HandLandmarkerGraph.
        /// </summary>
        List<NormalizedRect> ClipNormalizedRectVectorSizeCalculator(List<NormalizedRect> rects)
        {
            return ClipNormalizedRectVectorSizeCalculator(rects, _maxNumHands);
        }

        /// <summary>
        /// Equivalent to <c>MultipleHandLandmarksDetectorGraph</c>.
        /// This method only coordinates the loop and EndLoop stages and delegates
        /// each calculator to its own dedicated method.
        ///
        /// Mapping to the original <c>hand_landmarks_detector_graph.cc</c> (<c>MultipleHandLandmarksDetectorGraph</c>):
        /// - BeginLoopNormalizedRectCalculator → BeginLoopNormalizedRectCalculator
        /// - SingleHandLandmarksDetectorGraph → SingleHandLandmarksDetectorGraph
        /// - EndLoopClassificationListCalculator → EndLoopClassificationListCalculator
        /// - EndLoopBooleanCalculator → EndLoopBooleanCalculator
        /// - EndLoopFloatCalculator → EndLoopFloatCalculator
        /// - EndLoopNormalizedLandmarkListVectorCalculator → EndLoopNormalizedLandmarkListVectorCalculator
        /// - EndLoopLandmarkListVectorCalculator → EndLoopLandmarkListVectorCalculator
        /// - EndLoopNormalizedRectCalculator → EndLoopNormalizedRectCalculator
        /// - Merge of the vectorized EndLoop outputs → MergeEndLoopHandLandmarkOutputs
        /// - Iteration count matches the number of input HAND_RECTS, and inference failures still emit one placeholder slot to preserve the original EndLoop vector lengths
        /// </summary>
        (List<HandResult> HandResults, List<NormalizedRect> HandRectsNextFrame) MultipleHandLandmarksDetectorGraph(Mat image, List<NormalizedRect> handRects)
        {
            var handednessIterable = new List<float>();
            var presencesIterable = new List<bool>();
            var presenceScoresIterable = new List<float>();
            var landmarkListsIterable = new List<Vec3f[]>();
            var worldLandmarkListsIterable = new List<Vec3f[]>();
            var handRectsNextFrameIterable = new List<NormalizedRect>();

            foreach (var loopItem in BeginLoopNormalizedRectCalculator(image, handRects))
            {
                var single = SingleHandLandmarksDetectorGraph(loopItem.Image, loopItem.HandRect);
                HandResult h = single ?? CreateAbsentHandResultPlaceholder();
                EndLoopClassificationListCalculator(handednessIterable, h.Handedness);
                EndLoopBooleanCalculator(presencesIterable, h.HandPresence);
                EndLoopFloatCalculator(presenceScoresIterable, h.PresenceConfidence);
                EndLoopNormalizedLandmarkListVectorCalculator(landmarkListsIterable, h.Landmarks);
                EndLoopLandmarkListVectorCalculator(worldLandmarkListsIterable, h.WorldLandmarks);
                EndLoopNormalizedRectCalculator(handRectsNextFrameIterable, h.NextFrameRect);
            }

            return MergeEndLoopHandLandmarkOutputs(
                handednessIterable, presencesIterable, presenceScoresIterable,
                landmarkListsIterable, worldLandmarkListsIterable, handRectsNextFrameIterable);
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="MultipleHandLandmarksDetectorGraph"/> using the Sentis path with <see cref="SingleHandLandmarksDetectorGraphAsync"/>.
        /// </summary>
        async Task<(List<HandResult> HandResults, List<NormalizedRect> HandRectsNextFrame)> MultipleHandLandmarksDetectorGraphAsync(
            Mat image, List<NormalizedRect> handRects, CancellationToken cancellationToken)
        {
            var handednessIterable = new List<float>();
            var presencesIterable = new List<bool>();
            var presenceScoresIterable = new List<float>();
            var landmarkListsIterable = new List<Vec3f[]>();
            var worldLandmarkListsIterable = new List<Vec3f[]>();
            var handRectsNextFrameIterable = new List<NormalizedRect>();

            foreach (var loopItem in BeginLoopNormalizedRectCalculator(image, handRects))
            {
                var single = await SingleHandLandmarksDetectorGraphAsync(loopItem.Image, loopItem.HandRect, cancellationToken);
                HandResult h = single ?? CreateAbsentHandResultPlaceholder();
                EndLoopClassificationListCalculator(handednessIterable, h.Handedness);
                EndLoopBooleanCalculator(presencesIterable, h.HandPresence);
                EndLoopFloatCalculator(presenceScoresIterable, h.PresenceConfidence);
                EndLoopNormalizedLandmarkListVectorCalculator(landmarkListsIterable, h.Landmarks);
                EndLoopLandmarkListVectorCalculator(worldLandmarkListsIterable, h.WorldLandmarks);
                EndLoopNormalizedRectCalculator(handRectsNextFrameIterable, h.NextFrameRect);
            }

            return MergeEndLoopHandLandmarkOutputs(
                handednessIterable, presencesIterable, presenceScoresIterable,
                landmarkListsIterable, worldLandmarkListsIterable, handRectsNextFrameIterable);
        }

#endif
        /// <summary>
        /// Placeholder result used when preprocessing or inference fails for one iteration.
        /// Like the post-AllowIf path, it keeps 21 zero landmarks and <c>PRESENCE = false</c>
        /// so the 21-point assumption in <c>HandLandmarksDeduplication</c> remains satisfied.
        /// </summary>
        static HandResult CreateAbsentHandResultPlaceholder()
        {
            int L = HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            return new HandResult
            {
                HandPresence = false,
                Landmarks = new Vec3f[L],
                WorldLandmarks = new Vec3f[L],
                Handedness = 0f,
                PresenceConfidence = 0f,
                NextFrameRect = new NormalizedRect(),
            };
        }

        /// <summary>
        /// Final merge step of <c>MultipleHandLandmarksDetectorGraph</c>.
        /// Combines each EndLoop iterable into a list of per-hand <see cref="HandResult"/> values.
        /// In MediaPipe, multiple EndLoop nodes emit the same stream set, so the outputs are associated here.
        /// </summary>
        static (List<HandResult> HandResults, List<NormalizedRect> HandRectsNextFrame) MergeEndLoopHandLandmarkOutputs(
            List<float> handednessIterable,
            List<bool> presencesIterable,
            List<float> presenceScoresIterable,
            List<Vec3f[]> landmarkListsIterable,
            List<Vec3f[]> worldLandmarkListsIterable,
            List<NormalizedRect> handRectsNextFrameIterable)
        {
            int n = landmarkListsIterable.Count;
            var handResults = new List<HandResult>(n);
            for (int i = 0; i < n; i++)
            {
                handResults.Add(new HandResult
                {
                    HandPresence = presencesIterable[i],
                    Landmarks = landmarkListsIterable[i],
                    WorldLandmarks = worldLandmarkListsIterable[i],
                    Handedness = handednessIterable[i],
                    PresenceConfidence = presenceScoresIterable[i],
                    NextFrameRect = handRectsNextFrameIterable[i]
                });
            }
            return (handResults, handRectsNextFrameIterable);
        }

        /// <summary>
        /// Equivalent to <c>BeginLoopNormalizedRectCalculator</c>.
        /// Feeds <c>IMAGE</c> to the CLONE input and <c>HAND_RECTS</c> to ITERABLE,
        /// then enumerates each iteration item as a single-hand rectangle plus the shared image reference.
        /// </summary>
        static IEnumerable<(Mat Image, NormalizedRect HandRect)> BeginLoopNormalizedRectCalculator(Mat image, List<NormalizedRect> handRects)
        {
            if (image == null || handRects == null)
                yield break;
            foreach (var r in handRects)
                yield return (image, r);
        }

        /// <summary>
        /// Equivalent to <c>EndLoopClassificationListCalculator</c>.
        /// Appends the per-hand HANDEDNESS output, represented here as one scalar, to the output vector.
        /// </summary>
        static void EndLoopClassificationListCalculator(List<float> iterable, float handednessItem)
        {
            iterable.Add(handednessItem);
        }

        /// <summary>
        /// Equivalent to <c>EndLoopBooleanCalculator</c>.
        /// Appends the per-hand PRESENCE boolean to the output vector.
        /// </summary>
        static void EndLoopBooleanCalculator(List<bool> iterable, bool presenceItem)
        {
            iterable.Add(presenceItem);
        }

        /// <summary>
        /// Equivalent to <c>EndLoopFloatCalculator</c>.
        /// Appends the per-hand PRESENCE_SCORE value to the output vector.
        /// </summary>
        static void EndLoopFloatCalculator(List<float> iterable, float presenceScoreItem)
        {
            iterable.Add(presenceScoreItem);
        }

        /// <summary>
        /// Equivalent to <c>EndLoopNormalizedLandmarkListVectorCalculator</c>.
        /// Appends each hand's LANDMARKS list to the output vector.
        /// </summary>
        static void EndLoopNormalizedLandmarkListVectorCalculator(List<Vec3f[]> iterable, Vec3f[] landmarksItem)
        {
            iterable.Add(landmarksItem);
        }

        /// <summary>
        /// Equivalent to <c>EndLoopLandmarkListVectorCalculator</c>.
        /// Appends each hand's WORLD_LANDMARKS list to the output vector.
        /// </summary>
        static void EndLoopLandmarkListVectorCalculator(List<Vec3f[]> iterable, Vec3f[] worldLandmarksItem)
        {
            iterable.Add(worldLandmarksItem);
        }

        /// <summary>
        /// Equivalent to <c>EndLoopNormalizedRectCalculator</c>.
        /// Appends each hand's HAND_RECT_NEXT_FRAME to the output vector.
        /// </summary>
        static void EndLoopNormalizedRectCalculator(List<NormalizedRect> iterable, NormalizedRect rectItem)
        {
            iterable.Add(rectItem);
        }

        /// <summary>
        /// Equivalent to <c>SingleHandLandmarksDetectorGraph</c>.
        /// Only invokes the lower-level calculators and subgraphs in the original connection order,
        /// excluding branching glue and final return-value assembly.
        /// Even when presence is false, this method still returns a <see cref="HandResult"/>
        /// with zero-like landmark content through the AllowIf path so that the EndLoop vector lengths
        /// in the multiple-hand path remain aligned with the number of input hands.
        ///
        /// Mapping to the original <c>hand_landmarks_detector_graph.cc</c> (<c>BuildSingleHandLandmarksDetectorGraph</c>):
        /// - ImagePreprocessingGraph → ImagePreprocessingGraph_SingleHandLandmarks
        /// - Inference subgraph → InferenceSubgraph_SingleHandLandmarks
        /// - SplitTensorVectorCalculator → SplitTensorVectorCalculator
        /// - TensorsToLandmarksCalculator (normalized) → TensorsToLandmarksCalculator_NormalizedLandmarks
        /// - TensorsToLandmarksCalculator (world) → TensorsToLandmarksCalculator_WorldLandmarks
        /// - TensorsToFloatsCalculator → TensorsToFloatsCalculator_HandPresence
        /// - ThresholdingCalculator → ThresholdingCalculator_HandPresence
        /// - TensorsToClassificationCalculator → TensorsToClassificationCalculator_Handedness
        /// - AllowIf (handedness) → AllowIf_ClassificationListByHandPresence
        /// - LandmarkLetterboxRemovalCalculator → LandmarkLetterboxRemovalCalculator
        /// - LandmarkProjectionCalculator → LandmarkProjectionCalculator
        /// - AllowIf (landmarks) → AllowIf_NormalizedLandmarkListByHandPresence
        /// - WorldLandmarkProjectionCalculator → WorldLandmarkProjectionCalculator
        /// - AllowIf (world) → AllowIf_LandmarkListByHandPresence
        /// - HandLandmarksToRectCalculator → HandLandmarksToRectCalculator (inside the presence-true branch)
        /// - RectTransformationCalculator (next-frame ROI) → RectTransformationCalculator_SingleHandLandmarks
        /// - AllowIf (next rect) → AllowIf_NormalizedRectByHandPresence
        /// </summary>
        HandResult? SingleHandLandmarksDetectorGraph(Mat image, NormalizedRect handRect)
        {
            if (!ImagePreprocessingGraph_SingleHandLandmarks(image, handRect, out var pre))
                return null;

            Mat handBlob = pre.HandBlob;
            List<Mat> inferenceTensors = InferenceSubgraph_SingleHandLandmarks(handBlob);
            if (inferenceTensors == null || inferenceTensors.Count < 4)
                return null;

            if (!SplitTensorVectorCalculator(inferenceTensors,
                    out Mat landmarkTensors, out Mat handFlagTensors,
                    out Mat handednessTensors, out Mat worldLandmarkTensors))
            {
                return null;
            }

            // TensorsToLandmarksCalculator is represented by two nodes: one for normalized landmarks and one for world landmarks.
            var normLandmarksLetterboxed = TensorsToLandmarksCalculator_NormalizedLandmarks(
                landmarkTensors, pre.ModelW, pre.ModelH);
            var worldLandmarksRaw = TensorsToLandmarksCalculator_WorldLandmarks(worldLandmarkTensors);

            float handPresenceScore = TensorsToFloatsCalculator_HandPresence(handFlagTensors);
            bool handPresence = ThresholdingCalculator_HandPresence(handPresenceScore);
            float handednessClassification = TensorsToClassificationCalculator_Handedness(handednessTensors);
            float handednessGated = AllowIf_ClassificationListByHandPresence(
                handPresence, handednessClassification);

            var afterLetterbox = LandmarkLetterboxRemovalCalculator(
                normLandmarksLetterboxed, pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft,
                pre.LetterboxPaddingBottom, pre.LetterboxPaddingRight);

            // Follow hand_landmarks_detector_graph.cc exactly:
            // LandmarkProjection -> AllowIf(LANDMARKS) -> HandLandmarksToRect -> RectTransformation -> AllowIf(NEXT_FRAME)
            var projectedLandmarksRaw = LandmarkProjectionCalculator(afterLetterbox, handRect);
            var projectedLandmarks = AllowIf_NormalizedLandmarkListByHandPresence(
                handPresence, projectedLandmarksRaw);

            var projectedWorldRaw = WorldLandmarkProjectionCalculator(
                worldLandmarksRaw, handRect);
            var projectedWorld = AllowIf_LandmarkListByHandPresence(handPresence, projectedWorldRaw);

            NormalizedRect handRectNextFrame;
            if (handPresence)
            {
                var normRectFromLandmarks = HandLandmarksToRectCalculator(
                    projectedLandmarks, pre.ImageW, pre.ImageH);
                handRectNextFrame = RectTransformationCalculator_SingleHandLandmarks(
                    normRectFromLandmarks, pre.ImageW, pre.ImageH);
                // Carry the tracking ID into the next-frame rectangle, corresponding to the path that preserves BASE_RECTS.rect_id in the original graph.
                if (handRect.RectId.HasValue)
                    handRectNextFrame.RectId = handRect.RectId;
            }
            else
            {
                handRectNextFrame = new NormalizedRect();
            }

            var nextFrameGated = AllowIf_NormalizedRectByHandPresence(handPresence, handRectNextFrame);

            return new HandResult
            {
                HandPresence = handPresence,
                Landmarks = projectedLandmarks,
                WorldLandmarks = projectedWorld,
                Handedness = handednessGated,
                PresenceConfidence = handPresenceScore,
                NextFrameRect = nextFrameGated
            };
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="SingleHandLandmarksDetectorGraph"/> using the Sentis path with <see cref="InferenceSubgraph_SingleHandLandmarksAsync"/>.
        /// </summary>
        async Task<HandResult?> SingleHandLandmarksDetectorGraphAsync(Mat image, NormalizedRect handRect, CancellationToken cancellationToken)
        {
            if (!ImagePreprocessingGraph_SingleHandLandmarks(image, handRect, out var pre))
                return null;

            Mat handBlob = pre.HandBlob;
            List<Mat> inferenceTensors = await InferenceSubgraph_SingleHandLandmarksAsync(handBlob, cancellationToken);
            if (inferenceTensors == null || inferenceTensors.Count < 4)
                return null;

            if (!SplitTensorVectorCalculator(inferenceTensors,
                    out Mat landmarkTensors, out Mat handFlagTensors,
                    out Mat handednessTensors, out Mat worldLandmarkTensors))
            {
                return null;
            }

            var normLandmarksLetterboxed = TensorsToLandmarksCalculator_NormalizedLandmarks(
                landmarkTensors, pre.ModelW, pre.ModelH);
            var worldLandmarksRaw = TensorsToLandmarksCalculator_WorldLandmarks(worldLandmarkTensors);

            float handPresenceScore = TensorsToFloatsCalculator_HandPresence(handFlagTensors);
            bool handPresence = ThresholdingCalculator_HandPresence(handPresenceScore);
            float handednessClassification = TensorsToClassificationCalculator_Handedness(handednessTensors);
            float handednessGated = AllowIf_ClassificationListByHandPresence(
                handPresence, handednessClassification);

            var afterLetterbox = LandmarkLetterboxRemovalCalculator(
                normLandmarksLetterboxed, pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft,
                pre.LetterboxPaddingBottom, pre.LetterboxPaddingRight);

            var projectedLandmarksRaw = LandmarkProjectionCalculator(afterLetterbox, handRect);
            var projectedLandmarks = AllowIf_NormalizedLandmarkListByHandPresence(
                handPresence, projectedLandmarksRaw);

            var projectedWorldRaw = WorldLandmarkProjectionCalculator(
                worldLandmarksRaw, handRect);
            var projectedWorld = AllowIf_LandmarkListByHandPresence(handPresence, projectedWorldRaw);

            NormalizedRect handRectNextFrame;
            if (handPresence)
            {
                var normRectFromLandmarks = HandLandmarksToRectCalculator(
                    projectedLandmarks, pre.ImageW, pre.ImageH);
                handRectNextFrame = RectTransformationCalculator_SingleHandLandmarks(
                    normRectFromLandmarks, pre.ImageW, pre.ImageH);
                if (handRect.RectId.HasValue)
                    handRectNextFrame.RectId = handRect.RectId;
            }
            else
            {
                handRectNextFrame = new NormalizedRect();
            }

            var nextFrameGated = AllowIf_NormalizedRectByHandPresence(handPresence, handRectNextFrame);

            return new HandResult
            {
                HandPresence = handPresence,
                Landmarks = projectedLandmarks,
                WorldLandmarks = projectedWorld,
                Handedness = handednessGated,
                PresenceConfidence = handPresenceScore,
                NextFrameRect = nextFrameGated
            };
        }

#endif
        /// <summary>
        /// Equivalent to ImagePreprocessingGraph, following the Task API / ImageToTensorCalculator CPU path.
        /// As in MediaPipe <c>image_to_tensor_calculator.cc</c>, this performs
        /// GetRoi -> PadRoi -> OpenCV perspective transform:
        /// the normalized rectangle is converted to a pixel ROI, aspect-adjusted,
        /// and the four vertices of the rotated rectangle are projected into a 224x224 tensor.
        /// This differs from the older full-image affine rotation plus axis-aligned crop path.
        /// LETTERBOX_PADDING corresponds to the output of PadRoiLikeImageToTensorCalculator.
        /// </summary>
        bool ImagePreprocessingGraph_SingleHandLandmarks(Mat image, NormalizedRect handRect, out SingleHandLandmarkPreprocessOut pre)
        {
            pre = default;
            int imgW = image.cols();
            int imgH = image.rows();
            if (imgW <= 0 || imgH <= 0)
                return false;

            const int inputSize = 224;

            if (_singleHandLandmarkBlob == null)
            {
                _singleHandLandmarkSrcPts = new Mat(4, 2, CvType.CV_32FC1);
                _singleHandLandmarkDstPts = new Mat(4, 2, CvType.CV_32FC1);
                float dw = inputSize, dh = inputSize;
                // Same corner order as image_to_tensor_converter_opencv.cc (BL, TL, TR, BR). Written only once because 224 is fixed.
                Span<float> dstPtsArr = stackalloc float[8];
                dstPtsArr[0] = 0f; dstPtsArr[1] = dh;
                dstPtsArr[2] = 0f; dstPtsArr[3] = 0f;
                dstPtsArr[4] = dw; dstPtsArr[5] = 0f;
                dstPtsArr[6] = dw; dstPtsArr[7] = dh;
                _singleHandLandmarkDstPts.put(0, 0, dstPtsArr);

                _singleHandLandmarkWarpedBgr = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _singleHandLandmarkWarpedRgb = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _singleHandLandmarkBlob = new Mat(new int[] { 1, inputSize, inputSize, 3 }, CvType.CV_32FC1);
                _singleHandLandmarkBlobHxW = _singleHandLandmarkBlob.reshape(3, new int[] { inputSize, inputSize });
            }

            // GetRoi from image_to_tensor_utils.cc.
            float cx = handRect.XCenter * imgW;
            float cy = handRect.YCenter * imgH;
            float rw = handRect.Width * imgW;
            float rh = handRect.Height * imgH;
            if (rw <= 0f || rh <= 0f || float.IsNaN(rw) || float.IsNaN(rh))
                return false;

            // In the hand landmarks task, keep_aspect_ratio=true is typically implied by model metadata for ImagePreprocessingGraph.
            PadRoiLikeImageToTensorCalculator(inputSize, inputSize, keepAspectRatio: true, ref rw, ref rh,
                out float padL, out float padT, out float padR, out float padB);

            double angleDeg = handRect.Rotation * 180.0 / Math.PI;

            Imgproc.boxPoints(new Vec5d(cx, cy, rw, rh, angleDeg), _singleHandLandmarkSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_singleHandLandmarkSrcPts, _singleHandLandmarkDstPts))
            {
                Imgproc.warpPerspective(image, _singleHandLandmarkWarpedBgr, projMat, (inputSize, inputSize),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0, 0, 0, 0));
            }

            Imgproc.cvtColor(_singleHandLandmarkWarpedBgr, _singleHandLandmarkWarpedRgb, Imgproc.COLOR_BGR2RGB);
            // DebugMat.imshow("_singleHandLandmarkWarpedRgb", _singleHandLandmarkWarpedRgb);
            _singleHandLandmarkWarpedRgb.convertTo(_singleHandLandmarkBlobHxW, CvType.CV_32F, 1.0 / 255.0);

            pre = new SingleHandLandmarkPreprocessOut
            {
                HandBlob = _singleHandLandmarkBlob,
                ImageW = imgW,
                ImageH = imgH,
                ModelW = inputSize,
                ModelH = inputSize,
                LetterboxPaddingTop = padT,
                LetterboxPaddingLeft = padL,
                LetterboxPaddingRight = padR,
                LetterboxPaddingBottom = padB
            };
            return true;
        }

        /// <summary>
        /// Equivalent to <c>PadRoi</c> from <c>image_to_tensor_utils.cc</c>.
        /// Returns output padding as <c>[left, top, right, bottom]</c> in normalized tensor coordinates.
        /// </summary>
        static void PadRoiLikeImageToTensorCalculator(int tensorW, int tensorH, bool keepAspectRatio,
            ref float roiW, ref float roiH, out float padLeft, out float padTop, out float padRight, out float padBottom)
        {
            padLeft = padTop = padRight = padBottom = 0f;
            if (!keepAspectRatio || roiW <= 0f || roiH <= 0f)
                return;
            float tensorAr = (float)tensorH / tensorW;
            float roiAr = roiH / roiW;
            float horizontalPadding = 0f, verticalPadding = 0f;
            if (tensorAr > roiAr)
            {
                roiH = roiW * tensorAr;
                verticalPadding = (1f - roiAr / tensorAr) * 0.5f;
            }
            else
            {
                roiW = roiH / tensorAr;
                horizontalPadding = (1f - tensorAr / roiAr) * 0.5f;
            }
            padLeft = padRight = horizontalPadding;
            padTop = padBottom = verticalPadding;
        }

        /// <summary>
        /// Preprocessing result for landmark inference on one hand.
        /// Contains only the hand blob and the values required for projection and letterbox removal.
        /// </summary>
        struct SingleHandLandmarkPreprocessOut
        {
            public Mat HandBlob;
            public int ImageW;
            public int ImageH;
            public int ModelW;
            public int ModelH;
            public float LetterboxPaddingTop;
            public float LetterboxPaddingLeft;
            public float LetterboxPaddingRight;
            public float LetterboxPaddingBottom;
        }

        /// <summary>
        /// Equivalent to the inference subgraph.
        /// Feeds the preprocessed 224x224x3 tensor to <see cref="_handLandmarksNet"/> (OpenCV DNN or Unity Inference Engine) and
        /// returns the output tensor list (<c>TENSORS</c>). Callers do not dispose <see cref="Mat"/> entries in the returned list;
        /// <see cref="MultiBackendNet"/> owns OpenCV forward outputs across calls and reuses IE buffers in Sentis mode.
        /// </summary>
        List<Mat> InferenceSubgraph_SingleHandLandmarks(Mat handBlob)
        {
            _handLandmarksForwardOutputList.Clear();
            _handLandmarksNet.setInput(handBlob);
            _handLandmarksNet.forward(_handLandmarksForwardOutputList, _handLandmarksNetOutLayerNames);
            return _handLandmarksForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="InferenceSubgraph_SingleHandLandmarks"/> (via <see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// Invoked only from the <see cref="RunCoreProcessingTaskAsync"/> path; OpenCV inference uses <see cref="InferenceSubgraph_SingleHandLandmarks"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_SingleHandLandmarksAsync(Mat handBlob, CancellationToken cancellationToken)
        {
            _handLandmarksForwardOutputList.Clear();
            _handLandmarksNet.setInput(handBlob);
            await _handLandmarksNet.forwardTaskAsync(_handLandmarksForwardOutputList, _handLandmarksNetOutLayerNames, cancellationToken);
            return _handLandmarksForwardOutputList;
        }

#endif
        /// <summary>
        /// Equivalent to <c>SplitTensorVectorCalculator</c>.
        /// Splits inference output tensors into four branches:
        /// landmarks, presence, handedness, and world.
        /// In this implementation the network already returns four outputs, so this is just reference assignment.
        /// </summary>
        static bool SplitTensorVectorCalculator(List<Mat> inferenceTensors,
            out Mat landmarkTensors, out Mat handFlagTensors,
            out Mat handednessTensors, out Mat worldLandmarkTensors)
        {
            landmarkTensors = handFlagTensors = handednessTensors = worldLandmarkTensors = null;
            if (inferenceTensors == null || inferenceTensors.Count < 4)
                return false;
            landmarkTensors = inferenceTensors[0];
            handFlagTensors = inferenceTensors[1];
            handednessTensors = inferenceTensors[2];
            worldLandmarkTensors = inferenceTensors[3];
            return true;
        }

        /// <summary>
        /// Equivalent to the first <c>TensorsToLandmarksCalculator</c>.
        /// Decodes the landmark tensor into an array corresponding to <c>NORM_LANDMARKS</c>
        /// before letterbox removal, with x and y normalized by input resolution.
        /// z follows <c>raw / input_image_width / normalize_z</c> exactly as in <c>tensors_to_landmarks_calculator.cc</c>.
        /// </summary>
        float[] TensorsToLandmarksCalculator_NormalizedLandmarks(Mat tensor, int inputW, int inputH)
        {
            int need = HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT;
            if (_handTensorsToLmNorm == null || _handTensorsToLmNorm.Length < need)
                _handTensorsToLmNorm = new float[need];

            float zDenom = inputW * kLandmarksNormalizeZ;
            if (zDenom < 1e-8f)
                zDenom = 1f;
            using (var reshaped = tensor.reshape(1, HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT))
            {
                float[] arr = _handTensorsToLmNorm;
                reshaped.get(0, 0, arr.AsSpan(0, need));
                for (int i = 0; i < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT; i++)
                {
                    arr[i * 3 + 0] /= inputW;
                    arr[i * 3 + 1] /= inputH;
                    arr[i * 3 + 2] /= zDenom;
                }
                return arr;
            }
        }

        /// <summary>
        /// Equivalent to the second <c>TensorsToLandmarksCalculator</c>.
        /// Decodes the world tensor into a meter-space landmarks array,
        /// corresponding to <c>normalize=false</c>.
        /// </summary>
        float[] TensorsToLandmarksCalculator_WorldLandmarks(Mat tensor)
        {
            int need = HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT;
            if (_handTensorsToLmWorld == null || _handTensorsToLmWorld.Length < need)
                _handTensorsToLmWorld = new float[need];

            using (var reshaped = tensor.reshape(1, HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT))
            {
                float[] arr = _handTensorsToLmWorld;
                reshaped.get(0, 0, arr.AsSpan(0, need));
                return arr;
            }
        }

        /// <summary>
        /// Equivalent to <c>TensorsToFloatsCalculator</c>.
        /// Converts the hand-flag tensor into a hand presence score (<c>FLOAT</c>).
        /// </summary>
        static float TensorsToFloatsCalculator_HandPresence(Mat handFlagTensors)
        {
            return handFlagTensors.at<float>(0, 0)[0];
        }

        /// <summary>
        /// Equivalent to <c>ThresholdingCalculator</c>.
        /// Binarizes the presence score using <c>min_detection_confidence</c> (<c>minHandPresenceConfidence</c>).
        /// </summary>
        bool ThresholdingCalculator_HandPresence(float handPresenceScore)
        {
            return handPresenceScore >= _minHandPresenceConfidence;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToClassificationCalculator</c> with
        /// <c>binary_classification</c> and <c>top_k=1</c>.
        /// As in <c>tensors_to_classification_calculator.cc</c>, this forms <c>(s, 1-s)</c>
        /// and returns the winner's <c>score()</c> as a signed value via <see cref="PackHandednessBinaryTop1"/>.
        /// </summary>
        static float TensorsToClassificationCalculator_Handedness(Mat handednessTensors)
        {
            return PackHandednessBinaryTop1(handednessTensors.at<float>(0, 0)[0]);
        }

        /// <summary>
        /// Equivalent to an <c>AllowIf</c> gate.
        /// Passes HANDEDNESS only when PRESENCE is true; otherwise returns 0.
        /// </summary>
        static float AllowIf_ClassificationListByHandPresence(bool handPresence, float handednessWhenPresent)
        {
            return handPresence ? handednessWhenPresent : 0f;
        }

        /// <summary>
        /// Equivalent to <c>LandmarkLetterboxRemovalCalculator</c>.
        /// Adjusts normalized landmarks from the letterboxed input into coordinates after letterbox removal.
        /// When padding is zero, this is the identity transform.
        /// z is divided by <c>left + right</c> exactly as in <c>landmark_letterbox_removal_calculator.cc</c>.
        /// </summary>
        float[] LandmarkLetterboxRemovalCalculator(float[] normLandmarks, float padTop, float padLeft, float padBottom, float padRight)
        {
            if (padTop == 0f && padLeft == 0f && padBottom == 0f && padRight == 0f)
                return normLandmarks;
            float h = 1f - padTop - padBottom;
            float w = 1f - padLeft - padRight;
            if (h <= 1e-6f || w <= 1e-6f)
                return normLandmarks;
            float[] o = _handLetterboxRemovedNormScratch;
            for (int i = 0; i < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT; i++)
            {
                o[i * 3 + 0] = (normLandmarks[i * 3 + 0] - padLeft) / w;
                o[i * 3 + 1] = (normLandmarks[i * 3 + 1] - padTop) / h;
                o[i * 3 + 2] = normLandmarks[i * 3 + 2] / w;
            }
            return o;
        }

        /// <summary>
        /// Equivalent to <c>LandmarkProjectionCalculator</c>.
        /// Matches the <c>landmark_projection_calculator.cc</c> path where only <c>NORM_RECT</c> is connected
        /// and <c>IMAGE_DIMENSIONS</c> is absent.
        /// Output landmarks are normalized to the full image (<c>NormalizedLandmark</c>),
        /// with <c>new_z = landmark.z() * input_rect.width()</c>.
        /// </summary>
        static Vec3f[] LandmarkProjectionCalculator(float[] normLandmarksAfterLetterbox, NormalizedRect handRect)
        {
            float angle = handRect.Rotation;
            float ca = (float)Math.Cos(angle);
            float sa = (float)Math.Sin(angle);
            float cx = handRect.XCenter;
            float cy = handRect.YCenter;
            float nw = handRect.Width;
            float nh = handRect.Height;
            var projected = new Vec3f[HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT];
            for (int i = 0; i < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT; i++)
            {
                float x = normLandmarksAfterLetterbox[i * 3 + 0] - 0.5f;
                float y = normLandmarksAfterLetterbox[i * 3 + 1] - 0.5f;
                float z = normLandmarksAfterLetterbox[i * 3 + 2];
                float nx = ca * x - sa * y;
                float ny = sa * x + ca * y;
                float nxf = nx * nw + cx;
                float nyf = ny * nh + cy;
                float nzf = z * nw;
                projected[i] = new Vec3f(nxf, nyf, nzf);
            }
            return projected;
        }

        /// <summary>
        /// Equivalent to <c>WorldLandmarkProjectionCalculator</c>.
        /// Applies the rotation from <c>NORM_RECT</c> to world-space X and Y,
        /// matching <c>world_landmark_projection_calculator.cc</c>.
        /// </summary>
        static Vec3f[] WorldLandmarkProjectionCalculator(float[] worldLandmarksRaw, NormalizedRect handRect)
        {
            float ca = (float)Math.Cos(handRect.Rotation);
            float sa = (float)Math.Sin(handRect.Rotation);
            var v = new Vec3f[HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT];
            for (int i = 0; i < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT; i++)
            {
                int k = i * 3;
                float x = worldLandmarksRaw[k];
                float y = worldLandmarksRaw[k + 1];
                float z = worldLandmarksRaw[k + 2];
                v[i] = new Vec3f(ca * x - sa * y, sa * x + ca * y, z);
            }
            return v;
        }

        /// <summary>
        /// Equivalent to <c>AllowIf</c>.
        /// Returns projected LANDMARKS only when PRESENCE is true; otherwise returns an empty-array equivalent.
        /// </summary>
        static Vec3f[] AllowIf_NormalizedLandmarkListByHandPresence(bool handPresence, Vec3f[] landmarksWhenPresent)
        {
            if (!handPresence || landmarksWhenPresent == null)
                return new Vec3f[HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT];
            return landmarksWhenPresent;
        }

        /// <summary>
        /// Equivalent to <c>AllowIf</c>.
        /// Returns WORLD_LANDMARKS only when PRESENCE is true; otherwise returns an empty-array equivalent.
        /// </summary>
        static Vec3f[] AllowIf_LandmarkListByHandPresence(bool handPresence, Vec3f[] worldWhenPresent)
        {
            if (!handPresence || worldWhenPresent == null)
                return new Vec3f[HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT];
            return worldWhenPresent;
        }

        /// <summary>
        /// Equivalent to <c>HandLandmarksToRectCalculator</c>,
        /// specifically <c>NormalizedLandmarkListToRect</c> in <c>hand_landmarks_to_rect_calculator.cc</c>.
        /// Input landmarks are projected <b>normalized</b> landmarks (<c>NORM_LANDMARKS</c>).
        /// For 21 landmarks, the subset {0,1,2,3,5,6,9,10,13,14,17,18}
        /// is used in the same way as the original <c>GetPartialLandmarks</c>.
        /// </summary>
        static NormalizedRect HandLandmarksToRectCalculator(Vec3f[] normLandmarks, int imgW, int imgH)
        {
            if (normLandmarks == null || normLandmarks.Length < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT || imgW <= 0 || imgH <= 0)
                return new NormalizedRect();

            int[] partialIndices = { 0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18 };
            const int kWrist = 0;
            const int kIndexFingerMcp = 5;
            const int kMiddleFingerMcp = 9;
            const int kRingFingerMcp = 13;
            const float kTargetAngle = (float)(Math.PI * 0.5);
            const float twoPi = (float)(2.0 * Math.PI);

            float NormalizeRadians(float angle) =>
                angle - twoPi * (float)Math.Floor((angle + (float)Math.PI) / twoPi);

            // ComputeRotation: convert normalized coordinates into pixels via normalized_coordinate * image_size.
            float x0 = normLandmarks[kWrist].Item1 * imgW;
            float y0 = normLandmarks[kWrist].Item2 * imgH;
            float x1 = (normLandmarks[kIndexFingerMcp].Item1 + normLandmarks[kRingFingerMcp].Item1) * 0.5f;
            float y1 = (normLandmarks[kIndexFingerMcp].Item2 + normLandmarks[kRingFingerMcp].Item2) * 0.5f;
            x1 = (x1 + normLandmarks[kMiddleFingerMcp].Item1) * 0.5f * imgW;
            y1 = (y1 + normLandmarks[kMiddleFingerMcp].Item2) * 0.5f * imgH;
            float rotation = NormalizeRadians(kTargetAngle - (float)Math.Atan2(-(y1 - y0), x1 - x0));
            float reverseAngle = NormalizeRadians(-rotation);
            float cosR = (float)Math.Cos(rotation);
            float sinR = (float)Math.Sin(rotation);
            float cosRev = (float)Math.Cos(reverseAngle);
            float sinRev = (float)Math.Sin(reverseAngle);

            float minAx = float.MaxValue, minAy = float.MaxValue, maxAx = float.MinValue, maxAy = float.MinValue;
            foreach (int i in partialIndices)
            {
                float px = normLandmarks[i].Item1;
                float py = normLandmarks[i].Item2;
                if (px < minAx) minAx = px;
                if (py < minAy) minAy = py;
                if (px > maxAx) maxAx = px;
                if (py > maxAy) maxAy = py;
            }
            float axisAlignedCenterX = (minAx + maxAx) * 0.5f;
            float axisAlignedCenterY = (minAy + maxAy) * 0.5f;

            float minPx = float.MaxValue, minPy = float.MaxValue, maxPx = float.MinValue, maxPy = float.MinValue;
            foreach (int i in partialIndices)
            {
                float originalX = (normLandmarks[i].Item1 - axisAlignedCenterX) * imgW;
                float originalY = (normLandmarks[i].Item2 - axisAlignedCenterY) * imgH;
                float projX = originalX * cosRev - originalY * sinRev;
                float projY = originalX * sinRev + originalY * cosRev;
                if (projX < minPx) minPx = projX;
                if (projY < minPy) minPy = projY;
                if (projX > maxPx) maxPx = projX;
                if (projY > maxPy) maxPy = projY;
            }
            float projectedCenterX = (minPx + maxPx) * 0.5f;
            float projectedCenterY = (minPy + maxPy) * 0.5f;
            float centerXPixels = projectedCenterX * cosR - projectedCenterY * sinR + imgW * axisAlignedCenterX;
            float centerYPixels = projectedCenterX * sinR + projectedCenterY * cosR + imgH * axisAlignedCenterY;
            float widthNorm = (maxPx - minPx) / imgW;
            float heightNorm = (maxPy - minPy) / imgH;

            return new NormalizedRect
            {
                XCenter = centerXPixels / imgW,
                YCenter = centerYPixels / imgH,
                Width = widthNorm,
                Height = heightNorm,
                Rotation = rotation
            };
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> with only <c>NORM_RECT + IMAGE_SIZE</c>,
        /// using the same options as <c>hand_landmark_landmarks_to_roi.pbtxt</c>.
        /// As in the original <c>TransformNormalizedRect</c>, no clamping to [0,1] is performed.
        /// </summary>
        static NormalizedRect RectTransformationCalculator_SingleHandLandmarks(
            NormalizedRect handLandmarksToRect, int imgW, int imgH)
        {
            if (imgW <= 0 || imgH <= 0)
                return new NormalizedRect();

            float rotation = handLandmarksToRect.Rotation;
            float cosR = (float)Math.Cos(rotation);
            float sinR = (float)Math.Sin(rotation);

            float widthPx = handLandmarksToRect.Width * imgW;
            float heightPx = handLandmarksToRect.Height * imgH;
            float xCenterNorm = handLandmarksToRect.XCenter;
            float yCenterNorm = handLandmarksToRect.YCenter;
            float widthNorm = handLandmarksToRect.Width;
            float heightNorm = handLandmarksToRect.Height;

            const float shiftX = 0f, shiftY = -0.1f, scaleX = 2.0f, scaleY = 2.0f;
            float xShiftNorm = (imgW * widthNorm * shiftX * cosR - imgH * heightNorm * shiftY * sinR) / imgW;
            float yShiftNorm = (imgW * widthNorm * shiftX * sinR + imgH * heightNorm * shiftY * cosR) / imgH;
            xCenterNorm += xShiftNorm;
            yCenterNorm += yShiftNorm;

            float longSidePx = Mathf.Max(widthPx, heightPx);
            widthNorm = longSidePx / imgW;
            heightNorm = longSidePx / imgH;
            widthNorm *= scaleX;
            heightNorm *= scaleY;

            return new NormalizedRect
            {
                XCenter = xCenterNorm,
                YCenter = yCenterNorm,
                Width = widthNorm,
                Height = heightNorm,
                Rotation = rotation
            };
        }

        /// <summary>
        /// Equivalent to <c>AllowIf</c>.
        /// Returns HAND_RECT_NEXT_FRAME only when PRESENCE is true; otherwise returns an empty <see cref="NormalizedRect"/>.
        /// </summary>
        static NormalizedRect AllowIf_NormalizedRectByHandPresence(bool handPresence, NormalizedRect rectWhenPresent)
        {
            return handPresence ? rectWhenPresent : new NormalizedRect();
        }

        /// <summary>
        /// Equivalent to <c>ImagePropertiesCalculator</c>.
        /// Returns the input image width and height, matching the <c>SIZE</c> output from
        /// <c>mediapipe/calculators/image/image_properties_calculator.cc</c>.
        /// </summary>
        (int Width, int Height) ImagePropertiesCalculator(Mat image)
        {
            if (image == null || image.empty())
                return (0, 0);
            return (image.cols(), image.rows());
        }

        /// <summary>
        /// Equivalent to <c>HandLandmarksDeduplicationCalculator</c>.
        /// Removes duplicate hands using normalized bounding-box IoU and thresholds based on
        /// wrist-to-finger-MCP distances.
        /// See <c>mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_landmarks_deduplication_calculator.cc</c>.
        /// Uses the same constants as <c>HandDuplicatesFinder</c> / <c>landmarks_utils::CalculateIOU</c>.
        /// </summary>
        List<HandResult> HandLandmarksDeduplicationCalculator(List<HandResult> hands, int imageWidth, int imageHeight)
        {
            if (hands == null || hands.Count == 0)
                return hands != null ? new List<HandResult>() : new List<HandResult>();
            if (hands.Count == 1)
                return new List<HandResult> { hands[0] };
            if (imageWidth <= 0 || imageHeight <= 0)
                return new List<HandResult>(hands);

            for (int i = 0; i < hands.Count; i++)
            {
                if (hands[i].Landmarks == null || hands[i].Landmarks.Length != HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT)
                    return new List<HandResult>(hands);
            }

            int num = hands.Count;
            if (_handDedupBaselineScratch == null || _handDedupBaselineScratch.Length < num)
                _handDedupBaselineScratch = new float[num];
            if (_handDedupBoundsScratch == null || _handDedupBoundsScratch.Length < num)
                _handDedupBoundsScratch = new DedupRectF[num];
            float[] baselineDistances = _handDedupBaselineScratch;
            DedupRectF[] bounds = _handDedupBoundsScratch;
            for (int i = 0; i < num; i++)
            {
                baselineDistances[i] = HandBaselineDistanceForDedup(hands[i].Landmarks, imageWidth, imageHeight);
                bounds[i] = CalculateNormalizedLandmarkBounds(hands[i].Landmarks, imageWidth, imageHeight);
            }

            var retained = new HashSet<int>();
            var suppressed = new HashSet<int>();
            const float kAllowedBaselineDistanceRatio = 0.2f;
            const int kNumMatchedLandmarksToSuppressHand = 10;
            const float kMinIouThresholdToSuppressHand = 0.2f;

            // start_from_the_end = false, matching the traversal order of CreateHandDuplicatesFinder(false) in the original implementation.
            for (int i = 0; i < num; i++)
            {
                float stableI = baselineDistances[i];
                bool isSuppressed = false;
                foreach (int j in retained)
                {
                    float stableJ = baselineDistances[j];
                    float distanceThreshold = Mathf.Max(stableI, stableJ) * kAllowedBaselineDistanceRatio;
                    int matched = 0;
                    for (int k = 0; k < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT; k++)
                    {
                        if (LandmarkPairDistanceInImageSpace(hands[i].Landmarks[k], hands[j].Landmarks[k], imageWidth, imageHeight) < distanceThreshold)
                            matched++;
                    }
                    float iou = DedupCalculateIoU(bounds[i], bounds[j]);
                    if (matched >= kNumMatchedLandmarksToSuppressHand && iou > kMinIouThresholdToSuppressHand)
                    {
                        isSuppressed = true;
                        break;
                    }
                }
                if (isSuppressed)
                    suppressed.Add(i);
                else
                    retained.Add(i);
            }

            if (suppressed.Count == 0)
                return new List<HandResult>(hands);

            var filtered = new List<HandResult>();
            for (int i = 0; i < num; i++)
            {
                if (!suppressed.Contains(i))
                    filtered.Add(hands[i]);
            }
            return filtered;
        }

        /// <summary>
        /// AABB used for deduplication.
        /// <c>left</c>, <c>top</c>, <c>right</c>, and <c>bottom</c> are normalized coordinates in the 0..1 range.
        /// Equivalent to <c>CalculateBound</c> in <c>hand_landmarks_deduplication_calculator.cc</c>.
        /// </summary>
        struct DedupRectF
        {
            public float Left;
            public float Top;
            public float Right;
            public float Bottom;
        }

        static float LandmarkPairDistanceInImageSpace(Vec3f a, Vec3f b, int width, int height)
        {
            float dx = a.Item1 - b.Item1;
            float dy = a.Item2 - b.Item2;
            return Mathf.Sqrt(dx * dx + dy * dy);
        }

        static float HandBaselineDistanceForDedup(Vec3f[] lm, int width, int height)
        {
            const int kWrist = 0;
            const int kIndexFingerMcp = 5;
            const int kPinkyMcp = 17;
            float d = LandmarkPairDistanceInImageSpace(lm[kWrist], lm[kIndexFingerMcp], width, height);
            d = Mathf.Max(d, LandmarkPairDistanceInImageSpace(lm[kIndexFingerMcp], lm[kPinkyMcp], width, height));
            d = Mathf.Max(d, LandmarkPairDistanceInImageSpace(lm[kPinkyMcp], lm[kWrist], width, height));
            return d;
        }

        static DedupRectF CalculateNormalizedLandmarkBounds(Vec3f[] lm, int width, int height)
        {
            float minX = float.MaxValue, minY = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;
            for (int i = 0; i < lm.Length; i++)
            {
                float nx = lm[i].Item1 / width;
                float ny = lm[i].Item2 / height;
                minX = Mathf.Min(minX, nx);
                minY = Mathf.Min(minY, ny);
                maxX = Mathf.Max(maxX, nx);
                maxY = Mathf.Max(maxY, ny);
            }
            return new DedupRectF
            {
                Left = minX,
                Top = minY,
                Right = maxX,
                Bottom = maxY
            };
        }

        /// <summary>
        /// Equivalent to <c>CalculateIOU</c> in <c>landmarks_utils.cc</c> for axis-aligned rectangles in normalized coordinates.
        /// </summary>
        static float DedupCalculateIoU(DedupRectF a, DedupRectF b)
        {
            float areaA = Mathf.Max(0f, a.Right - a.Left) * Mathf.Max(0f, a.Bottom - a.Top);
            float areaB = Mathf.Max(0f, b.Right - b.Left) * Mathf.Max(0f, b.Bottom - b.Top);
            if (areaA <= 0f || areaB <= 0f)
                return 0f;
            float il = Mathf.Max(a.Left, b.Left);
            float it = Mathf.Max(a.Top, b.Top);
            float ir = Mathf.Min(a.Right, b.Right);
            float ib = Mathf.Min(a.Bottom, b.Bottom);
            float inter = Mathf.Max(0f, ib - it) * Mathf.Max(0f, ir - il);
            return inter / (areaA + areaB - inter);
        }

        /// <summary>
        /// Equivalent to the inference subgraph inside <c>HandDetectorGraph</c>,
        /// corresponding to <c>AddInference</c> in the original <c>hand_detector_graph.cc</c>
        /// and the <c>mediapipe.tasks.core.InferenceSubgraph</c> node.
        /// The name <c>InferenceSubgraph_PalmDetection</c> is paired with
        /// <see cref="InferenceSubgraph_SingleHandLandmarks"/> for this C# port,
        /// although the original code does not define a class with the same name.
        /// Feeds the preprocessed 192x192 letterboxed image to <see cref="_palmNet"/> (OpenCV DNN or Unity Inference Engine) and
        /// returns the output tensor list (<c>TENSORS</c>). Same ownership rules as <see cref="InferenceSubgraph_SingleHandLandmarks"/> for the returned list.
        /// </summary>
        List<Mat> InferenceSubgraph_PalmDetection(Mat preprocessedImage)
        {
            const int palmH = 192;
            const int palmW = 192;
            const int palmC = 3;

            if (_palmInferenceBlob == null)
            {
                _palmInferenceBlob = new Mat(new int[] { 1, palmH, palmW, palmC }, CvType.CV_32FC1);
                _palmInferenceBlobHxW = _palmInferenceBlob.reshape(palmC, new int[] { palmH, palmW });
                _palmInferenceInput8u = new Mat((palmW, palmH), CvType.CV_8UC3);
            }

            Imgproc.cvtColor(preprocessedImage, _palmInferenceInput8u, Imgproc.COLOR_BGR2RGB);
            _palmInferenceInput8u.convertTo(_palmInferenceBlobHxW, CvType.CV_32F, 1.0 / 255.0);

            _palmForwardOutputList.Clear();
            _palmNet.setInput(_palmInferenceBlob);
            _palmNet.forward(_palmForwardOutputList, _palmNetOutLayerNames);
            return _palmForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="InferenceSubgraph_PalmDetection"/> (via <see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// Invoked only from the <see cref="RunCoreProcessingTaskAsync"/> path; OpenCV inference uses <see cref="InferenceSubgraph_PalmDetection"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_PalmDetectionAsync(Mat preprocessedImage, CancellationToken cancellationToken)
        {
            const int palmH = 192;
            const int palmW = 192;
            const int palmC = 3;

            if (_palmInferenceBlob == null)
            {
                _palmInferenceBlob = new Mat(new int[] { 1, palmH, palmW, palmC }, CvType.CV_32FC1);
                _palmInferenceBlobHxW = _palmInferenceBlob.reshape(palmC, new int[] { palmH, palmW });
                _palmInferenceInput8u = new Mat((palmW, palmH), CvType.CV_8UC3);
            }

            Imgproc.cvtColor(preprocessedImage, _palmInferenceInput8u, Imgproc.COLOR_BGR2RGB);
            _palmInferenceInput8u.convertTo(_palmInferenceBlobHxW, CvType.CV_32F, 1.0 / 255.0);

            _palmForwardOutputList.Clear();
            _palmNet.setInput(_palmInferenceBlob);
            await _palmNet.forwardTaskAsync(_palmForwardOutputList, _palmNetOutLayerNames, cancellationToken);
            return _palmForwardOutputList;
        }

#endif
        /// <summary>
        /// Equivalent to <c>ImagePreprocessingGraph</c> inside <c>HandDetectorGraph</c>.
        /// Stores the same <c>GetRotatedSubRectToRectTransformMatrix</c> result from
        /// <c>image_to_tensor_utils.cc</c> into <see cref="_handDetectorProjectionMatrix16"/>,
        /// corresponding to the <c>PROJECTION_MATRIX</c> consumed by <c>DetectionProjectionCalculator</c>.
        /// <list type="bullet">
        /// <item><description>When <paramref name="normRect"/> is null, the entire input image is treated as the ROI and transformed into a letterboxed 192x192 image using the traditional resize-plus-padding path.</description></item>
        /// <item><description>When <paramref name="normRect"/> is provided, the rotated rectangle after GetRoi -> PadRoi is projected into 192x192 via <c>warpPerspective</c>, matching <c>image_to_tensor_converter_opencv.cc</c> with <c>INTER_LINEAR</c> and <c>BORDER_CONSTANT</c>.</description></item>
        /// </list>
        /// Even though current calls use only <c>HandDetectorGraph(..., null)</c>,
        /// passing a non-null value from internal code or tests still matches the original CPU path.
        /// </summary>
        void ImagePreprocessingGraph(Mat image, Mat maxSizeImg, NormalizedRect? normRect = null)
        {
            int origW = image.cols();
            int origH = image.rows();
            HandDetectorGetRoi(origW, origH, normRect, out float roiCx, out float roiCy, out float roiW, out float roiH, out float roiRot);
            HandDetectorPadRoi(192, 192, true, ref roiW, ref roiH);
            GetRotatedSubRectToRectTransformMatrix(roiCx, roiCy, roiW, roiH, roiRot, origW, origH, false, _handDetectorProjectionMatrix16);

            if (normRect.HasValue)
            {
                if (roiW > 0f && roiH > 0f &&
                    !float.IsNaN(roiCx) && !float.IsNaN(roiCy) &&
                    !float.IsNaN(roiW) && !float.IsNaN(roiH) && !float.IsNaN(roiRot))
                {
                    HandDetectorEnsurePalmWarpMats192();
                    double angleDeg = roiRot * (180.0 / Math.PI);
                    Imgproc.boxPoints(new Vec5d(roiCx, roiCy, roiW, roiH, angleDeg), _handDetectorWarpSrcPts);
                    using (Mat projMat = Imgproc.getPerspectiveTransform(_handDetectorWarpSrcPts, _handDetectorWarpDstPts))
                    {
                        Imgproc.warpPerspective(image, maxSizeImg, projMat, (192, 192),
                            Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0, 0, 0, 0));
                    }
                }
                else
                {
                    maxSizeImg.setTo((0, 0, 0, 255));
                }

                return;
            }

            double ratio = Math.Min(192.0 / origW, 192.0 / origH);
            int ratioSizeW = (int)(origW * ratio);
            int ratioSizeH = (int)(origH * ratio);

            maxSizeImg.setTo((0, 0, 0, 255));

            using (var resized = new Mat())
            {
                Imgproc.resize(image, resized, (ratioSizeW, ratioSizeH));
                int padLeft = (192 - ratioSizeW) / 2;
                int padTop = (192 - ratioSizeH) / 2;
                using (var roi = new Mat(maxSizeImg, (padLeft, padTop, ratioSizeW, ratioSizeH)))
                {
                    resized.copyTo(roi);
                }
            }
        }

        /// <summary>
        /// Allocates the fixed destination corners for the palm 192x192 perspective transform
        /// (the <c>dst_corners</c> from <c>image_to_tensor_converter_opencv.cc</c>)
        /// and the input-corner buffer updated every frame.
        /// </summary>
        void HandDetectorEnsurePalmWarpMats192()
        {
            if (_handDetectorWarpDstPts != null)
                return;

            const float dw = 192f;
            const float dh = 192f;
            _handDetectorWarpDstPts = new Mat(4, 2, CvType.CV_32FC1);
            Span<float> dstPtsArr = stackalloc float[8];
            dstPtsArr[0] = 0f;
            dstPtsArr[1] = dh;
            dstPtsArr[2] = 0f;
            dstPtsArr[3] = 0f;
            dstPtsArr[4] = dw;
            dstPtsArr[5] = 0f;
            dstPtsArr[6] = dw;
            dstPtsArr[7] = dh;
            _handDetectorWarpDstPts.put(0, 0, dstPtsArr);
            _handDetectorWarpSrcPts = new Mat(4, 2, CvType.CV_32FC1);
        }

        /// <summary>
        /// Equivalent to <c>GetRoi</c> in <c>image_to_tensor_utils.cc</c>, returning a rotated rectangle in pixel units.
        /// </summary>
        static void HandDetectorGetRoi(int inputWidth, int inputHeight, NormalizedRect? normRect, out float centerX, out float centerY, out float width, out float height, out float rotation)
        {
            if (normRect.HasValue)
            {
                var n = normRect.Value;
                centerX = n.XCenter * inputWidth;
                centerY = n.YCenter * inputHeight;
                width = n.Width * inputWidth;
                height = n.Height * inputHeight;
                rotation = n.Rotation;
            }
            else
            {
                centerX = 0.5f * inputWidth;
                centerY = 0.5f * inputHeight;
                width = inputWidth;
                height = inputHeight;
                rotation = 0f;
            }
        }

        /// <summary>
        /// Equivalent to <c>PadRoi</c> in <c>image_to_tensor_utils.cc</c>.
        /// When <c>keep_aspect_ratio</c> is enabled, expands the ROI to match the tensor aspect ratio and overwrites width and height.
        /// </summary>
        static void HandDetectorPadRoi(int inputTensorWidth, int inputTensorHeight, bool keepAspectRatio, ref float roiWidth, ref float roiHeight)
        {
            if (!keepAspectRatio)
                return;

            float tensorAspectRatio = (float)inputTensorHeight / inputTensorWidth;
            float roiAspectRatio = roiHeight / roiWidth;

            if (tensorAspectRatio > roiAspectRatio)
            {
                float newWidth = roiWidth;
                float newHeight = roiWidth * tensorAspectRatio;
                roiWidth = newWidth;
                roiHeight = newHeight;
            }
            else
            {
                float newWidth = roiHeight / tensorAspectRatio;
                float newHeight = roiHeight;
                roiWidth = newWidth;
                roiHeight = newHeight;
            }
        }

        /// <summary>
        /// Uses the same formula as <c>GetRotatedSubRectToRectTransformMatrix</c> in <c>image_to_tensor_utils.cc</c> (row-major 4x4).
        /// This matrix maps normalized coordinates on the tensor (0..1) to normalized coordinates on the input image (0..1),
        /// i.e. from the sub-rectangle back to the full image.
        /// </summary>
        static void GetRotatedSubRectToRectTransformMatrix(
            float centerX,
            float centerY,
            float subWidth,
            float subHeight,
            float rotation,
            int rectWidth,
            int rectHeight,
            bool flipHorizontally,
            float[] matrix16)
        {
            float a = subWidth;
            float b = subHeight;
            float flip = flipHorizontally ? -1f : 1f;
            float c = Mathf.Cos(rotation);
            float d = Mathf.Sin(rotation);
            float e = centerX;
            float f = centerY;
            float g = 1f / rectWidth;
            float h = 1f / rectHeight;

            matrix16[0] = a * c * flip * g;
            matrix16[1] = -b * d * g;
            matrix16[2] = 0f;
            matrix16[3] = (-0.5f * a * c * flip + 0.5f * b * d + e) * g;

            matrix16[4] = a * d * flip * h;
            matrix16[5] = b * c * h;
            matrix16[6] = 0f;
            matrix16[7] = (-0.5f * b * c - 0.5f * a * d * flip + f) * h;

            matrix16[8] = 0f;
            matrix16[9] = 0f;
            matrix16[10] = a * g;
            matrix16[11] = 0f;

            matrix16[12] = 0f;
            matrix16[13] = 0f;
            matrix16[14] = 0f;
            matrix16[15] = 1f;
        }

        /// <summary>
        /// Equivalent to the <c>min_score_thresh</c> step in
        /// <c>TensorsToDetectionsCalculator::ConvertToDetection</c>
        /// (task-level <c>min_detection_confidence</c>).
        /// Returns matrices for NMS after removing rows whose score is below the threshold.
        /// If the threshold is non-positive, or the filtered row count is unchanged,
        /// the input buffers are referenced directly.
        /// </summary>
        void PalmDetectionsFilterByMinScoreThresh(
            Mat boxXywh,
            Mat scoreNx1,
            Mat boxAndLandmarkNx18,
            float minScoreThresh,
            out Mat boxOut,
            out Mat scoreOut,
            out Mat lmOut)
        {
            int num = boxXywh.rows();
            if (num <= 0 || minScoreThresh <= 0f)
            {
                boxOut = boxXywh;
                scoreOut = scoreNx1;
                lmOut = boxAndLandmarkNx18;
                return;
            }

            int kept = 0;
            for (int i = 0; i < num; i++)
            {
                if (scoreNx1.at<float>(i, 0)[0] >= minScoreThresh)
                    kept++;
            }

            if (kept == num)
            {
                boxOut = boxXywh;
                scoreOut = scoreNx1;
                lmOut = boxAndLandmarkNx18;
                return;
            }

            if (_palmScoreFilteredBoxXywh == null)
                _palmScoreFilteredBoxXywh = new Mat();
            if (_palmScoreFilteredScore == null)
                _palmScoreFilteredScore = new Mat();
            if (_palmScoreFilteredLm18 == null)
                _palmScoreFilteredLm18 = new Mat();

            _palmScoreFilteredBoxXywh.create(kept, 4, CvType.CV_32FC1);
            _palmScoreFilteredScore.create(kept, 1, CvType.CV_32FC1);
            _palmScoreFilteredLm18.create(kept, 18, CvType.CV_32FC1);

            int r = 0;
            for (int i = 0; i < num; i++)
            {
                if (scoreNx1.at<float>(i, 0)[0] < minScoreThresh)
                    continue;
                using (var srcRow = boxXywh.row(i))
                using (var dstRow = _palmScoreFilteredBoxXywh.row(r))
                    srcRow.copyTo(dstRow);
                using (var srcRow = scoreNx1.row(i))
                using (var dstRow = _palmScoreFilteredScore.row(r))
                    srcRow.copyTo(dstRow);
                using (var srcRow = boxAndLandmarkNx18.row(i))
                using (var dstRow = _palmScoreFilteredLm18.row(r))
                    srcRow.copyTo(dstRow);
                r++;
            }

            boxOut = _palmScoreFilteredBoxXywh;
            scoreOut = _palmScoreFilteredScore;
            lmOut = _palmScoreFilteredLm18;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToDetectionsCalculator</c>.
        /// Builds the pre-NMS bbox matrix <c>boxXywh</c> in preprocessing coordinates from
        /// <c>TENSORS</c> (model outputs) and <c>ANCHORS</c>.
        /// As in the original implementation, score thresholding is applied before NMS via
        /// <see cref="PalmDetectionsFilterByMinScoreThresh"/>.
        /// <paramref name="output0"/> and <paramref name="output1"/> are updated in place.
        /// <c>boxXywh</c> references a worker-owned buffer and must not be disposed by the caller.
        /// </summary>
        void TensorsToDetectionsCalculator(Mat output0, Mat output1, Mat anchors, Mat anchorsNx14, out Mat boxXywh)
        {
            int num = output0.size(1);
            if (_tensorsToDetectionsBoxXywh == null)
                _tensorsToDetectionsBoxXywh = new Mat();
            _tensorsToDetectionsBoxXywh.create(num, 4, CvType.CV_32FC1);
            boxXywh = _tensorsToDetectionsBoxXywh;
            using (var score = output1.reshape(1, num))
            using (var boxAndLandmark = output0.reshape(1, num))
            {
                Core.multiply(score, (-1.0, 0, 0, 0), score);
                Core.exp(score, score);
                Core.add(score, (1.0, 0, 0, 0), score);
                Core.divide(1.0, score, score);

                using (var boxAndLandmarkNx1c2 = boxAndLandmark.reshape(2, num))
                {
                    Core.divide(boxAndLandmarkNx1c2, (192.0, 192.0, 0, 0), boxAndLandmarkNx1c2);
                }

                using (var cxy = boxAndLandmark.colRange(0, 2))
                {
                    Core.add(cxy, anchors, cxy);
                }

                using (var lm = boxAndLandmark.colRange(4, 18))
                {
                    Core.add(lm, anchorsNx14, lm);
                }

                using (var cxy2 = boxAndLandmark.colRange(0, 2))
                using (var wh2 = boxAndLandmark.colRange(2, 4))
                using (var dstXy = boxXywh.colRange(0, 2))
                using (var dstWh = boxXywh.colRange(2, 4))
                {
                    cxy2.copyTo(dstWh);
                    Core.divide(wh2, (2.0, 0, 0, 0), dstXy);
                    Core.subtract(dstWh, dstXy, cxy2);
                    Core.add(dstWh, dstXy, wh2);

                    cxy2.copyTo(dstXy);
                    Core.subtract(wh2, cxy2, dstWh);
                }
            }
        }

        /// <summary>
        /// Equivalent to <c>DetectionLabelIdToTextCalculator</c>.
        /// This is the stage that would assign the label "Palm" to palm detections.
        /// In this C# implementation it is effectively a no-op because detections are represented as <c>float[]</c> rows.
        /// </summary>
        static void DetectionLabelIdToTextCalculator()
        {
        }

        /// <summary>
        /// Equivalent to <c>SsdAnchorsCalculator</c>.
        /// Builds the SSD anchor matrix for palm detection according to
        /// <c>ConfigureSsdAnchorsCalculator</c> in <c>hand_detector_graph.cc</c> and
        /// <c>GenerateAnchors</c> in <c>ssd_anchors_calculator.cc</c>,
        /// and also prepares the 7x-expanded matrix used for landmarks.
        /// </summary>
        void SsdAnchorsCalculator(int num, out Mat anchors, out Mat anchorsNx14)
        {
            _anchors ??= BuildPalmAnchors();

            // Reuse the landmark matrix as well (the 7x-expanded variant) inside the worker.
            if (_anchorsNx14 == null)
            {
                _anchorsNx14 = new Mat();
                Core.repeat(_anchors, 1, 7, _anchorsNx14);
            }

            anchors = _anchors;
            anchorsNx14 = _anchorsNx14;
        }

        /// <summary>
        /// Equivalent to <c>NonMaxSuppressionCalculator</c>, specifically
        /// <c>WeightedNonMaxSuppression</c> in the original <c>non_max_suppression_calculator.cc</c>.
        /// </summary>
        /// <remarks>
        /// In <c>hand_detector_graph.cc</c>, <c>ConfigureNonMaxSuppressionCalculator</c> uses
        /// <c>min_suppression_threshold=0.3</c>, <c>overlap_type=INTERSECTION_OVER_UNION</c>,
        /// and <c>algorithm=WEIGHTED</c>.
        /// The original <c>min_score_threshold</c> NMS option is disabled by default,
        /// so this method assumes score thresholding has already been applied upstream by
        /// <see cref="PalmDetectionsFilterByMinScoreThresh"/>,
        /// corresponding to <c>TensorsToDetectionsCalculatorOptions.min_score_thresh</c>.
        /// The merged tensor rows are written into
        /// <see cref="_handWnmsMergedBoxXywh"/>, <see cref="_handWnmsMergedLm18"/>, and <see cref="_handWnmsMergedScore"/>.
        /// The returned <see cref="_nmsIndices"/> contains <c>0 .. K-1</c> because
        /// <see cref="DetectionProjectionCalculator"/> references rows by these indices.
        /// </remarks>
        MatOfInt NonMaxSuppressionCalculator(Mat boxXywh, Mat score, Mat boxAndLandmarkNx18)
        {
            const float kHandMinSuppressionThreshold = 0.3f;

            if (_nmsIndices == null)
                _nmsIndices = new MatOfInt();
            if (_handWnmsMergedBoxXywh == null)
                _handWnmsMergedBoxXywh = new Mat();
            if (_handWnmsMergedLm18 == null)
                _handWnmsMergedLm18 = new Mat();
            if (_handWnmsMergedScore == null)
                _handWnmsMergedScore = new Mat();

            int num = boxXywh.rows();
            if (num <= 0 || score == null || score.rows() < num || boxAndLandmarkNx18 == null || boxAndLandmarkNx18.rows() < num)
            {
                _handWnmsMergedBoxXywh.create(0, 4, CvType.CV_32FC1);
                _handWnmsMergedLm18.create(0, 18, CvType.CV_32FC1);
                _handWnmsMergedScore.create(0, 1, CvType.CV_32FC1);
                _nmsIndices.create(0, 1, CvType.CV_32SC1);
                return _nmsIndices;
            }

            _handWnmsIndexed.Clear();
            for (int i = 0; i < num; i++)
                _handWnmsIndexed.Add((i, score.at<float>(i, 0)[0]));
            _handWnmsIndexed.Sort((a, b) => b.sc.CompareTo(a.sc));

            _handWnmsRemained.Clear();
            _handWnmsRemained.AddRange(_handWnmsIndexed);

            _handNmsMergedBoxScratch.Clear();
            _handNmsMergedLmScratch.Clear();
            _handNmsMergedScScratch.Clear();

            if (_palmWnmsKpAccumulator14 == null || _palmWnmsKpAccumulator14.Length < 14)
                _palmWnmsKpAccumulator14 = new float[14];
            _palmNmsRowBuf18 ??= new float[18];

            float[] rowBuf = _palmNmsRowBuf18;
            while (_handWnmsRemained.Count > 0)
            {
                int originalSize = _handWnmsRemained.Count;
                var anchor = _handWnmsRemained[0];

                float ax = boxXywh.at<float>(anchor.idx, 0)[0];
                float ay = boxXywh.at<float>(anchor.idx, 1)[0];
                float aw = boxXywh.at<float>(anchor.idx, 2)[0];
                float ah = boxXywh.at<float>(anchor.idx, 3)[0];

                _handWnmsNextRemained.Clear();
                for (int t = 0; t < _handWnmsRemained.Count; t++)
                {
                    var item = _handWnmsRemained[t];
                    float bx = boxXywh.at<float>(item.idx, 0)[0];
                    float by = boxXywh.at<float>(item.idx, 1)[0];
                    float bw = boxXywh.at<float>(item.idx, 2)[0];
                    float bh = boxXywh.at<float>(item.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) > kHandMinSuppressionThreshold)
                        continue;
                    _handWnmsNextRemained.Add(item);
                }

                float wXmin = 0f, wYmin = 0f, wXmax = 0f, wYmax = 0f;
                float totalScore = 0f;
                float[] kpAcc = _palmWnmsKpAccumulator14;
                Array.Clear(kpAcc, 0, 14);
                for (int t = 0; t < _handWnmsRemained.Count; t++)
                {
                    var c = _handWnmsRemained[t];
                    float bx = boxXywh.at<float>(c.idx, 0)[0];
                    float by = boxXywh.at<float>(c.idx, 1)[0];
                    float bw = boxXywh.at<float>(c.idx, 2)[0];
                    float bh = boxXywh.at<float>(c.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) <= kHandMinSuppressionThreshold)
                        continue;

                    float s = c.sc;
                    totalScore += s;
                    wXmin += bx * s;
                    wYmin += by * s;
                    wXmax += (bx + bw) * s;
                    wYmax += (by + bh) * s;
                    boxAndLandmarkNx18.get(c.idx, 0, rowBuf.AsSpan(0, 18));
                    for (int k = 0; k < 14; k++)
                        kpAcc[k] += rowBuf[4 + k] * s;
                }

                if (totalScore <= 0f)
                    break;

                float outXmin = wXmin / totalScore;
                float outYmin = wYmin / totalScore;
                float outW = wXmax / totalScore - outXmin;
                float outH = wYmax / totalScore - outYmin;

                float[] lmRow = RentHandPalmNmsLm18();
                lmRow[0] = outXmin;
                lmRow[1] = outYmin;
                lmRow[2] = outXmin + outW;
                lmRow[3] = outYmin + outH;
                for (int k = 0; k < 14; k++)
                    lmRow[4 + k] = kpAcc[k] / totalScore;

                float[] box4 = RentHandPalmNmsBox4();
                box4[0] = outXmin;
                box4[1] = outYmin;
                box4[2] = outW;
                box4[3] = outH;
                _handNmsMergedBoxScratch.Add(box4);
                _handNmsMergedLmScratch.Add(lmRow);
                _handNmsMergedScScratch.Add(anchor.sc);

                if (originalSize == _handWnmsNextRemained.Count)
                    break;

                (_handWnmsRemained, _handWnmsNextRemained) = (_handWnmsNextRemained, _handWnmsRemained);
            }

            int kOut = _handNmsMergedScScratch.Count;
            _handWnmsMergedBoxXywh.create(kOut, 4, CvType.CV_32FC1);
            _handWnmsMergedLm18.create(kOut, 18, CvType.CV_32FC1);
            _handWnmsMergedScore.create(kOut, 1, CvType.CV_32FC1);
            Span<float> putScore1 = stackalloc float[1];
            Span<int> putIdx1 = stackalloc int[1];
            for (int r = 0; r < kOut; r++)
            {
                _handWnmsMergedBoxXywh.put(r, 0, _handNmsMergedBoxScratch[r].AsSpan(0, 4));
                _handWnmsMergedLm18.put(r, 0, _handNmsMergedLmScratch[r].AsSpan(0, 18));
                putScore1[0] = _handNmsMergedScScratch[r];
                _handWnmsMergedScore.put(r, 0, putScore1);
            }

            _nmsIndices.create(kOut, 1, CvType.CV_32SC1);
            for (int r = 0; r < kOut; r++)
            {
                putIdx1[0] = r;
                _nmsIndices.put(r, 0, putIdx1);
            }

            ReleaseHandPalmNmsMergedScratchLists();

            return _nmsIndices;
        }

        /// <summary>
        /// IoU for axis-aligned rectangles (<c>xmin, ymin, width, height</c>).
        /// Equivalent to the original <c>OverlapSimilarity(..., INTERSECTION_OVER_UNION)</c>.
        /// </summary>
        static float NonMaxSuppressionCalculator_ComputeIouXywh(
            float ax, float ay, float aw, float ah,
            float bx, float by, float bw, float bh)
        {
            float ax2 = ax + aw;
            float ay2 = ay + ah;
            float bx2 = bx + bw;
            float by2 = by + bh;

            float ix1 = Mathf.Max(ax, bx);
            float iy1 = Mathf.Max(ay, by);
            float ix2 = Mathf.Min(ax2, bx2);
            float iy2 = Mathf.Min(ay2, by2);

            if (ix2 <= ix1 || iy2 <= iy1)
                return 0f;

            float intersection = (ix2 - ix1) * (iy2 - iy1);
            float areaA = aw * ah;
            float areaB = bw * bh;
            float union = areaA + areaB - intersection;
            return union > 0f ? intersection / union : 0f;
        }

        /// <summary>
        /// Equivalent to <c>DetectionProjectionCalculator</c>,
        /// specifically <c>ProjectDetection</c> in <c>detection_projection_calculator.cc</c>.
        /// Uses <c>PROJECTION_MATRIX</c> (the MATRIX from ImageToTensor) to map tensor-normalized coordinates
        /// into normalized input-image coordinates.
        /// The bbox becomes the axis-aligned envelope of the four projected corners,
        /// and keypoints are projected point by point and then converted to pixel coordinates.
        /// </summary>
        void DetectionProjectionCalculator(
            Mat boxXywh,
            Mat score,
            Mat boxAndLandmark,
            MatOfInt indices,
            float[] projectionMatrix16,
            int origW,
            int origH,
            List<float[]> detections)
        {
            if (indices.empty() || projectionMatrix16 == null || projectionMatrix16.Length < 16)
                return;

            ReadOnlySpan<float> m = projectionMatrix16;
            int selected = indices.rows();
            Span<float> box = stackalloc float[4];
            float[] allBuf = _palmNmsRowBuf18 ??= new float[18];
            Span<float> all = allBuf.AsSpan(0, 18);
            Span<float> dst = stackalloc float[PalmDetectionRowElementCount];
            for (int i = 0; i < selected; i++)
            {
                int idx = indices.at<int>(i, 0)[0];

                boxXywh.get(idx, 0, box);
                float xmin = box[0];
                float ymin = box[1];
                float wBox = box[2];
                float hBox = box[3];

                // RELATIVE_BOUNDING_BOX: project the four corners and take the AABB exactly as in detection_projection_calculator.cc.
                float minNx = float.MaxValue;
                float minNy = float.MaxValue;
                float maxNx = float.MinValue;
                float maxNy = float.MinValue;
                DetectionProjection_ProjectTensorNormalized(m, xmin, ymin, out float p0x, out float p0y);
                DetectionProjection_ProjectTensorNormalized(m, xmin + wBox, ymin, out float p1x, out float p1y);
                DetectionProjection_ProjectTensorNormalized(m, xmin + wBox, ymin + hBox, out float p2x, out float p2y);
                DetectionProjection_ProjectTensorNormalized(m, xmin, ymin + hBox, out float p3x, out float p3y);
                minNx = Mathf.Min(Mathf.Min(p0x, p1x), Mathf.Min(p2x, p3x));
                minNy = Mathf.Min(Mathf.Min(p0y, p1y), Mathf.Min(p2y, p3y));
                maxNx = Mathf.Max(Mathf.Max(p0x, p1x), Mathf.Max(p2x, p3x));
                maxNy = Mathf.Max(Mathf.Max(p0y, p1y), Mathf.Max(p2y, p3y));

                float x1 = minNx * origW;
                float y1 = minNy * origH;
                float x2 = maxNx * origW;
                float y2 = maxNy * origH;

                boxAndLandmark.get(idx, 0, all);

                dst[0] = x1;
                dst[1] = y1;
                dst[2] = x2;
                dst[3] = y2;
                for (int j = 0; j < 14; j += 2)
                {
                    DetectionProjection_ProjectTensorNormalized(m, all[4 + j], all[4 + j + 1], out float nx, out float ny);
                    dst[4 + j] = nx * origW;
                    dst[4 + j + 1] = ny * origH;
                }
                dst[18] = score.at<float>(idx, 0)[0];

                float[] row = RentHandDetectionProjRow19();
                dst.CopyTo(row);
                detections.Add(row);
            }
        }

        /// <summary>
        /// Affine projection from <c>detection_projection_calculator.cc</c>.
        /// Uses only the first two rows of the 4x4 matrix.
        /// </summary>
        static void DetectionProjection_ProjectTensorNormalized(ReadOnlySpan<float> m, float tx, float ty, out float nx, out float ny)
        {
            nx = tx * m[0] + ty * m[1] + m[3];
            ny = tx * m[4] + ty * m[5] + m[7];
        }

        /// <summary>
        /// Generates the SSD anchor matrix for palm detection using options aligned with
        /// <c>ConfigureSsdAnchorsCalculator</c> in MediaPipe Tasks <c>hand_detector_graph.cc</c>,
        /// following the same procedure as <c>SsdAnchorsCalculator::GenerateAnchors</c> in
        /// <c>ssd_anchors_calculator.cc</c> when <c>multiscale_anchor_generation</c> is disabled.
        /// Each row stores <c>x_center</c> and <c>y_center</c> in tensor-normalized coordinates.
        /// Because <c>interpolated_scale_aspect_ratio</c> is not set in the original graph,
        /// the default value 1.0 from <c>ssd_anchors_calculator.proto</c> is applied.
        /// </summary>
        Mat BuildPalmAnchors()
        {
            // Same parameters as ConfigureSsdAnchorsCalculator with has_metadata=false.
            const int numLayers = 4;
            const float minScale = 0.1484375f;
            const float maxScale = 0.75f;
            const int inputSizeHeight = 192;
            const int inputSizeWidth = 192;
            const float anchorOffsetX = 0.5f;
            const float anchorOffsetY = 0.5f;
            const bool reduceBoxesInLowestLayer = false;
            const float interpolatedScaleAspectRatio = 1.0f;
            float[] aspectRatiosOptions = { 1.0f };
            int[] strides = { 8, 16, 16, 16 };

            int stridesLen = strides.Length;
            if (stridesLen != numLayers)
                throw new InvalidOperationException("The lengths of SSD strides and num_layers do not match.");

            var aspectRatios = new List<float>(8);
            var scales = new List<float>(8);
            var anchorHeight = new List<float>(8);
            var anchorWidth = new List<float>(8);
            const int expectedRows = 2016;
            var xy = new float[expectedRows * 2];
            int outIx = 0;

            int layerId = 0;
            while (layerId < numLayers)
            {
                aspectRatios.Clear();
                scales.Clear();
                int lastSameStrideLayer = layerId;
                while (lastSameStrideLayer < stridesLen &&
                       strides[lastSameStrideLayer] == strides[layerId])
                {
                    float scale = SsdAnchorsCalculator_CalculateScale(
                        minScale, maxScale, lastSameStrideLayer, stridesLen);
                    if (lastSameStrideLayer == 0 && reduceBoxesInLowestLayer)
                    {
                        aspectRatios.Add(1.0f);
                        aspectRatios.Add(2.0f);
                        aspectRatios.Add(0.5f);
                        scales.Add(0.1f);
                        scales.Add(scale);
                        scales.Add(scale);
                    }
                    else
                    {
                        for (int arId = 0; arId < aspectRatiosOptions.Length; arId++)
                        {
                            aspectRatios.Add(aspectRatiosOptions[arId]);
                            scales.Add(scale);
                        }
                        if (interpolatedScaleAspectRatio > 0f)
                        {
                            float scaleNext = lastSameStrideLayer == stridesLen - 1
                                ? 1.0f
                                : SsdAnchorsCalculator_CalculateScale(
                                    minScale, maxScale, lastSameStrideLayer + 1, stridesLen);
                            scales.Add(Mathf.Sqrt(scale * scaleNext));
                            aspectRatios.Add(interpolatedScaleAspectRatio);
                        }
                    }
                    lastSameStrideLayer++;
                }

                anchorHeight.Clear();
                anchorWidth.Clear();
                for (int i = 0; i < aspectRatios.Count; i++)
                {
                    float ratioSqrt = Mathf.Sqrt(aspectRatios[i]);
                    anchorHeight.Add(scales[i] / ratioSqrt);
                    anchorWidth.Add(scales[i] * ratioSqrt);
                }

                int stride = strides[layerId];
                int featureMapHeight = Mathf.CeilToInt(inputSizeHeight / (float)stride);
                int featureMapWidth = Mathf.CeilToInt(inputSizeWidth / (float)stride);

                for (int y = 0; y < featureMapHeight; y++)
                {
                    for (int x = 0; x < featureMapWidth; x++)
                    {
                        for (int anchorId = 0; anchorId < anchorHeight.Count; anchorId++)
                        {
                            float xCenter = (x + anchorOffsetX) / featureMapWidth;
                            float yCenter = (y + anchorOffsetY) / featureMapHeight;
                            xy[outIx++] = xCenter;
                            xy[outIx++] = yCenter;
                        }
                    }
                }

                layerId = lastSameStrideLayer;
            }

            if (outIx != expectedRows * 2)
                throw new InvalidOperationException(
                    $"The number of SSD anchors does not match the original implementation: expected {expectedRows}, actual {outIx / 2}.");

            Mat anchors = new Mat(expectedRows, 2, CvType.CV_32FC1);
            anchors.put(0, 0, xy.AsSpan(0, expectedRows * 2));
            return anchors;
        }

        /// <summary>
        /// Uses the same formula as <c>CalculateScale</c> in <c>ssd_anchors_calculator.cc</c>.
        /// </summary>
        static float SsdAnchorsCalculator_CalculateScale(
            float minScale, float maxScale, int strideIndex, int numStrides)
        {
            if (numStrides == 1)
                return (minScale + maxScale) * 0.5f;
            return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1.0f);
        }

        /// <summary>
        /// Equivalent to <c>DetectionsToRectsCalculator</c>,
        /// specifically <c>Process</c>, <c>DetectionToNormalizedRect</c>, and <c>ComputeRotation</c>
        /// in <c>detections_to_rects_calculator.cc</c>.
        /// Generates the <c>PALM_RECTS</c> list of rotated normalized palm rectangles
        /// from <c>DETECTIONS</c> (one row contains <see cref="PalmDetectionRowElementCount"/> elements:
        /// bbox image coordinates, 7 keypoints, and score) and the image size.
        /// <list type="bullet">
        /// <item><description>This calculator does not apply a score threshold, matching the original implementation. Thresholding is expected to have already been applied upstream by <see cref="PalmDetectionsFilterByMinScoreThresh"/>, corresponding to <c>TensorsToDetectionsCalculatorOptions.min_score_thresh</c> / task-level <c>min_detection_confidence</c>.</description></item>
        /// <item><description>Like <c>ConfigureDetectionsToRectsCalculator</c> with <c>output_zero_rect_for_empty_detections(true)</c>, when <c>DETECTIONS</c> is empty this method returns one zero <see cref="NormalizedRect"/>, provided the image size is valid.</description></item>
        /// </list>
        /// </summary>
        List<NormalizedRect> DetectionsToRectsCalculator(List<float[]> palmDetections, int imgW, int imgH)
        {
            if (imgW <= 0 || imgH <= 0)
                return new List<NormalizedRect>();

            if (palmDetections == null || palmDetections.Count == 0)
            {
                // In detections_to_rects_calculator.cc, NORM_RECTS with an empty vector produces one default NormalizedRect.
                return new List<NormalizedRect> { default };
            }

            var palmRects = new List<NormalizedRect>();
            foreach (var row in palmDetections)
            {
                if (row == null || row.Length < PalmDetectionRowElementCount)
                    continue;

                palmRects.Add(DetectionsToRectsCalculator_OneRow(row, imgW, imgH));
            }

            return palmRects;
        }

        /// <summary>
        /// Uses the same formula as <c>NormalizeRadians</c> in <c>detections_to_rects_calculator.h</c>,
        /// normalizing to the range <c>(-pi, pi]</c>.
        /// </summary>
        static float DetectionsToRectsCalculator_NormalizeRadians(float angle)
        {
            const float twoPi = 2f * Mathf.PI;
            return angle - twoPi * Mathf.Floor((angle - (-Mathf.PI)) / twoPi);
        }

        /// <summary>
        /// Converts one detection row for <c>DetectionsToRectsCalculator</c>.
        /// Applies the center, width, and height from the <c>RELATIVE_BOUNDING_BOX</c> equivalent,
        /// together with <c>ComputeRotation</c> using keypoints 0 -> 2 and the target angle
        /// <see cref="kDetectionPalmRotationTargetAngleRadians"/>.
        /// </summary>
        static NormalizedRect DetectionsToRectsCalculator_OneRow(ReadOnlySpan<float> detectionRow, int imgW, int imgH)
        {
            float x1 = detectionRow[0];
            float y1 = detectionRow[1];
            float x2 = detectionRow[2];
            float y2 = detectionRow[3];
            float centerX = (x1 + x2) * 0.5f;
            float centerY = (y1 + y2) * 0.5f;
            float widthPx = x2 - x1;
            float heightPx = y2 - y1;

            float xKp0 = detectionRow[4];
            float yKp0 = detectionRow[5];
            float xKp2 = detectionRow[8];
            float yKp2 = detectionRow[9];
            float rotation = DetectionsToRectsCalculator_NormalizeRadians(
                kDetectionPalmRotationTargetAngleRadians - Mathf.Atan2(-(yKp2 - yKp0), xKp2 - xKp0));

            return new NormalizedRect
            {
                XCenter = centerX / imgW,
                YCenter = centerY / imgH,
                Width = widthPx / imgW,
                Height = heightPx / imgH,
                Rotation = rotation,
            };
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> in the HandDetectorGraph path.
        /// Expands and shifts each palm rectangle to obtain full-hand <c>HAND_RECTS</c>.
        /// </summary>
        List<NormalizedRect> RectTransformationCalculator(List<NormalizedRect> palmRects, int imgW, int imgH)
        {
            var handRects = new List<NormalizedRect>(palmRects.Count);
            foreach (var p in palmRects)
                handRects.Add(RectTransformationCalculator(p, imgW, imgH));
            return handRects;
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> for one palm normalized rectangle.
        /// Converts the rectangle into a hand ROI.
        /// As in <c>TransformNormalizedRect</c> from <c>rect_transformation_calculator.cc</c>,
        /// normalized center, width, and height are not clamped to [0,1];
        /// width and height may exceed 1 for large hands.
        /// </summary>
        NormalizedRect RectTransformationCalculator(NormalizedRect rawRoi, int imgW, int imgH)
        {
            const float shiftX = 0f;
            const float shiftY = -0.5f;
            const float scaleX = 2.6f;
            const float scaleY = 2.6f;

            float xCenterNorm = rawRoi.XCenter;
            float yCenterNorm = rawRoi.YCenter;
            float widthNorm = rawRoi.Width;
            float heightNorm = rawRoi.Height;
            float rotation = rawRoi.Rotation;

            float cosR = Mathf.Cos(rotation);
            float sinR = Mathf.Sin(rotation);
            float xShiftNorm = (imgW * widthNorm * shiftX * cosR - imgH * heightNorm * shiftY * sinR) / imgW;
            float yShiftNorm = (imgW * widthNorm * shiftX * sinR + imgH * heightNorm * shiftY * cosR) / imgH;
            xCenterNorm += xShiftNorm;
            yCenterNorm += yShiftNorm;

            float widthPx = widthNorm * imgW;
            float heightPx = heightNorm * imgH;
            float longSidePx = Mathf.Max(widthPx, heightPx);
            widthNorm = longSidePx / imgW;
            heightNorm = longSidePx / imgH;

            widthNorm *= scaleX;
            heightNorm *= scaleY;

            return new NormalizedRect
            {
                XCenter = xCenterNorm,
                YCenter = yCenterNorm,
                Width = widthNorm,
                Height = heightNorm,
                Rotation = rotation,
            };
        }

        /// <summary>
        /// Packs per-hand results into one <see cref="Mat"/> where rows correspond to hand indices.
        /// Each row contains <see cref="HandLandmarkerEstimationData.ELEMENT_COUNT"/> elements
        /// in the same layout as <see cref="HandLandmarkerEstimationData"/>.
        /// </summary>
        Mat[] PackResultsToMats(List<HandResult> hands)
        {
            int handCount = hands?.Count ?? 0;
            int L = HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            int R = HandLandmarkerEstimationData.ELEMENT_COUNT;
            if (handCount == 0)
                return new Mat[] { new Mat() };

            lock (_lockObject)
            {
                // Reallocate only when the row count is insufficient. Column count and type are fixed.
                if (_outputBuffer == null
                    || _outputBuffer.rows() < handCount
                    || _outputBuffer.cols() != R
                    || _outputBuffer.type() != CvType.CV_32FC1)
                {
                    _outputBuffer?.Dispose();
                    // When possible, allocate up to maxNumHands to reduce reallocation frequency.
                    int rows = Math.Max(handCount, _maxNumHands);
                    _outputBuffer = new Mat(rows, R, CvType.CV_32FC1);
                }

                var packed = _outputBuffer;

                Span<float> row = _handPackRowScratch.AsSpan(0, HandLandmarkerEstimationData.ELEMENT_COUNT);
                for (int i = 0; i < handCount; i++)
                {
                    row.Clear();

                    var h = hands[i];
                    var lm = h.Landmarks;
                    var wm = h.WorldLandmarks;

                    if (lm != null && lm.Length == L)
                    {
                        for (int j = 0; j < L; j++)
                        {
                            int o = j * 3;
                            row[o] = lm[j].Item1;
                            row[o + 1] = lm[j].Item2;
                            row[o + 2] = lm[j].Item3;
                        }
                    }

                    if (wm != null && wm.Length == L)
                    {
                        for (int j = 0; j < L; j++)
                        {
                            int o = HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT + j * 3;
                            row[o] = wm[j].Item1;
                            row[o + 1] = wm[j].Item2;
                            row[o + 2] = wm[j].Item3;
                        }
                    }

                    row[HandLandmarkerEstimationData.ELEMENT_COUNT - 1] = h.Handedness;
                    packed.put(i, 0, row);
                }

                // Return a submat view referencing the internal output buffer.
                // Valid until the next Execute() call (PeekOutput contract).
                Mat result = packed.rowRange(0, handCount);
                return new Mat[] { result };
            }
        }
        /// <summary>
        /// Draws one <see cref="HandLandmarkerEstimationData"/> value onto the image.
        /// Called once per hand from the <c>Mat[]</c> overload of <c>Visualize</c>.
        /// </summary>
        /// <param name="handIndex">Hand index, used for logging.</param>
        internal static void VisualizeHandLandmarkerEstimationData(Mat image, in HandLandmarkerEstimationData data, int handIndex,
            bool printResult, bool isRGB)
        {
            Vec3f[] landmarksScreen = data.GetNormLandmarksArray();
            Vec3f[] landmarksWorld = data.GetWorldLandmarksArray();
            float handedness = data.Handedness;
            string handednessText = handedness == 0f
                ? "?"
                : (HandLandmarkerEstimationData.IsRightHandDominant(handedness) ? "Right" : "Left");
            float handednessScore = HandLandmarkerEstimationData.HandednessScore(handedness);

            int imgW = image.cols();
            int imgH = image.rows();
            float minXN = float.MaxValue, minYN = float.MaxValue;
            for (int i = 0; i < landmarksScreen.Length; i++)
            {
                ref readonly var p = ref landmarksScreen[i];
                if (p.Item1 < minXN) minXN = p.Item1;
                if (p.Item2 < minYN) minYN = p.Item2;
            }
            int left = (int)(minXN * imgW);
            int top = (int)Mathf.Max(0, minYN * imgH - 30);

            Vec4d pointColor = isRGB ? VizBlueBgra : VizRedBgra;

            Imgproc.putText(image, handednessText, new Vec2d(left, top + 12), Imgproc.FONT_HERSHEY_DUPLEX, 0.5, in pointColor);
            Imgproc.putText(image, handednessScore.ToString("F3"), new Vec2d(left, top + 24), Imgproc.FONT_HERSHEY_DUPLEX, 0.5, in pointColor);

            DrawHandSkeleton(image, landmarksScreen, imgW, imgH, in VizWhiteBgra, in pointColor, thickness: 2);

            if (!printResult)
                return;

            var sb = new StringBuilder(1024);
            sb.AppendFormat("[MediaPipeHandLandmarker] Hand {0}: ", handIndex);
            sb.AppendLine();
            sb.AppendFormat("Handedness: {0} (score {1:F3})", handednessText, handednessScore);
            sb.AppendLine();
            sb.Append("Hand NormLandmarks: ");
            sb.Append("{");
            for (int i = 0; i < landmarksScreen.Length; i++)
            {
                ref readonly var p = ref landmarksScreen[i];
                sb.AppendFormat("({0:F3}, {1:F3}, {2:F3})", p.Item1, p.Item2, p.Item3);
                if (i < landmarksScreen.Length - 1)
                    sb.Append(", ");
            }
            sb.Append("}");
            sb.AppendLine();
            sb.Append("Hand WorldLandmarks: ");
            sb.Append("{");
            for (int i = 0; i < landmarksWorld.Length; i++)
            {
                ref readonly var p = ref landmarksWorld[i];
                sb.AppendFormat("({0:F3}, {1:F3}, {2:F3})", p.Item1, p.Item2, p.Item3);
                if (i < landmarksWorld.Length - 1)
                    sb.Append(", ");
            }
            sb.Append("}");
            sb.AppendLine();
            Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Converts normalized image landmarks (<see cref="Vec3f"/>, corresponding to the task-level <c>NormalizedLandmark</c>)
        /// into pixel coordinates and draws the skeleton and joint points.
        /// </summary>
        /// <param name="imageWidth">Input image width in pixels.</param>
        /// <param name="imageHeight">Input image height in pixels.</param>
        internal static void DrawHandSkeleton(Mat image, Vec3f[] landmarksNorm, int imageWidth, int imageHeight,
            in Vec4d lineColor,
            in Vec4d pointColor,
            int thickness = 2)
        {
            if (landmarksNorm == null || landmarksNorm.Length < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT)
                return;
            if (imageWidth <= 0 || imageHeight <= 0)
                return;
            for (int c = 0; c < HAND_LANDMARK_CONNECTIONS.Length; c++)
            {
                int i = (int)HAND_LANDMARK_CONNECTIONS[c].from;
                int j = (int)HAND_LANDMARK_CONNECTIONS[c].to;
                ref readonly var a = ref landmarksNorm[i];
                ref readonly var b = ref landmarksNorm[j];
                Imgproc.line(image,
                    (a.Item1 * imageWidth, a.Item2 * imageHeight),
                    (b.Item1 * imageWidth, b.Item2 * imageHeight),
                    (lineColor.Item1, lineColor.Item2, lineColor.Item3, lineColor.Item4), thickness);
            }
            int dMin = Mathf.Min(imageWidth, imageHeight);
            for (int i = 0; i < HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT; i++)
            {
                ref readonly var p = ref landmarksNorm[i];
                float zAbs = Mathf.Min(Mathf.Abs(p.Item3), kHandSkeletonCircleZAbsCap);
                int r = Mathf.Clamp((int)Mathf.Round(4f + zAbs * dMin * kHandSkeletonCircleZPixelScale), 2, 14);
                Imgproc.circle(image, (p.Item1 * imageWidth, p.Item2 * imageHeight), r,
                    (pointColor.Item1, pointColor.Item2, pointColor.Item3, pointColor.Item4), -1);
            }
        }
    }
}
#endif
#endif
