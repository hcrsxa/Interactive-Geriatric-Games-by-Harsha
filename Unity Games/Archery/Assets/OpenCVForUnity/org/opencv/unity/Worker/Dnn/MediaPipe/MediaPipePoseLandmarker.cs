#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
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
    /// Processing worker that reproduces the pose landmarking graph logic of
    /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) PoseLandmarker
    /// on top of the OpenCV for Unity Dnn module.
    /// </summary>
    public class MediaPipePoseLandmarker : DnnInferenceWorkerBase
    {
        /// <summary>
        /// Landmark indices for the 33-body-point pose topology used by
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) PoseLandmarker.
        /// These values correspond to the ordering of the original pose landmark list.
        /// </summary>
        public enum KeyPoint : byte
        {
            Nose, LeftEyeInner, LeftEye, LeftEyeOuter, RightEyeInner, RightEye, RightEyeOuter, LeftEar, RightEar,
            MouthLeft, MouthRight,
            LeftShoulder, RightShoulder, LeftElbow, RightElbow, LeftWrist, RightWrist, LeftPinky, RightPinky, LeftIndex, RightIndex, LeftThumb, RightThumb,
            LeftHip, RightHip, LeftKnee, RightKnee, LeftAnkle, RightAnkle, LeftHeel, RightHeel, LeftFootIndex, RightFootIndex
        }

        /// <summary>
        /// Execution modes compatible with the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) PoseLandmarker task.
        /// This enum corresponds to the task running mode that switches between
        /// per-image processing and stateful video processing.
        /// </summary>
        public enum MediaPipePoseRunningMode : byte
        {
            /// <summary>
            /// IMAGE mode.
            /// Runs pose detection and pose landmarking for each input image without
            /// reusing loopback tracking state from previous frames.
            /// </summary>
            IMAGE = 0,

            /// <summary>
            /// VIDEO mode.
            /// Assumes a frame sequence and reuses pose rectangles from the previous
            /// frame so the detector can be skipped on frames where tracking remains valid.
            /// </summary>
            VIDEO = 1,
        }

        readonly MediaPipePoseRunningMode _runningMode;
        readonly int _numPoses;
        readonly float _minPoseDetectionConfidence;
        readonly float _minPosePresenceConfidence;
        readonly float _minPoseTrackingConfidence;
        readonly bool _smoothLandmarks;
        /// <summary>
        /// Equivalent to the original <c>output_segmentation_masks</c> option.
        /// When true, the worker emits segmentation masks using <see cref="TensorsToSegmentationCalculator_Pose"/>
        /// and inverse projection from the letterboxed ROI back to the source image
        /// through the <c>warpPerspective</c> path corresponding to <c>WarpAffineCalculator</c>.
        /// </summary>
        readonly bool _outputSegmentationMasks;
        readonly MultiBackendNet _poseDetectorNet;
        /// <summary>Output layer names for pose detection inference. Cached to avoid calling <c>getUnconnectedOutLayersNames()</c> every frame.</summary>
        readonly List<string> _poseDetectorOutLayerNames;

        readonly MultiBackendNet _poseLandmarksNet;
        /// <summary>Output layer names for pose landmark inference. Cached to avoid calling <c>getUnconnectedOutLayersNames()</c> every frame.</summary>
        readonly List<string> _poseLandmarksNetOutLayerNames;

        static readonly Vec4d kVisualizeScalarWhite = new Vec4d(255, 255, 255, 255);
        static readonly Vec4d kVisualizeScalarRed = new Vec4d(0, 0, 255, 255);
        static readonly Vec4d kVisualizeScalarBlue = new Vec4d(255, 0, 0, 255);

        /// <summary>BGR warp destination for one pose ROI in <see cref="ImagePreprocessingGraph_SinglePoseLandmarks"/>. The original model input size is 256x256.</summary>
        Mat _singlePoseLandmarkWarpedBgr;
        Mat _singlePoseLandmarkWarpedRgb;
        Mat _singlePoseLandmarkBlob;
        Mat _singlePoseLandmarkBlobHxW;
        Mat _singlePoseLandmarkSrcPts;
        Mat _singlePoseLandmarkDstPts;

        /// <summary>Original <c>kModelOutputTensorSplitNum</c>, i.e. the split count used by <c>SplitTensorVectorCalculator</c>.</summary>
        const int kPoseLandmarkModelTensorSplitCount = 5;

        /// <summary>Original <c>kLandmarksNum</c>, i.e. the number of joints per landmark tensor.</summary>
        const int kPoseLandmarkModelLandmarkCount = 39;

        /// <summary>Pose landmark input resolution aligned with the task-level <c>ImagePreprocessingGraph</c>, typically 256.</summary>
        const int kPoseLandmarkModelInputSize = 256;

        /// <summary><c>kernel_size</c> from the original <c>RefineLandmarksFromHeatmapCalculator</c> in <c>pose_landmarks_detector_graph.cc</c>.</summary>
        const int kPoseLandmarkHeatmapKernelSize = 7;

        /// <summary>224x224 letterboxed BGR image used by <see cref="ImagePreprocessingGraph"/>.</summary>
        Mat _poseDetectorLetterbox224;

        /// <summary>
        /// Matrix obtained by repeating the SSD anchors for four keypoints,
        /// used by <c>TensorsToDetectionsCalculator</c> via <c>Core.repeat</c>.
        /// </summary>
        Mat _poseDetectorAnchorsNx8;

        Mat _poseTensorsToDetectionsBoxXywh;

        /// <summary>
        /// NMS input (<c>bbox xywh</c>) after <see cref="TensorsToDetectionsCalculator"/>,
        /// containing only rows that pass the original <c>ConvertToDetection</c> <c>min_score_thresh</c>.
        /// </summary>
        Mat _poseTensorsToDetectionsNmsBoxXywh;

        /// <summary>Score column corresponding to the filtered NMS input (K x 1).</summary>
        Mat _poseTensorsToDetectionsNmsScore;

        /// <summary>Bbox plus keypoint rows corresponding to the filtered NMS input (K x 12).</summary>
        Mat _poseTensorsToDetectionsNmsBoxLm;

        /// <summary>
        /// Scratch lists for weighted <see cref="NonMaxSuppressionCalculator"/>,
        /// cleared and reused across frames.
        /// </summary>
        readonly List<(int idx, float sc)> _poseWnmsIndexed = new List<(int, float)>();
        List<(int idx, float sc)> _poseWnmsRemained = new List<(int, float)>();
        List<(int idx, float sc)> _poseWnmsNextRemained = new List<(int, float)>();

        readonly List<Mat> _poseDetectorForwardOutputList = new List<Mat>();
        readonly List<Mat> _poseLandmarksForwardOutputList = new List<Mat>();

        readonly float[] _poseDetectorLetterboxPadding4 = new float[4];
        readonly float[] _poseWnmsRowBuf12 = new float[12];
        readonly float[] _poseWnmsKpAcc8 = new float[8];

        float[] _poseLandmarksTensorFlatScratch;
        float[] _poseHeatmapReadScratch;

        PoseLandmarkDecoded[] _poseDecodedLandmarkScratch;
        readonly PoseLandmarkDecoded[] _poseHeatmapRefineDecodedScratch =
            new PoseLandmarkDecoded[kPoseLandmarkModelLandmarkCount];
        readonly PoseLandmarkDecoded[] _poseWorldDecodedLandmarkScratch =
            new PoseLandmarkDecoded[kPoseLandmarkModelLandmarkCount];

        readonly float[] _poseLandmarksToDetKp8 = new float[8];
        readonly float[] _posePackOutputRowScratch = new float[PoseLandmarkerEstimationData.ELEMENT_COUNT];

        /// <summary>
        /// Output buffer for inference results where rows correspond to pose indices.
        /// Reused as the backing store for packed pose rows returned from detection.
        /// </summary>
        Mat _outputBuffer;

        /// <summary>
        /// 3x3 projection matrix from a single-pose ROI in image space to the 256-space tensor plane.
        /// Corresponds to the homography part of the <c>MATRIX</c> output from <c>ImagePreprocessingGraph</c>.
        /// </summary>
        Mat _singlePoseLandmarkProjMat3x3;

        /// <summary>
        /// Matrix corresponding to the output after the original <c>InverseMatrixCalculator</c>.
        /// This is the inverse of <see cref="_singlePoseLandmarkProjMat3x3"/> and is used
        /// when warping segmentation data from 256-space back to the original image size via <see cref="Imgproc.warpPerspective"/>.
        /// </summary>
        Mat _segmentationFullWarpInvMat3x3;

        /// <summary>Scratch mask in tensor resolution (<c>CV_32FC1</c>) produced after <c>TensorsToSegmentationCalculator</c>.</summary>
        Mat _segmentationScratchSmall;

        /// <summary>
        /// Output buffer for <see cref="PackResultsToMats"/> index 1.
        /// Stores vertically stacked pose masks as <c>CV_32FC1</c> with <c>rows = image.rows * poseCount</c>.
        /// </summary>
        Mat _segmentationStackOutput;

        /// <summary>
        /// Full-image segmentation planes (<c>CV_32FC1</c>) for pose slots <c>0 .. _numPoses - 1</c>
        /// when <c>output_segmentation_masks</c> is enabled.
        /// Null when disabled. <see cref="EnsureSegmentationMaskFullSlot"/> reuses the existing <see cref="Mat"/>
        /// when the image size has not changed.
        /// </summary>
        Mat[] _segmentationMaskFullBySlot;

        /// <summary>
        /// Reusable visualization buffers for stacked pose segmentation.
        /// Non-null only when <c>output_segmentation_masks</c> is enabled.
        /// <see cref="PoseSegmentationVisualizationBuffers.VisualizeStackAllPoses"/> reuses fused, u8, and pseudo-color <see cref="Mat"/> buffers.
        /// </summary>
        readonly PoseSegmentationVisualizationBuffers _poseSegmentationVisualizationBuffers;

        /// <summary>
        /// VIDEO-mode loopback storage for the previous frame's <c>POSE_RECTS_NEXT_FRAME</c>,
        /// i.e. the next-frame ROI produced by the landmark subgraph.
        /// </summary>
        readonly List<NormalizedRect> _prevPoseRectsFromLandmarks = new List<NormalizedRect>();

        /// <summary>
        /// Active only when <c>smooth_landmarks</c> is enabled.
        /// Corresponds to the original <c>VisibilitySmoothingCalculator</c> and
        /// <c>LandmarksSmoothingCalculator</c> stages that live outside the
        /// <c>MultiplePoseLandmarksDetectorGraph</c> loop in <c>pose_landmarks_detector_graph.cc</c>.
        /// </summary>
        readonly PoseLandmarkSmoothingPipeline _poseLandmarkSmoothingPipeline;

        /// <summary>
        /// Pose ROI looped back from the previous frame.
        /// Equivalent to a [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>NormalizedRect</c>.
        /// </summary>
        private struct NormalizedRect
        {
            public float XCenter;
            public float YCenter;
            public float Width;
            public float Height;
            public float Rotation;
            /// <summary>Corresponds to the original <c>NormalizedRect.rect_id</c>. Unset is represented as <c>null</c>.</summary>
            public long? RectId;
        }

        /// <summary>
        /// Intermediate result for one pose before packing.
        /// Filled by the lower-level subgraph implementations.
        /// </summary>
        private struct PoseResult
        {
            /// <summary>Pose presence flag after the original <c>ThresholdingCalculator</c>.</summary>
            public bool PosePresence;

            /// <summary>
            /// Raw presence scalar extracted by <c>TensorsToFloatsCalculator</c> inside the original
            /// <c>SinglePoseLandmarksDetectorGraph</c>, before thresholding.
            /// Zero when inference did not run.
            /// </summary>
            public float PosePresenceScore;
            /// <summary>33 points corresponding to the original <c>NormalizedLandmark</c> output. x and y are normalized to the full image, and z follows <c>landmark.z * NORM_RECT.width</c>.</summary>
            public Vec3f[] NormLandmarks;
            public Vec3f[] WorldLandmarks;
            /// <summary>Two auxiliary landmarks in full-image normalized coordinates, using the same z convention as the main landmarks.</summary>
            public Vec3f[] AuxiliaryLandmarks;
            /// <summary>Visibility values for the normalized landmark list. Before smoothing they come from <c>TensorsToLandmarks</c>; after smoothing they come from the normalized-side <c>VisibilitySmoothing</c> output.</summary>
            public float[] LandmarkVisibility;
            /// <summary>
            /// Visibility values for the world landmark list.
            /// After <c>VisibilityCopyCalculator</c> they start from the same values as the normalized landmarks,
            /// but may diverge after the world-side <c>VisibilitySmoothingCalculator</c>.
            /// </summary>
            public float[] LandmarkVisibilityWorld;
            /// <summary>Per-landmark presence values corresponding to the original <c>NormalizedLandmark.presence</c>. These values are not smoothed.</summary>
            public float[] LandmarkPresence;
            public NormalizedRect NextFrameRect;
            /// <summary>
            /// Buffer index into <see cref="_segmentationMaskFullBySlot"/> when segmentation is enabled (<c>0 .. numPoses - 1</c>).
            /// This is <c>-1</c> when segmentation is disabled, the image size is zero, or the plane was not used in the current iteration.
            /// The actual <see cref="Mat"/> is owned by the worker.
            /// </summary>
            public int SegmentationMaskSlotIndex;
        }

        /// <summary>
        /// Creates a pose landmarker worker backed by a pose detector model and a pose landmark model.
        /// This public API maps to the model assets and runtime options used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) pose detector graph and
        /// pose landmarks detector graph.
        /// </summary>
        /// <param name="poseDetectorModelFilepath">
        /// File path to the pose detection model.
        /// Corresponds to the detector model asset used by [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>pose_detector</c>.
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, pass the full path to a serialized model that <see cref="Unity.InferenceEngine.ModelLoader.Load(string)"/> can load (e.g. <c>.sentis</c>); the caller may rewrite the path from ONNX.
        /// </param>
        /// <param name="poseLandmarksModelFilepath">
        /// File path to the pose landmark model.
        /// Corresponds to the landmark model asset used by [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>pose_landmarks_detector</c>.
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, as for the detector, pass the full path to the Inference Engine serialized model.
        /// </param>
        /// <param name="runningMode">
        /// Task running mode.
        /// Corresponds to whether the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) task behaves like single-image processing
        /// or stateful video processing with loopback tracking state.
        /// </param>
        /// <param name="numPoses">
        /// Maximum number of poses to return.
        /// Corresponds to the <c>num_poses</c> option used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) PoseLandmarker task.
        /// </param>
        /// <param name="minPoseDetectionConfidence">
        /// Minimum confidence for pose detections to be kept before later stages.
        /// Corresponds to the pose detector minimum detection confidence used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) task configuration.
        /// </param>
        /// <param name="minPosePresenceConfidence">
        /// Minimum presence confidence required for landmark results to be treated as present.
        /// Corresponds to the pose presence threshold used after the landmark model in
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe).
        /// </param>
        /// <param name="minTrackingConfidence">
        /// Minimum tracking confidence required to reuse the previous-frame rectangle.
        /// Corresponds to the pose tracking confidence gate used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) video pipeline.
        /// </param>
        /// <param name="outputSegmentationMasks">
        /// When true, enables per-pose segmentation mask output.
        /// Corresponds to the task option that enables segmentation mask outputs in
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) PoseLandmarker.
        /// </param>
        /// <param name="dnnBackend">
        /// Inference backend: an OpenCV <see cref="Dnn"/> <c>DNN_BACKEND_*</c> constant, or <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>.
#if OPENCV_SENTIS_AVAILABLE
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, both models use Unity Inference Engine; <paramref name="dnnTarget"/> is interpreted as an integer <see cref="Unity.InferenceEngine.BackendType"/> value. Assumes Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer.
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
        public MediaPipePoseLandmarker(
            string poseDetectorModelFilepath,
            string poseLandmarksModelFilepath,
            MediaPipePoseRunningMode runningMode = MediaPipePoseRunningMode.IMAGE,
            int numPoses = 1,
            float minPoseDetectionConfidence = 0.5f,
            float minPosePresenceConfidence = 0.5f,
            float minTrackingConfidence = 0.5f,
            bool outputSegmentationMasks = false,
            int dnnBackend = Dnn.DNN_BACKEND_OPENCV,
            int dnnTarget = Dnn.DNN_TARGET_CPU)
            : base(dnnBackend, dnnTarget)
        {
            if (string.IsNullOrEmpty(poseDetectorModelFilepath))
                throw new ArgumentException("The pose detection model file path is not specified.", nameof(poseDetectorModelFilepath));
            if (string.IsNullOrEmpty(poseLandmarksModelFilepath))
                throw new ArgumentException("The pose landmarks model file path is not specified.", nameof(poseLandmarksModelFilepath));
            if (numPoses <= 0)
                throw new ArgumentOutOfRangeException(nameof(numPoses), "numPoses must be greater than or equal to 1.");

            // In the original pose_landmarker_graph, smoothing is enabled in the subgraph only for stream mode with num_poses == 1.
            bool smoothingEnabled = runningMode == MediaPipePoseRunningMode.VIDEO
                && numPoses == 1;

            _runningMode = runningMode;
            _numPoses = numPoses;
            _minPoseDetectionConfidence = Mathf.Clamp01(minPoseDetectionConfidence);
            _minPosePresenceConfidence = Mathf.Clamp01(minPosePresenceConfidence);
            _minPoseTrackingConfidence = Mathf.Clamp01(minTrackingConfidence);
            _smoothLandmarks = smoothingEnabled;
            _outputSegmentationMasks = outputSegmentationMasks;
            _poseLandmarkSmoothingPipeline = smoothingEnabled ? new PoseLandmarkSmoothingPipeline() : null;
            _segmentationMaskFullBySlot = _outputSegmentationMasks ? new Mat[_numPoses] : null;
            _poseSegmentationVisualizationBuffers = outputSegmentationMasks ? new PoseSegmentationVisualizationBuffers() : null;

#if !OPENCV_SENTIS_AVAILABLE
            if (DnnBackend == MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS)
            {
                throw new NotSupportedException(
                    "DNN_BACKEND_UNITY_SENTIS requires Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer in the project and the OPENCV_SENTIS_AVAILABLE define.");
            }
#endif

            try
            {
                _poseDetectorNet = MultiBackendDnn.readNet(poseDetectorModelFilepath);
                _poseDetectorNet.setPreferableBackend(DnnBackend);
                _poseDetectorNet.setPreferableTarget(DnnTarget);
                _poseDetectorOutLayerNames = _poseDetectorNet.getUnconnectedOutLayersNames();

                _poseLandmarksNet = MultiBackendDnn.readNet(poseLandmarksModelFilepath);
                _poseLandmarksNet.setPreferableBackend(DnnBackend);
                _poseLandmarksNet.setPreferableTarget(DnnTarget);
                _poseLandmarksNetOutLayerNames = _poseLandmarksNet.getUnconnectedOutLayersNames();
            }
            catch (Exception e)
            {
                throw new ArgumentException("Failed to initialize the DNN models for Pose Landmarker. Check the model paths and file contents.", e);
            }
        }

        /// <summary>
        /// High-level inference API for
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) PoseLandmarker.
        /// Returns one or two packed matrices (see <see cref="Detect(Mat, bool)"/> remarks in the public API docs).
        /// </summary>
        public Mat[] Detect(Mat image, bool useCopyOutput = false)
        {
            if (image != null) image.ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            Execute(image);
            return BuildDetectReturnArray(useCopyOutput);
        }

        /// <summary>
        /// Asynchronous inference; returns copied <see cref="Mat"/> instances (same layout as <see cref="Detect(Mat, bool)"/> with copy).
        /// </summary>
        /// <remarks>
        /// For the OpenCV Dnn module, inference is scheduled on a background thread when thread-pool scheduling is available.
        /// Web builds cannot use thread pools; only then does the OpenCV Dnn path run synchronously on the caller thread.
        /// When <c>OPENCV_SENTIS_AVAILABLE</c> and Sentis is selected, inference uses Sentis forward APIs asynchronously on every platform, including Web.
        /// </remarks>
        public async Task<Mat[]> DetectTaskAsync(Mat image, CancellationToken cancellationToken = default)
        {
            if (image != null) image.ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            await ExecuteTaskAsync(new[] { image }, cancellationToken);
            return BuildDetectReturnArray(useCopyOutput: true);
        }

        /// <summary>
        /// Asynchronous inference; returns copied <see cref="Mat"/> instances (same layout as <see cref="Detect(Mat, bool)"/> with copy).
        /// </summary>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task{TResult}"/>.
        /// See <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. Web synchronous fallback applies only to the OpenCV Dnn backend; Sentis remains asynchronous on every platform, including Web.
        /// </remarks>
        [Obsolete("Use DetectTaskAsync(). DetectAsync() will return Awaitable in a future version.")]
        public Task<Mat[]> DetectAsync(Mat image, CancellationToken cancellationToken = default) =>
            DetectTaskAsync(image, cancellationToken);

        /// <summary>
        /// Returns an array that stores output <see cref="Mat"/> values by output index.
        /// The array length is 1 or 2 depending on whether segmentation output is enabled.
        /// </summary>
        Mat[] BuildDetectReturnArray(bool useCopyOutput)
        {
            int n = _outputSegmentationMasks ? 2 : 1;
            var arr = new Mat[n];
            for (int i = 0; i < n; i++)
                arr[i] = useCopyOutput ? CopyOutput(i) : PeekOutput(i);
            return arr;
        }

        /// <summary>
        /// Converts a packed result matrix into a managed array of <see cref="PoseLandmarkerEstimationData"/>.
        /// Each returned element corresponds to one row from <see cref="Detect(Mat, bool)"/>.
        /// </summary>
        /// <param name="result">
        /// Packed output matrix returned by <see cref="Detect(Mat, bool)"/> or a compatible source.
        /// Each row corresponds to one pose and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) pose landmarks and
        /// pose world landmarks outputs.
        /// </param>
        /// <returns>
        /// Managed array of pose estimation data.
        /// Returns an empty array when no poses are present.
        /// </returns>
        public virtual PoseLandmarkerEstimationData[] ToStructuredData(Mat result)
        {
            ThrowIfDisposed();

            if (result != null)
                result.ThrowIfDisposed();
            if (result.empty())
                return Array.Empty<PoseLandmarkerEstimationData>();

            int elementCount = PoseLandmarkerEstimationData.ELEMENT_COUNT;
            if (result.cols() < elementCount)
                throw new ArgumentException("Invalid result matrix. It must have at least " + elementCount + " columns.");

            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            int poseCount = result.rows();
            var dst = new PoseLandmarkerEstimationData[poseCount];
            OpenCVMatUtils.CopyFromMat(result, dst);

            return dst;
        }

        /// <summary>
        /// Views a packed result matrix as a zero-allocation <see cref="Span{T}"/> of
        /// <see cref="PoseLandmarkerEstimationData"/>.
        /// </summary>
        /// <remarks>
        /// The returned span remains valid only while <paramref name="result"/> stays allocated
        /// and unchanged.
        /// If the matrix has more than <see cref="PoseLandmarkerEstimationData.ELEMENT_COUNT"/> columns,
        /// interpreting the underlying memory as contiguous rows of
        /// <see cref="PoseLandmarkerEstimationData"/> can cross row boundaries.
        /// The worker-generated packed matrices use the exact expected column count.
        /// </remarks>
        /// <param name="result">
        /// Packed output matrix returned by <see cref="Detect(Mat, bool)"/> or a compatible source.
        /// Each row corresponds to one pose and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) pose landmarks and
        /// pose world landmarks outputs.
        /// </param>
        /// <returns>
        /// Span whose elements correspond to poses in row order.
        /// Returns an empty span when the matrix is empty.
        /// </returns>
        public virtual Span<PoseLandmarkerEstimationData> ToStructuredDataAsSpan(Mat result)
        {
            ThrowIfDisposed();

            if (result != null)
                result.ThrowIfDisposed();
            if (result.empty())
                return Span<PoseLandmarkerEstimationData>.Empty;

            int elementCount = PoseLandmarkerEstimationData.ELEMENT_COUNT;
            if (result.cols() < elementCount)
                throw new ArgumentException("Invalid result matrix. It must have at least " + elementCount + " columns.");

            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            return result.AsSpan<PoseLandmarkerEstimationData>();
        }

        /// <summary>
        /// Draws pose landmarks from a <see cref="Mat"/> array whose layout matches
        /// <see cref="Detect(Mat, bool)"/>.
        /// Element <c>[0]</c> is required and contains one row per pose.
        /// When present, element <c>[1]</c> contains vertically stacked segmentation output.
        /// Array input is supported for compatibility with <see cref="MediaPipeHolisticLandmarker"/>.
        /// </summary>
        /// <param name="image">Destination image for visualization.</param>
        /// <param name="results">
        /// Array of output matrices.
        /// <c>results[0]</c> corresponds to the packed pose output derived from the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) pose landmark outputs.
        /// <c>results[1]</c>, when present, corresponds to the segmentation-mask output enabled by
        /// the PoseLandmarker segmentation option.
        /// </param>
        /// <param name="printResult">If true, prints the decoded result to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public void Visualize(Mat image, Mat[] results, bool printResult = false, bool isRGB = false)
        {
            ThrowIfDisposed();
            VisualizePackedPoseOutputs(image, results, printResult, isRGB, _poseSegmentationVisualizationBuffers);
        }

        /// <summary>
        /// Same drawing logic as <see cref="Visualize(Mat, Mat[], bool, bool)"/> without worker disposal checks.
        /// Called from other workers such as <see cref="MediaPipeHolisticLandmarker"/>.
        /// </summary>
        /// <param name="segmentationVisualization">
        /// Reusable buffer set for segmentation overlay visualization.
        /// Pass a non-null value when <c>output_segmentation_masks</c> is enabled.
        /// If a non-null segmentation <see cref="Mat"/> is supplied while this argument is null,
        /// segmentation drawing is skipped and an error is logged.
        /// </param>
        internal static void VisualizePackedPoseOutputs(Mat image, Mat[] results, bool printResult, bool isRGB,
            PoseSegmentationVisualizationBuffers segmentationVisualization = null)
        {
            if (image != null)
                image.ThrowIfDisposed();
            if (results == null || results.Length == 0 || results[0] == null)
                return;

            Mat main = results[0];
            if (main.empty() || main.rows() <= 0)
                return;

            if (main.cols() < PoseLandmarkerEstimationData.ELEMENT_COUNT)
                throw new ArgumentException(
                    "The result Mat at index 0 does not have enough columns. It must have at least " + PoseLandmarkerEstimationData.ELEMENT_COUNT + " columns.",
                    nameof(results));

            if (!main.isContinuous())
                throw new ArgumentException("The result Mat at index 0 is not stored in a continuous buffer.", nameof(results));

            Span<PoseLandmarkerEstimationData> dataSpan = main.AsSpan<PoseLandmarkerEstimationData>();
            for (int p = 0; p < dataSpan.Length; p++)
            {
                ref readonly PoseLandmarkerEstimationData row = ref dataSpan[p];
                VisualizePoseLandmarkerEstimationData(image, in row, poseIndex: p, printResult: printResult && p == 0, isRGB);
            }

            if (results.Length > 1 && results[1] != null && !results[1].empty() && image != null && !image.empty())
            {
                if (segmentationVisualization == null)
                {
                    Debug.LogError(
                        "[MediaPipePoseLandmarker] A segmentation result Mat (results[1]) was provided, but no segmentation visualization buffer is available."
                        + " Enable outputSegmentationMasks for PoseLandmarker or outputPoseSegmentationMasks for Holistic."
                        + " If you do not want to draw segmentation, pass a results array of length 1 or leave results[1] empty.");
                }
                else
                {
                    segmentationVisualization.VisualizeStackAllPoses(image, results[1], dataSpan.Length);
                }
            }
        }

        /// <summary>
        /// Visualizes the packed pose output returned by <see cref="Detect(Mat, bool)"/>.
        /// Each row is decoded as one <see cref="PoseLandmarkerEstimationData"/> value.
        /// </summary>
        /// <param name="image">Destination image for visualization.</param>
        /// <param name="results">
        /// Packed result matrix with one row per pose.
        /// This matrix stores the public packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) pose landmark output.
        /// </param>
        /// <param name="printResult">If true, prints the decoded result to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public override void Visualize(Mat image, Mat results, bool printResult = false, bool isRGB = false)
        {
            Visualize(image, results == null ? null : new[] { results }, printResult, isRGB);
        }

        /// <summary>
        /// Draws one pose worth of <see cref="PoseLandmarkerEstimationData"/>.
        /// </summary>
        internal static void VisualizePoseLandmarkerEstimationData(Mat image, in PoseLandmarkerEstimationData data, int poseIndex,
            bool printResult, bool isRGB)
        {
            ReadOnlySpan<Vec5f> norm5 = data.GetNormLandmarks();
            ReadOnlySpan<Vec5f> world5 = data.GetWorldLandmarks();

            int iw = image.cols();
            int ih = image.rows();
            float minX = float.MaxValue, minY = float.MaxValue;
            for (int i = 0; i < norm5.Length; i++)
            {
                ref readonly Vec5f lm = ref norm5[i];
                float px = lm.Item1 * iw;
                float py = lm.Item2 * ih;
                if (px < minX) minX = px;
                if (py < minY) minY = py;
            }
            int left = (int)minX;
            int top = (int)Mathf.Max(0, minY - 30);

            var lineColor = kVisualizeScalarWhite.ToValueTuple();
            var pointColor = isRGB ? kVisualizeScalarBlue.ToValueTuple() : kVisualizeScalarRed.ToValueTuple();

            Imgproc.putText(image, "Pose " + poseIndex, (left, top + 12), Imgproc.FONT_HERSHEY_DUPLEX, 0.5, pointColor);

            DrawPoseLandmarkBody(image, norm5, iw, ih, lineColor, pointColor, lineThickness: 2);

            if (!printResult)
                return;

            var sb = new StringBuilder(2048);
            sb.Append("[MediaPipePoseLandmarker] Pose ").Append(poseIndex).AppendLine();
            sb.Append("NormLandmarks (x, y, z, visibility, presence): {");
            for (int i = 0; i < norm5.Length; i++)
            {
                ref readonly Vec5f p = ref norm5[i];
                sb.AppendFormat("({0:F4}, {1:F4}, {2:F4}, {3:F4}, {4:F4})", p.Item1, p.Item2, p.Item3, p.Item4, p.Item5);
                if (i < norm5.Length - 1)
                    sb.Append(", ");
            }
            sb.AppendLine("}");
            sb.Append("WorldLandmarks (x, y, z, visibility, presence): {");
            for (int i = 0; i < world5.Length; i++)
            {
                ref readonly Vec5f p = ref world5[i];
                sb.AppendFormat("({0:F4}, {1:F4}, {2:F4}, {3:F4}, {4:F4})", p.Item1, p.Item2, p.Item3, p.Item4, p.Item5);
                if (i < world5.Length - 1)
                    sb.Append(", ");
            }
            sb.AppendLine("}");
            Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Helper for <see cref="DrawPoseLandmarkBody"/>.
        /// Split out as a static method because a local function cannot capture <see cref="ReadOnlySpan{T}"/>.
        /// </summary>
        static void DrawPoseLandmarkBodySegment(Mat image, ReadOnlySpan<Vec5f> lm, int i, int j,
            int imageWidth, int imageHeight,
            (double, double, double, double) lineColor, int lineThickness)
        {
            ref readonly Vec5f a = ref lm[i];
            ref readonly Vec5f b = ref lm[j];
            Imgproc.line(image,
                (a.Item1 * imageWidth, a.Item2 * imageHeight),
                (b.Item1 * imageWidth, b.Item2 * imageHeight),
                lineColor, lineThickness);
        }

        /// <summary>
        /// Draws lines and points using the same skeleton connectivity as Blaze Pose visualization.
        /// Like the hand visualization in <see cref="MediaPipeHandLandmarker"/>, connections are drawn whenever coordinates are valid.
        /// </summary>
        /// <param name="normLandmarks">Equivalent to the original <c>NormalizedLandmark</c> list. Item1 and Item2 are image-normalized coordinates, and Item4 is visibility.</param>
        internal static void DrawPoseLandmarkBody(Mat image, ReadOnlySpan<Vec5f> normLandmarks,
            int imageWidth, int imageHeight,
            (double, double, double, double) lineColor,
            (double, double, double, double) pointColor,
            int lineThickness)
        {
            if (normLandmarks.Length < PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT)
                return;
            if (imageWidth <= 0 || imageHeight <= 0)
                return;

            // Same connectivity as MediaPipePoseEstimator.Visualize, following KeyPoint order.
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.Nose, (int)KeyPoint.LeftEyeInner, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftEyeInner, (int)KeyPoint.LeftEye, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftEye, (int)KeyPoint.LeftEyeOuter, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftEyeOuter, (int)KeyPoint.LeftEar, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.Nose, (int)KeyPoint.RightEyeInner, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightEyeInner, (int)KeyPoint.RightEye, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightEye, (int)KeyPoint.RightEyeOuter, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightEyeOuter, (int)KeyPoint.RightEar, imageWidth, imageHeight, lineColor, lineThickness);

            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.MouthLeft, (int)KeyPoint.MouthRight, imageWidth, imageHeight, lineColor, lineThickness);

            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightShoulder, (int)KeyPoint.RightElbow, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightElbow, (int)KeyPoint.RightWrist, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightWrist, (int)KeyPoint.RightThumb, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightWrist, (int)KeyPoint.RightPinky, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightWrist, (int)KeyPoint.RightIndex, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightPinky, (int)KeyPoint.RightIndex, imageWidth, imageHeight, lineColor, lineThickness);

            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftShoulder, (int)KeyPoint.LeftElbow, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftElbow, (int)KeyPoint.LeftWrist, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftWrist, (int)KeyPoint.LeftThumb, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftWrist, (int)KeyPoint.LeftIndex, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftWrist, (int)KeyPoint.LeftPinky, imageWidth, imageHeight, lineColor, lineThickness);

            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftShoulder, (int)KeyPoint.RightShoulder, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftShoulder, (int)KeyPoint.LeftHip, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftHip, (int)KeyPoint.RightHip, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightHip, (int)KeyPoint.RightShoulder, imageWidth, imageHeight, lineColor, lineThickness);

            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightHip, (int)KeyPoint.RightKnee, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightKnee, (int)KeyPoint.RightAnkle, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightAnkle, (int)KeyPoint.RightHeel, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightAnkle, (int)KeyPoint.RightFootIndex, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.RightHeel, (int)KeyPoint.RightFootIndex, imageWidth, imageHeight, lineColor, lineThickness);

            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftHip, (int)KeyPoint.LeftKnee, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftKnee, (int)KeyPoint.LeftAnkle, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftAnkle, (int)KeyPoint.LeftFootIndex, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftAnkle, (int)KeyPoint.LeftHeel, imageWidth, imageHeight, lineColor, lineThickness);
            DrawPoseLandmarkBodySegment(image, normLandmarks, (int)KeyPoint.LeftHeel, (int)KeyPoint.LeftFootIndex, imageWidth, imageHeight, lineColor, lineThickness);

            const float visibilityThreshold = 0.5f;
            for (int i = 0; i < PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT; i++)
            {
                ref readonly Vec5f p = ref normLandmarks[i];
                if (p.Item4 < visibilityThreshold)
                    continue;
                Imgproc.circle(image, (p.Item1 * imageWidth, p.Item2 * imageHeight), 2, pointColor, -1);
            }
        }

        /// <summary>
        /// Packed result for one pose from Pose Landmarker.
        /// Memory-compatible with each row produced by <see cref="PackResultsToMats"/>.
        /// Order: 33 elements corresponding to the original <c>NormalizedLandmark</c> values
        /// (x, y, z, visibility, presence) followed by 33 elements corresponding to the original
        /// <c>Landmark</c> values with the same five components.
        /// The user-facing [MediaPipe](https://github.com/google-ai-edge/mediapipe) PoseLandmarker result
        /// does not expose a pose-level raw presence score, so this packed representation does not include it either;
        /// the threshold is applied inside the graph using the constructor's <c>minPosePresenceConfidence</c>.
        /// </summary>
        [Serializable]
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public unsafe struct PoseLandmarkerEstimationData
        {
            /// <summary>
            /// Number of main Pose landmarks in [MediaPipe](https://github.com/google-ai-edge/mediapipe): 33.
            /// Each element is represented as <see cref="Vec5f"/> and indices run from 0 to 32.
            /// </summary>
            public const int LANDMARK_VEC5F_COUNT = 33;

            /// <summary>
            /// Number of float values per landmark, matching the original
            /// <c>NormalizedLandmark</c> / <c>Landmark</c> layout.
            /// </summary>
            public const int LANDMARK_FLOAT_STRIDE = 5;

            /// <summary>Total number of float values occupied by the normalized landmark block for one pose.</summary>
            public const int NORM_LANDMARKS_FLOAT_COUNT = LANDMARK_VEC5F_COUNT * LANDMARK_FLOAT_STRIDE;
            /// <summary>Total number of float values occupied by the world-landmark block for one pose.</summary>
            public const int WORLD_LANDMARKS_FLOAT_COUNT = LANDMARK_VEC5F_COUNT * LANDMARK_FLOAT_STRIDE;

            /// <summary>
            /// Total float element count per packed row:
            /// normalized landmarks (<c>5 x 33</c>) plus world landmarks (<c>5 x 33</c>).
            /// </summary>
            public const int ELEMENT_COUNT = NORM_LANDMARKS_FLOAT_COUNT + WORLD_LANDMARKS_FLOAT_COUNT;

            /// <summary>Total byte size of one packed pose row.</summary>
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Fixed buffer storing 33 normalized pose landmarks in the packed order
            /// <c>(x, y, z, visibility, presence)</c>.
            /// Corresponds to the original [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>NormalizedLandmark</c> list.
            /// </summary>
            public fixed float NormLandmarks[NORM_LANDMARKS_FLOAT_COUNT];
            /// <summary>
            /// Fixed buffer storing 33 world pose landmarks in the packed order
            /// <c>(x, y, z, visibility, presence)</c>.
            /// Corresponds to the original [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>Landmark</c> list.
            /// </summary>
            public fixed float WorldLandmarks[WORLD_LANDMARKS_FLOAT_COUNT];

            /// <summary>
            /// Creates one packed pose result from arrays of <see cref="Vec5f"/> values.
            /// Each <see cref="Vec5f"/> corresponds to the five-element layout of the original
            /// <c>NormalizedLandmark</c> / <c>Landmark</c>: x, y, z, visibility, and presence.
            /// </summary>
            public PoseLandmarkerEstimationData(Vec5f[] normLandmarks, Vec5f[] worldLandmarks)
            {
                if (normLandmarks == null || normLandmarks.Length != LANDMARK_VEC5F_COUNT)
                    throw new ArgumentException("normLandmarks must be a Vec5f[" + LANDMARK_VEC5F_COUNT + "]");
                if (worldLandmarks == null || worldLandmarks.Length != LANDMARK_VEC5F_COUNT)
                    throw new ArgumentException("worldLandmarks must be a Vec5f[" + LANDMARK_VEC5F_COUNT + "]");

                for (int i = 0; i < LANDMARK_VEC5F_COUNT; i++)
                {
                    int o = i * LANDMARK_FLOAT_STRIDE;
                    ref readonly Vec5f n = ref normLandmarks[i];
                    NormLandmarks[o + 0] = n.Item1;
                    NormLandmarks[o + 1] = n.Item2;
                    NormLandmarks[o + 2] = n.Item3;
                    NormLandmarks[o + 3] = n.Item4;
                    NormLandmarks[o + 4] = n.Item5;
                    ref readonly Vec5f w = ref worldLandmarks[i];
                    WorldLandmarks[o + 0] = w.Item1;
                    WorldLandmarks[o + 1] = w.Item2;
                    WorldLandmarks[o + 2] = w.Item3;
                    WorldLandmarks[o + 3] = w.Item4;
                    WorldLandmarks[o + 4] = w.Item5;
                }
            }

            /// <summary>
            /// Returns 33 normalized-landmark elements (x, y, z, visibility, presence) as a
            /// <see cref="ReadOnlySpan{T}"/> that is memory-compatible with the fixed buffer and does not copy.
            /// </summary>
            public readonly ReadOnlySpan<Vec5f> GetNormLandmarks()
            {
                unsafe
                {
                    fixed (float* p = NormLandmarks)
                    {
                        return MemoryMarshal.Cast<float, Vec5f>(new ReadOnlySpan<float>(p, NORM_LANDMARKS_FLOAT_COUNT));
                    }
                }
            }

            /// <summary>
            /// Returns 33 world-landmark elements (x, y, z, visibility, presence) as a
            /// <see cref="ReadOnlySpan{T}"/> that is memory-compatible with the fixed buffer and does not copy.
            /// </summary>
            public readonly ReadOnlySpan<Vec5f> GetWorldLandmarks()
            {
                unsafe
                {
                    fixed (float* p = WorldLandmarks)
                    {
                        return MemoryMarshal.Cast<float, Vec5f>(new ReadOnlySpan<float>(p, WORLD_LANDMARKS_FLOAT_COUNT));
                    }
                }
            }

            /// <summary>
            /// Returns a heap-allocated copy of the 33 normalized landmarks,
            /// useful as a snapshot of <see cref="GetNormLandmarks"/>.
            /// </summary>
            public readonly Vec5f[] GetNormLandmarksArray()
            {
                var a = new Vec5f[LANDMARK_VEC5F_COUNT];
                GetNormLandmarks().CopyTo(a);
                return a;
            }

            /// <summary>
            /// Returns a heap-allocated copy of the 33 world landmarks,
            /// useful as a snapshot of <see cref="GetWorldLandmarks"/>.
            /// </summary>
            public readonly Vec5f[] GetWorldLandmarksArray()
            {
                var a = new Vec5f[LANDMARK_VEC5F_COUNT];
                GetWorldLandmarks().CopyTo(a);
                return a;
            }

            public readonly override string ToString()
            {
                var sb = new StringBuilder(2048);
                sb.Append("PoseLandmarkerEstimationData(");
                sb.Append("Norm:");
                foreach (var v in GetNormLandmarks())
                    sb.Append(v).Append(',');
                sb.Append(" World:");
                foreach (var v in GetWorldLandmarks())
                    sb.Append(v).Append(',');
                sb.Append(')');
                return sb.ToString();
            }
        }

        protected override Mat[] RunCoreProcessing(Mat[] inputs)
        {
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("MediaPipePoseLandmarker accepts only a single input image at index 0.", nameof(inputs));

            var image = inputs[0];
            if (image != null) image.ThrowIfDisposed();
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            List<PoseResult> poses = _runningMode == MediaPipePoseRunningMode.IMAGE
                ? DetectPipeline(image)
                : DetectForVideoPipeline(image);

            return PackResultsToMats(poses, image);
        }

        protected override async Task<Mat[]> RunCoreProcessingTaskAsync(Mat[] inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("MediaPipePoseLandmarker accepts only a single input image at index 0.", nameof(inputs));

            var image = inputs[0];
            if (image != null) image.ThrowIfDisposed();
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

#if OPENCV_SENTIS_AVAILABLE
            if (_poseLandmarksNet.UsesSentis)
            {
                List<PoseResult> poses = _runningMode == MediaPipePoseRunningMode.IMAGE
                    ? await ProcessImageDataAsync(image, cancellationToken)
                    : await ProcessVideoDataAsync(image, cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                return PackResultsToMats(poses, image);
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

        /// <summary>
        /// Holds reusable <see cref="Mat"/> buffers used when visualizing vertically stacked pose segmentation,
        /// avoiding per-frame allocation.
        /// </summary>
        internal sealed class PoseSegmentationVisualizationBuffers : IDisposable
        {
            Mat _fused32f;
            Mat _u8;
            Mat _colorBgr;
            Mat _colorRgba;

            /// <summary>
            /// Fuses all pose slices with per-pixel max and overlays the result on <paramref name="image"/> using pseudo color.
            /// Internal <see cref="Mat"/> buffers are reused until the resolution changes.
            /// </summary>
            public void VisualizeStackAllPoses(Mat image, Mat stackFloat01, int poseCount)
            {
                if (image == null || stackFloat01 == null || poseCount <= 0)
                    return;
                int ih = image.rows();
                int iw = image.cols();
                if (ih <= 0 || iw <= 0 || stackFloat01.cols() != iw)
                    return;
                if (stackFloat01.type() != CvType.CV_32FC1)
                    return;

                int stackSlices = stackFloat01.rows() / ih;
                if (stackSlices <= 0)
                    return;
                int n = Math.Min(poseCount, stackSlices);

                EnsureMat(ref _fused32f, ih, iw, CvType.CV_32FC1);
                _fused32f.setTo((0d, 0d, 0d, 0d));
                for (int p = 0; p < n; p++)
                {
                    using (Mat slice = stackFloat01.rowRange(p * ih, (p + 1) * ih))
                        Core.max(slice, _fused32f, _fused32f);
                }

                EnsureMat(ref _u8, ih, iw, CvType.CV_8UC1);
                EnsureMat(ref _colorBgr, ih, iw, CvType.CV_8UC3);
                _fused32f.convertTo(_u8, CvType.CV_8UC1, 255.0);
                Imgproc.applyColorMap(_u8, _colorBgr, Imgproc.COLORMAP_TURBO);

                int ch = image.channels();
                if (ch == 4)
                {
                    EnsureMat(ref _colorRgba, ih, iw, CvType.CV_8UC4);
                    Imgproc.cvtColor(_colorBgr, _colorRgba, Imgproc.COLOR_BGR2RGBA);
                    Core.addWeighted(image, 0.78, _colorRgba, 0.22, 0, image);
                }
                else if (ch == 3)
                {
                    Core.addWeighted(image, 0.78, _colorBgr, 0.22, 0, image);
                }
            }

            static void EnsureMat(ref Mat m, int rows, int cols, int type)
            {
                if (m == null)
                {
                    m = new Mat(rows, cols, type);
                    return;
                }

                if (m.rows() != rows || m.cols() != cols || m.type() != type)
                {
                    m.release();
                    m.create(rows, cols, type);
                }
            }

            /// <inheritdoc />
            public void Dispose()
            {
                _fused32f?.Dispose();
                _fused32f = null;
                _u8?.Dispose();
                _u8 = null;
                _colorBgr?.Dispose();
                _colorBgr = null;
                _colorRgba?.Dispose();
                _colorRgba = null;
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                _poseSegmentationVisualizationBuffers?.Dispose();
                _poseDetectorNet?.Dispose();
                _poseLandmarksNet?.Dispose();
                _poseDetectorForwardOutputList.Clear();
                _poseLandmarksForwardOutputList.Clear();
                _outputBuffer?.Dispose();
                _outputBuffer = null;
                _poseDetectorLetterbox224?.Dispose();
                _poseDetectorLetterbox224 = null;
                _poseDetectorAnchorsNx8?.Dispose();
                _poseDetectorAnchorsNx8 = null;
                _poseTensorsToDetectionsBoxXywh?.Dispose();
                _poseTensorsToDetectionsBoxXywh = null;
                _poseTensorsToDetectionsNmsBoxXywh?.Dispose();
                _poseTensorsToDetectionsNmsBoxXywh = null;
                _poseTensorsToDetectionsNmsScore?.Dispose();
                _poseTensorsToDetectionsNmsScore = null;
                _poseTensorsToDetectionsNmsBoxLm?.Dispose();
                _poseTensorsToDetectionsNmsBoxLm = null;

                _singlePoseLandmarkWarpedBgr?.Dispose();
                _singlePoseLandmarkWarpedBgr = null;
                _singlePoseLandmarkWarpedRgb?.Dispose();
                _singlePoseLandmarkWarpedRgb = null;
                _singlePoseLandmarkBlob?.Dispose();
                _singlePoseLandmarkBlob = null;
                _singlePoseLandmarkBlobHxW = null;
                _singlePoseLandmarkSrcPts?.Dispose();
                _singlePoseLandmarkSrcPts = null;
                _singlePoseLandmarkDstPts?.Dispose();
                _singlePoseLandmarkDstPts = null;

                _singlePoseLandmarkProjMat3x3?.Dispose();
                _singlePoseLandmarkProjMat3x3 = null;
                _segmentationFullWarpInvMat3x3?.Dispose();
                _segmentationFullWarpInvMat3x3 = null;
                _segmentationScratchSmall?.Dispose();
                _segmentationScratchSmall = null;
                _segmentationStackOutput?.Dispose();
                _segmentationStackOutput = null;

                if (_segmentationMaskFullBySlot != null)
                {
                    for (int i = 0; i < _segmentationMaskFullBySlot.Length; i++)
                    {
                        _segmentationMaskFullBySlot[i]?.Dispose();
                        _segmentationMaskFullBySlot[i] = null;
                    }
                }

                _poseLandmarkSmoothingPipeline?.ResetAll();
            }

            base.Dispose(disposing);
        }

        /// <summary>
        /// Internal IMAGE-mode entry point.
        /// Equivalent to <c>PoseLandmarker::Detect</c> and called from <see cref="RunCoreProcessing"/>.
        /// </summary>
        List<PoseResult> DetectPipeline(Mat image)
        {
            return ProcessImageData(image);
        }

        /// <summary>
        /// Internal VIDEO-mode entry point.
        /// Equivalent to <c>PoseLandmarker::DetectForVideo</c>.
        /// </summary>
        List<PoseResult> DetectForVideoPipeline(Mat image)
        {
            return ProcessVideoData(image);
        }

        /// <summary>
        /// IMAGE-mode pipeline entry corresponding to Task API <c>ProcessImageData</c>.
        /// Equivalent to the IMAGE path of <c>PoseLandmarkerGraph</c>.
        /// </summary>
        /// <remarks>
        /// In Pose IMAGE mode, the graph does not include <c>PreviousLoopbackCalculator</c>.
        /// Unlike Hand IMAGE mode, this path starts directly from <c>PoseDetectorGraph</c> without assuming loopback state.
        ///
        /// Mapping to the original <c>pose_landmarker_graph.cc</c>:
        /// - PoseDetectorGraph → ClipNormalizedRectVectorSizeCalculator → MultiplePoseLandmarksDetectorGraph
        /// </remarks>
        List<PoseResult> ProcessImageData(Mat image)
        {
            _prevPoseRectsFromLandmarks.Clear();

            var det = PoseDetectorGraph(image, null);
            var clipped = ClipNormalizedRectVectorSizeCalculator(det.ExpandedPoseRects);
            return MultiplePoseLandmarksDetectorGraph(image, clipped);
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="ProcessImageData"/> using the Unity Inference Engine path with <see cref="InferenceSubgraph_PoseDetectionAsync"/> and <see cref="InferenceSubgraph_PoseLandmarksAsync"/> (<see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// </summary>
        async Task<List<PoseResult>> ProcessImageDataAsync(Mat image, CancellationToken cancellationToken)
        {
            _prevPoseRectsFromLandmarks.Clear();

            var det = await PoseDetectorGraphAsync(image, null, cancellationToken);
            var clipped = ClipNormalizedRectVectorSizeCalculator(det.ExpandedPoseRects);
            return await MultiplePoseLandmarksDetectorGraphAsync(image, clipped, cancellationToken);
        }

#endif
        /// <summary>
        /// VIDEO-mode pipeline entry corresponding to Task API <c>ProcessVideoData</c>.
        /// Equivalent to <c>PoseLandmarker::DetectForVideo</c>.
        /// </summary>
        /// <remarks>
        /// Mapping to the original <c>pose_landmarker_graph.cc</c> stream-mode path:
        /// - PreviousLoopbackCalculator → NormalizedRectVectorHasMinSizeCalculator → (DisallowIf equivalent) PoseDetectorGraph
        ///   → AssociationNormRectCalculator → ClipNormalizedRectVectorSizeCalculator → MultiplePoseLandmarksDetectorGraph
        /// - Output <c>POSE_RECTS_NEXT_FRAME</c> is fed into the next frame's loop input via <see cref="_prevPoseRectsFromLandmarks"/>.
        /// </remarks>
        List<PoseResult> ProcessVideoData(Mat image)
        {
            // 1. PreviousLoopbackCalculator: get the previous frame's POSE_RECTS_NEXT_FRAME values as PREV_LOOP.
            var prevPoseRects = PreviousLoopbackCalculator(image, _prevPoseRectsFromLandmarks);

            // 2. NormalizedRectVectorHasMinSizeCalculator: if the previous-frame rectangle count is at least num_poses, detector execution may be skipped.
            bool hasEnoughPoses = NormalizedRectVectorHasMinSizeCalculator(prevPoseRects, _numPoses);

            // 3. DisallowIf + PoseDetectorGraph: run PoseDetectorGraph only when tracking is insufficient. The original graph uses an empty packet on the skipped branch.
            List<NormalizedRect> expandedPoseRectsFromDetector = new List<NormalizedRect>();
            if (!hasEnoughPoses)
            {
                var det = PoseDetectorGraph(image, null);
                if (det.ExpandedPoseRects != null)
                    expandedPoseRectsFromDetector = det.ExpandedPoseRects;
            }

            // 4. AssociationNormRectCalculator with inputs [0] = previous rects and [1] = detected EXPANDED_POSE_RECTS, using min_similarity_threshold = min_tracking_confidence.
            var associatedPoseRects = AssociationNormRectCalculator(prevPoseRects, expandedPoseRectsFromDetector);

            // 5. ClipNormalizedRectVectorSizeCalculator -> MultiplePoseLandmarksDetectorGraph.
            var clipped = ClipNormalizedRectVectorSizeCalculator(associatedPoseRects);
            var poses = MultiplePoseLandmarksDetectorGraph(image, clipped);

            // 6. As in the original graph, poses that are not present are excluded from the next-frame loopback, matching the same ordering convention used in Hand ProcessVideoData.
            for (int ri = poses.Count - 1; ri >= 0; ri--)
            {
                if (poses[ri].PosePresence)
                    continue;
                // Segmentation Mats are worker-owned through _segmentationMaskFullBySlot, so they must not be disposed when removing absent poses.
                poses.RemoveAt(ri);
            }

            // 7. Back edge to PreviousLoopbackCalculator: store POSE_RECTS_NEXT_FRAME for the next frame.
            _prevPoseRectsFromLandmarks.Clear();
            foreach (var p in poses)
                _prevPoseRectsFromLandmarks.Add(p.NextFrameRect);

            return poses;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="ProcessVideoData"/> using the Unity Inference Engine path with <see cref="InferenceSubgraph_PoseDetectionAsync"/> and <see cref="InferenceSubgraph_PoseLandmarksAsync"/> (<see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// </summary>
        async Task<List<PoseResult>> ProcessVideoDataAsync(Mat image, CancellationToken cancellationToken)
        {
            var prevPoseRects = PreviousLoopbackCalculator(image, _prevPoseRectsFromLandmarks);
            bool hasEnoughPoses = NormalizedRectVectorHasMinSizeCalculator(prevPoseRects, _numPoses);
            List<NormalizedRect> expandedPoseRectsFromDetector = new List<NormalizedRect>();
            if (!hasEnoughPoses)
            {
                var det = await PoseDetectorGraphAsync(image, null, cancellationToken);
                if (det.ExpandedPoseRects != null)
                    expandedPoseRectsFromDetector = det.ExpandedPoseRects;
            }

            var associatedPoseRects = AssociationNormRectCalculator(prevPoseRects, expandedPoseRectsFromDetector);
            var clipped = ClipNormalizedRectVectorSizeCalculator(associatedPoseRects);
            var poses = await MultiplePoseLandmarksDetectorGraphAsync(image, clipped, cancellationToken);

            for (int ri = poses.Count - 1; ri >= 0; ri--)
            {
                if (poses[ri].PosePresence)
                    continue;
                poses.RemoveAt(ri);
            }

            _prevPoseRectsFromLandmarks.Clear();
            foreach (var p in poses)
                _prevPoseRectsFromLandmarks.Add(p.NextFrameRect);

            return poses;
        }

#endif
        /// <summary>
        /// Equivalent to <c>PreviousLoopbackCalculator</c>.
        /// Returns the previous frame's rectangle list as the LOOP input.
        /// </summary>
        List<NormalizedRect> PreviousLoopbackCalculator(Mat image, List<NormalizedRect> loopPoseRects)
        {
            _ = image;
            return loopPoseRects != null ? new List<NormalizedRect>(loopPoseRects) : new List<NormalizedRect>();
        }

        /// <summary>
        /// Equivalent to <c>NormalizedRectVectorHasMinSizeCalculator</c>.
        /// </summary>
        bool NormalizedRectVectorHasMinSizeCalculator(List<NormalizedRect> rects, int minSize)
        {
            if (rects == null) return false;
            return rects.Count >= minSize;
        }

        /// <summary>
        /// Equivalent to <c>AssociationNormRectCalculator</c>,
        /// following <c>association_norm_rect_calculator.cc</c> and the base
        /// <c>GetNonOverlappingElements</c> logic in <c>association_calculator.h</c>.
        /// </summary>
        /// <remarks>
        /// In the original stream-mode <c>pose_landmarker_graph.cc</c>, input [0] is connected to previous-frame rectangles
        /// and input [1] is connected to <c>EXPANDED_POSE_RECTS</c> from <c>PoseDetectorGraph</c>,
        /// with <c>min_similarity_threshold = min_tracking_confidence</c>.
        /// The Pose graph does not use the <c>PREV</c> tagged input, so
        /// <c>PropagateIdsFromPreviousToCurrent</c> does not occur here.
        /// </remarks>
        /// <param name="prevPoseRects">Stream 0, i.e. previous-frame <c>POSE_RECTS_NEXT_FRAME</c>.</param>
        /// <param name="expandedPoseRectsFromDetector">Stream 1, i.e. expanded rectangles from the detector. Empty when detection is skipped.</param>
        List<NormalizedRect> AssociationNormRectCalculator(
            List<NormalizedRect> prevPoseRects,
            List<NormalizedRect> expandedPoseRectsFromDetector)
        {
            float minSim = _minPoseTrackingConfidence;

            bool prevEmpty = prevPoseRects == null || prevPoseRects.Count == 0;
            bool detEmpty = expandedPoseRectsFromDetector == null || expandedPoseRectsFromDetector.Count == 0;

            var result = new List<NormalizedRect>();

            if (!prevEmpty)
            {
                result.Add(prevPoseRects[0]);
                for (int j = 1; j < prevPoseRects.Count; j++)
                    AssociationNormRectCalculator_AddElementToList(prevPoseRects[j], result, minSim);
                if (!detEmpty)
                {
                    foreach (var r in expandedPoseRectsFromDetector)
                        AssociationNormRectCalculator_AddElementToList(r, result, minSim);
                }
            }
            else if (!detEmpty)
            {
                result.Add(expandedPoseRectsFromDetector[0]);
                for (int j = 1; j < expandedPoseRectsFromDetector.Count; j++)
                    AssociationNormRectCalculator_AddElementToList(expandedPoseRectsFromDetector[j], result, minSim);
            }

            return result;
        }

        /// <summary>
        /// Equivalent to <c>AssociationCalculator::AddElementToList</c>,
        /// using <c>CalculateIou</c> from <c>rectangle_util.cc</c> for overlap comparison.
        /// </summary>
        static void AssociationNormRectCalculator_AddElementToList(
            NormalizedRect element, List<NormalizedRect> current, float minSimilarityThreshold)
        {
            for (int i = current.Count - 1; i >= 0; i--)
            {
                if (AssociationNormRectCalculator_ComputeIoU(element, current[i]) > minSimilarityThreshold)
                    current.RemoveAt(i);
            }
            current.Add(element);
        }

        /// <summary>
        /// Equivalent to <c>CalculateIou</c> in <c>rectangle_util.cc</c> for axis-aligned rectangles, ignoring rotation.
        /// </summary>
        static float AssociationNormRectCalculator_ComputeIoU(NormalizedRect a, NormalizedRect b)
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
        /// One pose detection after letterbox removal, expressed in coordinates relative to the input image.
        /// Corresponds to the <c>LOCATION_DATA</c> payload of the original <c>Detection</c>.
        /// </summary>
        sealed class PoseDetectionData
        {
            public float RelXmin;
            public float RelYmin;
            public float RelWidth;
            public float RelHeight;
            public float[] RelKeypointsXy;
            public float Score;
        }

        /// <summary>
        /// Bundled outputs of <c>PoseDetectorGraph</c>.
        /// Corresponds to <c>DETECTIONS</c>, <c>POSE_RECTS</c>, and <c>EXPANDED_POSE_RECTS</c> in the original <c>pose_detector_graph.cc</c>.
        /// </summary>
        struct PoseDetectorGraphResult
        {
            public List<PoseDetectionData> PoseDetections;
            public List<NormalizedRect> PoseRects;
            public List<NormalizedRect> ExpandedPoseRects;
        }

        /// <summary>
        /// Equivalent to <c>PoseDetectorGraph</c> from
        /// <c>mediapipe/tasks/cc/vision/pose_detector/pose_detector_graph.cc</c>.
        /// This method only invokes lower-level calculators and subgraphs in the original connection order.
        ///
        /// Mapping to the original <c>pose_detector_graph.cc</c>:
        /// - ImagePreprocessingGraph → <see cref="ImagePreprocessingGraph"/>
        /// - Inference subgraph (<c>AddInference</c>) → <see cref="InferenceSubgraph_PoseDetection"/>
        /// - SsdAnchorsCalculator → <see cref="SsdAnchorsCalculator"/>
        /// - TensorsToDetectionsCalculator → <see cref="TensorsToDetectionsCalculator"/>
        /// - NonMaxSuppressionCalculator → <see cref="NonMaxSuppressionCalculator"/>
        /// - DetectionLetterboxRemovalCalculator → <see cref="DetectionLetterboxRemovalCalculator"/>
        /// - AlignmentPointsRectsCalculator → <see cref="AlignmentPointsRectsCalculator"/>
        /// - RectTransformationCalculator → <see cref="RectTransformationCalculator"/>
        /// - ClipDetectionVectorSizeCalculator (when <c>num_poses</c> is set) → <see cref="ClipDetectionVectorSizeCalculator"/>
        /// </summary>
        /// <param name="normRect">Original <c>NORM_RECT</c> input. Use null for the current full-image-only implementation.</param>
        PoseDetectorGraphResult PoseDetectorGraph(Mat image, NormalizedRect? normRect)
        {
            var empty = new PoseDetectorGraphResult
            {
                PoseDetections = new List<PoseDetectionData>(),
                PoseRects = new List<NormalizedRect>(),
                ExpandedPoseRects = new List<NormalizedRect>(),
            };

            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;

            if (normRect.HasValue)
                throw new NotSupportedException("Non-null NORM_RECT is not implemented in PoseDetectorGraph yet. Phase 1 supports full-image input only.");

            Mat inputBlob = null;
            try
            {
                ImagePreprocessingGraph(image, out _, out inputBlob, out int imageW, out int imageH, out float[] letterboxPadding);

                var outputBlobs = InferenceSubgraph_PoseDetection(inputBlob);
                if (outputBlobs == null || outputBlobs.Count < 2)
                    return empty;

                Mat output0 = outputBlobs[0];
                Mat output1 = outputBlobs[1];
                int num = output0.size(1);
                if (num <= 0)
                    return empty;

                Mat anchors = SsdAnchorsCalculator(out Mat anchorsNx8);
                TensorsToDetectionsCalculator(output0, output1, anchors, anchorsNx8, out Mat boxXywh, out Mat scoreForNms, out Mat boxLmForNms);
                var letterboxed = new List<PoseDetectionData>();
                NonMaxSuppressionCalculator(boxXywh, scoreForNms, boxLmForNms, letterboxed);

                var afterLetterbox = DetectionLetterboxRemovalCalculator(letterboxed, letterboxPadding);
                List<NormalizedRect> poseRects = AlignmentPointsRectsCalculator(afterLetterbox, imageW, imageH);
                List<NormalizedRect> expanded = RectTransformationCalculator(poseRects, imageW, imageH);

                List<PoseDetectionData> clippedDetections = ClipDetectionVectorSizeCalculator(afterLetterbox, _numPoses);

                return new PoseDetectorGraphResult
                {
                    PoseDetections = clippedDetections,
                    PoseRects = poseRects,
                    ExpandedPoseRects = expanded,
                };
            }
            finally
            {
                inputBlob?.Dispose();
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="PoseDetectorGraph"/> using the Unity Inference Engine path with <see cref="InferenceSubgraph_PoseDetectionAsync"/> (<see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// </summary>
        async Task<PoseDetectorGraphResult> PoseDetectorGraphAsync(Mat image, NormalizedRect? normRect, CancellationToken cancellationToken)
        {
            var empty = new PoseDetectorGraphResult
            {
                PoseDetections = new List<PoseDetectionData>(),
                PoseRects = new List<NormalizedRect>(),
                ExpandedPoseRects = new List<NormalizedRect>(),
            };

            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;

            if (normRect.HasValue)
                throw new NotSupportedException("Non-null NORM_RECT is not implemented in PoseDetectorGraph yet. Phase 1 supports full-image input only.");

            Mat inputBlob = null;
            try
            {
                ImagePreprocessingGraph(image, out _, out inputBlob, out int imageW, out int imageH, out float[] letterboxPadding);

                var outputBlobs = await InferenceSubgraph_PoseDetectionAsync(inputBlob, cancellationToken);
                if (outputBlobs == null || outputBlobs.Count < 2)
                    return empty;

                Mat output0 = outputBlobs[0];
                Mat output1 = outputBlobs[1];
                int num = output0.size(1);
                if (num <= 0)
                    return empty;

                Mat anchors = SsdAnchorsCalculator(out Mat anchorsNx8);
                TensorsToDetectionsCalculator(output0, output1, anchors, anchorsNx8, out Mat boxXywh, out Mat scoreForNms, out Mat boxLmForNms);
                var letterboxed = new List<PoseDetectionData>();
                NonMaxSuppressionCalculator(boxXywh, scoreForNms, boxLmForNms, letterboxed);

                var afterLetterbox = DetectionLetterboxRemovalCalculator(letterboxed, letterboxPadding);
                List<NormalizedRect> poseRects = AlignmentPointsRectsCalculator(afterLetterbox, imageW, imageH);
                List<NormalizedRect> expanded = RectTransformationCalculator(poseRects, imageW, imageH);

                List<PoseDetectionData> clippedDetections = ClipDetectionVectorSizeCalculator(afterLetterbox, _numPoses);

                return new PoseDetectorGraphResult
                {
                    PoseDetections = clippedDetections,
                    PoseRects = poseRects,
                    ExpandedPoseRects = expanded,
                };
            }
            finally
            {
                inputBlob?.Dispose();
            }
        }

#endif
        /// <summary>
        /// Equivalent to <c>ImagePreprocessingGraph</c> in the Tasks path, i.e. the
        /// <c>ImagePreprocessingGraph</c> / <c>ImageToTensorCalculator</c> stage.
        /// Produces the 224x224 input blob with aspect-ratio preservation, zero padding,
        /// and [-1, 1] normalization, together with the letterbox padding values.
        /// </summary>
        /// <remarks>
        /// Integer sizes after resize follow the same truncation rule as the full-image palm detector
        /// letterbox path in <see cref="MediaPipeHandLandmarker"/>:
        /// use <c>(int)(width * ratio)</c> and do not use <c>Mathf.RoundToInt</c>.
        /// </remarks>
        void ImagePreprocessingGraph(Mat image, out Mat letterbox224, out Mat inputBlob, out int imageW, out int imageH, out float[] letterboxPadding)
        {
            const int tensorSize = 224;
            imageW = image.cols();
            imageH = image.rows();

            if (_poseDetectorLetterbox224 == null)
                _poseDetectorLetterbox224 = new Mat(tensorSize, tensorSize, image.type());
            letterbox224 = _poseDetectorLetterbox224;

            double ratio = Math.Min((double)tensorSize / imageW, (double)tensorSize / imageH);
            int newW = Math.Max(1, (int)(imageW * ratio));
            int newH = Math.Max(1, (int)(imageH * ratio));

            int padX = (tensorSize - newW) / 2;
            int padY = (tensorSize - newH) / 2;

            letterbox224.setTo((0d, 0d, 0d, 0d));
            using (Mat resized = new Mat())
            {
                Imgproc.resize(image, resized, (newW, newH));
                using (Mat roi = new Mat(letterbox224, (padX, padY, newW, newH)))
                {
                    resized.copyTo(roi);
                }
            }

            float padLeft = padX / (float)tensorSize;
            float padTop = padY / (float)tensorSize;
            float padRight = (tensorSize - padX - newW) / (float)tensorSize;
            float padBottom = (tensorSize - padY - newH) / (float)tensorSize;
            _poseDetectorLetterboxPadding4[0] = padLeft;
            _poseDetectorLetterboxPadding4[1] = padTop;
            _poseDetectorLetterboxPadding4[2] = padRight;
            _poseDetectorLetterboxPadding4[3] = padBottom;
            letterboxPadding = _poseDetectorLetterboxPadding4;

            inputBlob = Dnn.blobFromImage(
                letterbox224,
                1.0 / 127.5,
                (tensorSize, tensorSize),
                (127.5, 127.5, 127.5, 0),
                true,
                false,
                CvType.CV_32F);
        }

        /// <summary>
        /// Equivalent to the inference subgraph, corresponding to the original
        /// <c>AddInference</c> / <c>mediapipe.tasks.core.InferenceSubgraph</c>.
        /// Feeds the preprocessed 224x224 blob into <see cref="_poseDetectorNet"/> (OpenCV DNN or Unity Inference Engine) and returns <c>TENSORS</c>.
        /// Callers do not dispose <see cref="Mat"/> entries in the returned list; <see cref="MultiBackendNet"/> owns OpenCV forward outputs across calls and reuses IE buffers in Sentis mode.
        /// </summary>
        List<Mat> InferenceSubgraph_PoseDetection(Mat inputBlob)
        {
            _poseDetectorForwardOutputList.Clear();
            _poseDetectorNet.setInput(inputBlob);
            _poseDetectorNet.forward(_poseDetectorForwardOutputList, _poseDetectorOutLayerNames);
            return _poseDetectorForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Asynchronous <see cref="InferenceSubgraph_PoseDetection"/> for the Unity Inference Engine path (<see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// Only <see cref="RunCoreProcessingTaskAsync"/> uses this; the OpenCV path uses <see cref="InferenceSubgraph_PoseDetection"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_PoseDetectionAsync(Mat inputBlob, CancellationToken cancellationToken)
        {
            _poseDetectorForwardOutputList.Clear();
            _poseDetectorNet.setInput(inputBlob);
            await _poseDetectorNet.forwardTaskAsync(_poseDetectorForwardOutputList, _poseDetectorOutLayerNames, cancellationToken);
            return _poseDetectorForwardOutputList;
        }

#endif
        /// <summary>
        /// Equivalent to <c>SsdAnchorsCalculator</c>.
        /// Retrieves the shared cached 2254x2 anchor matrix according to
        /// <c>ConfigureSsdAnchorsCalculator</c> in <c>pose_detector_graph.cc</c> and
        /// <c>GenerateAnchors</c> in <c>ssd_anchors_calculator.cc</c>,
        /// and also prepares the 4x-repeated matrix for keypoints.
        /// </summary>
        Mat SsdAnchorsCalculator(out Mat anchorsNx8)
        {
            Mat anchors = GetPoseDetectorSsdAnchors2254Shared();
            if (_poseDetectorAnchorsNx8 == null)
            {
                _poseDetectorAnchorsNx8 = new Mat();
                Core.repeat(anchors, 1, 4, _poseDetectorAnchorsNx8);
            }
            anchorsNx8 = _poseDetectorAnchorsNx8;
            return anchors;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToDetectionsCalculator</c>, following
        /// <c>pose_detection_gpu.pbtxt</c> and <c>ConfigureTensorsToDetectionsCalculator</c>.
        /// After decoding, rows below <c>min_score_thresh</c> (task-level <c>min_detection_confidence</c>)
        /// are removed just like the original <c>ConvertToDetection</c>, and the filtered
        /// <paramref name="boxXywh"/>, <paramref name="scoreForNms"/>, and <paramref name="boxLmForNms"/>
        /// are returned as NMS inputs in letterbox-normalized coordinates.
        /// </summary>
        void TensorsToDetectionsCalculator(Mat output0, Mat output1, Mat anchors, Mat anchorsNx8, out Mat boxXywh, out Mat scoreForNms, out Mat boxLmForNms)
        {
            const int inputSize = 224;
            int num = output0.size(1);
            if (_poseTensorsToDetectionsBoxXywh == null)
                _poseTensorsToDetectionsBoxXywh = new Mat();
            _poseTensorsToDetectionsBoxXywh.create(num, 4, CvType.CV_32FC1);

            using (Mat score = output1.reshape(1, num))
            using (var boxAndLandmark = output0.reshape(1, num))
            {
                NumpyClip(score, -100.0, 100.0);
                Core.multiply(score, (-1.0, 0, 0, 0), score);
                Core.exp(score, score);
                Core.add(score, (1.0, 0, 0, 0), score);
                Core.divide(1.0, score, score);

                using (var boxAndLandmarkNx1c2 = boxAndLandmark.reshape(2, num))
                {
                    Core.divide(boxAndLandmarkNx1c2, (inputSize, inputSize, 0, 0), boxAndLandmarkNx1c2);
                }

                using (var cxy = boxAndLandmark.colRange(0, 2))
                {
                    Core.add(cxy, anchors, cxy);
                }

                using (var lm = boxAndLandmark.colRange(4, 12))
                {
                    Core.add(lm, anchorsNx8, lm);
                }

                using (var cxy2 = boxAndLandmark.colRange(0, 2))
                using (var wh2 = boxAndLandmark.colRange(2, 4))
                using (var dstXy = _poseTensorsToDetectionsBoxXywh.colRange(0, 2))
                using (var dstWh = _poseTensorsToDetectionsBoxXywh.colRange(2, 4))
                {
                    cxy2.copyTo(dstWh);
                    Core.divide(wh2, (2.0, 0, 0, 0), dstXy);
                    Core.subtract(dstWh, dstXy, cxy2);
                    Core.add(dstWh, dstXy, wh2);

                    cxy2.copyTo(dstXy);
                    Core.subtract(wh2, cxy2, dstWh);
                }

                if (_poseTensorsToDetectionsNmsBoxXywh == null)
                    _poseTensorsToDetectionsNmsBoxXywh = new Mat();
                if (_poseTensorsToDetectionsNmsScore == null)
                    _poseTensorsToDetectionsNmsScore = new Mat();
                if (_poseTensorsToDetectionsNmsBoxLm == null)
                    _poseTensorsToDetectionsNmsBoxLm = new Mat();

                int k = 0;
                for (int src = 0; src < num; src++)
                {
                    float sc = score.at<float>(src, 0)[0];
                    if (sc < _minPoseDetectionConfidence)
                        continue;
                    k++;
                }

                _poseTensorsToDetectionsNmsBoxXywh.create(k, 4, CvType.CV_32FC1);
                _poseTensorsToDetectionsNmsScore.create(k, 1, CvType.CV_32FC1);
                _poseTensorsToDetectionsNmsBoxLm.create(k, 12, CvType.CV_32FC1);

                int dst = 0;
                for (int src = 0; src < num; src++)
                {
                    float sc = score.at<float>(src, 0)[0];
                    if (sc < _minPoseDetectionConfidence)
                        continue;
                    _poseTensorsToDetectionsBoxXywh.row(src).copyTo(_poseTensorsToDetectionsNmsBoxXywh.row(dst));
                    _poseTensorsToDetectionsNmsScore.at<float>(dst, 0)[0] = sc;
                    boxAndLandmark.row(src).copyTo(_poseTensorsToDetectionsNmsBoxLm.row(dst));
                    dst++;
                }

                boxXywh = _poseTensorsToDetectionsNmsBoxXywh;
                scoreForNms = _poseTensorsToDetectionsNmsScore;
                boxLmForNms = _poseTensorsToDetectionsNmsBoxLm;
            }
        }

        /// <summary>
        /// Equivalent to <c>NonMaxSuppressionCalculator</c>,
        /// specifically <c>WeightedNonMaxSuppression</c> in the original <c>non_max_suppression_calculator.cc</c>.
        /// </summary>
        /// <remarks>
        /// In <c>PoseDetectorGraphOptions</c>, the original graph uses
        /// <c>overlap_type=INTERSECTION_OVER_UNION</c> and <c>algorithm=WEIGHTED</c>.
        /// Candidates whose overlap exceeds <c>min_suppression_threshold</c> are merged into one detection
        /// by score-weighted averaging, while the output score keeps the anchor detection score
        /// with the highest confidence, matching the original behavior.
        /// Score thresholding is assumed to have already been applied upstream by
        /// <see cref="TensorsToDetectionsCalculator"/> through <c>min_score_thresh</c>.
        /// </remarks>
        /// <param name="boxXywh">Rows of <c>(xmin, ymin, width, height)</c> in letterbox-normalized coordinates.</param>
        /// <param name="score">One scalar confidence per row after <c>TensorsToDetectionsCalculator</c>.</param>
        /// <param name="boxAndLandmarkNx12">Rows of 12 elements: 4 bbox values plus 8 keypoint values in letterbox-normalized coordinates.</param>
        void NonMaxSuppressionCalculator(Mat boxXywh, Mat score, Mat boxAndLandmarkNx12, List<PoseDetectionData> outLetterboxed)
        {
            const float kPoseMinSuppressionThreshold = 0.5f;

            outLetterboxed.Clear();
            int num = boxXywh.rows();
            if (num <= 0 || score == null || score.rows() < num)
                return;

            _poseWnmsIndexed.Clear();
            for (int i = 0; i < num; i++)
            {
                float sc = score.at<float>(i, 0)[0];
                _poseWnmsIndexed.Add((i, sc));
            }

            _poseWnmsIndexed.Sort((a, b) => b.sc.CompareTo(a.sc));

            _poseWnmsRemained.Clear();
            _poseWnmsRemained.AddRange(_poseWnmsIndexed);

            while (_poseWnmsRemained.Count > 0)
            {
                int originalSize = _poseWnmsRemained.Count;
                var anchor = _poseWnmsRemained[0];

                float ax = boxXywh.at<float>(anchor.idx, 0)[0];
                float ay = boxXywh.at<float>(anchor.idx, 1)[0];
                float aw = boxXywh.at<float>(anchor.idx, 2)[0];
                float ah = boxXywh.at<float>(anchor.idx, 3)[0];

                _poseWnmsNextRemained.Clear();
                for (int t = 0; t < _poseWnmsRemained.Count; t++)
                {
                    var item = _poseWnmsRemained[t];
                    float bx = boxXywh.at<float>(item.idx, 0)[0];
                    float by = boxXywh.at<float>(item.idx, 1)[0];
                    float bw = boxXywh.at<float>(item.idx, 2)[0];
                    float bh = boxXywh.at<float>(item.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) > kPoseMinSuppressionThreshold)
                        continue;
                    _poseWnmsNextRemained.Add(item);
                }

                float wXmin = 0f, wYmin = 0f, wXmax = 0f, wYmax = 0f;
                float totalScore = 0f;
                Span<float> kpAcc = _poseWnmsKpAcc8.AsSpan(0, 8);
                kpAcc.Clear();
                for (int t = 0; t < _poseWnmsRemained.Count; t++)
                {
                    var c = _poseWnmsRemained[t];
                    float bx = boxXywh.at<float>(c.idx, 0)[0];
                    float by = boxXywh.at<float>(c.idx, 1)[0];
                    float bw = boxXywh.at<float>(c.idx, 2)[0];
                    float bh = boxXywh.at<float>(c.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) <= kPoseMinSuppressionThreshold)
                        continue;

                    float s = c.sc;
                    totalScore += s;
                    float xmin = bx;
                    float ymin = by;
                    float w = bw;
                    float h = bh;
                    wXmin += xmin * s;
                    wYmin += ymin * s;
                    wXmax += (xmin + w) * s;
                    wYmax += (ymin + h) * s;
                    boxAndLandmarkNx12.get(c.idx, 0, _poseWnmsRowBuf12);
                    for (int k = 0; k < 8; k++)
                        kpAcc[k] += _poseWnmsRowBuf12[4 + k] * s;
                }

                if (totalScore <= 0f)
                    break;

                float outXmin = wXmin / totalScore;
                float outYmin = wYmin / totalScore;
                float outW = wXmax / totalScore - outXmin;
                float outH = wYmax / totalScore - outYmin;
                var relKp = new float[8];
                for (int k = 0; k < 8; k++)
                    relKp[k] = kpAcc[k] / totalScore;

                outLetterboxed.Add(new PoseDetectionData
                {
                    RelXmin = outXmin,
                    RelYmin = outYmin,
                    RelWidth = outW,
                    RelHeight = outH,
                    Score = anchor.sc,
                    RelKeypointsXy = relKp,
                });

                if (originalSize == _poseWnmsNextRemained.Count)
                    break;

                (_poseWnmsRemained, _poseWnmsNextRemained) = (_poseWnmsNextRemained, _poseWnmsRemained);
            }
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
        /// Equivalent to <c>DetectionLetterboxRemovalCalculator</c>.
        /// Converts coordinates in the letterboxed tensor space into normalized image coordinates after letterbox removal.
        /// </summary>
        List<PoseDetectionData> DetectionLetterboxRemovalCalculator(List<PoseDetectionData> detections, float[] letterboxPadding)
        {
            float left = letterboxPadding[0];
            float top = letterboxPadding[1];
            float lr = letterboxPadding[0] + letterboxPadding[2];
            float tb = letterboxPadding[1] + letterboxPadding[3];
            float invW = 1.0f / (1.0f - lr);
            float invH = 1.0f / (1.0f - tb);

            var result = new List<PoseDetectionData>(detections.Count);
            for (int i = 0; i < detections.Count; i++)
            {
                var d = detections[i];
                var o = new PoseDetectionData
                {
                    Score = d.Score,
                    RelKeypointsXy = new float[8],
                };
                o.RelXmin = (d.RelXmin - left) * invW;
                o.RelYmin = (d.RelYmin - top) * invH;
                o.RelWidth = d.RelWidth * invW;
                o.RelHeight = d.RelHeight * invH;
                for (int k = 0; k < 8; k += 2)
                {
                    o.RelKeypointsXy[k] = (d.RelKeypointsXy[k] - left) * invW;
                    o.RelKeypointsXy[k + 1] = (d.RelKeypointsXy[k + 1] - top) * invH;
                }
                result.Add(o);
            }
            return result;
        }

        /// <summary>
        /// Equivalent to <c>AlignmentPointsRectsCalculator</c>,
        /// specifically <c>DetectionToNormalizedRect</c> in <c>alignment_points_to_rects_calculator.cc</c>.
        /// </summary>
        List<NormalizedRect> AlignmentPointsRectsCalculator(List<PoseDetectionData> detections, int imageWidth, int imageHeight)
        {
            var rects = new List<NormalizedRect>(detections.Count);
            for (int i = 0; i < detections.Count; i++)
            {
                var d = detections[i];
                float kx0 = d.RelKeypointsXy[0] * imageWidth;
                float ky0 = d.RelKeypointsXy[1] * imageHeight;
                float kx1 = d.RelKeypointsXy[2] * imageWidth;
                float ky1 = d.RelKeypointsXy[3] * imageHeight;

                float boxSize = Mathf.Sqrt((kx1 - kx0) * (kx1 - kx0) + (ky1 - ky0) * (ky1 - ky0)) * 2.0f;

                float rot = NormalizePoseRadians(Mathf.PI * 0.5f - Mathf.Atan2(-(ky1 - ky0), kx1 - kx0));

                rects.Add(new NormalizedRect
                {
                    XCenter = d.RelKeypointsXy[0],
                    YCenter = d.RelKeypointsXy[1],
                    Width = boxSize / imageWidth,
                    Height = boxSize / imageHeight,
                    Rotation = rot,
                    RectId = null,
                });
            }
            return rects;
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> with
        /// <c>ConfigureRectTransformationCalculator</c> options <c>scale = 1.25</c> and <c>square_long</c>.
        /// </summary>
        List<NormalizedRect> RectTransformationCalculator(List<NormalizedRect> poseRects, int imageWidth, int imageHeight)
        {
            var result = new List<NormalizedRect>(poseRects.Count);
            for (int i = 0; i < poseRects.Count; i++)
            {
                result.Add(RectTransformationCalculator(poseRects[i], imageWidth, imageHeight));
            }
            return result;
        }

        /// <summary>
        /// Single-rectangle overload of <see cref="RectTransformationCalculator"/>.
        /// Follows the same order as the original <c>TransformNormalizedRect</c> in <c>rect_transformation_calculator.cc</c>.
        /// </summary>
        /// <remarks>
        /// Equivalent options: <c>shift_x/y = 0</c>, <c>square_long = true</c>, and <c>scale_x/y = 1.25</c>.
        /// </remarks>
        NormalizedRect RectTransformationCalculator(NormalizedRect rect, int imageWidth, int imageHeight)
        {
            float width = rect.Width;
            float height = rect.Height;
            float rotation = rect.Rotation;
            const float shiftXOpt = 0f;
            const float shiftYOpt = 0f;
            const float scaleX = 1.25f;
            const float scaleY = 1.25f;

            float cx = rect.XCenter;
            float cy = rect.YCenter;
            if (rotation == 0f)
            {
                cx += width * shiftXOpt;
                cy += height * shiftYOpt;
            }
            else
            {
                float xShift = (imageWidth * width * shiftXOpt * Mathf.Cos(rotation) - imageHeight * height * shiftYOpt * Mathf.Sin(rotation)) / imageWidth;
                float yShift = (imageWidth * width * shiftXOpt * Mathf.Sin(rotation) + imageHeight * height * shiftYOpt * Mathf.Cos(rotation)) / imageHeight;
                cx += xShift;
                cy += yShift;
            }

            float longSide = Mathf.Max(width * imageWidth, height * imageHeight);
            width = longSide / imageWidth;
            height = longSide / imageHeight;

            return new NormalizedRect
            {
                XCenter = cx,
                YCenter = cy,
                Width = width * scaleX,
                Height = height * scaleY,
                Rotation = rotation,
                RectId = rect.RectId,
            };
        }

        /// <summary>
        /// Equivalent to <c>ClipDetectionVectorSizeCalculator</c>,
        /// i.e. <c>ClipVectorSizeCalculator</c> for detection vectors.
        /// </summary>
        List<PoseDetectionData> ClipDetectionVectorSizeCalculator(List<PoseDetectionData> detections, int maxVecSize)
        {
            if (detections == null || detections.Count <= maxVecSize)
                return detections != null ? new List<PoseDetectionData>(detections) : new List<PoseDetectionData>();

            var clipped = new List<PoseDetectionData>(detections);
            clipped.RemoveRange(maxVecSize, clipped.Count - maxVecSize);
            return clipped;
        }

        static float NormalizePoseRadians(float angleRadians)
        {
            return angleRadians - 2f * Mathf.PI * Mathf.Floor((angleRadians - (-Mathf.PI)) / (2f * Mathf.PI));
        }

        static void NumpyClip(Mat a, double aMin, double aMax)
        {
            Core.min(a, (aMax, 0, 0, 0), a);
            Core.max(a, (aMin, 0, 0, 0), a);
        }

        /// <summary>
        /// Equivalent to <c>GetImageSize</c> from <c>mediapipe/framework/api2/stream/image_size.h</c>.
        /// In the original graph, one <c>ImagePropertiesCalculator</c> node is added during graph construction
        /// and its <c>SIZE</c> output is connected as a <c>std::pair&lt;int, int&gt;</c> stream.
        /// In C#, this method returns the input <see cref="Mat"/> column count and row count
        /// as width and height, matching the order of the <c>ImagePropertiesCalculator</c> SIZE output.
        /// </summary>
        /// <returns>Width (<see cref="Mat.cols"/>) and height (<see cref="Mat.rows"/>). Returns (0, 0) when <paramref name="image"/> is null or empty.</returns>
        static (int Width, int Height) GetImageSize(Mat image)
        {
            if (image == null || image.empty())
                return (0, 0);
            return (image.cols(), image.rows());
        }

        /// <summary>
        /// Equivalent to <c>ConstantSidePacketCalculator</c> from
        /// <c>mediapipe/calculators/core/constant_side_packet_calculator.cc</c>.
        /// The original graph registers <c>int_value</c> as a side packet.
        /// In C#, this method simply returns that integer directly.
        /// </summary>
        static int ConstantSidePacketCalculator(int intValue) => intValue;

        /// <summary>
        /// Equivalent to <c>SidePacketToStreamCalculator</c> from
        /// <c>mediapipe/calculators/core/side_packet_to_stream_calculator.cc</c>.
        /// The original graph emits the side-packet value to <c>AT_TICK</c> in sync with the TICK stream.
        /// Here, one call to <see cref="PoseLandmarkSmoothingPipeline.ApplyPostEndLoop"/> corresponds to one frame boundary,
        /// so <paramref name="packetValue"/> is returned unchanged.
        /// </summary>
        static int SidePacketToStreamCalculator(int packetValue) => packetValue;

        /// <summary>
        /// Uses the same composition order as <c>CreateIntConstantStream</c> in
        /// <c>pose_landmarks_detector_graph.cc</c>:
        /// <see cref="ConstantSidePacketCalculator"/> → <see cref="SidePacketToStreamCalculator"/>.
        /// As in the original comments, smoothing is applied only to the first pose,
        /// so <paramref name="constantInt"/> is typically <c>0</c>.
        /// </summary>
        static int CreateIntConstantStream(int constantInt)
        {
            int packet = ConstantSidePacketCalculator(constantInt);
            return SidePacketToStreamCalculator(packet);
        }

        /// <summary>
        /// Equivalent to <c>GetNormalizedLandmarkListVectorItemCalculator</c>,
        /// i.e. api2 <c>GetItem</c> as registered in <c>get_vector_item_calculator.cc</c>.
        /// The original graph extracts one item from <c>VECTOR</c> (<c>std::vector&lt;NormalizedLandmarkList&gt;</c>)
        /// using <c>INDEX</c>.
        /// In C#, this returns <see cref="PoseResult.NormLandmarks"/> at <paramref name="index"/> from <c>List&lt;PoseResult&gt;</c>.
        /// </summary>
        static Vec3f[] GetNormalizedLandmarkListVectorItemCalculator(List<PoseResult> poseResults, int index)
        {
            if (poseResults == null || (uint)index >= (uint)poseResults.Count)
                return null;
            return poseResults[index].NormLandmarks;
        }

        /// <summary>
        /// Equivalent to <c>GetLandmarkListVectorItemCalculator</c>,
        /// i.e. api2 <c>GetItem</c> as registered in <c>get_vector_item_calculator.cc</c>.
        /// The original graph extracts one item from <c>std::vector&lt;LandmarkList&gt;</c> by index.
        /// In C#, this returns <see cref="PoseResult.WorldLandmarks"/> at <paramref name="index"/>.
        /// </summary>
        static Vec3f[] GetLandmarkListVectorItemCalculator(List<PoseResult> poseResults, int index)
        {
            if (poseResults == null || (uint)index >= (uint)poseResults.Count)
                return null;
            return poseResults[index].WorldLandmarks;
        }

        /// <summary>
        /// Equivalent to <c>GetNormalizedRectVectorItemCalculator</c>,
        /// i.e. api2 <c>GetItem</c> as registered in <c>get_vector_item_calculator.cc</c>.
        /// The original graph extracts one item from <c>std::vector&lt;NormalizedRect&gt;</c>
        /// for <c>pose_rects_next_frame</c>.
        /// In C#, this returns <see cref="PoseResult.NextFrameRect"/> at <paramref name="index"/>.
        /// </summary>
        static NormalizedRect GetNormalizedRectVectorItemCalculator(List<PoseResult> poseResults, int index)
        {
            if (poseResults == null || (uint)index >= (uint)poseResults.Count)
                return default;
            return poseResults[index].NextFrameRect;
        }

        /// <summary>
        /// Equivalent to <c>ClipNormalizedRectVectorSizeCalculator</c>.
        /// Clips the normalized rectangle vector to at most <paramref name="maxVecSize"/> elements,
        /// matching the original <c>ClipVectorSizeCalculator</c>.
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
        /// Equivalent to <c>ClipNormalizedRectVectorSizeCalculator</c> using <c>num_poses</c>.
        /// </summary>
        List<NormalizedRect> ClipNormalizedRectVectorSizeCalculator(List<NormalizedRect> rects)
        {
            return ClipNormalizedRectVectorSizeCalculator(rects, _numPoses);
        }

        /// <summary>
        /// Preprocessing result for one pose ROI from <c>ImagePreprocessingGraph_SinglePoseLandmarks</c>.
        /// </summary>
        struct SinglePoseLandmarkPreprocessOut
        {
            public Mat PoseBlob;
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
        /// Decoded data for one joint from the original <c>TensorsToLandmarksCalculator</c>, with up to five dimensions.
        /// </summary>
        struct PoseLandmarkDecoded
        {
            public float X, Y, Z, Visibility, Presence;
        }

        /// <summary>
        /// Equivalent to <c>MultiplePoseLandmarksDetectorGraph</c> from
        /// <c>mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarks_detector_graph.cc</c>.
        /// This method only coordinates the BeginLoop / EndLoop structure and delegates
        /// each calculator to its dedicated method.
        ///
        /// Mapping to the original graph:
        /// - BeginLoopNormalizedRectCalculator → <see cref="BeginLoopNormalizedRectCalculator"/>
        /// - SinglePoseLandmarksDetectorGraph → <see cref="SinglePoseLandmarksDetectorGraph"/>
        /// - EndLoopNormalizedLandmarkListVectorCalculator (LANDMARKS) → <see cref="EndLoopNormalizedLandmarkListVectorCalculator"/>
        /// - EndLoopLandmarkListVectorCalculator (WORLD_LANDMARKS) → <see cref="EndLoopLandmarkListVectorCalculator"/>
        /// - EndLoopNormalizedLandmarkListVectorCalculator (AUXILIARY) → <see cref="EndLoopNormalizedLandmarkListVectorCalculator"/>
        /// - EndLoopNormalizedRectCalculator (POSE_RECTS_NEXT_FRAME) → <see cref="EndLoopNormalizedRectCalculator"/>
        /// - EndLoopBooleanCalculator (PRESENCE) → <see cref="EndLoopBooleanCalculator"/>
        /// - EndLoopImageCalculator (SEGMENTATION_MASK) → <see cref="EndLoopImageCalculator(int, int, int)"/> / <see cref="EndLoopImageCalculator(int, int, int, Mat, bool, ref PoseResult)"/>
        /// - EndLoop vector association is done in this C# port by constructing one <see cref="PoseResult"/> per loop iteration.
        /// - The equivalent of <c>output_segmentation_masks</c> writes into slots of <see cref="_segmentationMaskFullBySlot"/>, referenced via <see cref="PoseResult.SegmentationMaskSlotIndex"/>.
        /// - When <c>smooth_landmarks</c> is enabled and <c>num_poses == 1</c>, loop-external smoothing is handled by <c>PoseLandmarkSmoothingPipeline</c> (<c>VisibilitySmoothingCalculator</c> → <c>LandmarksSmoothingCalculator</c> for two streams). In the original graph, <c>Concatenate*VectorCalculator</c> operates on a single vector element, so this implementation replaces the first result list item.
        /// - GetImageSize (api2) → <see cref="GetImageSize"/> at the start of this method and in <see cref="PoseLandmarkSmoothingPipeline.ApplyPostEndLoop"/>.
        /// - ConstantSidePacketCalculator / SidePacketToStreamCalculator via <see cref="CreateIntConstantStream"/> → smoothing index, fixed to <c>0</c> as in the original graph.
        /// - GetItem → <see cref="GetNormalizedLandmarkListVectorItemCalculator"/> / <see cref="GetLandmarkListVectorItemCalculator"/> / <see cref="GetNormalizedRectVectorItemCalculator"/> in <see cref="PoseLandmarkSmoothingPipeline.ApplyPostEndLoop"/>.
        /// </summary>
        List<PoseResult> MultiplePoseLandmarksDetectorGraph(Mat image, List<NormalizedRect> poseRects)
        {
            var (iw, ih) = GetImageSize(image);
            var merged = new List<PoseResult>();
            int slot = 0;

            foreach (var loopItem in BeginLoopNormalizedRectCalculator(image, poseRects))
            {
                Mat segPlane = EndLoopImageCalculator(slot, iw, ih);

                PoseResult? singleOpt = SinglePoseLandmarksDetectorGraph(loopItem.Image, loopItem.PoseRect, segPlane);
                PoseResult pr = singleOpt ?? CreateAbsentPoseResultPlaceholder();
                EndLoopImageCalculator(slot, iw, ih, segPlane, singleOpt.HasValue, ref pr);
                merged.Add(pr);
                slot++;
            }

            if (_poseLandmarkSmoothingPipeline != null)
            {
                if (_numPoses == 1 && merged.Count >= 1)
                    _poseLandmarkSmoothingPipeline.ApplyPostEndLoop(image, merged);
                else if (merged.Count == 0)
                    _poseLandmarkSmoothingPipeline.ResetAll();
            }

            return merged;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="MultiplePoseLandmarksDetectorGraph"/> using the Unity Inference Engine path with <see cref="SinglePoseLandmarksDetectorGraphAsync"/>.
        /// </summary>
        async Task<List<PoseResult>> MultiplePoseLandmarksDetectorGraphAsync(Mat image, List<NormalizedRect> poseRects, CancellationToken cancellationToken)
        {
            var (iw, ih) = GetImageSize(image);
            var merged = new List<PoseResult>();
            int slot = 0;

            foreach (var loopItem in BeginLoopNormalizedRectCalculator(image, poseRects))
            {
                Mat segPlane = EndLoopImageCalculator(slot, iw, ih);

                PoseResult? singleOpt = await SinglePoseLandmarksDetectorGraphAsync(loopItem.Image, loopItem.PoseRect, segPlane, cancellationToken);
                PoseResult pr = singleOpt ?? CreateAbsentPoseResultPlaceholder();
                EndLoopImageCalculator(slot, iw, ih, segPlane, singleOpt.HasValue, ref pr);
                merged.Add(pr);
                slot++;
            }

            if (_poseLandmarkSmoothingPipeline != null)
            {
                if (_numPoses == 1 && merged.Count >= 1)
                    _poseLandmarkSmoothingPipeline.ApplyPostEndLoop(image, merged);
                else if (merged.Count == 0)
                    _poseLandmarkSmoothingPipeline.ResetAll();
            }

            return merged;
        }

#endif
        /// <summary>
        /// Placeholder used when preprocessing or inference fails.
        /// Preserves the original EndLoop vector length so it still matches the number of input poses.
        /// </summary>
        static PoseResult CreateAbsentPoseResultPlaceholder()
        {
            int L = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            return new PoseResult
            {
                PosePresence = false,
                PosePresenceScore = 0f,
                NormLandmarks = new Vec3f[L],
                WorldLandmarks = new Vec3f[L],
                AuxiliaryLandmarks = new Vec3f[2],
                LandmarkVisibility = new float[L],
                LandmarkVisibilityWorld = new float[L],
                LandmarkPresence = new float[L],
                NextFrameRect = new NormalizedRect(),
                SegmentationMaskSlotIndex = -1,
            };
        }

        /// <summary>
        /// Ensures that <see cref="_segmentationMaskFullBySlot"/>[<paramref name="slotIndex"/>]
        /// is a <c>CV_32FC1</c> buffer of size <paramref name="width"/> x <paramref name="height"/>.
        /// If the existing buffer already has the same type and resolution, the <see cref="Mat"/> is reused.
        /// </summary>
        void EnsureSegmentationMaskFullSlot(int slotIndex, int width, int height)
        {
            if (_segmentationMaskFullBySlot == null || (uint)slotIndex >= (uint)_segmentationMaskFullBySlot.Length)
                return;
            if (width <= 0 || height <= 0)
                return;

            Mat m = _segmentationMaskFullBySlot[slotIndex];
            if (m == null || m.rows() != height || m.cols() != width || m.type() != CvType.CV_32FC1)
            {
                m?.Dispose();
                _segmentationMaskFullBySlot[slotIndex] = new Mat(height, width, CvType.CV_32FC1);
            }
        }

        /// <summary>
        /// Equivalent to <c>BeginLoopNormalizedRectCalculator</c>.
        /// Feeds IMAGE to <c>CLONE</c> and the pose rectangle list to <c>ITERABLE</c>,
        /// then enumerates each iteration item together with the shared image reference.
        /// </summary>
        static IEnumerable<(Mat Image, NormalizedRect PoseRect)> BeginLoopNormalizedRectCalculator(Mat image, List<NormalizedRect> poseRects)
        {
            if (image == null || poseRects == null)
                yield break;
            foreach (var r in poseRects)
                yield return (image, r);
        }

        static void EndLoopNormalizedLandmarkListVectorCalculator(List<Vec3f[]> iterable, Vec3f[] item)
        {
            iterable.Add(item ?? new Vec3f[PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT]);
        }

        static void EndLoopLandmarkListVectorCalculator(List<Vec3f[]> iterable, Vec3f[] item)
        {
            iterable.Add(item ?? new Vec3f[PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT]);
        }

        static void EndLoopNormalizedRectCalculator(List<NormalizedRect> iterable, NormalizedRect item)
        {
            iterable.Add(item);
        }

        static void EndLoopBooleanCalculator(List<bool> iterable, bool item)
        {
            iterable.Add(item);
        }

        static void EndLoopFloatCalculator(List<float> iterable, float item)
        {
            iterable.Add(item);
        }

        /// <summary>
        /// Equivalent to <c>EndLoopImageCalculator</c> at iteration start.
        /// In the original <c>EndLoopCalculator&lt;std::vector&lt;Image&gt;&gt;</c> from
        /// <c>mediapipe/calculators/core/end_loop_calculator.cc</c>, each iteration image is aggregated
        /// into a vector at <c>BATCH_END</c>.
        /// In this C# port, full-image masks of identical resolution are held in <see cref="_segmentationMaskFullBySlot"/>,
        /// so this method allocates and returns the writable <see cref="Mat"/> plane (<c>CV_32FC1</c>)
        /// for iteration <paramref name="slotIndex"/> via <see cref="EnsureSegmentationMaskFullSlot"/>.
        /// </summary>
        /// <returns>The plane into which <see cref="SinglePoseLandmarksDetectorGraph"/> writes segmentation data. Returns null when disabled.</returns>
        Mat EndLoopImageCalculator(int slotIndex, int imageWidth, int imageHeight)
        {
            if (!_outputSegmentationMasks || imageWidth <= 0 || imageHeight <= 0 || slotIndex >= _numPoses)
                return null;
            EnsureSegmentationMaskFullSlot(slotIndex, imageWidth, imageHeight);
            return _segmentationMaskFullBySlot?[slotIndex];
        }

        /// <summary>
        /// Equivalent to <c>EndLoopImageCalculator</c> at iteration end.
        /// As the finalize step corresponding to one vector element in the original graph,
        /// this method zero-fills the plane when the single-pose subgraph failed and sets
        /// <see cref="PoseResult.SegmentationMaskSlotIndex"/> to the slot number,
        /// or <c>-1</c> when segmentation is disabled.
        /// </summary>
        void EndLoopImageCalculator(int slotIndex, int imageWidth, int imageHeight, Mat segmentationFullPlane, bool singlePoseGraphSucceeded, ref PoseResult poseResult)
        {
            if (segmentationFullPlane != null && !singlePoseGraphSucceeded)
                segmentationFullPlane.setTo((0d, 0d, 0d, 0d));
            if (_outputSegmentationMasks && imageWidth > 0 && imageHeight > 0 && slotIndex < _numPoses)
                poseResult.SegmentationMaskSlotIndex = slotIndex;
            else
                poseResult.SegmentationMaskSlotIndex = -1;
        }

        /// <summary>
        /// Equivalent to <c>SinglePoseLandmarksDetectorGraph</c>, corresponding to
        /// <c>BuildSinglePoseLandmarksDetectorGraph</c> in
        /// <c>mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarks_detector_graph.cc</c>.
        /// This method only invokes lower-level calculators in the same order as the original graph.
        ///
        /// Subgraph-to-method mapping:
        /// - ImagePreprocessingGraph → <see cref="ImagePreprocessingGraph_SinglePoseLandmarks"/>
        /// - Inference → <see cref="InferenceSubgraph_PoseLandmarks"/>
        /// - SplitTensorVectorCalculator → <see cref="SplitTensorVectorCalculator_PoseLandmarks"/>
        /// - TensorsToFloatsCalculator → <see cref="TensorsToFloatsCalculator_PosePresence"/>
        /// - ThresholdingCalculator → <see cref="ThresholdingCalculator_PosePresence"/>
        /// - GateCalculator → <see cref="GateCalculator_PoseLandmarkTensors"/> (branching only)
        /// - TensorsToLandmarksCalculator (image landmarks) → <see cref="TensorsToLandmarksCalculator_PoseImage"/>
        /// - RefineLandmarksFromHeatmapCalculator → <see cref="RefineLandmarksFromHeatmapCalculator"/>
        /// - SplitNormalizedLandmarkListCalculator → <see cref="SplitNormalizedLandmarkListCalculator"/>
        /// - TensorsToLandmarksCalculator (world landmarks) → <see cref="TensorsToLandmarksCalculator_PoseWorld"/>
        /// - SplitLandmarkListCalculator → <see cref="SplitLandmarkListCalculator_PoseWorld"/>
        /// - VisibilityCopyCalculator → <see cref="VisibilityCopyCalculator_PoseWorld"/>
        /// - LandmarkLetterboxRemovalCalculator → <see cref="LandmarkLetterboxRemovalCalculator_Pose"/>
        /// - LandmarkProjectionCalculator → <see cref="LandmarkProjectionCalculator_Pose"/>
        /// - WorldLandmarkProjectionCalculator → <see cref="WorldLandmarkProjectionCalculator_Pose"/>
        /// - LandmarksToDetectionCalculator → <see cref="LandmarksToDetectionCalculator"/>
        /// - AlignmentPointsRectsCalculator → <see cref="AlignmentPointsRectsCalculator"/>
        /// - RectTransformationCalculator → <see cref="RectTransformationCalculator"/> (single-rectangle overload)
        /// - TensorsToSegmentationCalculator → <see cref="TensorsToSegmentationCalculator_Pose"/> when <c>output_segmentation_masks</c> is enabled
        /// - Inverse transform plus projection back to the full image: equivalent to the original <c>InverseMatrixCalculator</c> + <c>WarpAffineCalculator</c>, combined here into a 3x3-projective <c>warpPerspective</c>.
        /// </summary>
        /// <param name="segmentationFullPlane">
        /// Non-null only when segmentation output is enabled.
        /// Holds a <c>CV_32FC1</c> plane with the same size as the input image and is zero-filled when no pose is detected.
        /// </param>
        PoseResult? SinglePoseLandmarksDetectorGraph(Mat image, NormalizedRect poseRect, Mat segmentationFullPlane)
        {
            void ZeroSegmentationPlane()
            {
                segmentationFullPlane?.setTo((0d, 0d, 0d, 0d));
            }

            if (!ImagePreprocessingGraph_SinglePoseLandmarks(image, poseRect, out SinglePoseLandmarkPreprocessOut pre))
            {
                ZeroSegmentationPlane();
                return null;
            }

            var inferenceTensors = InferenceSubgraph_PoseLandmarks(pre.PoseBlob);
            if (inferenceTensors == null || inferenceTensors.Count < kPoseLandmarkModelTensorSplitCount)
            {
                ZeroSegmentationPlane();
                return null;
            }

            if (!SplitTensorVectorCalculator_PoseLandmarks(inferenceTensors,
                    out Mat landmarkTensors, out Mat poseFlagTensors, out Mat segmentationTensors,
                    out Mat heatmapTensors, out Mat worldLandmarkTensors))
            {
                ZeroSegmentationPlane();
                return null;
            }

            float posePresenceScore = TensorsToFloatsCalculator_PosePresence(poseFlagTensors);
            bool posePresence = ThresholdingCalculator_PosePresence(posePresenceScore);

            if (!GateCalculator_PoseLandmarkTensors(posePresence))
            {
                ZeroSegmentationPlane();
                int Lz = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
                return new PoseResult
                {
                    PosePresence = false,
                    PosePresenceScore = posePresenceScore,
                    NormLandmarks = new Vec3f[Lz],
                    WorldLandmarks = new Vec3f[Lz],
                    AuxiliaryLandmarks = new Vec3f[2],
                    LandmarkVisibility = new float[Lz],
                    LandmarkVisibilityWorld = new float[Lz],
                    LandmarkPresence = new float[Lz],
                    NextFrameRect = default,
                    SegmentationMaskSlotIndex = -1,
                };
            }

            PoseLandmarkDecoded[] decoded = TensorsToLandmarksCalculator_PoseImage(landmarkTensors, pre.ModelW, pre.ModelH);
            PoseLandmarkDecoded[] refined = RefineLandmarksFromHeatmapCalculator(decoded, heatmapTensors);

            SplitNormalizedLandmarkListCalculator(refined, out PoseLandmarkDecoded[] mainLm, out PoseLandmarkDecoded[] auxLm);

            PoseLandmarkDecoded[] worldRaw = TensorsToLandmarksCalculator_PoseWorld(worldLandmarkTensors);
            SplitLandmarkListCalculator_PoseWorld(worldRaw, out PoseLandmarkDecoded[] world33);
            VisibilityCopyCalculator_PoseWorld(mainLm, world33);

            PoseLandmarkDecoded[] mainAfterLb = LandmarkLetterboxRemovalCalculator_Pose(mainLm, pre);
            PoseLandmarkDecoded[] auxAfterLb = LandmarkLetterboxRemovalCalculator_Pose(auxLm, pre);

            Vec3f[] projected = LandmarkProjectionCalculator_Pose(mainAfterLb, poseRect);
            Vec3f[] auxProjected = LandmarkProjectionCalculator_PoseAux(auxAfterLb, poseRect);
            Vec3f[] worldProj = WorldLandmarkProjectionCalculator_Pose(world33, poseRect);

            int Lm = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            var visibility = new float[Lm];
            var visibilityWorld = new float[Lm];
            var presenceLm = new float[Lm];
            for (int i = 0; i < Lm; i++)
            {
                visibility[i] = mainAfterLb[i].Visibility;
                visibilityWorld[i] = world33[i].Visibility;
                presenceLm[i] = mainAfterLb[i].Presence;
            }

            var det = LandmarksToDetectionCalculator(auxProjected);
            var poseRects = AlignmentPointsRectsCalculator(new List<PoseDetectionData> { det }, pre.ImageW, pre.ImageH);
            NormalizedRect nextFrame = poseRects.Count > 0
                ? RectTransformationCalculator(poseRects[0], pre.ImageW, pre.ImageH)
                : new NormalizedRect();

            if (segmentationFullPlane != null)
                SegmentationMaskFromTensorToFullImage(segmentationTensors, pre, segmentationFullPlane);

            return new PoseResult
            {
                PosePresence = true,
                PosePresenceScore = posePresenceScore,
                NormLandmarks = projected,
                WorldLandmarks = worldProj,
                AuxiliaryLandmarks = auxProjected,
                LandmarkVisibility = visibility,
                LandmarkVisibilityWorld = visibilityWorld,
                LandmarkPresence = presenceLm,
                NextFrameRect = nextFrame,
                SegmentationMaskSlotIndex = -1,
            };
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="SinglePoseLandmarksDetectorGraph"/> using the Unity Inference Engine path with <see cref="InferenceSubgraph_PoseLandmarksAsync"/> (<see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// </summary>
        async Task<PoseResult?> SinglePoseLandmarksDetectorGraphAsync(Mat image, NormalizedRect poseRect, Mat segmentationFullPlane, CancellationToken cancellationToken)
        {
            void ZeroSegmentationPlane()
            {
                segmentationFullPlane?.setTo((0d, 0d, 0d, 0d));
            }

            if (!ImagePreprocessingGraph_SinglePoseLandmarks(image, poseRect, out SinglePoseLandmarkPreprocessOut pre))
            {
                ZeroSegmentationPlane();
                return null;
            }

            var inferenceTensors = await InferenceSubgraph_PoseLandmarksAsync(pre.PoseBlob, cancellationToken);
            if (inferenceTensors == null || inferenceTensors.Count < kPoseLandmarkModelTensorSplitCount)
            {
                ZeroSegmentationPlane();
                return null;
            }

            if (!SplitTensorVectorCalculator_PoseLandmarks(inferenceTensors,
                    out Mat landmarkTensors, out Mat poseFlagTensors, out Mat segmentationTensors,
                    out Mat heatmapTensors, out Mat worldLandmarkTensors))
            {
                ZeroSegmentationPlane();
                return null;
            }

            float posePresenceScore = TensorsToFloatsCalculator_PosePresence(poseFlagTensors);
            bool posePresence = ThresholdingCalculator_PosePresence(posePresenceScore);

            if (!GateCalculator_PoseLandmarkTensors(posePresence))
            {
                ZeroSegmentationPlane();
                int Lz = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
                return new PoseResult
                {
                    PosePresence = false,
                    PosePresenceScore = posePresenceScore,
                    NormLandmarks = new Vec3f[Lz],
                    WorldLandmarks = new Vec3f[Lz],
                    AuxiliaryLandmarks = new Vec3f[2],
                    LandmarkVisibility = new float[Lz],
                    LandmarkVisibilityWorld = new float[Lz],
                    LandmarkPresence = new float[Lz],
                    NextFrameRect = default,
                    SegmentationMaskSlotIndex = -1,
                };
            }

            PoseLandmarkDecoded[] decoded = TensorsToLandmarksCalculator_PoseImage(landmarkTensors, pre.ModelW, pre.ModelH);
            PoseLandmarkDecoded[] refined = RefineLandmarksFromHeatmapCalculator(decoded, heatmapTensors);

            SplitNormalizedLandmarkListCalculator(refined, out PoseLandmarkDecoded[] mainLm, out PoseLandmarkDecoded[] auxLm);

            PoseLandmarkDecoded[] worldRaw = TensorsToLandmarksCalculator_PoseWorld(worldLandmarkTensors);
            SplitLandmarkListCalculator_PoseWorld(worldRaw, out PoseLandmarkDecoded[] world33);
            VisibilityCopyCalculator_PoseWorld(mainLm, world33);

            PoseLandmarkDecoded[] mainAfterLb = LandmarkLetterboxRemovalCalculator_Pose(mainLm, pre);
            PoseLandmarkDecoded[] auxAfterLb = LandmarkLetterboxRemovalCalculator_Pose(auxLm, pre);

            Vec3f[] projected = LandmarkProjectionCalculator_Pose(mainAfterLb, poseRect);
            Vec3f[] auxProjected = LandmarkProjectionCalculator_PoseAux(auxAfterLb, poseRect);
            Vec3f[] worldProj = WorldLandmarkProjectionCalculator_Pose(world33, poseRect);

            int Lm = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            var visibility = new float[Lm];
            var visibilityWorld = new float[Lm];
            var presenceLm = new float[Lm];
            for (int i = 0; i < Lm; i++)
            {
                visibility[i] = mainAfterLb[i].Visibility;
                visibilityWorld[i] = world33[i].Visibility;
                presenceLm[i] = mainAfterLb[i].Presence;
            }

            var det = LandmarksToDetectionCalculator(auxProjected);
            var poseRects = AlignmentPointsRectsCalculator(new List<PoseDetectionData> { det }, pre.ImageW, pre.ImageH);
            NormalizedRect nextFrame = poseRects.Count > 0
                ? RectTransformationCalculator(poseRects[0], pre.ImageW, pre.ImageH)
                : new NormalizedRect();

            if (segmentationFullPlane != null)
                SegmentationMaskFromTensorToFullImage(segmentationTensors, pre, segmentationFullPlane);

            return new PoseResult
            {
                PosePresence = true,
                PosePresenceScore = posePresenceScore,
                NormLandmarks = projected,
                WorldLandmarks = worldProj,
                AuxiliaryLandmarks = auxProjected,
                LandmarkVisibility = visibility,
                LandmarkVisibilityWorld = visibilityWorld,
                LandmarkPresence = presenceLm,
                NextFrameRect = nextFrame,
                SegmentationMaskSlotIndex = -1,
            };
        }

#endif
        /// <summary>
        /// Projects the segmentation mask back to the full image,
        /// equivalent to <c>TensorsToSegmentationCalculator</c> with <c>activation: SIGMOID</c>
        /// followed by the original <c>InverseMatrixCalculator</c> / <c>WarpAffineCalculator</c>.
        /// Because the original graph inverts the projection matrix before warping,
        /// this implementation passes the <strong>inverse</strong> of <see cref="_singlePoseLandmarkProjMat3x3"/>;
        /// using the forward matrix directly would shrink the mask into the top-left corner.
        /// </summary>
        void SegmentationMaskFromTensorToFullImage(Mat segmentationTensor, in SinglePoseLandmarkPreprocessOut pre, Mat dstFullImageFloat01)
        {
            if (dstFullImageFloat01 == null || dstFullImageFloat01.empty())
                return;
            if (segmentationTensor == null || segmentationTensor.empty()
                || _singlePoseLandmarkProjMat3x3 == null || _singlePoseLandmarkProjMat3x3.empty())
            {
                dstFullImageFloat01.setTo((0d, 0d, 0d, 0d));
                return;
            }

            if (!TensorsToSegmentationCalculator_Pose(segmentationTensor, ref _segmentationScratchSmall))
            {
                dstFullImageFloat01.setTo((0d, 0d, 0d, 0d));
                return;
            }

            if (_segmentationFullWarpInvMat3x3 == null || _segmentationFullWarpInvMat3x3.rows() != 3 || _segmentationFullWarpInvMat3x3.cols() != 3
                || _segmentationFullWarpInvMat3x3.type() != CvType.CV_32FC1)
            {
                _segmentationFullWarpInvMat3x3?.Dispose();
                _segmentationFullWarpInvMat3x3 = new Mat(3, 3, CvType.CV_32FC1);
            }

            if (Core.invert(_singlePoseLandmarkProjMat3x3, _segmentationFullWarpInvMat3x3, Core.DECOMP_LU) == 0)
            {
                dstFullImageFloat01.setTo((0d, 0d, 0d, 0d));
                return;
            }

            Imgproc.warpPerspective(_segmentationScratchSmall, dstFullImageFloat01, _segmentationFullWarpInvMat3x3,
                (pre.ImageW, pre.ImageH), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
        }

        /// <summary>
        /// Equivalent to <c>TensorsToSegmentationCalculator</c> in
        /// <c>tensors_to_segmentation_converter_opencv.cc</c>, using a single channel and SIGMOID activation.
        /// Writes the tensor-space <c>H x W</c> mask into <paramref name="dstSmallMask"/> as <c>CV_32FC1</c>.
        /// </summary>
        /// <remarks>
        /// This assumes that the <c>1 x flatLen</c> reshape result of <paramref name="tensor"/>
        /// and <paramref name="dstSmallMask"/> both provide contiguous 32-bit float storage
        /// (<c>elemSize() == 4</c>) required by <see cref="Mat.AsSpan{T}"/>, which is normally true for OpenCV DNN outputs.
        /// If not, <see cref="Mat.AsSpan{T}"/> may throw.
        /// </remarks>
        static bool TensorsToSegmentationCalculator_Pose(Mat tensor, ref Mat dstSmallMask)
        {
            if (tensor == null || tensor.empty())
                return false;
            if (!TryGetSegmentationTensorHw(tensor, out int th, out int tw))
                return false;

            if (dstSmallMask == null || dstSmallMask.rows() != th || dstSmallMask.cols() != tw
                || dstSmallMask.type() != CvType.CV_32FC1)
            {
                dstSmallMask?.Dispose();
                dstSmallMask = new Mat(th, tw, CvType.CV_32FC1);
            }

            int planePixels = th * tw;
            int flatLen = (int)tensor.total();
            if (flatLen < planePixels)
                return false;

            using (Mat flat = tensor.reshape(1, flatLen))
            {
                Span<float> dst = dstSmallMask.AsSpan<float>();
                ReadOnlySpan<float> src = flat.AsSpan<float>();
                if (dst.Length < planePixels || src.Length < planePixels)
                    return false;

                for (int i = 0; i < planePixels; i++)
                    dst[i] = Sigmoid(src[i]);
            }

            return true;
        }

        /// <summary>
        /// Extracts the plane height and width from a segmentation tensor in NHWC, HWC, or HW layout.
        /// </summary>
        static bool TryGetSegmentationTensorHw(Mat t, out int h, out int w)
        {
            h = w = 0;
            int d = t.dims();
            if (d == 4)
            {
                int d0 = (int)t.size(0);
                int d1 = (int)t.size(1);
                int d2 = (int)t.size(2);
                int d3 = (int)t.size(3);
                // NCHW: 1x1xHxW
                if (d0 == 1 && d1 == 1 && d2 > 1 && d3 > 1)
                {
                    h = d2;
                    w = d3;
                    return true;
                }
                // NHWC: 1xHxWxC (C >= 1)
                if (d0 == 1 && d1 > 1 && d2 > 1)
                {
                    h = d1;
                    w = d2;
                    return true;
                }
                h = d2;
                w = d3;
                return h > 0 && w > 0;
            }
            if (d == 3)
            {
                h = t.size(0);
                w = t.size(1);
                return h > 0 && w > 0;
            }
            if (d == 2)
            {
                long tot = t.total();
                int guessH = t.rows();
                int guessW = t.cols();
                if (guessH > 1 && guessW > 1)
                {
                    h = guessH;
                    w = guessW;
                    return true;
                }
                // Flattened into a single row.
                if (tot > 0)
                {
                    int side = (int)Math.Sqrt(tot);
                    if (side * side == tot)
                    {
                        h = w = side;
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Equivalent to the inference subgraph for pose landmarks.
        /// Feeds a 256x256x3 (NHWC) tensor to <see cref="_poseLandmarksNet"/> (OpenCV DNN or Unity Inference Engine) and
        /// returns the output tensor list (<c>TENSORS</c>).
        /// Callers do not dispose <see cref="Mat"/> entries in the returned list; <see cref="MultiBackendNet"/> owns OpenCV forward outputs across calls and reuses IE buffers in Sentis mode.
        /// </summary>
        List<Mat> InferenceSubgraph_PoseLandmarks(Mat poseBlob)
        {
            _poseLandmarksForwardOutputList.Clear();
            _poseLandmarksNet.setInput(poseBlob);
            _poseLandmarksNet.forward(_poseLandmarksForwardOutputList, _poseLandmarksNetOutLayerNames);
            return _poseLandmarksForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Asynchronous <see cref="InferenceSubgraph_PoseLandmarks"/> for the Unity Inference Engine path (<see cref="MultiBackendNet.forwardTaskAsync"/>).
        /// Only <see cref="RunCoreProcessingTaskAsync"/> uses this; the OpenCV path uses <see cref="InferenceSubgraph_PoseLandmarks"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_PoseLandmarksAsync(Mat poseBlob, CancellationToken cancellationToken)
        {
            _poseLandmarksForwardOutputList.Clear();
            _poseLandmarksNet.setInput(poseBlob);
            await _poseLandmarksNet.forwardTaskAsync(_poseLandmarksForwardOutputList, _poseLandmarksNetOutLayerNames, cancellationToken);
            return _poseLandmarksForwardOutputList;
        }

#endif
        /// <summary>
        /// Equivalent to <c>SplitTensorVectorCalculator</c> with the
        /// five-range layout configured by <c>ConfigureSplitTensorVectorCalculator</c>.
        /// </summary>
        static bool SplitTensorVectorCalculator_PoseLandmarks(List<Mat> tensors,
            out Mat landmarkTensors, out Mat poseFlagTensors, out Mat segmentationTensors,
            out Mat heatmapTensors, out Mat worldLandmarkTensors)
        {
            landmarkTensors = poseFlagTensors = segmentationTensors = heatmapTensors = worldLandmarkTensors = null;
            if (tensors == null || tensors.Count < kPoseLandmarkModelTensorSplitCount)
                return false;
            landmarkTensors = tensors[0];
            poseFlagTensors = tensors[1];
            segmentationTensors = tensors[2];
            heatmapTensors = tensors[3];
            worldLandmarkTensors = tensors[4];
            return true;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToFloatsCalculator</c>.
        /// Converts the pose-flag tensor into a scalar.
        /// </summary>
        static float TensorsToFloatsCalculator_PosePresence(Mat poseFlagTensors)
        {
            return poseFlagTensors.at<float>(0, 0)[0];
        }

        /// <summary>
        /// Equivalent to <c>ThresholdingCalculator</c>, using the same
        /// <c>score &gt; threshold</c> rule as the original <c>thresholding_calculator.cc</c>.
        /// The threshold corresponds to the landmarks-subgraph <c>min_detection_confidence</c>
        /// and is stored in <see cref="_minPosePresenceConfidence"/>.
        /// </summary>
        bool ThresholdingCalculator_PosePresence(float score)
        {
            return score > _minPosePresenceConfidence;
        }

        /// <summary>
        /// Equivalent to <c>GateCalculator</c>.
        /// When ALLOW is false, downstream tensors do not propagate; this C# port expresses that behavior as a branch.
        /// </summary>
        static bool GateCalculator_PoseLandmarkTensors(bool allow)
        {
            return allow;
        }

        /// <summary>
        /// Grows <see cref="_poseLandmarksTensorFlatScratch"/> to the required length
        /// for reading landmark tensors via <c>reshape(1, total)</c>.
        /// </summary>
        void EnsurePoseLandmarksTensorFlatScratch(int need)
        {
            if (_poseLandmarksTensorFlatScratch == null || _poseLandmarksTensorFlatScratch.Length < need)
                _poseLandmarksTensorFlatScratch = new float[need];
        }

        /// <summary>
        /// Equivalent to the original <c>TensorsToLandmarksCalculator</c> for image-space normalized landmarks,
        /// with <c>sigmoid_activation = true</c>.
        /// The Z value follows <c>z / input_image_width / normalize_z</c> as in
        /// <c>tensors_to_landmarks_calculator.cc</c>.
        /// In Tasks <c>pose_landmarks_detector_graph.cc</c>,
        /// <c>ConfigureTensorsToLandmarksCalculator(..., normalize=false, ...)</c> leaves
        /// <c>normalize_z</c> unset, so the proto default of 1.0 applies
        /// and the result is effectively <c>z / input_image_width</c>.
        /// </summary>
        PoseLandmarkDecoded[] TensorsToLandmarksCalculator_PoseImage(Mat tensor, int inputW, int inputH)
        {
            int total = (int)tensor.total();
            int numDims = total / kPoseLandmarkModelLandmarkCount;
            if (numDims < 3)
                numDims = 3;

            if (_poseDecodedLandmarkScratch == null)
                _poseDecodedLandmarkScratch = new PoseLandmarkDecoded[kPoseLandmarkModelLandmarkCount];
            var arr = _poseDecodedLandmarkScratch;
            using (var flat = tensor.reshape(1, total))
            {
                EnsurePoseLandmarksTensorFlatScratch(total);
                flat.get(0, 0, _poseLandmarksTensorFlatScratch.AsSpan(0, total));
                ReadOnlySpan<float> buf = _poseLandmarksTensorFlatScratch.AsSpan(0, total);
                for (int i = 0; i < kPoseLandmarkModelLandmarkCount; i++)
                {
                    int o = i * numDims;
                    float x = buf[o];
                    float y = o + 1 < total ? buf[o + 1] : 0f;
                    float z = o + 2 < total ? buf[o + 2] : 0f;
                    float vis = numDims > 3 && o + 3 < total ? Sigmoid(buf[o + 3]) : 0f;
                    float pres = numDims > 4 && o + 4 < total ? Sigmoid(buf[o + 4]) : 0f;
                    arr[i] = new PoseLandmarkDecoded
                    {
                        X = x / inputW,
                        Y = y / inputH,
                        Z = z / inputW,
                        Visibility = vis,
                        Presence = pres,
                    };
                }
            }
            return arr;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToLandmarksCalculator</c> for absolute LANDMARKS coordinates,
        /// with <c>sigmoid_activation = false</c>.
        /// </summary>
        PoseLandmarkDecoded[] TensorsToLandmarksCalculator_PoseWorld(Mat tensor)
        {
            int total = (int)tensor.total();
            int numDims = total / kPoseLandmarkModelLandmarkCount;
            if (numDims < 3) numDims = 3;
            var arr = _poseWorldDecodedLandmarkScratch;
            using (var flat = tensor.reshape(1, total))
            {
                EnsurePoseLandmarksTensorFlatScratch(total);
                flat.get(0, 0, _poseLandmarksTensorFlatScratch.AsSpan(0, total));
                ReadOnlySpan<float> buf = _poseLandmarksTensorFlatScratch.AsSpan(0, total);
                for (int i = 0; i < kPoseLandmarkModelLandmarkCount; i++)
                {
                    int o = i * numDims;
                    arr[i] = new PoseLandmarkDecoded
                    {
                        X = buf[o],
                        Y = o + 1 < total ? buf[o + 1] : 0f,
                        Z = o + 2 < total ? buf[o + 2] : 0f,
                        Visibility = numDims > 3 && o + 3 < total ? buf[o + 3] : 0f,
                        Presence = numDims > 4 && o + 4 < total ? buf[o + 4] : 0f,
                    };
                }
            }
            return arr;
        }

        /// <summary>
        /// Equivalent to <c>SplitNormalizedLandmarkListCalculator</c>,
        /// following <c>ConfigureSplitNormalizedLandmarkListCalculator</c>.
        /// </summary>
        static void SplitNormalizedLandmarkListCalculator(PoseLandmarkDecoded[] all, out PoseLandmarkDecoded[] main33, out PoseLandmarkDecoded[] aux2)
        {
            main33 = new PoseLandmarkDecoded[33];
            aux2 = new PoseLandmarkDecoded[2];
            for (int i = 0; i < 33; i++)
                main33[i] = all[i];
            aux2[0] = all[33];
            aux2[1] = all[34];
        }

        /// <summary>
        /// Equivalent to <c>SplitLandmarkListCalculator</c> for the first 33 world landmarks.
        /// </summary>
        static void SplitLandmarkListCalculator_PoseWorld(PoseLandmarkDecoded[] all, out PoseLandmarkDecoded[] world33)
        {
            world33 = new PoseLandmarkDecoded[33];
            for (int i = 0; i < 33; i++)
                world33[i] = all[i];
        }

        /// <summary>
        /// Equivalent to <c>VisibilityCopyCalculator</c>.
        /// Copies visibility and presence from the 33 image-space landmarks to the world-space landmarks.
        /// </summary>
        static void VisibilityCopyCalculator_PoseWorld(PoseLandmarkDecoded[] fromNorm, PoseLandmarkDecoded[] toWorld)
        {
            for (int i = 0; i < 33; i++)
            {
                var w = toWorld[i];
                w.Visibility = fromNorm[i].Visibility;
                w.Presence = fromNorm[i].Presence;
                toWorld[i] = w;
            }
        }

        /// <summary>
        /// Equivalent to <c>LandmarkLetterboxRemovalCalculator</c>.
        /// </summary>
        static PoseLandmarkDecoded[] LandmarkLetterboxRemovalCalculator_Pose(PoseLandmarkDecoded[] lm, SinglePoseLandmarkPreprocessOut pre)
        {
            float padTop = pre.LetterboxPaddingTop;
            float padLeft = pre.LetterboxPaddingLeft;
            float padRight = pre.LetterboxPaddingRight;
            float padBottom = pre.LetterboxPaddingBottom;
            if (padTop == 0f && padLeft == 0f && padBottom == 0f && padRight == 0f)
                return lm;

            float h = 1f - padTop - padBottom;
            float w = 1f - padLeft - padRight;
            if (h <= 1e-6f || w <= 1e-6f)
                return lm;

            var o = new PoseLandmarkDecoded[lm.Length];
            for (int i = 0; i < lm.Length; i++)
            {
                o[i] = new PoseLandmarkDecoded
                {
                    X = (lm[i].X - padLeft) / w,
                    Y = (lm[i].Y - padTop) / h,
                    Z = lm[i].Z / w,
                    Visibility = lm[i].Visibility,
                    Presence = lm[i].Presence,
                };
            }
            return o;
        }

        /// <summary>
        /// Equivalent to <c>LandmarkProjectionCalculator</c> for the main 33 landmarks.
        /// This matches the original <c>landmark_projection_calculator.cc</c> path where only
        /// <c>NORM_RECT</c> is connected, so the output is full-image normalized coordinates
        /// as <c>NormalizedLandmark</c>.
        /// The Z projection follows <c>new_z = landmark.z() * input_rect.width()</c>,
        /// which is the same formula used for hands in <see cref="MediaPipeHandLandmarker"/>.
        /// </summary>
        static Vec3f[] LandmarkProjectionCalculator_Pose(PoseLandmarkDecoded[] lm, NormalizedRect roi)
        {
            var projected = new Vec3f[lm.Length];
            float angle = roi.Rotation;
            float ca = (float)Math.Cos(angle);
            float sa = (float)Math.Sin(angle);
            float cx = roi.XCenter;
            float cy = roi.YCenter;
            float nw = roi.Width;
            float nh = roi.Height;
            for (int i = 0; i < lm.Length; i++)
            {
                float x = lm[i].X - 0.5f;
                float y = lm[i].Y - 0.5f;
                float z = lm[i].Z;
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
        /// Equivalent to <c>LandmarkProjectionCalculator</c> for the two auxiliary landmarks.
        /// </summary>
        static Vec3f[] LandmarkProjectionCalculator_PoseAux(PoseLandmarkDecoded[] lm, NormalizedRect roi)
        {
            return LandmarkProjectionCalculator_Pose(lm, roi);
        }

        /// <summary>
        /// Equivalent to <c>WorldLandmarkProjectionCalculator</c>.
        /// </summary>
        static Vec3f[] WorldLandmarkProjectionCalculator_Pose(PoseLandmarkDecoded[] world, NormalizedRect roi)
        {
            float ca = (float)Math.Cos(roi.Rotation);
            float sa = (float)Math.Sin(roi.Rotation);
            var v = new Vec3f[world.Length];
            for (int i = 0; i < world.Length; i++)
            {
                float x = world[i].X;
                float y = world[i].Y;
                float z = world[i].Z;
                v[i] = new Vec3f(ca * x - sa * y, sa * x + ca * y, z);
            }
            return v;
        }

        /// <summary>
        /// Equivalent to <c>LandmarksToDetectionCalculator</c>.
        /// Builds <see cref="PoseDetectionData"/> from the two auxiliary landmarks in full-image normalized coordinates.
        /// </summary>
        PoseDetectionData LandmarksToDetectionCalculator(Vec3f[] auxNorm)
        {
            float xmin = float.MaxValue, ymin = float.MaxValue, xmax = float.MinValue, ymax = float.MinValue;
            for (int i = 0; i < 2 && i < auxNorm.Length; i++)
            {
                float rx = auxNorm[i].Item1;
                float ry = auxNorm[i].Item2;
                xmin = Mathf.Min(xmin, rx);
                xmax = Mathf.Max(xmax, rx);
                ymin = Mathf.Min(ymin, ry);
                ymax = Mathf.Max(ymax, ry);
                _poseLandmarksToDetKp8[i * 2] = rx;
                _poseLandmarksToDetKp8[i * 2 + 1] = ry;
            }
            return new PoseDetectionData
            {
                RelXmin = xmin,
                RelYmin = ymin,
                RelWidth = xmax - xmin,
                RelHeight = ymax - ymin,
                RelKeypointsXy = _poseLandmarksToDetKp8,
                Score = 1f,
            };
        }

        /// <summary>
        /// Equivalent to <c>RefineLandmarksFromHeatmapCalculator</c>,
        /// using <c>kernel_size = 7</c> with heatmaps interpreted as HWC.
        /// </summary>
        PoseLandmarkDecoded[] RefineLandmarksFromHeatmapCalculator(PoseLandmarkDecoded[] landmarks, Mat heatmapTensor)
        {
            if (heatmapTensor == null || heatmapTensor.empty())
                return landmarks;

            if (!TryGetHeatmapHwc(heatmapTensor, out int hmH, out int hmW, out int hmC))
                return landmarks;

            if (hmC != kPoseLandmarkModelLandmarkCount)
                return landmarks;

            int kernel = kPoseLandmarkHeatmapKernelSize;
            int offset = (kernel - 1) / 2;
            float minConf = 0.5f;
            var outLm = _poseHeatmapRefineDecodedScratch;
            int nCopy = Math.Min(landmarks.Length, kPoseLandmarkModelLandmarkCount);
            for (int i = 0; i < nCopy; i++)
                outLm[i] = landmarks[i];
            for (int i = nCopy; i < kPoseLandmarkModelLandmarkCount; i++)
                outLm[i] = default;

            int hmRowSize = hmW * hmC;
            int hmPixelSize = hmC;
            int hmTotal = (int)heatmapTensor.total();
            using (var hmFlat = heatmapTensor.reshape(1, hmTotal))
            {
                if (_poseHeatmapReadScratch == null || _poseHeatmapReadScratch.Length < hmTotal)
                    _poseHeatmapReadScratch = new float[hmTotal];
                hmFlat.get(0, 0, _poseHeatmapReadScratch.AsSpan(0, hmTotal));
                ReadOnlySpan<float> hm = _poseHeatmapReadScratch.AsSpan(0, hmTotal);

                for (int lmIndex = 0; lmIndex < kPoseLandmarkModelLandmarkCount; lmIndex++)
                {
                    int centerCol = (int)(outLm[lmIndex].X * hmW);
                    int centerRow = (int)(outLm[lmIndex].Y * hmH);
                    if (centerCol < 0 || centerCol >= hmW || centerRow < 0 || centerRow >= hmH)
                        continue;

                    int beginCol = Math.Max(0, centerCol - offset);
                    int endCol = Math.Min(hmW, centerCol + offset + 1);
                    int beginRow = Math.Max(0, centerRow - offset);
                    int endRow = Math.Min(hmH, centerRow + offset + 1);

                    float sum = 0f;
                    float weightedCol = 0f;
                    float weightedRow = 0f;
                    float maxConf = 0f;

                    for (int row = beginRow; row < endRow; row++)
                    {
                        for (int col = beginCol; col < endCol; col++)
                        {
                            int idx = hmRowSize * row + hmPixelSize * col + lmIndex;
                            if (idx < 0 || idx >= hm.Length) continue;
                            float conf = Sigmoid(hm[idx]);
                            sum += conf;
                            maxConf = Mathf.Max(maxConf, conf);
                            weightedCol += col * conf;
                            weightedRow += row * conf;
                        }
                    }

                    if (maxConf >= minConf && sum > 0f)
                    {
                        outLm[lmIndex].X = weightedCol / hmW / sum;
                        outLm[lmIndex].Y = weightedRow / hmH / sum;
                    }
                }
            }

            return outLm;
        }

        static bool TryGetHeatmapHwc(Mat t, out int h, out int w, out int c)
        {
            h = w = c = 0;
            int d = t.dims();
            if (d == 4)
            {
                h = t.size(1);
                w = t.size(2);
                c = t.size(3);
                return true;
            }
            if (d == 3)
            {
                h = t.size(0);
                w = t.size(1);
                c = t.size(2);
                return true;
            }
            if (d == 2 && t.rows() > 1 && t.cols() > 1)
            {
                h = t.rows();
                w = t.cols();
                c = (int)(t.total() / (h * w));
                return c > 0;
            }
            return false;
        }

        static float Sigmoid(float v)
        {
            return 1f / (1f + Mathf.Exp(-v));
        }

        /// <summary>
        /// Equivalent to the Tasks <c>ImagePreprocessingGraph</c> for the single-pose landmark stage.
        /// Uses the same 256x256 perspective-warp path as <see cref="MediaPipeHandLandmarker"/>.
        /// </summary>
        bool ImagePreprocessingGraph_SinglePoseLandmarks(Mat image, NormalizedRect poseRect, out SinglePoseLandmarkPreprocessOut pre)
        {
            pre = default;
            int imgW = image.cols();
            int imgH = image.rows();
            if (imgW <= 0 || imgH <= 0)
                return false;

            const int inputSize = kPoseLandmarkModelInputSize;

            if (_singlePoseLandmarkBlob == null)
            {
                _singlePoseLandmarkSrcPts = new Mat(4, 2, CvType.CV_32FC1);
                _singlePoseLandmarkDstPts = new Mat(4, 2, CvType.CV_32FC1);
                float dw = inputSize, dh = inputSize;
                Span<float> dstPtsArr = stackalloc float[8];
                dstPtsArr[0] = 0f; dstPtsArr[1] = dh;
                dstPtsArr[2] = 0f; dstPtsArr[3] = 0f;
                dstPtsArr[4] = dw; dstPtsArr[5] = 0f;
                dstPtsArr[6] = dw; dstPtsArr[7] = dh;
                _singlePoseLandmarkDstPts.put(0, 0, dstPtsArr);

                _singlePoseLandmarkWarpedBgr = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _singlePoseLandmarkWarpedRgb = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _singlePoseLandmarkBlob = new Mat(new int[] { 1, inputSize, inputSize, 3 }, CvType.CV_32FC1);
                _singlePoseLandmarkBlobHxW = _singlePoseLandmarkBlob.reshape(3, new int[] { inputSize, inputSize });
            }

            float cx = poseRect.XCenter * imgW;
            float cy = poseRect.YCenter * imgH;
            float rw = poseRect.Width * imgW;
            float rh = poseRect.Height * imgH;
            if (rw <= 0f || rh <= 0f || float.IsNaN(rw) || float.IsNaN(rh))
                return false;

            PadRoiLikeImageToTensorCalculator(inputSize, inputSize, keepAspectRatio: true, ref rw, ref rh,
                out float padL, out float padT, out float padR, out float padB);

            double angleDeg = poseRect.Rotation * 180.0 / Math.PI;

            Imgproc.boxPoints((cx, cy, rw, rh, angleDeg), _singlePoseLandmarkSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_singlePoseLandmarkSrcPts, _singlePoseLandmarkDstPts))
            {
                if (_singlePoseLandmarkProjMat3x3 == null)
                    _singlePoseLandmarkProjMat3x3 = new Mat(3, 3, CvType.CV_32FC1);
                projMat.copyTo(_singlePoseLandmarkProjMat3x3);
                Imgproc.warpPerspective(image, _singlePoseLandmarkWarpedBgr, projMat, (inputSize, inputSize),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
            }

            Imgproc.cvtColor(_singlePoseLandmarkWarpedBgr, _singlePoseLandmarkWarpedRgb, Imgproc.COLOR_BGR2RGB);
            _singlePoseLandmarkWarpedRgb.convertTo(_singlePoseLandmarkBlobHxW, CvType.CV_32F, 1.0 / 255.0);

            pre = new SinglePoseLandmarkPreprocessOut
            {
                PoseBlob = _singlePoseLandmarkBlob,
                ImageW = imgW,
                ImageH = imgH,
                ModelW = inputSize,
                ModelH = inputSize,
                LetterboxPaddingTop = padT,
                LetterboxPaddingLeft = padL,
                LetterboxPaddingRight = padR,
                LetterboxPaddingBottom = padB,
            };
            return true;
        }

        /// <summary>
        /// Equivalent to <c>PadRoi</c> in <c>image_to_tensor_utils.cc</c>.
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
        /// Packs per-pose results into one <see cref="Mat"/>, with rows indexed by pose.
        /// When <c>_outputSegmentationMasks</c> is enabled, index 1 returns a vertically stacked
        /// <c>CV_32FC1</c> mask image containing one full-image mask per pose.
        /// Each pose plane is read from <see cref="_segmentationMaskFullBySlot"/> via
        /// <see cref="PoseResult.SegmentationMaskSlotIndex"/>, and the slot <see cref="Mat"/> instances
        /// are not disposed here because they are owned by the worker.
        /// </summary>
        Mat[] PackResultsToMats(List<PoseResult> poses, Mat sourceImage)
        {
            int poseCount = poses?.Count ?? 0;
            int L = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            int R = PoseLandmarkerEstimationData.ELEMENT_COUNT;
            int imgW = sourceImage != null ? sourceImage.cols() : 0;
            int imgH = sourceImage != null ? sourceImage.rows() : 0;

            if (poseCount == 0)
            {
                if (!_outputSegmentationMasks)
                    return new Mat[] { new Mat() };
                return new Mat[] { new Mat(), new Mat(0, imgW, CvType.CV_32FC1) };
            }

            lock (_lockObject)
            {
                if (_outputBuffer == null
                    || _outputBuffer.rows() < poseCount
                    || _outputBuffer.cols() != R
                    || _outputBuffer.type() != CvType.CV_32FC1)
                {
                    _outputBuffer?.Dispose();
                    int rows = Math.Max(poseCount, _numPoses);
                    _outputBuffer = new Mat(rows, R, CvType.CV_32FC1);
                }

                var packed = _outputBuffer;
                Span<float> row = _posePackOutputRowScratch.AsSpan(0, PoseLandmarkerEstimationData.ELEMENT_COUNT);
                int stride = PoseLandmarkerEstimationData.LANDMARK_FLOAT_STRIDE;
                int offWorld = PoseLandmarkerEstimationData.NORM_LANDMARKS_FLOAT_COUNT;

                for (int i = 0; i < poseCount; i++)
                {
                    row.Clear();

                    var p = poses[i];
                    var lm = p.NormLandmarks;
                    var wm = p.WorldLandmarks;
                    var visN = p.LandmarkVisibility;
                    var visW = p.LandmarkVisibilityWorld;
                    var pres = p.LandmarkPresence;

                    for (int j = 0; j < L; j++)
                    {
                        int o = j * stride;
                        float vx = visN != null && j < visN.Length ? visN[j] : 0f;
                        float pr = pres != null && j < pres.Length ? pres[j] : 0f;
                        if (lm != null && j < lm.Length)
                        {
                            row[o + 0] = lm[j].Item1;
                            row[o + 1] = lm[j].Item2;
                            row[o + 2] = lm[j].Item3;
                        }
                        row[o + 3] = vx;
                        row[o + 4] = pr;
                    }

                    for (int j = 0; j < L; j++)
                    {
                        int o = offWorld + j * stride;
                        float vw = visW != null && j < visW.Length
                            ? visW[j]
                            : (visN != null && j < visN.Length ? visN[j] : 0f);
                        float pr = pres != null && j < pres.Length ? pres[j] : 0f;
                        if (wm != null && j < wm.Length)
                        {
                            row[o + 0] = wm[j].Item1;
                            row[o + 1] = wm[j].Item2;
                            row[o + 2] = wm[j].Item3;
                        }
                        row[o + 3] = vw;
                        row[o + 4] = pr;
                    }

                    packed.put(i, 0, row);
                }

                Mat result = packed.rowRange(0, poseCount);

                if (!_outputSegmentationMasks)
                    return new Mat[] { result };

                int stackRows = poseCount * imgH;
                if (stackRows <= 0 || imgW <= 0)
                    return new Mat[] { result, new Mat(0, Math.Max(imgW, 1), CvType.CV_32FC1) };

                if (_segmentationStackOutput == null
                    || _segmentationStackOutput.rows() != stackRows
                    || _segmentationStackOutput.cols() != imgW
                    || _segmentationStackOutput.type() != CvType.CV_32FC1)
                {
                    _segmentationStackOutput?.Dispose();
                    _segmentationStackOutput = new Mat(stackRows, imgW, CvType.CV_32FC1);
                }

                for (int i = 0; i < poseCount; i++)
                {
                    using (Mat roi = _segmentationStackOutput.rowRange(i * imgH, (i + 1) * imgH))
                    {
                        int si = poses[i].SegmentationMaskSlotIndex;
                        Mat sm = _segmentationMaskFullBySlot != null && si >= 0 && si < _segmentationMaskFullBySlot.Length
                            ? _segmentationMaskFullBySlot[si]
                            : null;
                        if (sm != null && !sm.empty())
                            sm.copyTo(roi);
                        else
                            roi.setTo((0d, 0d, 0d, 0d));
                    }
                }

                // Execute disposes the previous _outputs as a batch, so return a submat header rather than the owned backing Mat itself, matching the PeekOutput contract and packed.rowRange usage.
                return new Mat[] { result, _segmentationStackOutput.rowRange(0, stackRows) };
            }
        }

        /// <summary>
        /// Builds the SSD anchor matrix for pose detection using options aligned with
        /// <c>ConfigureSsdAnchorsCalculator</c> in Tasks <c>pose_detector_graph.cc</c>,
        /// derived from <c>pose_detection_gpu.pbtxt</c>.
        /// The generation procedure matches <c>SsdAnchorsCalculator::GenerateAnchors</c>
        /// in <c>ssd_anchors_calculator.cc</c> with <c>multiscale_anchor_generation</c> disabled.
        /// Each row stores tensor-normalized <c>x_center</c> and <c>y_center</c>.
        /// Because <c>interpolated_scale_aspect_ratio</c> is unset in the original graph,
        /// the default value 1.0 from <c>ssd_anchors_calculator.proto</c> applies here as well.
        /// </summary>
        static Mat BuildPoseDetectorSsdAnchors2254()
        {
            const int numLayers = 5;
            const float minScale = 0.1484375f;
            const float maxScale = 0.75f;
            const int inputSizeHeight = 224;
            const int inputSizeWidth = 224;
            const float anchorOffsetX = 0.5f;
            const float anchorOffsetY = 0.5f;
            const bool reduceBoxesInLowestLayer = false;
            const float interpolatedScaleAspectRatio = 1.0f;
            float[] aspectRatiosOptions = { 1.0f };
            int[] strides = { 8, 16, 32, 32, 32 };

            int stridesLen = strides.Length;
            if (stridesLen != numLayers)
                throw new InvalidOperationException("The lengths of SSD strides and num_layers do not match.");

            var aspectRatios = new List<float>(8);
            var scales = new List<float>(8);
            var anchorHeight = new List<float>(8);
            var anchorWidth = new List<float>(8);
            const int expectedRows = 2254;
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
                    float scale = PoseDetectorSsdAnchors_CalculateScale(
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
                                : PoseDetectorSsdAnchors_CalculateScale(
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
            anchors.put(0, 0, xy);
            return anchors;
        }

        /// <summary>
        /// Uses the same formula as <c>CalculateScale</c> in <c>ssd_anchors_calculator.cc</c> for <c>PoseDetectorGraph</c>.
        /// </summary>
        static float PoseDetectorSsdAnchors_CalculateScale(
            float minScale, float maxScale, int strideIndex, int numStrides)
        {
            if (numStrides == 1)
                return (minScale + maxScale) * 0.5f;
            return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1.0f);
        }

        /// <summary>
        /// Shared cache for the same 2254x2 anchors produced by <c>SsdAnchorsCalculator</c> in <c>PoseDetectorGraph</c>.
        /// </summary>
        static Mat _poseDetectorSsdAnchors2254Cache;

        static Mat GetPoseDetectorSsdAnchors2254Shared()
        {
            if (_poseDetectorSsdAnchors2254Cache != null)
                return _poseDetectorSsdAnchors2254Cache;
            lock (typeof(MediaPipePoseLandmarker))
            {
                if (_poseDetectorSsdAnchors2254Cache != null)
                    return _poseDetectorSsdAnchors2254Cache;
                _poseDetectorSsdAnchors2254Cache = BuildPoseDetectorSsdAnchors2254();
                return _poseDetectorSsdAnchors2254Cache;
            }
        }


        /// <summary>
        /// Equivalent to the <c>smooth_landmarks</c> block in the original
        /// <c>MultiplePoseLandmarksDetectorGraph</c> from <c>pose_landmarks_detector_graph.cc</c>.
        /// Smoothing is applied outside the loop to one pose only, then written back to the vector.
        /// The equivalent of <c>CreateIntConstantStream</c> is <see cref="CreateIntConstantStream"/>,
        /// which internally uses <see cref="ConstantSidePacketCalculator"/> and <see cref="SidePacketToStreamCalculator"/>.
        /// The equivalent of <c>GetItem</c> is provided by
        /// <see cref="GetNormalizedLandmarkListVectorItemCalculator"/>,
        /// <see cref="GetLandmarkListVectorItemCalculator"/>,
        /// and <see cref="GetNormalizedRectVectorItemCalculator"/>.
        /// </summary>
        sealed class PoseLandmarkSmoothingPipeline
        {
            const float kVisibilityLowPassAlpha = 0.1f;
            const float kNormMinCutoff = 0.05f;
            const float kNormBeta = 80f;
            const float kNormDerivateCutoff = 1f;
            const float kWorldMinCutoff = 0.1f;
            const float kWorldBeta = 40f;
            const float kWorldDerivateCutoff = 1f;
            const double kDefaultFrequency = 30.0;
            const float kMinAllowedObjectScale = 1e-6f;

            readonly MediapipeLowPassFilter[] _normVisibilityFilters;
            readonly MediapipeLowPassFilter[] _worldVisibilityFilters;
            readonly MediapipeOneEuroFilter[] _normFiltersX;
            readonly MediapipeOneEuroFilter[] _normFiltersY;
            readonly MediapipeOneEuroFilter[] _normFiltersZ;
            readonly MediapipeOneEuroFilter[] _worldFiltersX;
            readonly MediapipeOneEuroFilter[] _worldFiltersY;
            readonly MediapipeOneEuroFilter[] _worldFiltersZ;

            public PoseLandmarkSmoothingPipeline()
            {
                int n = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
                _normVisibilityFilters = new MediapipeLowPassFilter[n];
                _worldVisibilityFilters = new MediapipeLowPassFilter[n];
                for (int i = 0; i < n; i++)
                {
                    _normVisibilityFilters[i] = MediapipeLowPassFilter.Create(kVisibilityLowPassAlpha);
                    _worldVisibilityFilters[i] = MediapipeLowPassFilter.Create(kVisibilityLowPassAlpha);
                }

                _normFiltersX = new MediapipeOneEuroFilter[n];
                _normFiltersY = new MediapipeOneEuroFilter[n];
                _normFiltersZ = new MediapipeOneEuroFilter[n];
                _worldFiltersX = new MediapipeOneEuroFilter[n];
                _worldFiltersY = new MediapipeOneEuroFilter[n];
                _worldFiltersZ = new MediapipeOneEuroFilter[n];
                for (int i = 0; i < n; i++)
                {
                    _normFiltersX[i] = MediapipeOneEuroFilter.Create(kDefaultFrequency, kNormMinCutoff, kNormBeta, kNormDerivateCutoff);
                    _normFiltersY[i] = MediapipeOneEuroFilter.Create(kDefaultFrequency, kNormMinCutoff, kNormBeta, kNormDerivateCutoff);
                    _normFiltersZ[i] = MediapipeOneEuroFilter.Create(kDefaultFrequency, kNormMinCutoff, kNormBeta, kNormDerivateCutoff);
                    _worldFiltersX[i] = MediapipeOneEuroFilter.Create(kDefaultFrequency, kWorldMinCutoff, kWorldBeta, kWorldDerivateCutoff);
                    _worldFiltersY[i] = MediapipeOneEuroFilter.Create(kDefaultFrequency, kWorldMinCutoff, kWorldBeta, kWorldDerivateCutoff);
                    _worldFiltersZ[i] = MediapipeOneEuroFilter.Create(kDefaultFrequency, kWorldMinCutoff, kWorldBeta, kWorldDerivateCutoff);
                }
            }

            public void ResetAll()
            {
                foreach (var f in _normVisibilityFilters) f.Reset();
                foreach (var f in _worldVisibilityFilters) f.Reset();
                foreach (var f in _normFiltersX) f.Reset();
                foreach (var f in _normFiltersY) f.Reset();
                foreach (var f in _normFiltersZ) f.Reset();
                foreach (var f in _worldFiltersX) f.Reset();
                foreach (var f in _worldFiltersY) f.Reset();
                foreach (var f in _worldFiltersZ) f.Reset();
            }

            public void ApplyPostEndLoop(Mat image, List<PoseResult> merged)
            {
                if (image == null || merged == null || merged.Count < 1)
                    return;

                var (iw, ih) = GetImageSize(image);
                long timestampNs = (long)Environment.TickCount * 1_000_000L;
                int n = PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;

                const int kSmoothingLandmarkListIndex = 0;
                int smoothingIdx = CreateIntConstantStream(kSmoothingLandmarkListIndex);
                if ((uint)smoothingIdx >= (uint)merged.Count)
                    return;

                PoseResult p = merged[smoothingIdx];
                float[] rawVis = p.LandmarkVisibility != null && p.LandmarkVisibility.Length == n
                    ? (float[])p.LandmarkVisibility.Clone()
                    : new float[n];
                float[] rawVisWorld = p.LandmarkVisibilityWorld != null && p.LandmarkVisibilityWorld.Length == n
                    ? (float[])p.LandmarkVisibilityWorld.Clone()
                    : rawVis;
                Vec3f[] lm = CloneVec3f(GetNormalizedLandmarkListVectorItemCalculator(merged, smoothingIdx), n);
                Vec3f[] wm = CloneVec3f(GetLandmarkListVectorItemCalculator(merged, smoothingIdx), n);
                var roi = GetNormalizedRectVectorItemCalculator(merged, smoothingIdx);

                // The original pose_landmarks_detector_graph smooths norm and world visibility on separate streams, both using the pre-smoothed visibility values as input.
                float[] visNorm = VisibilitySmoothingCalculator_PoseLandmarks(rawVis, _normVisibilityFilters);
                float[] visWorld = VisibilitySmoothingCalculator_PoseLandmarks(rawVisWorld, _worldVisibilityFilters);

                lm = LandmarksSmoothingCalculator_PoseNormalized(lm, timestampNs, iw, ih, roi, _normFiltersX, _normFiltersY, _normFiltersZ);
                wm = LandmarksSmoothingCalculator_PoseWorld(wm, timestampNs, _worldFiltersX, _worldFiltersY, _worldFiltersZ);

                p.NormLandmarks = lm;
                p.WorldLandmarks = wm;
                p.LandmarkVisibility = visNorm;
                p.LandmarkVisibilityWorld = visWorld;
                merged[smoothingIdx] = p;
            }

            static Vec3f[] CloneVec3f(Vec3f[] src, int n)
            {
                if (src == null || src.Length != n)
                    return new Vec3f[n];
                var d = new Vec3f[n];
                Array.Copy(src, d, n);
                return d;
            }

            /// <summary>
            /// Equivalent to <c>VisibilitySmoothingCalculator</c>,
            /// following <c>visibility_smoothing_calculator.cc</c> with <c>low_pass_filter.alpha</c>.
            /// </summary>
            static float[] VisibilitySmoothingCalculator_PoseLandmarks(float[] rawVisibility, MediapipeLowPassFilter[] filters)
            {
                int n = rawVisibility.Length;
                var o = new float[n];
                for (int i = 0; i < n; i++)
                    o[i] = filters[i].Apply(rawVisibility[i]);
                return o;
            }

            /// <summary>
            /// Equivalent to <c>LandmarksSmoothingCalculator</c> for normalized landmarks,
            /// using the One Euro implementation from <c>landmarks_smoothing_calculator.cc</c> and <c>smoothing.cc</c>.
            /// </summary>
            static Vec3f[] LandmarksSmoothingCalculator_PoseNormalized(Vec3f[] normLm, long timestampNs, int imageWidth, int imageHeight,
                NormalizedRect roi, MediapipeOneEuroFilter[] fx, MediapipeOneEuroFilter[] fy, MediapipeOneEuroFilter[] fz)
            {
                int n = normLm.Length;
                float objectScale = GetObjectScaleNormalizedRoi(roi, imageWidth, imageHeight);
                if (objectScale < kMinAllowedObjectScale)
                    return normLm;

                double valueScale = 1.0 / objectScale;
                var o = new Vec3f[n];
                for (int i = 0; i < n; i++)
                {
                    double xPx = normLm[i].Item1 * imageWidth;
                    double yPx = normLm[i].Item2 * imageHeight;
                    double zPx = normLm[i].Item3 * imageWidth;
                    xPx = fx[i].Apply(timestampNs, xPx, valueScale, 1.0);
                    yPx = fy[i].Apply(timestampNs, yPx, valueScale, 1.0);
                    zPx = fz[i].Apply(timestampNs, zPx, valueScale, 1.0);
                    o[i] = new Vec3f(
                        (float)(xPx / imageWidth),
                        (float)(yPx / imageHeight),
                        (float)(zPx / imageWidth));
                }
                return o;
            }

            /// <summary>
            /// Equivalent to <c>LandmarksSmoothingCalculator</c> for world landmarks,
            /// with <c>disable_value_scaling = true</c>.
            /// </summary>
            static Vec3f[] LandmarksSmoothingCalculator_PoseWorld(Vec3f[] worldLm, long timestampNs,
                MediapipeOneEuroFilter[] fx, MediapipeOneEuroFilter[] fy, MediapipeOneEuroFilter[] fz)
            {
                int n = worldLm.Length;
                const double valueScale = 1.0;
                var o = new Vec3f[n];
                for (int i = 0; i < n; i++)
                {
                    double x = fx[i].Apply(timestampNs, worldLm[i].Item1, valueScale, 1.0);
                    double y = fy[i].Apply(timestampNs, worldLm[i].Item2, valueScale, 1.0);
                    double z = fz[i].Apply(timestampNs, worldLm[i].Item3, valueScale, 1.0);
                    o[i] = new Vec3f((float)x, (float)y, (float)z);
                }
                return o;
            }

            static float GetObjectScaleNormalizedRoi(NormalizedRect roi, int imageWidth, int imageHeight)
            {
                float w = roi.Width * imageWidth;
                float h = roi.Height * imageHeight;
                return (w + h) * 0.5f;
            }

            /// <summary>
            /// Equivalent to <c>mediapipe/util/filtering/low_pass_filter.cc</c>,
            /// as embedded by <c>VisibilitySmoothingCalculator</c>.
            /// </summary>
            struct MediapipeLowPassFilter
            {
                float _alpha;
                bool _initialized;
                float _rawValue;
                float _storedValue;

                public static MediapipeLowPassFilter Create(float alpha)
                {
                    var f = new MediapipeLowPassFilter();
                    f.Init(alpha);
                    return f;
                }

                void Init(float alpha)
                {
                    SetAlpha(alpha);
                    _initialized = false;
                }

                void SetAlpha(float alpha)
                {
                    if (alpha < 0f || alpha > 1f)
                        alpha = Mathf.Clamp01(alpha);
                    _alpha = alpha;
                }

                public void Reset()
                {
                    _initialized = false;
                }

                public float Apply(float value)
                {
                    float result;
                    if (_initialized)
                        result = _alpha * value + (1f - _alpha) * _storedValue;
                    else
                    {
                        result = value;
                        _initialized = true;
                    }
                    _rawValue = value;
                    _storedValue = result;
                    return result;
                }

                public float ApplyWithAlpha(float value, float alpha)
                {
                    SetAlpha(alpha);
                    return Apply(value);
                }

                public bool HasLastRawValue => _initialized;

                public float LastRawValue() => _rawValue;
            }

            /// <summary>
            /// Equivalent to <c>mediapipe/util/filtering/one_euro_filter.cc</c>,
            /// i.e. the One Euro filter used by <c>LandmarksSmoothingCalculator</c>.
            /// </summary>
            struct MediapipeOneEuroFilter
            {
                const long kUninitializedTimestamp = -1;
                const double kEpsilon = 1e-6;

                double _frequency;
                double _minCutoff;
                double _beta;
                double _derivateCutoff;
                long _lastTimeNs;
                MediapipeLowPassFilter _x;
                MediapipeLowPassFilter _dx;

                public static MediapipeOneEuroFilter Create(double frequency, double minCutoff, double beta, double derivateCutoff)
                {
                    if (frequency <= kEpsilon || minCutoff <= kEpsilon || derivateCutoff <= kEpsilon)
                        throw new ArgumentException("OneEuroFilter: frequency, min_cutoff, and derivate_cutoff must be positive.");
                    MediapipeOneEuroFilter f;
                    f._frequency = frequency;
                    f._minCutoff = minCutoff;
                    f._beta = beta;
                    f._derivateCutoff = derivateCutoff;
                    f._lastTimeNs = kUninitializedTimestamp;
                    f._x = MediapipeLowPassFilter.Create((float)GetAlpha(minCutoff, frequency));
                    f._dx = MediapipeLowPassFilter.Create((float)GetAlpha(derivateCutoff, frequency));
                    return f;
                }

                public void Reset()
                {
                    _frequency = kDefaultFrequency;
                    _lastTimeNs = kUninitializedTimestamp;
                    _x = MediapipeLowPassFilter.Create((float)GetAlpha(_minCutoff, _frequency));
                    _dx = MediapipeLowPassFilter.Create((float)GetAlpha(_derivateCutoff, _frequency));
                }

                public double Apply(long timestampNs, double value, double valueScale, double betaScale)
                {
                    if (_lastTimeNs >= timestampNs)
                        return value;

                    if (_lastTimeNs != 0 && timestampNs != 0)
                        _frequency = 1.0 / ((timestampNs - _lastTimeNs) * 1e-9);

                    _lastTimeNs = timestampNs;

                    double dvalue = _x.HasLastRawValue
                        ? (value - _x.LastRawValue()) * valueScale * _frequency
                        : 0.0;
                    double edvalue = _dx.ApplyWithAlpha((float)dvalue, (float)GetAlpha(_derivateCutoff, _frequency));
                    double scaledBeta = betaScale * _beta;
                    double cutoff = _minCutoff + scaledBeta * Math.Abs(edvalue);
                    return _x.ApplyWithAlpha((float)value, (float)GetAlpha(cutoff, _frequency));
                }

                static double GetAlpha(double cutoff, double frequency)
                {
                    double te = 1.0 / frequency;
                    double tau = 1.0 / (2.0 * Math.PI * cutoff);
                    return 1.0 / (1.0 + tau / te);
                }
            }
        }
    }
}
#endif
