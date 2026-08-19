#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.Calib3dModule;
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
    /// Processing worker that reproduces the face landmarking graph logic of
    /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceLandmarker
    /// on top of the OpenCV for Unity Dnn module.
    /// </summary>
    public class MediaPipeFaceLandmarker : DnnInferenceWorkerBase
    {
        /// <summary>
        /// Execution modes compatible with the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceLandmarker task.
        /// This enum corresponds to the task running mode that switches between
        /// per-image processing and stateful video processing.
        /// </summary>
        public enum MediaPipeFaceRunningMode : byte
        {
            /// <summary>
            /// IMAGE mode.
            /// Runs face detection and face landmarking for each input image without
            /// reusing loopback tracking state from previous frames.
            /// </summary>
            IMAGE = 0,

            /// <summary>
            /// VIDEO mode.
            /// Assumes a frame sequence and reuses face rectangles from the previous
            /// frame so the detector can be skipped on frames where tracking remains valid.
            /// </summary>
            VIDEO = 1,
        }

        #region Constants corresponding to upstream face_detector_graph.cc

        /// <summary>Metadata buffer name used by <c>GetFaceDetectorOptionsFromMetadata</c>.</summary>
        internal const string kFaceDetectorMetadataName = "FACE_DETECTOR_METADATA";

        /// <summary>Short-range face detector input side length (<c>kShortRangeImageSize</c>).</summary>
        internal const int kFaceDetectorShortRangeImageSize = 128;

        /// <summary>Long-range face detector input side length (<c>kLongRangeImageSize</c>).</summary>
        internal const int kFaceDetectorLongRangeImageSize = 192;

        /// <summary><c>TensorsToDetectionsCalculator.num_boxes</c> for the legacy 128-input path.</summary>
        internal const int kFaceDetectorLegacyShortRangeNumBoxes = 896;

        /// <summary><c>TensorsToDetectionsCalculator.num_boxes</c> for the legacy 192-input path.</summary>
        internal const int kFaceDetectorLegacyLongRangeNumBoxes = 2304;

        /// <summary>Legacy decoder <c>num_classes</c>.</summary>
        internal const int kFaceDetectorTensorsToDetectionsNumClasses = 1;

        /// <summary>Legacy decoder <c>num_coords</c> (4 box values + 6 keypoints x 2).</summary>
        internal const int kFaceDetectorTensorsToDetectionsNumCoords = 16;

        /// <summary>Legacy decoder <c>box_coord_offset</c>.</summary>
        internal const int kFaceDetectorTensorsToDetectionsBoxCoordOffset = 0;

        /// <summary>Legacy decoder <c>keypoint_coord_offset</c>.</summary>
        internal const int kFaceDetectorTensorsToDetectionsKeypointCoordOffset = 4;

        /// <summary>Legacy decoder <c>num_keypoints</c> (6 points: both eyes, nose, mouth, and ears).</summary>
        internal const int kFaceDetectorTensorsToDetectionsNumKeypoints = 6;

        /// <summary>Legacy decoder <c>num_values_per_keypoint</c>.</summary>
        internal const int kFaceDetectorTensorsToDetectionsNumValuesPerKeypoint = 2;

        /// <summary>Legacy decoder <c>score_clipping_thresh</c>.</summary>
        internal const float kFaceDetectorTensorsToDetectionsScoreClippingThresh = 100f;

        /// <summary>
        /// Upper bound for the SSD anchor row count, used to size buffers to the legacy 192-input <c>num_boxes</c>.
        /// Metadata-backed models can vary, so buffers are recreated when needed.
        /// </summary>
        internal const int kFaceDetectorMaxAnchorCount = kFaceDetectorLegacyLongRangeNumBoxes;

        // --- Legacy <c>ConfigureSsdAnchorsCalculator</c> defaults for metadata-free models (128x128 path) ---

        /// <summary>SSD <c>num_layers</c> for the 128-input path.</summary>
        internal const int kFaceDetectorSsdLegacy128NumLayers = 4;

        /// <summary>Legacy <c>min_scale</c> shared by the 128 and 192 paths.</summary>
        internal const float kFaceDetectorSsdLegacyMinScale = 0.1484375f;

        /// <summary>Legacy <c>max_scale</c> shared by the 128 and 192 paths.</summary>
        internal const float kFaceDetectorSsdLegacyMaxScale = 0.75f;

        /// <summary>SSD stride sequence for the 128-input path (4 layers).</summary>
        internal static readonly int[] kFaceDetectorSsdLegacy128Strides = { 8, 16, 16, 16 };

        /// <summary><c>interpolated_scale_aspect_ratio</c> for the 128-input path.</summary>
        internal const float kFaceDetectorSsdLegacy128InterpolatedScaleAspectRatio = 1f;

        // --- Legacy <c>ConfigureSsdAnchorsCalculator</c> defaults for metadata-free models (192x192 path) ---

        /// <summary>SSD <c>num_layers</c> for the 192-input path.</summary>
        internal const int kFaceDetectorSsdLegacy192NumLayers = 1;

        /// <summary>SSD stride for the 192-input path (single layer).</summary>
        internal const int kFaceDetectorSsdLegacy192Stride = 4;

        /// <summary><c>interpolated_scale_aspect_ratio</c> for the 192-input path.</summary>
        internal const float kFaceDetectorSsdLegacy192InterpolatedScaleAspectRatio = 0f;

        /// <summary>Legacy SSD <c>aspect_ratios</c> (single ratio 1.0).</summary>
        internal const float kFaceDetectorSsdLegacyAspectRatio = 1f;

        /// <summary>Legacy SSD <c>anchor_offset_x</c> and <c>anchor_offset_y</c>.</summary>
        internal const float kFaceDetectorSsdLegacyAnchorOffset = 0.5f;

        /// <summary>Legacy SSD <c>fixed_anchor_size</c>.</summary>
        internal const bool kFaceDetectorSsdLegacyFixedAnchorSize = true;

        /// <summary>
        /// <c>DetectionsToRectsCalculator</c> (face detection): keypoint index for the left eye from the observer view.</summary>
        internal const int kFaceDetectorDetectionsToRectsRotationStartKeypointIndex = 0;

        /// <summary>
        /// <c>DetectionsToRectsCalculator</c> (face detection): keypoint index for the right eye.</summary>
        internal const int kFaceDetectorDetectionsToRectsRotationEndKeypointIndex = 1;

        /// <summary><c>DetectionsToRectsCalculator.rotation_vector_target_angle_degrees</c> for face detection.</summary>
        internal const float kFaceDetectorDetectionsToRectsTargetAngleDegrees = 0f;

        /// <summary><c>RectTransformationCalculator.scale_x</c> and <c>scale_y</c> for detector-to-landmark ROI expansion.</summary>
        internal const float kFaceDetectorExpandedRoiScale = 1.5f;

        #endregion

        #region Constants corresponding to upstream face_landmarks_detector_graph.cc

        /// <summary>
        /// Number of model output tensors before <c>SplitTensorVectorCalculator</c> splits them (landmarks + presence).
        /// </summary>
        internal const int kFaceLandmarksOutputTensorsNum = 2;

        /// <summary>Input side length for the Tasks face_landmarks_detector (default 192x192).</summary>
        internal const int kFaceLandmarksDetectorImageSize = 192;

        /// <summary>
        /// <c>DetectionsToRectsCalculator</c> for the next-frame ROI: landmark index corresponding to the outer left eye corner.
        /// </summary>
        internal const int kFaceLandmarksDetectionsToRectsRotationStartKeypointIndex = 33;

        /// <summary>
        /// <c>DetectionsToRectsCalculator</c> for the next-frame ROI: landmark index corresponding to the outer right eye corner.
        /// </summary>
        internal const int kFaceLandmarksDetectionsToRectsRotationEndKeypointIndex = 263;

        /// <summary>The corresponding <c>rotation_vector_target_angle_degrees</c> for the next-frame ROI path.</summary>
        internal const float kFaceLandmarksDetectionsToRectsTargetAngleDegrees = 0f;

        /// <summary><c>RectTransformationCalculator.scale_x</c> and <c>scale_y</c> for next-frame rect expansion.</summary>
        internal const float kFaceLandmarksNextFrameRoiScale = 1.5f;

        /// <summary><c>LandmarksSmoothingCalculator</c> One Euro <c>min_cutoff</c>.</summary>
        internal const float kFaceLandmarksSmoothingOneEuroMinCutoff = 0.05f;

        /// <summary><c>LandmarksSmoothingCalculator</c> One Euro <c>beta</c>.</summary>
        internal const float kFaceLandmarksSmoothingOneEuroBeta = 80f;

        /// <summary><c>LandmarksSmoothingCalculator</c> One Euro <c>derivate_cutoff</c>.</summary>
        internal const float kFaceLandmarksSmoothingOneEuroDerivateCutoff = 1f;

        /// <summary>
        /// Number of blendshape classification coefficients (indices 0-51 in the category list, used by Phase B).
        /// </summary>
        internal const int kFaceBlendshapeCoefficientCount = 52;

        /// <summary><c>GetNormalizedLandmarkListVectorItemCalculator.item_index</c> used for single-face smoothing.</summary>
        internal const int kFaceSmoothLandmarksVectorItemIndex = 0;

        /// <summary>
        /// <c>kLandmarksSubsetIdxs</c> from <c>face_blendshapes_graph.cc</c> (146 selected landmarks out of the 478-point HUND input).
        /// </summary>
        internal static readonly int[] kFaceBlendshapesLandmarkSubsetIndices =
        {
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
            40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
            136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
            161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
            249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
            296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
            336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
            384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
            466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
        };

        #endregion

        #region Constants corresponding to upstream tensors_to_face_landmarks_graph.cc

        /// <summary>Number of face mesh landmarks excluding iris landmarks.</summary>
        internal const int kFaceMeshLandmarksNum = 468;

        /// <summary>Total number of output landmarks including iris landmarks. Corresponds to <c>TensorsToLandmarksCalculator.num_landmarks</c>.</summary>
        internal const int kFaceMeshWithIrisLandmarksNum = 478;

        /// <summary>
        /// Origin landmark for drawing the facial pose axes in <see cref="Visualize"/>. This is the nose tip in the upstream 468-point face mesh, and IDs 0-467 are unchanged even in the 478-point output.
        /// </summary>
        internal const int kVisualizeFacialPoseAxesOriginLandmarkIndex = 1;

        /// <summary>Number of landmarks in the lip region, referenced by refinement-related logic.</summary>
        internal const int kFaceLipsLandmarksNum = 80;

        /// <summary>Number of landmarks in a single eye region.</summary>
        internal const int kFaceEyeLandmarksNum = 71;

        /// <summary>Number of landmarks in one iris.</summary>
        internal const int kFaceIrisLandmarksNum = 5;

        /// <summary>Number of contour points used to average the iris region.</summary>
        internal const int kFaceContoursNumForIrisAvg = 16;

        #endregion

        #region Task proto defaults (face_detector_graph_options / face_landmarker_graph_options)

        /// <summary>Default value of <c>FaceDetectorGraphOptions.min_detection_confidence</c>.</summary>
        internal const float kDefaultMinFaceDetectionConfidence = 0.5f;

        /// <summary>Default value of <c>FaceLandmarksDetectorGraphOptions.min_detection_confidence</c> (presence threshold).</summary>
        internal const float kDefaultMinFacePresenceConfidence = 0.5f;

        /// <summary>Default value of <c>FaceLandmarkerGraphOptions.min_tracking_confidence</c>.</summary>
        internal const float kDefaultMinFaceTrackingConfidence = 0.5f;

        #endregion

        #region Visualization (equivalent to MediaPipe Python face_mesh_connections)

        static readonly Vec4d kVisualizeScalarWhite = new Vec4d(255, 255, 255, 255);
        static readonly Vec4d kVisualizeScalarRed = new Vec4d(0, 0, 255, 255);
        static readonly Vec4d kVisualizeScalarBlue = new Vec4d(255, 0, 0, 255);

        /// <summary>Ratio of the facial pose axis length relative to <c>max(w,h)</c> of the 468-point face bounding box.</summary>
        const float kVisualizeFacialPoseAxisLengthFractionOfFaceSize = 0.28f;

        /// <summary>Minimum axis length in pixels when the above ratio would make the axis too short.</summary>
        const int kVisualizeFacialPoseAxisLengthMinPx = 14;

        /// <summary>
        /// Same order as <c>kBlendshapeNames</c> in <c>face_blendshapes_graph.cc</c>, used for visualization labels.
        /// </summary>
        static readonly string[] kVisualizeBlendshapeCategoryNames =
        {
            "_neutral",
            "browDownLeft",
            "browDownRight",
            "browInnerUp",
            "browOuterUpLeft",
            "browOuterUpRight",
            "cheekPuff",
            "cheekSquintLeft",
            "cheekSquintRight",
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "eyeLookDownLeft",
            "eyeLookDownRight",
            "eyeLookInLeft",
            "eyeLookInRight",
            "eyeLookOutLeft",
            "eyeLookOutRight",
            "eyeLookUpLeft",
            "eyeLookUpRight",
            "eyeSquintLeft",
            "eyeSquintRight",
            "eyeWideLeft",
            "eyeWideRight",
            "jawForward",
            "jawLeft",
            "jawOpen",
            "jawRight",
            "mouthClose",
            "mouthDimpleLeft",
            "mouthDimpleRight",
            "mouthFrownLeft",
            "mouthFrownRight",
            "mouthFunnel",
            "mouthLeft",
            "mouthLowerDownLeft",
            "mouthLowerDownRight",
            "mouthPressLeft",
            "mouthPressRight",
            "mouthPucker",
            "mouthRight",
            "mouthRollLower",
            "mouthRollUpper",
            "mouthShrugLower",
            "mouthShrugUpper",
            "mouthSmileLeft",
            "mouthSmileRight",
            "mouthStretchLeft",
            "mouthStretchRight",
            "mouthUpperUpLeft",
            "mouthUpperUpRight",
            "noseSneerLeft",
            "noseSneerRight",
        };

        /// <summary>
        /// Blendshape coefficient indices used for simple bar visualization,
        /// focusing on categories that usually show large motion such as mouth, jaw, and blink.
        /// Display order follows top-to-bottom order and matches <see cref="kVisualizeBlendshapeCategoryNames"/>.
        /// </summary>
        static readonly int[] kVisualizeBlendshapeHighMotionIndices =
        {
            25, // jawOpen
            44, // mouthSmileLeft
            45, // mouthSmileRight
            32, // mouthFunnel
            38, // mouthPucker
            9, // eyeBlinkLeft
            10, // eyeBlinkRight
            3, // browInnerUp
            23, // jawForward
            6, // cheekPuff
            21, // eyeWideLeft
            22, // eyeWideRight
            34, // mouthLowerDownLeft
            35, // mouthLowerDownRight
        };

        /// <summary>
        /// Same edge set as <c>FACEMESH_CONTOURS | FACEMESH_IRISES</c> in
        /// Python <c>mediapipe/python/solutions/face_mesh_connections.py</c>.
        /// Sorted, 132 edges total.
        /// </summary>
        static readonly (int from, int to)[] kFaceMeshContoursAndIrisConnections = new (int, int)[]
        {
            (0, 267), (7, 163), (10, 338), (13, 312), (14, 317), (17, 314), (21, 54), (33, 7), (33, 246), (37, 0),
            (39, 37), (40, 39), (46, 53), (52, 65), (53, 52), (54, 103), (58, 132), (61, 146), (61, 185), (63, 105),
            (65, 55), (66, 107), (67, 109), (70, 63), (78, 95), (78, 191), (80, 81), (81, 82), (82, 13), (84, 17),
            (87, 14), (88, 178), (91, 181), (93, 234), (95, 88), (103, 67), (105, 66), (109, 10), (127, 162),
            (132, 93), (136, 172), (144, 145), (145, 153), (146, 91), (148, 176), (149, 150), (150, 136), (152, 148),
            (153, 154), (154, 155), (155, 133), (157, 173), (158, 157), (159, 158), (160, 159), (161, 160), (162, 21),
            (163, 144), (172, 58), (173, 133), (176, 149), (178, 87), (181, 84), (185, 40), (191, 80), (234, 127),
            (246, 161), (249, 390), (251, 389), (263, 249), (263, 466), (267, 269), (269, 270), (270, 409), (276, 283),
            (282, 295), (283, 282), (284, 251), (288, 397), (293, 334), (295, 285), (296, 336), (297, 332), (300, 293),
            (310, 415), (311, 310), (312, 311), (314, 405), (317, 402), (318, 324), (321, 375), (323, 361), (324, 308),
            (332, 284), (334, 296), (338, 297), (356, 454), (361, 288), (365, 379), (373, 374), (374, 380), (375, 291),
            (377, 152), (378, 400), (379, 378), (380, 381), (381, 382), (382, 362), (384, 398), (385, 384), (386, 385),
            (387, 386), (388, 387), (389, 356), (390, 373), (397, 365), (398, 362), (400, 377), (402, 318), (405, 321),
            (409, 291), (415, 308), (454, 323), (466, 388), (469, 470), (470, 471), (471, 472), (472, 469), (474, 475),
            (475, 476), (476, 477), (477, 474),
        };

        #endregion

        readonly MediaPipeFaceRunningMode _runningMode;
        readonly int _numFaces;
        readonly float _minFaceDetectionConfidence;
        readonly float _minFacePresenceConfidence;
        readonly float _minFaceTrackingConfidence;
        readonly bool _smoothLandmarks;

        readonly bool _outputFaceBlendshapes;
        readonly bool _outputFacialTransformationMatrixes;
        readonly string _faceBlendshapesModelFilepath;
        readonly string _faceGeometryPipelineMetadataFilepath;

        readonly MultiBackendNet _faceDetectorNet;
        readonly MultiBackendNet _faceLandmarksNet;
        readonly MultiBackendNet _faceBlendshapesNet;

        /// <summary>True when blendshape model was loaded (OpenCV or Unity Inference Engine).</summary>
        readonly bool _hasFaceBlendshapesInference;

        /// <summary>Degrees-to-radians conversion factor used for face geometry frustum calculations.</summary>
        const float kFaceGeometryDegreesToRadians = (float)(Math.PI / 180.0);

        /// <summary>
        /// Vertical FOV in degrees configured by <see cref="FaceGeometryEnvGeneratorCalculator"/>.
        /// Zero when face geometry output is disabled.
        /// </summary>
        readonly float _faceGeometryVerticalFovDegrees;

        /// <summary>Near-plane distance. Zero when face geometry output is disabled.</summary>
        readonly float _faceGeometryNearPlane;

        /// <summary>Canonical mesh vertex xyz values in <c>landmark_id</c> order. Null when face geometry output is disabled.</summary>
        readonly float[] _faceGeometryCanonicalMetricLandmarks;

        readonly float[] _faceGeometryLandmarkWeights;
        readonly int _faceGeometryNumVertices;

        /// <summary>Face detector model path passed to the constructor, kept for logging and debugging.</summary>
        readonly string _faceDetectorModelFilepath;

        /// <summary>Output names for the face detector inference subgraph, in the same order as OpenCV <c>forward</c> (regressors then classificators).</summary>
        List<string> _faceDetectorOutLayerNames;

        /// <summary>Output names for face landmarks, in the same order as OpenCV (presence then landmarks).</summary>
        List<string> _faceLandmarksNetOutLayerNames;

        /// <summary>Output names for FaceBlendshapes, matching OpenCV <c>forward</c> order.</summary>
        List<string> _faceBlendshapesNetOutLayerNames;

        /// <summary>Reusable detector letterboxed BGR buffer, sized up to <see cref="kFaceDetectorLongRangeImageSize"/>.</summary>
        Mat _faceDetectorLetterboxBgr;

        /// <summary>
        /// Face detector inference input used by <see cref="InferenceSubgraph_FaceDetection"/>.
        /// Stored as <c>[1,H,W,3]</c> float32 in RGB with normalization <c>(x-127.5)/127.5</c>.
        /// Reused in the same way as Hand's <c>_palmInferenceBlob</c>.
        /// </summary>
        Mat _faceDetectorInferenceBlob;

        /// <summary>H x W x C view of <see cref="_faceDetectorInferenceBlob"/> used as the <c>convertTo</c> destination. Same role as Hand's <c>_palmInferenceBlobHxW</c>.</summary>
        Mat _faceDetectorInferenceBlobHxW;

        /// <summary>Temporary RGB 8-bit buffer for the NHWC path, used as the <c>cvtColor</c> destination.</summary>
        Mat _faceDetectorInferenceRgb8u;

        /// <summary>Reusable buffer for <c>TensorsToDetectionsCalculator</c>, including SSD anchors and repeated keypoint anchor data. Row count depends on the model.</summary>
        Mat _faceDetectorAnchorsBuffer;

        Mat _faceTensorsToDetectionsWorking;
        MatOfInt _faceNmsIndices;

        /// <summary>Reusable list of face detector <c>forward</c> output <see cref="Mat"/> values, cleared and refilled every frame.</summary>
        readonly List<Mat> _faceDetectorForwardOutputList = new List<Mat>();

        /// <summary>Reusable list of face landmarks <c>forward</c> output <see cref="Mat"/> values.</summary>
        readonly List<Mat> _faceLandmarksForwardOutputList = new List<Mat>();

        /// <summary>Reusable list of FaceBlendshapes <c>forward</c> output <see cref="Mat"/> values.</summary>
        readonly List<Mat> _faceBlendshapesForwardOutputList = new List<Mat>();

        /// <summary>Normalized padding buffer used by <see cref="ImagePreprocessingGraph_FillLetterbox"/>, overwritten on each call and shared across uses.</summary>
        readonly float[] _faceDetectorLetterboxPaddingNormReuse = new float[4];

        /// <summary>Lower-bound tensor for <see cref="NumpyClip"/>, reused via <c>create</c> to match the input shape.</summary>
        Mat _faceNumpyClipLo;

        /// <summary>Upper-bound tensor for <see cref="NumpyClip"/>.</summary>
        Mat _faceNumpyClipHi;

        /// <summary>Transpose buffer for face detector box rows, used by <see cref="FaceDetectorGraph_PrepareBoxMajorRows"/>.</summary>
        Mat _faceDetectorTransposeBuffer;

        /// <summary>Transpose buffer for the face detector score column, used by <see cref="FaceDetectorGraph_PrepareScoreColumn"/>.</summary>
        Mat _faceDetectorScoreColumnBuffer;

        /// <summary>Resize destination used by the full-frame letterbox path in <see cref="ImagePreprocessingGraph_FillLetterbox"/>.</summary>
        Mat _faceLetterboxResizeScratch;

        /// <summary>Temporary list of merged boxes and decoded rows used by <see cref="NonMaxSuppressionCalculator"/>.</summary>
        readonly List<float[]> _faceNmsMergedBoxScratch = new List<float[]>();

        readonly List<float[]> _faceNmsMergedDecScratch = new List<float[]>();
        readonly List<float> _faceNmsMergedScScratch = new List<float>();

        /// <summary>Pool of 17-element rows used by <see cref="DetectionProjectionCalculator"/>.</summary>
        readonly Stack<float[]> _poolFaceDetectorProjRow17 = new Stack<float[]>();

        /// <summary>Pool of 16-element decoded rows for WEIGHTED NMS merges.</summary>
        readonly Stack<float[]> _poolFaceDetectorNmsDec16 = new Stack<float[]>();

        /// <summary>Pool of 4-element merged boxes for WEIGHTED NMS.</summary>
        readonly Stack<float[]> _poolFaceDetectorNmsBox4 = new Stack<float[]>();

        /// <summary>Keypoint weighted-sum buffer used inside <see cref="NonMaxSuppressionCalculator"/>.</summary>
        float[] _faceWnmsKpAccumulator;

        /// <summary>Scratch buffer for raw tensor reads in <see cref="TensorsToLandmarksCalculator_Face"/>.</summary>
        float[] _faceTensorsToLmRaw;

        /// <summary>Normalized output buffer for <see cref="TensorsToLandmarksCalculator_Face"/>, returned by reference.</summary>
        float[] _faceTensorsToLmNorm;

        /// <summary>Output buffer used by <see cref="LandmarkLetterboxRemovalCalculator_Face"/> when removing padding.</summary>
        readonly float[] _faceLetterboxRemovedNormScratch = new float[kFaceMeshWithIrisLandmarksNum * 3];

        /// <summary>Reusable EndLoop aggregation buffer for <see cref="MultiFaceLandmarksDetectorGraph"/>, cleared every frame.</summary>
        readonly List<bool> _multiFacePresencesScratch = new List<bool>();

        readonly List<float> _multiFacePresenceScoresScratch = new List<float>();
        readonly List<Vec3f[]> _multiFaceLandmarkListsScratch = new List<Vec3f[]>();
        readonly List<NormalizedRect> _multiFaceNextFrameRectsScratch = new List<NormalizedRect>();

        /// <summary>Temporary list of detector-expanded rects used by <see cref="ProcessVideoData"/>.</summary>
        readonly List<NormalizedRect> _processVideoExpandedDetectorScratch = new List<NormalizedRect>();

        /// <summary>Rect copy destination used by <see cref="PreviousLoopbackCalculator"/>.</summary>
        readonly List<NormalizedRect> _previousLoopbackCopyScratch = new List<NormalizedRect>();

        /// <summary>Reusable result list for <see cref="AssociationNormRectCalculator"/>.</summary>
        readonly List<NormalizedRect> _associationNormRectScratch = new List<NormalizedRect>();

        /// <summary>Reusable pose-row list used by <see cref="FaceGeometryFromLandmarksGraph"/>.</summary>
        readonly List<float[]> _faceGeometryPoseRowsScratch = new List<float[]>();

        /// <summary>Reusable aggregation list for multi-face landmark arrays (468/478) inside <see cref="FaceGeometryFromLandmarksGraph"/>.</summary>
        readonly List<Vec3f[]> _faceGeomMultiNoIrisScratch = new List<Vec3f[]>();

        /// <summary>NMS input after applying upstream <c>TensorsToDetectionsCalculatorOptions.min_score_thresh</c>, which corresponds to Tasks <c>min_detection_confidence</c>. The row count shrinks only when thresholding removes entries. This serves the same role as Hand's <c>_palmScoreFiltered*</c>.</summary>
        Mat _faceScoreFilteredBoxXywh;
        Mat _faceScoreFilteredScore;
        Mat _faceScoreFilteredDecodedNx16;

        /// <summary>Tensor-normalized bounding boxes after WEIGHTED NMS (K x 4, <c>xmin ymin w h</c>).</summary>
        Mat _faceWnmsMergedBoxXywh;
        /// <summary>Decoded rows after WEIGHTED NMS (K x 16).</summary>
        Mat _faceWnmsMergedDecodedNx16;
        /// <summary>Scores after WEIGHTED NMS (K x 1).</summary>
        Mat _faceWnmsMergedScore;

        readonly List<(int idx, float sc)> _faceWnmsIndexed = new List<(int, float)>();
        List<(int idx, float sc)> _faceWnmsRemained = new List<(int, float)>();
        List<(int idx, float sc)> _faceWnmsNextRemained = new List<(int, float)>();

        /// <summary>Input side length corresponding to upstream <c>BuildInputImageTensorSpecs</c>. Fixed to short-range 128x128 for the Tasks ONNX path.</summary>
        readonly int _faceDetectorTensorSize = kFaceDetectorShortRangeImageSize;

        /// <summary>Equivalent to upstream <c>TensorsToDetectionsCalculatorOptions.num_boxes</c>. Fixed to 896 for the short-range 128-input path.</summary>
        readonly int _faceDetectorNumBoxes = kFaceDetectorLegacyShortRangeNumBoxes;

        /// <summary>Whether the detector is the 192x192 long-range model. Always false because the fixed ONNX path only supports the short-range model.</summary>
        readonly bool _faceDetectorIsLongRange = false;

        /// <summary>The <c>MATRIX</c> output of <c>ImagePreprocessingGraph</c>, corresponding to the <c>PROJECTION_MATRIX</c> input of <c>DetectionProjectionCalculator</c>.</summary>
        readonly float[] _faceDetectorProjectionMatrix16 = new float[16];

        /// <summary>Decoded 16-value rows from <c>TensorsToDetectionsCalculator</c>, in letterboxed tensor-normalized coordinates.</summary>
        Mat _faceDetectorDecodedBoxesNx16;

        /// <summary>Perspective-transform buffers used when <c>NORM_RECT</c> is present.</summary>
        Mat _faceDetectorWarpSrcPts;
        Mat _faceDetectorWarpDstPts;

        /// <summary>Row buffers for <c>TensorsToDetectionsCalculator</c>, using <c>float[]</c> for <c>get</c> and <c>put</c>.</summary>
        float[] _faceDetectorDecodeRowSrc;
        float[] _faceDetectorDecodeRowDst;
        float[] _faceDetectorAnchorRow4;

        /// <summary>Projected row for one detection (4 normalized bbox values + 12 keypoint values + score), used from <see cref="DetectionProjectionCalculator"/> through <see cref="DetectionsToRectsCalculator"/>.</summary>
        const int FaceDetectorProjectedDetectionRowLength = 17;

        /// <summary>Shared 478-point placeholder used by <c>EndLoopNormalizedLandmarkListVectorCalculator</c> and related paths.</summary>
        static readonly Vec3f[] s_emptyNormLandmarks478 = new Vec3f[kFaceMeshWithIrisLandmarksNum];

        /// <summary>Shared cache of legacy SSD anchor matrices with rows <c>(x_center, y_center, w, h)</c>.</summary>
        static Mat _faceDetectorSsdAnchors128Cache;
        static Mat _faceDetectorSsdAnchors192Cache;

        /// <summary>Input side length for the face_landmarks model, fixed at 192x192.</summary>
        readonly int _faceLandmarkTensorSize = kFaceLandmarksDetectorImageSize;

        /// <summary>Face landmarks model path passed to the constructor, kept for logging and debugging.</summary>
        readonly string _faceLandmarksModelFilepath;

        Mat _faceLmWarpSrcPts;
        Mat _faceLmWarpDstPts;
        Mat _faceLmWarpedBgr;
        Mat _faceLmWarpedRgb;
        /// <summary>
        /// Face landmarks inference input used by <see cref="InferenceSubgraph_SingleFaceLandmarks"/>.
        /// Reused NHWC RGB tensor normalized to 0-1, analogous to Hand's <c>_singleHandLandmarkBlob</c>.
        /// </summary>
        Mat _faceLandmarksInferenceBlob;

        /// <summary>H x W x C view of <see cref="_faceLandmarksInferenceBlob"/>, with the same role as Hand's <c>_singleHandLandmarkBlobHxW</c>.</summary>
        Mat _faceLandmarksInferenceBlobHxW;

        /// <summary>Row-major 4x4 matrix used by <c>LandmarkProjectionCalculator</c> with <c>NORM_RECT</c> and <c>IMAGE_DIMENSIONS</c>.</summary>
        readonly float[] _faceLmProjectionMatrix16 = new float[16];

        /// <summary>Implements <c>smooth_landmarks</c> inside <c>MultiFaceLandmarksDetectorGraph</c> using an outer-loop One Euro filter.</summary>
        readonly FaceLandmarksSmoothingPipeline _faceLandmarksSmoothingPipeline;

        const double kFaceLandmarksSmoothingDefaultFrequency = 30.0;

        /// <summary>Reusable row-packed buffer for the main face landmark <see cref="Mat"/> written by detection.</summary>
        Mat _outputBuffer;

        /// <summary>Optional output buffer for blendshape coefficients, with rows as faces and 52 columns.</summary>
        Mat _outputBlendshapesBuffer;

        /// <summary>Optional output buffer for facial pose 4x4 matrices, with rows as faces and 16 row-major columns.</summary>
        Mat _outputFacialPoseBuffer;

        /// <summary>Reusable <c>[1,146,2]</c> input tensor for <c>LandmarksToTensorCalculator</c>.</summary>
        Mat _faceBlendshapesInputBlob;

        /// <summary>In VIDEO mode, stores the previous frame's <c>FACE_RECTS_NEXT_FRAME</c> values produced by the landmarks subgraph.</summary>
        readonly List<NormalizedRect> _prevFaceRectsFromLandmarks = new List<NormalizedRect>();

        /// <summary>Face ROI looped back from the previous frame, corresponding to the upstream <c>NormalizedRect</c> packet.</summary>
        private struct NormalizedRect
        {
            public float XCenter;
            public float YCenter;
            public float Width;
            public float Height;
            public float Rotation;
            /// <summary>Corresponds to upstream <c>NormalizedRect.rect_id</c>. Null when unset.</summary>
            public long? RectId;
        }

        /// <summary>
        /// Bundled outputs of <c>FaceDetectorGraph</c>, corresponding to the upstream <c>DETECTIONS</c>, <c>FACE_RECTS</c>, and <c>EXPANDED_FACE_RECTS</c> streams in <c>face_detector_graph.cc</c>.
        /// </summary>
        struct FaceDetectorGraphResult
        {
            /// <summary>Pixel-space detections corresponding to the upstream <c>DETECTIONS</c> stream after <c>DetectionTransformationCalculator</c>, stored in row form.</summary>
            public List<float[]> PixelDetections;

            /// <summary>Corresponds to the upstream <c>FACE_RECTS</c> stream of normalized rotated rects.</summary>
            public List<NormalizedRect> FaceRects;

            /// <summary>Corresponds to the upstream <c>EXPANDED_FACE_RECTS</c> stream, used as the landmark subgraph ROI input.</summary>
            public List<NormalizedRect> ExpandedFaceRects;
        }

        /// <summary>
        /// Intermediate result for one face before packing. Downstream subgraph implementations fill each field before packing.
        /// </summary>
        private struct FaceResult
        {
            /// <summary>Face landmark presence flag after the upstream-equivalent <c>ThresholdingCalculator</c>.</summary>
            public bool FacePresence;

            /// <summary>
            /// Presence scalar produced by the upstream-equivalent <c>TensorsToFloatsCalculator</c>, before <c>ThresholdingCalculator</c>.
            /// </summary>
            public float FacePresenceScore;

            /// <summary>Landmarks in normalized image coordinates, corresponding to the Tasks <c>face_landmarks</c> output with 478 points.</summary>
            public Vec3f[] NormLandmarks;

            /// <summary>VIDEO loopback ROI for the next frame, corresponding to the upstream <c>FACE_RECTS_NEXT_FRAME</c> stream.</summary>
            public NormalizedRect NextFrameRect;

            /// <summary>Phase B output from <c>FaceBlendshapesGraph</c>: 52 coefficients, or null when disabled.</summary>
            public float[] BlendshapeCoefficients;

            /// <summary>Phase B output from <c>FaceGeometryFromLandmarksGraph</c>: row-major 4x4 pose matrix, or null when disabled.</summary>
            public float[] FacialPoseTransformRowMajor16;
        }

        /// <summary>
        /// Packed result for one detected face.
        /// The memory layout matches one row produced by <see cref="BuildPackedOutputMats"/>.
        /// <see cref="NormLandmarks"/> corresponds to the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) Task API
        /// <c>face_landmarks</c> output represented as 478 <see cref="Vec3f"/> values.
        /// Face-level raw presence scores are not included; the internal threshold is configured by
        /// the <see cref="MediaPipeFaceLandmarker"/> constructor.
        /// </summary>
        [Serializable]
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public unsafe struct FaceLandmarkerEstimationData
        {
            /// <summary>
            /// Number of face landmarks in [MediaPipe](https://github.com/google-ai-edge/mediapipe): 478.
            /// Each element is represented as <see cref="Vec3f"/> and indices follow the original face mesh with iris ordering.
            /// </summary>
            public const int LANDMARK_VEC3F_COUNT = kFaceMeshWithIrisLandmarksNum;

            /// <summary>Total number of float values occupied by the normalized landmark block for one face.</summary>
            public const int LANDMARK_ELEMENT_COUNT = LANDMARK_VEC3F_COUNT * 3;

            /// <summary>Total float element count per packed row, containing only normalized landmarks.</summary>
            public const int ELEMENT_COUNT = LANDMARK_ELEMENT_COUNT;

            /// <summary>Total byte size of one packed face row.</summary>
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Packed normalized face landmarks.
            /// Corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>face_landmarks</c> output flattened as 478 xyz triplets in row-major float order.
            /// </summary>
            public fixed float NormLandmarks[LANDMARK_ELEMENT_COUNT];

            /// <summary>
            /// Creates one packed face result from an array of <see cref="Vec3f"/> values.
            /// Each <see cref="Vec3f"/> corresponds to one landmark in the original
            /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>face_landmarks</c> output.
            /// </summary>
            public FaceLandmarkerEstimationData(Vec3f[] normLandmarks)
            {
                if (normLandmarks == null || normLandmarks.Length != LANDMARK_VEC3F_COUNT)
                    throw new ArgumentException("normLandmarks must be a Vec3f[" + LANDMARK_VEC3F_COUNT + "]");

                for (int i = 0; i < normLandmarks.Length; i++)
                {
                    int offset = i * 3;
                    ref readonly var v = ref normLandmarks[i];
                    NormLandmarks[offset + 0] = v.Item1;
                    NormLandmarks[offset + 1] = v.Item2;
                    NormLandmarks[offset + 2] = v.Item3;
                }
            }

            /// <summary>
            /// Returns 478 normalized-landmark elements (x, y, z) as a
            /// <see cref="ReadOnlySpan{T}"/> that is memory-compatible with the fixed buffer and does not copy.
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
            /// Returns a heap-allocated copy of the 478 normalized landmarks,
            /// useful as a snapshot of <see cref="GetNormLandmarks"/>.
            /// </summary>
            public readonly Vec3f[] GetNormLandmarksArray()
            {
                var landmarks = new Vec3f[LANDMARK_VEC3F_COUNT];
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

            public readonly override string ToString()
            {
                var sb = new StringBuilder(2048);
                sb.Append("FaceLandmarkerEstimationData(");
                sb.Append("NormLandmarks:");
                foreach (var p in GetNormLandmarks())
                    sb.Append(p.ToString());
                sb.Append(')');
                return sb.ToString();
            }
        }

        /// <summary>
        /// Packed result for one detected face blendshape output.
        /// The memory layout matches one row of the packed blendshape result matrix when
        /// <c>output_face_blendshapes</c> is enabled.
        /// <see cref="Coefficients"/> corresponds to the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) Task API
        /// <c>face_blendshapes</c> output represented as 52 coefficients.
        /// </summary>
        [Serializable]
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public unsafe struct FaceBlendshapeEstimationData
        {
            /// <summary>Number of face blendshape coefficients in [MediaPipe](https://github.com/google-ai-edge/mediapipe): 52.</summary>
            public const int COEFFICIENT_COUNT = kFaceBlendshapeCoefficientCount;
            /// <summary>Total byte size of one packed blendshape row.</summary>
            public const int DATA_SIZE = COEFFICIENT_COUNT * 4;

            /// <summary>
            /// Packed face blendshape coefficients.
            /// Corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// <c>face_blendshapes</c> output flattened in category order.
            /// </summary>
            public fixed float Coefficients[COEFFICIENT_COUNT];

            /// <summary>
            /// Returns the 52 blendshape coefficients as a
            /// <see cref="ReadOnlySpan{T}"/> that is memory-compatible with the fixed buffer and does not copy.
            /// </summary>
            public readonly ReadOnlySpan<float> GetCoefficients()
            {
                unsafe
                {
                    fixed (float* p = Coefficients)
                        return new ReadOnlySpan<float>(p, COEFFICIENT_COUNT);
                }
            }

            /// <summary>
            /// Returns a heap-allocated copy of the 52 blendshape coefficients,
            /// useful as a snapshot of <see cref="GetCoefficients"/>.
            /// </summary>
            public readonly float[] GetCoefficientsArray()
            {
                var coeffs = new float[COEFFICIENT_COUNT];
                unsafe
                {
                    for (int i = 0; i < coeffs.Length; i++)
                        coeffs[i] = Coefficients[i];
                }

                return coeffs;
            }

            public readonly override string ToString()
            {
                var sb = new StringBuilder(512);
                sb.Append("FaceBlendshapeEstimationData(");
                sb.Append("Coefficients:");
                foreach (float c in GetCoefficients())
                    sb.Append(c.ToString());
                sb.Append(')');
                return sb.ToString();
            }
        }

        /// <summary>
        /// Packed result for one detected facial transformation matrix.
        /// The memory layout matches one row of the packed facial transformation matrix result when
        /// <c>output_facial_transformation_matrixes</c> is enabled.
        /// <see cref="RowMajor4x4"/> corresponds to the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face geometry output
        /// stored as a row-major 4x4 matrix in the same ordering as <c>MatrixData</c>.
        /// </summary>
        [Serializable]
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public unsafe struct FacialTransformationMatrixEstimationData
        {
            /// <summary>Number of float values in one row-major 4x4 transformation matrix.</summary>
            public const int MATRIX_ELEMENT_COUNT = 16;
            /// <summary>Total byte size of one packed facial transformation row.</summary>
            public const int DATA_SIZE = MATRIX_ELEMENT_COUNT * 4;

            /// <summary>
            /// Packed row-major 4x4 facial transformation matrix.
            /// Corresponds to the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
            /// facial transformation matrix output.
            /// </summary>
            public fixed float RowMajor4x4[MATRIX_ELEMENT_COUNT];

            /// <summary>
            /// Returns the 16 row-major 4x4 matrix entries as a
            /// <see cref="ReadOnlySpan{T}"/> that is memory-compatible with the fixed buffer and does not copy.
            /// Translation in a <c>[R|t; 0 0 0 1]</c> layout is stored in the <strong>fourth column</strong> (indices 3, 7, and 11).
            /// Indices 12 to 14 belong to the first three elements of the last row, which are normally 0, 0, 0, so reading them would incorrectly make translation appear to be zero.
            /// </summary>
            public readonly ReadOnlySpan<float> GetRowMajor4x4()
            {
                unsafe
                {
                    fixed (float* p = RowMajor4x4)
                        return new ReadOnlySpan<float>(p, MATRIX_ELEMENT_COUNT);
                }
            }

            /// <summary>
            /// Returns a heap-allocated copy of the 16 row-major 4x4 matrix entries,
            /// useful as a snapshot of <see cref="GetRowMajor4x4"/>.
            /// </summary>
            public readonly float[] GetRowMajor4x4Array()
            {
                var m = new float[MATRIX_ELEMENT_COUNT];
                GetRowMajor4x4().CopyTo(m);
                return m;
            }

            /// <summary>
            /// Returns a single-line summary string that concatenates the main fields, in the same style as <see cref="FaceLandmarkerEstimationData.ToString"/>.
            /// </summary>
            public readonly override string ToString()
            {
                var sb = new StringBuilder(512);
                sb.Append("FacialTransformationMatrixEstimationData(");
                sb.Append("Translation:");
                ReadOnlySpan<float> rm = GetRowMajor4x4();
                sb.AppendFormat("({0:F4},{1:F4},{2:F4})", rm[3], rm[7], rm[11]);
                sb.Append(",RowMajor4x4:");
                for (int i = 0; i < MATRIX_ELEMENT_COUNT; i++)
                    sb.Append(rm[i].ToString());

                sb.Append(')');
                return sb.ToString();
            }
        }

        /// <summary>
        /// Creates a face landmarker worker backed by a face detector model and a face landmark model.
        /// This public API maps to the model assets and runtime options used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face detector graph,
        /// face landmarks detector graph, optional FaceBlendshapesGraph, and optional face geometry path.
        /// </summary>
        /// <param name="faceDetectorModelFilepath">
        /// File path to the face detector model.
        /// Corresponds to the detector model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face detector graph.
        /// </param>
        /// <param name="faceLandmarksModelFilepath">
        /// File path to the face landmarks model.
        /// Corresponds to the landmark model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) SingleFaceLandmarksDetectorGraph path.
        /// </param>
        /// <param name="runningMode">
        /// Task running mode.
        /// Corresponds to whether the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) task behaves like single-image processing
        /// or stateful video processing with loopback tracking state.
        /// </param>
        /// <param name="numFaces">
        /// Maximum number of faces to return.
        /// Corresponds to the max number of faces option used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceLandmarker task.
        /// </param>
        /// <param name="minFaceDetectionConfidence">
        /// Minimum confidence for face detections to be kept before later stages.
        /// Corresponds to the face detector minimum detection confidence used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) task configuration.
        /// </param>
        /// <param name="minFacePresenceConfidence">
        /// Minimum presence confidence required for landmark results to be treated as present.
        /// Corresponds to the face presence threshold used after the landmark model in
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe).
        /// </param>
        /// <param name="minTrackingConfidence">
        /// Minimum tracking confidence required to reuse the previous-frame rectangle.
        /// Corresponds to the face tracking confidence gate used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) video pipeline.
        /// </param>
        /// <param name="outputFaceBlendshapes">
        /// When true, enables the blendshape output; packed rows use the memory layout of <see cref="FaceBlendshapeEstimationData"/> (second slot when three outputs are returned).
        /// Corresponds to the task option that enables <c>FACE_BLENDSHAPES</c> in
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceLandmarker.
        /// </param>
        /// <param name="faceBlendshapesModelFilepath">
        /// File path to the face blendshapes model.
        /// Corresponds to the model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceBlendshapesGraph path.
        /// </param>
        /// <param name="outputFacialTransformationMatrixes">
        /// When true, enables the facial transformation matrix output.
        /// Corresponds to the task option that enables face geometry output in
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceLandmarker.
        /// </param>
        /// <param name="faceGeometryPipelineMetadataFilepath">
        /// File path to the upstream <c>geometry_pipeline_metadata_landmarks_including_iris.pbtxt</c> file.
        /// This corresponds to the geometry pipeline metadata consumed by face geometry calculators.
        /// </param>
        /// <param name="dnnBackend">
        /// OpenCV DNN <c>DNN_BACKEND_*</c> constant, or <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/> for Unity Inference Engine. When Sentis, use a path loadable by <c>ModelLoader.Load</c> (e.g. <c>.sentis</c>); <paramref name="dnnTarget"/> is cast to <c>BackendType</c>.
        /// </param>
        /// <param name="dnnTarget">
        /// OpenCV DNN <c>DNN_TARGET_*</c> constant, or for Sentis an integer cast to Unity Inference Engine <c>BackendType</c>.
        /// </param>
        public MediaPipeFaceLandmarker(
            string faceDetectorModelFilepath,
            string faceLandmarksModelFilepath,
            MediaPipeFaceRunningMode runningMode = MediaPipeFaceRunningMode.IMAGE,
            int numFaces = 1,
            float minFaceDetectionConfidence = kDefaultMinFaceDetectionConfidence,
            float minFacePresenceConfidence = kDefaultMinFacePresenceConfidence,
            float minTrackingConfidence = kDefaultMinFaceTrackingConfidence,
            bool outputFaceBlendshapes = false,
            string faceBlendshapesModelFilepath = null,
            bool outputFacialTransformationMatrixes = false,
            string faceGeometryPipelineMetadataFilepath = null,
            int dnnBackend = Dnn.DNN_BACKEND_OPENCV,
            int dnnTarget = Dnn.DNN_TARGET_CPU)
            : base(dnnBackend, dnnTarget)
        {
            if (string.IsNullOrEmpty(faceDetectorModelFilepath))
                throw new ArgumentException("The face detector model file path is not specified.", nameof(faceDetectorModelFilepath));
            if (string.IsNullOrEmpty(faceLandmarksModelFilepath))
                throw new ArgumentException("The face landmarks model file path is not specified.", nameof(faceLandmarksModelFilepath));
            if (numFaces <= 0)
                throw new ArgumentOutOfRangeException(nameof(numFaces), "numFaces must be at least 1.");

            if (outputFaceBlendshapes && string.IsNullOrWhiteSpace(faceBlendshapesModelFilepath))
                throw new ArgumentException(
                    "A face_blendshapes model path is required when output_face_blendshapes is true, matching the upstream Task InvalidArgument intent.",
                    nameof(faceBlendshapesModelFilepath));

            if (outputFacialTransformationMatrixes &&
                string.IsNullOrWhiteSpace(faceGeometryPipelineMetadataFilepath))
                throw new ArgumentException(
                    "A geometry metadata pbtxt path is required when output_facial_transformation_matrixes is true.",
                    nameof(faceGeometryPipelineMetadataFilepath));

            _runningMode = runningMode;
            _numFaces = numFaces;
            _minFaceDetectionConfidence = Mathf.Clamp01(minFaceDetectionConfidence);
            _minFacePresenceConfidence = Mathf.Clamp01(minFacePresenceConfidence);
            _minFaceTrackingConfidence = Mathf.Clamp01(minTrackingConfidence);
            _smoothLandmarks = runningMode == MediaPipeFaceRunningMode.VIDEO && numFaces == 1;
            _outputFaceBlendshapes = outputFaceBlendshapes;
            _outputFacialTransformationMatrixes = outputFacialTransformationMatrixes;
            _faceBlendshapesModelFilepath = faceBlendshapesModelFilepath;
            _faceGeometryPipelineMetadataFilepath = faceGeometryPipelineMetadataFilepath;
            _faceDetectorModelFilepath = faceDetectorModelFilepath;
            _faceLandmarksModelFilepath = faceLandmarksModelFilepath;

#if !OPENCV_SENTIS_AVAILABLE
            if (DnnBackend == MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS)
            {
                throw new NotSupportedException(
                    "DNN_BACKEND_UNITY_SENTIS requires Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer in the project and the OPENCV_SENTIS_AVAILABLE define.");
            }
#endif

            try
            {
                _faceDetectorNet = MultiBackendDnn.readNet(faceDetectorModelFilepath);
                _faceDetectorNet.setPreferableBackend(DnnBackend);
                _faceDetectorNet.setPreferableTarget(DnnTarget);
                _faceDetectorOutLayerNames = _faceDetectorNet.getUnconnectedOutLayersNames();

                _faceLandmarksNet = MultiBackendDnn.readNet(faceLandmarksModelFilepath);
                _faceLandmarksNet.setPreferableBackend(DnnBackend);
                _faceLandmarksNet.setPreferableTarget(DnnTarget);
                _faceLandmarksNetOutLayerNames = _faceLandmarksNet.getUnconnectedOutLayersNames();

                if (outputFaceBlendshapes)
                {
                    _faceBlendshapesNet = MultiBackendDnn.readNet(faceBlendshapesModelFilepath);
                    _faceBlendshapesNet.setPreferableBackend(DnnBackend);
                    _faceBlendshapesNet.setPreferableTarget(DnnTarget);
                    _faceBlendshapesNetOutLayerNames = _faceBlendshapesNet.getUnconnectedOutLayersNames();
                    _hasFaceBlendshapesInference = true;
                }
                else
                {
                    _faceBlendshapesNet = null;
                    _faceBlendshapesNetOutLayerNames = new List<string>();
                }

                if (outputFacialTransformationMatrixes)
                {
                    if (!File.Exists(faceGeometryPipelineMetadataFilepath))
                        throw new FileNotFoundException(
                            "Face geometry metadata was not found: " + faceGeometryPipelineMetadataFilepath,
                            faceGeometryPipelineMetadataFilepath);

                    FaceGeometryEnvGeneratorCalculator(true, out _faceGeometryVerticalFovDegrees,
                        out _faceGeometryNearPlane);
                    FaceGeometryLoadPbtxt(faceGeometryPipelineMetadataFilepath,
                        out _faceGeometryCanonicalMetricLandmarks, out _faceGeometryLandmarkWeights,
                        out _faceGeometryNumVertices);
                }
                else
                {
                    FaceGeometryEnvGeneratorCalculator(false, out _faceGeometryVerticalFovDegrees,
                        out _faceGeometryNearPlane);
                    _faceGeometryCanonicalMetricLandmarks = null;
                    _faceGeometryLandmarkWeights = null;
                    _faceGeometryNumVertices = 0;
                }

                _faceLandmarksSmoothingPipeline = (_smoothLandmarks
                    && runningMode == MediaPipeFaceRunningMode.VIDEO
                    && numFaces == 1)
                    ? new FaceLandmarksSmoothingPipeline()
                    : null;
            }
            catch (Exception e)
            {
                throw new ArgumentException("Failed to initialize the Face Landmarker DNN models. Check the model paths and file contents.", e);
            }
        }

        /// <summary>
        /// High-level inference API equivalent to the synchronous detect entry points of
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceLandmarker.
        /// Returns one to three packed output matrices depending on which optional outputs are enabled.
        /// </summary>
        /// <param name="image">
        /// Input image in BGR 3-channel format.
        /// Corresponds to the input image consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face landmarker graph.
        /// </param>
        /// <param name="useCopyOutput">
        /// If true, returns a copied output matrix.
        /// If false, returns a view backed by the worker's reusable output buffer.
        /// </param>
        /// <returns>
        /// Packed result matrices with length 1 or 3.
        /// The length is 3 when either <c>outputFaceBlendshapes</c> or <c>outputFacialTransformationMatrixes</c> is enabled, otherwise 1.
        /// Element [0] is the packed face result matrix with one row per detected face and
        /// <see cref="FaceLandmarkerEstimationData.ELEMENT_COUNT"/> columns per row.
        /// Element [0] is returned as <c>CV_32FC1</c>, and each row matches the memory layout of
        /// <see cref="FaceLandmarkerEstimationData"/>:
        /// columns <c>[0 .. 1433]</c> store 478 normalized face landmarks as xyz triplets.
        /// When the array length is 3, element [1] contains packed blendshape coefficients or an empty <see cref="Mat"/> when disabled,
        /// with each row returned as <c>CV_32FC1</c>, containing 52 coefficients in category order,
        /// and matching the memory layout of <see cref="FaceBlendshapeEstimationData"/>.
        /// Element [2] contains packed row-major 4x4 facial transformation matrices or an empty <see cref="Mat"/> when disabled,
        /// with each row returned as <c>CV_32FC1</c>, containing 16 float values,
        /// and matching the memory layout of <see cref="FacialTransformationMatrixEstimationData"/>.
        /// </returns>
        public Mat[] Detect(Mat image, bool useCopyOutput = false)
        {
            if (image != null) image.ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            Execute(image);
            return BuildDetectReturnArray(useCopyOutput);
        }

        /// <summary>Asynchronous single-image detection; returns copied output mats (see <see cref="Detect(Mat, bool)"/>).</summary>
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

        /// <summary>Asynchronous single-image detection; returns copied output mats (see <see cref="Detect(Mat, bool)"/>).</summary>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task{TResult}"/>.
        /// See <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. Web synchronous fallback applies only to the OpenCV Dnn backend; Sentis remains asynchronous on every platform, including Web.
        /// </remarks>
        [Obsolete("Use DetectTaskAsync(). DetectAsync() will return Awaitable in a future version.")]
        public Task<Mat[]> DetectAsync(Mat image, CancellationToken cancellationToken = default) =>
            DetectTaskAsync(image, cancellationToken);

        /// <summary>
        /// Returns output Mats packed into an array by output index order, with length 1 or 3.
        /// </summary>
        Mat[] BuildDetectReturnArray(bool useCopyOutput)
        {
            int n = (_outputFaceBlendshapes || _outputFacialTransformationMatrixes) ? 3 : 1;
            var arr = new Mat[n];
            for (int i = 0; i < n; i++)
                arr[i] = useCopyOutput ? CopyOutput(i) : PeekOutput(i);
            return arr;
        }

        /// <summary>
        /// Converts a packed result matrix into a managed array of <see cref="FaceLandmarkerEstimationData"/>.
        /// Each returned element corresponds to one row from <see cref="Detect(Mat, bool)"/>.
        /// </summary>
        /// <param name="result">
        /// Packed output matrix returned by <see cref="Detect(Mat, bool)"/> or a compatible source.
        /// Each row corresponds to one face and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face landmarks output.
        /// </param>
        /// <returns>
        /// Managed array of face estimation data.
        /// Returns an empty array when no faces are present.
        /// </returns>
        public virtual FaceLandmarkerEstimationData[] ToStructuredData(Mat result)
        {
            ThrowIfDisposed();

            if (result != null)
                result.ThrowIfDisposed();
            if (result.empty())
                return Array.Empty<FaceLandmarkerEstimationData>();

            int elementCount = FaceLandmarkerEstimationData.ELEMENT_COUNT;
            if (result.cols() < elementCount)
                throw new ArgumentException("Invalid result matrix. It must have at least " + elementCount + " columns.");

            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            int faceCount = result.rows();
            var dst = new FaceLandmarkerEstimationData[faceCount];
            OpenCVMatUtils.CopyFromMat(result, dst);

            return dst;
        }

        /// <summary>
        /// Views a packed result matrix as a zero-allocation <see cref="Span{T}"/> of
        /// <see cref="FaceLandmarkerEstimationData"/>.
        /// </summary>
        /// <remarks>
        /// The returned span remains valid only while <paramref name="result"/> stays allocated
        /// and unchanged.
        /// If the matrix has more than <see cref="FaceLandmarkerEstimationData.ELEMENT_COUNT"/> columns,
        /// interpreting the underlying memory as contiguous rows of
        /// <see cref="FaceLandmarkerEstimationData"/> can cross row boundaries.
        /// The worker-generated packed matrices use the exact expected column count.
        /// </remarks>
        /// <param name="result">
        /// Packed output matrix returned by <see cref="Detect(Mat, bool)"/> or a compatible source.
        /// Each row corresponds to one face and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face landmarks output.
        /// </param>
        /// <returns>
        /// Span whose elements correspond to faces in row order.
        /// Returns an empty span when the matrix is empty.
        /// </returns>
        public virtual Span<FaceLandmarkerEstimationData> ToStructuredDataAsSpan(Mat result)
        {
            ThrowIfDisposed();

            if (result != null)
                result.ThrowIfDisposed();
            if (result.empty())
                return Span<FaceLandmarkerEstimationData>.Empty;

            int elementCount = FaceLandmarkerEstimationData.ELEMENT_COUNT;
            if (result.cols() < elementCount)
                throw new ArgumentException("Invalid result matrix. It must have at least " + elementCount + " columns.");

            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            return result.AsSpan<FaceLandmarkerEstimationData>();
        }

        /// <summary>
        /// Draws face landmarks and any enabled optional outputs from a <see cref="Mat"/> array whose layout matches
        /// <see cref="Detect(Mat, bool)"/>.
        /// Element <c>[0]</c> is required and contains one row per face.
        /// When present, element <c>[1]</c> contains blendshape output and element <c>[2]</c> contains facial transformation matrices.
        /// Array input is supported for compatibility with <see cref="MediaPipeHolisticLandmarker"/>.
        /// </summary>
        /// <param name="image">Destination image for visualization.</param>
        /// <param name="results">
        /// Array of output matrices.
        /// <c>results[0]</c> corresponds to the packed face output derived from the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face landmark output.
        /// <c>results[1]</c>, when present, corresponds to the blendshape output enabled by the FaceLandmarker blendshape option.
        /// <c>results[2]</c>, when present, corresponds to the facial transformation matrix output enabled by the face geometry option.
        /// </param>
        /// <param name="printResult">If true, prints the decoded result to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public void Visualize(Mat image, Mat[] results, bool printResult = false, bool isRGB = false)
        {
            ThrowIfDisposed();
            VisualizePackedFaceOutputs(image, results, printResult, isRGB);
        }

        /// <summary>
        /// Same drawing routine as <see cref="Visualize(Mat, Mat[], bool, bool)"/>, but without checking whether the worker has been disposed.
        /// Used for the face-related slots of <see cref="MediaPipeHolisticLandmarker"/>.
        /// </summary>
        internal static void VisualizePackedFaceOutputs(Mat image, Mat[] results, bool printResult, bool isRGB)
        {
            if (image != null)
                image.ThrowIfDisposed();
            if (results == null || results.Length == 0 || results[0] == null)
                return;

            Mat main = results[0];
            if (main.empty() || main.rows() <= 0)
                return;

            if (main.cols() < FaceLandmarkerEstimationData.ELEMENT_COUNT)
                throw new ArgumentException(
                    "The result Mat at index 0 does not have enough columns. It must have at least " + FaceLandmarkerEstimationData.ELEMENT_COUNT + " columns.",
                    nameof(results));

            if (!main.isContinuous())
                throw new ArgumentException("The result Mat at index 0 is not stored in a continuous buffer.", nameof(results));

            Span<FaceLandmarkerEstimationData> dataSpan = main.AsSpan<FaceLandmarkerEstimationData>();
            for (int f = 0; f < dataSpan.Length; f++)
            {
                ref readonly FaceLandmarkerEstimationData row = ref dataSpan[f];
                VisualizeFaceLandmarkerEstimationData(image, in row, faceIndex: f, printResult, isRGB);
            }

            if (results.Length > 1
                && TryGetBlendshapeSpanForVisualize(results[1], out Span<FaceBlendshapeEstimationData> blendSpan))
            {
                int nb = Math.Min(dataSpan.Length, blendSpan.Length);
                for (int f = 0; f < nb; f++)
                    VisualizeBlendshapeCoefficientsForFace(image, blendSpan[f].GetCoefficients(), f, printResult, isRGB);
            }

            if (results.Length > 2
                && TryGetFacialPoseSpanForVisualize(results[2], out Span<FacialTransformationMatrixEstimationData> poseSpan))
            {
                int np = Math.Min(dataSpan.Length, poseSpan.Length);
                for (int f = 0; f < np; f++)
                    VisualizeFacialTransformationMatrixForFace(image, poseSpan[f], f, printResult, isRGB);

                for (int f = 0; f < np; f++)
                {
                    ref readonly FaceLandmarkerEstimationData faceRow = ref dataSpan[f];
                    DrawFacialPoseAxes2DFromLandmarksAndPose(image, in faceRow, in poseSpan[f], isRGB);
                }
            }
        }

        /// <summary>
        /// Visualizes the packed face output returned by <see cref="Detect(Mat, bool)"/>.
        /// Each row is decoded as one <see cref="FaceLandmarkerEstimationData"/> value.
        /// </summary>
        /// <param name="image">Destination image for visualization.</param>
        /// <param name="results">
        /// Packed result matrix with one row per face.
        /// This matrix stores the public packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) face landmark output.
        /// </param>
        /// <param name="printResult">If true, prints the decoded result to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public override void Visualize(Mat image, Mat results, bool printResult = false, bool isRGB = false)
        {
            Visualize(image, results == null ? null : new[] { results }, printResult, isRGB);
        }

        internal static bool TryGetBlendshapeSpanForVisualize(Mat m, out Span<FaceBlendshapeEstimationData> span)
        {
            span = Span<FaceBlendshapeEstimationData>.Empty;
            if (m == null || m.empty() || m.rows() <= 0)
                return false;
            if (m.cols() < FaceBlendshapeEstimationData.COEFFICIENT_COUNT)
                return false;
            if (!m.isContinuous())
                return false;

            span = m.AsSpan<FaceBlendshapeEstimationData>();
            return true;
        }

        internal static bool TryGetFacialPoseSpanForVisualize(Mat m, out Span<FacialTransformationMatrixEstimationData> span)
        {
            span = Span<FacialTransformationMatrixEstimationData>.Empty;
            if (m == null || m.empty() || m.rows() <= 0)
                return false;
            if (m.cols() < FacialTransformationMatrixEstimationData.MATRIX_ELEMENT_COUNT)
                return false;
            if (!m.isContinuous())
                return false;

            span = m.AsSpan<FacialTransformationMatrixEstimationData>();
            return true;
        }

        /// <summary>
        /// Using <see cref="kVisualizeFacialPoseAxesOriginLandmarkIndex"/> as the origin, draws three line segments from the row-major 4x4 <paramref name="pose"/> matrix by
        /// taking the image-plane (x,y) components of each column of the upper-left 3x3 after removing uniform scale (X = red, Y = green, Z = blue, with BGR/RGB chosen by <paramref name="isRGB"/>).
        /// Axis length is proportional to <c>max(w,h)</c> of the 468-point face bounds, with a lower bound of <see cref="kVisualizeFacialPoseAxisLengthMinPx"/>.
        /// </summary>
        internal static void DrawFacialPoseAxes2DFromLandmarksAndPose(Mat image, in FaceLandmarkerEstimationData faceRow,
            in FacialTransformationMatrixEstimationData pose, bool isRGB)
        {
            if (image == null)
                return;
            image.ThrowIfDisposed();

            int w = image.cols();
            int h = image.rows();
            if (w <= 0 || h <= 0)
                return;

            Vec3f[] normLm = faceRow.GetNormLandmarksArray();
            if (normLm == null || normLm.Length != FaceLandmarkerEstimationData.LANDMARK_VEC3F_COUNT)
                return;

            int originIdx = kVisualizeFacialPoseAxesOriginLandmarkIndex;
            if ((uint)originIdx >= (uint)kFaceMeshLandmarksNum)
                return;

            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
            for (int i = 0; i < kFaceMeshLandmarksNum; i++)
            {
                ref readonly Vec3f n = ref normLm[i];
                float lx = n.Item1 * w;
                float ly = n.Item2 * h;
                if (lx < minX)
                    minX = lx;
                if (ly < minY)
                    minY = ly;
                if (lx > maxX)
                    maxX = lx;
                if (ly > maxY)
                    maxY = ly;
            }

            float faceW = maxX - minX;
            float faceH = maxY - minY;
            float axisLen = Mathf.Max(kVisualizeFacialPoseAxisLengthMinPx,
                Mathf.Max(faceW, faceH) * kVisualizeFacialPoseAxisLengthFractionOfFaceSize);

            ref readonly Vec3f originN = ref normLm[originIdx];
            double ox = originN.Item1 * w;
            double oy = originN.Item2 * h;

            ReadOnlySpan<float> rowMajor = pose.GetRowMajor4x4();

            float m00 = rowMajor[0], m01 = rowMajor[1], m02 = rowMajor[2];
            float m10 = rowMajor[4], m11 = rowMajor[5], m12 = rowMajor[6];
            float m20 = rowMajor[8], m21 = rowMajor[9], m22 = rowMajor[10];

            float n0 = Mathf.Sqrt(m00 * m00 + m10 * m10 + m20 * m20);
            float n1 = Mathf.Sqrt(m01 * m01 + m11 * m11 + m21 * m21);
            float n2 = Mathf.Sqrt(m02 * m02 + m12 * m12 + m22 * m22);
            float sc = (n0 + n1 + n2) / 3f;
            if (sc < 1e-8f)
                return;

            m00 /= sc;
            m01 /= sc;
            m02 /= sc;
            m10 /= sc;
            m11 /= sc;
            m12 /= sc;
            m20 /= sc;
            m21 /= sc;
            m22 /= sc;

            // X = red, Y = green, Z = blue (OpenCV default BGR, or RGB when isRGB is true).
            var colorX = isRGB ? (255d, 0d, 0d, 255d) : (0d, 0d, 255d, 255d);
            var colorY = (0d, 255d, 0d, 255d);
            var colorZ = isRGB ? (0d, 0d, 255d, 255d) : (255d, 0d, 0d, 255d);

            void DrawAxis(float dx, float dy, (double, double, double, double) color)
            {
                float len = Mathf.Sqrt(dx * dx + dy * dy);
                if (len < 1e-6f)
                    return;
                float nx = dx / len;
                // Flip the y direction so the direction vector looks natural in OpenCV drawing coordinates.
                float ny = -dy / len;
                double ex = ox + nx * axisLen;
                double ey = oy + ny * axisLen;
                Imgproc.line(image, (ox, oy), (ex, ey), color, 2, Imgproc.LINE_AA, 0);
            }

            // The image-plane components of column c are (m0c, m1c).
            DrawAxis(m00, m10, colorX);
            DrawAxis(m01, m11, colorY);
            DrawAxis(m02, m12, colorZ);
        }

        /// <summary>
        /// Shows coefficients for high-motion categories (mouth, jaw, blink, etc.) listed in <see cref="kVisualizeBlendshapeHighMotionIndices"/> using bars and labels.
        /// Stacks one block per face from the bottom of the image, aligns the bottom of each bar with the text baseline so they overlap as before,
        /// and aligns the left X of the label column to the widest string (including the title).
        /// </summary>
        /// <param name="printResult">When true, prints all coefficients for face <paramref name="faceIndex"/> to the console.</param>
        internal static void VisualizeBlendshapeCoefficientsForFace(Mat image, ReadOnlySpan<float> coeffs, int faceIndex, bool printResult,
            bool isRGB)
        {
            if (coeffs.Length < kFaceBlendshapeCoefficientCount)
                return;

            int w = image.cols();
            int h = image.rows();
            if (w <= 8 || h <= 12)
            {
                if (printResult)
                    Debug.Log("[MediaPipeFaceLandmarker] Blendshapes Face " + faceIndex + " (image too small, skipping overlay; coefficients below)\n" +
                              FormatBlendshapeCoefficientsForConsole(coeffs, faceIndex));
                return;
            }

            ReadOnlySpan<int> hi = kVisualizeBlendshapeHighMotionIndices;
            const int lineStep = 11;
            const int gapBetweenFaces = 14;
            const int margin = 4;
            const int barMaxW = 125;
            const int barHeightPx = 6;
            const double fontTitle = 0.35;
            const double fontLabel = 0.30;
            const int thickness = 1;

            int nRows = hi.Length;
            int lineCount = nRows + 1;
            int blockHeight = lineCount * lineStep + gapBetweenFaces;
            int baseY = h - 6 - faceIndex * blockHeight;
            if (baseY < lineCount * lineStep + 4)
                baseY = lineCount * lineStep + 4 + faceIndex * 2;

            var textColor = isRGB ? kVisualizeScalarBlue.ToValueTuple() : kVisualizeScalarRed.ToValueTuple();
            int[] baseLineArr = new int[1];

            string title = "BlendshapeCoefficients[" + faceIndex + "]";
            int maxTextPx = 0;
            for (int k = 0; k < nRows; k++)
            {
                int idx = hi[k];
                if ((uint)idx >= (uint)coeffs.Length)
                    continue;

                float bestV = coeffs[idx];
                string name = (uint)idx < (uint)kVisualizeBlendshapeCategoryNames.Length
                    ? kVisualizeBlendshapeCategoryNames[idx]
                    : ("c" + idx);
                if (name.Length > 14)
                    name = name.Substring(0, 14);

                string label = name + ":" + bestV.ToString("F2");
                var labelSize = Imgproc.getTextSizeAsValueTuple(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontLabel, thickness, baseLineArr);
                int iw = (int)System.Math.Ceiling(labelSize.width);
                if (iw > maxTextPx)
                    maxTextPx = iw;
            }

            var titleSize = Imgproc.getTextSizeAsValueTuple(title, Imgproc.FONT_HERSHEY_SIMPLEX, fontTitle, thickness, baseLineArr);
            int titleW = (int)System.Math.Ceiling(titleSize.width);
            if (titleW > maxTextPx)
                maxTextPx = titleW;

            int xCol = w - margin - maxTextPx;
            if (xCol < margin)
                xCol = margin;

            int titleY = baseY - (lineCount - 1) * lineStep;
            Imgproc.putText(image, title, (xCol, titleY), Imgproc.FONT_HERSHEY_SIMPLEX, fontTitle, textColor, thickness,
                Imgproc.LINE_AA, false);

            for (int k = 0; k < nRows; k++)
            {
                int idx = hi[k];
                if ((uint)idx >= (uint)coeffs.Length)
                    continue;

                float bestV = coeffs[idx];
                float barLen = bestV <= 1f && bestV >= 0f
                    ? bestV
                    : 1f / (1f + Mathf.Exp(-bestV));

                barLen = Mathf.Clamp01(barLen);
                int bw = (int)(barLen * barMaxW);

                string name = (uint)idx < (uint)kVisualizeBlendshapeCategoryNames.Length
                    ? kVisualizeBlendshapeCategoryNames[idx]
                    : ("c" + idx);
                if (name.Length > 14)
                    name = name.Substring(0, 14);

                string label = name + ":" + bestV.ToString("F2");
                int y = baseY - (nRows - 1 - k) * lineStep;

                Imgproc.rectangle(image, (xCol, y - barHeightPx), (xCol + bw, y), (0.0, 200.0, 0.0, 255.0), -1, Imgproc.LINE_AA, 0);

                Imgproc.putText(image, label, (xCol, y), Imgproc.FONT_HERSHEY_SIMPLEX, fontLabel, textColor, thickness,
                    Imgproc.LINE_AA, false);
            }

            if (!printResult)
                return;

            Debug.Log(FormatBlendshapeCoefficientsForConsole(coeffs, faceIndex));
        }

        /// <summary>
        /// Formats all blendshape coefficients into a single console block similar to the log style of <see cref="VisualizeFaceLandmarkerEstimationData"/>.
        /// </summary>
        static string FormatBlendshapeCoefficientsForConsole(ReadOnlySpan<float> coeffs, int faceIndex)
        {
            var sb = new StringBuilder(Math.Max(1536, coeffs.Length * 28));
            sb.Append("[MediaPipeFaceLandmarker] Blendshapes Face ").Append(faceIndex).AppendLine();
            sb.Append("Coefficients: ");
            for (int i = 0; i < coeffs.Length; i++)
            {
                string nm = (uint)i < (uint)kVisualizeBlendshapeCategoryNames.Length
                    ? kVisualizeBlendshapeCategoryNames[i]
                    : ("c" + i);
                sb.AppendFormat("{0}={1:F4} ", nm, coeffs[i]);
            }

            sb.AppendLine();
            sb.Append("Total ").Append(coeffs.Length).Append(" coefficients");
            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// Removes uniform scale from the rotation by averaging the column norms of the upper-left 3x3, then computes the rotation vector and Euler angles in degrees using
        /// OpenCV <c>Rodrigues</c> and <c>decomposeProjectionMatrix</c> on the 3x4 matrix <c>[R|t]</c>.
        /// </summary>
        static bool TryFacialPoseAnglesViaOpenCv(ReadOnlySpan<float> rowMajor16, out float rodX, out float rodY,
            out float rodZ, out float rodAngleDeg, out float euler0Deg, out float euler1Deg, out float euler2Deg)
        {
            rodX = rodY = rodZ = rodAngleDeg = euler0Deg = euler1Deg = euler2Deg = 0f;

            if (rowMajor16.Length < 16)
                return false;

            float m00 = rowMajor16[0], m01 = rowMajor16[1], m02 = rowMajor16[2], tx = rowMajor16[3];
            float m10 = rowMajor16[4], m11 = rowMajor16[5], m12 = rowMajor16[6], ty = rowMajor16[7];
            float m20 = rowMajor16[8], m21 = rowMajor16[9], m22 = rowMajor16[10], tz = rowMajor16[11];

            float n0 = Mathf.Sqrt(m00 * m00 + m10 * m10 + m20 * m20);
            float n1 = Mathf.Sqrt(m01 * m01 + m11 * m11 + m21 * m21);
            float n2 = Mathf.Sqrt(m02 * m02 + m12 * m12 + m22 * m22);
            float s = (n0 + n1 + n2) / 3f;
            if (s < 1e-8f)
                return false;

            m00 /= s;
            m01 /= s;
            m02 /= s;
            m10 /= s;
            m11 /= s;
            m12 /= s;
            m20 /= s;
            m21 /= s;
            m22 /= s;

            Span<float> rot9 = stackalloc float[9];
            rot9[0] = m00;
            rot9[1] = m01;
            rot9[2] = m02;
            rot9[3] = m10;
            rot9[4] = m11;
            rot9[5] = m12;
            rot9[6] = m20;
            rot9[7] = m21;
            rot9[8] = m22;

            Span<float> proj12 = stackalloc float[12];
            proj12[0] = m00;
            proj12[1] = m01;
            proj12[2] = m02;
            proj12[3] = tx;
            proj12[4] = m10;
            proj12[5] = m11;
            proj12[6] = m12;
            proj12[7] = ty;
            proj12[8] = m20;
            proj12[9] = m21;
            proj12[10] = m22;
            proj12[11] = tz;

            using (var rMat = new Mat(3, 3, CvType.CV_32FC1))
            using (var rvec = new Mat(3, 1, CvType.CV_32FC1))
            using (var projMat = new Mat(3, 4, CvType.CV_32FC1))
            using (var cameraMatrix = new Mat())
            using (var rotOut = new Mat())
            using (var transOut = new Mat())
            using (var rotMatrixX = new Mat())
            using (var rotMatrixY = new Mat())
            using (var rotMatrixZ = new Mat())
            using (var eulerAngles = new Mat())
            {
                rMat.put(0, 0, rot9);
                Calib3d.Rodrigues(rMat, rvec);
                Span<float> rv = stackalloc float[3];
                rvec.get(0, 0, rv);
                rodX = rv[0];
                rodY = rv[1];
                rodZ = rv[2];
                float thetaRad = Mathf.Sqrt(rodX * rodX + rodY * rodY + rodZ * rodZ);
                rodAngleDeg = thetaRad * Mathf.Rad2Deg;

                projMat.put(0, 0, proj12);
                Calib3d.decomposeProjectionMatrix(projMat, cameraMatrix, rotOut, transOut, rotMatrixX, rotMatrixY,
                    rotMatrixZ, eulerAngles);

                int eulerCh = eulerAngles.channels();
                int eulerDepth = eulerAngles.depth();
                if (eulerAngles.rows() >= 3 && eulerAngles.cols() >= 1)
                {
                    if (eulerDepth == CvType.CV_64F && eulerCh == 1)
                    {
                        Span<double> eu = stackalloc double[3];
                        eulerAngles.get(0, 0, eu);
                        euler0Deg = (float)eu[0];
                        euler1Deg = (float)eu[1];
                        euler2Deg = (float)eu[2];
                    }
                    else
                    {
                        Span<float> eu = stackalloc float[3];
                        eulerAngles.get(0, 0, eu);
                        euler0Deg = eu[0];
                        euler1Deg = eu[1];
                        euler2Deg = eu[2];
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Displays a per-face title plus the translation, four matrix rows, and OpenCV-derived rotation vector / Euler angles
        /// of the row-major 4x4 pose matrix near the lower-left corner of the image, stacking blocks by face index (same layout idea as <see cref="VisualizeBlendshapeCoefficientsForFace"/>).
        /// </summary>
        /// <param name="printResult">When true, prints a matrix summary for face <paramref name="faceIndex"/> to the console.</param>
        internal static void VisualizeFacialTransformationMatrixForFace(Mat image, FacialTransformationMatrixEstimationData data,
            int faceIndex, bool printResult, bool isRGB)
        {
            int w = image.cols();
            int h = image.rows();
            if (w <= 0 || h <= 0)
            {
                if (printResult)
                    Debug.Log(FormatFacialTransformationMatrixForConsole(in data, faceIndex));
                return;
            }

            var textColor = isRGB ? kVisualizeScalarBlue.ToValueTuple() : kVisualizeScalarRed.ToValueTuple();
            const int lineStep = 11;
            // One title line (same role as "BlendshapeCoefficients[i]" in VisualizeBlendshapeCoefficientsForFace) plus seven data lines.
            const int lineCount = 8;
            const int gapBetweenFaces = 14;
            const double fontTitle = 0.35;
            int blockHeight = lineCount * lineStep + gapBetweenFaces;
            int baseY = h - 6 - faceIndex * blockHeight;
            if (baseY < lineCount * lineStep + 4)
                baseY = lineCount * lineStep + 4 + faceIndex * 2;

            ReadOnlySpan<float> rowMajor = data.GetRowMajor4x4();

            bool haveAngles = TryFacialPoseAnglesViaOpenCv(rowMajor, out float rodX, out float rodY, out float rodZ,
                out float rodAngleDeg, out float euler0Deg, out float euler1Deg, out float euler2Deg);
            string rodLine = haveAngles
                ? string.Format(CultureInfo.InvariantCulture,
                    "rod {0:F2} {1:F2} {2:F2} |theta|={3:F1}deg", rodX, rodY, rodZ, rodAngleDeg)
                : "rod (scale~0 or OpenCV failed)";
            string eulerLine = haveAngles
                ? string.Format(CultureInfo.InvariantCulture,
                    "euler(proj,deg) {0:F1} {1:F1} {2:F1}", euler0Deg, euler1Deg, euler2Deg)
                : "euler(proj) —";

            string title = "FacialTransformationMatrix[" + faceIndex + "]";
            int titleY = baseY - (lineCount - 1) * lineStep;
            Imgproc.putText(image, title, (4, titleY), Imgproc.FONT_HERSHEY_SIMPLEX, fontTitle, textColor, 1,
                Imgproc.LINE_AA, false);

            Imgproc.putText(image, eulerLine, (4, baseY - 6 * lineStep), Imgproc.FONT_HERSHEY_SIMPLEX, 0.26, textColor, 1,
                Imgproc.LINE_AA, false);
            Imgproc.putText(image, rodLine, (4, baseY - 5 * lineStep), Imgproc.FONT_HERSHEY_SIMPLEX, 0.26, textColor, 1,
                Imgproc.LINE_AA, false);

            float tx = rowMajor[3], ty = rowMajor[7], tz = rowMajor[11];
            string transLine = string.Format(CultureInfo.InvariantCulture,
                "pose[{0}] t=({1:F2},{2:F2},{3:F2})", faceIndex, tx, ty, tz);
            Imgproc.putText(image, transLine, (4, baseY - 4 * lineStep), Imgproc.FONT_HERSHEY_SIMPLEX, 0.30, textColor, 1,
                Imgproc.LINE_AA, false);

            for (int r = 0; r < 4; r++)
            {
                int o = r * 4;
                string rowLine = string.Format(CultureInfo.InvariantCulture,
                    "m{0} {1:F2} {2:F2} {3:F2} {4:F2}",
                    r, rowMajor[o], rowMajor[o + 1], rowMajor[o + 2], rowMajor[o + 3]);
                int yLine = baseY - (3 - r) * lineStep;
                Imgproc.putText(image, rowLine, (4, yLine), Imgproc.FONT_HERSHEY_SIMPLEX, 0.28, textColor, 1,
                    Imgproc.LINE_AA, false);
            }

            if (!printResult)
                return;

            Debug.Log(FormatFacialTransformationMatrixForConsole(in data, faceIndex));
        }

        /// <summary>
        /// Formats the facial pose 4x4 row-major matrix as a single console block.
        /// </summary>
        static string FormatFacialTransformationMatrixForConsole(in FacialTransformationMatrixEstimationData data, int faceIndex)
        {
            var sb = new StringBuilder(512);
            sb.Append("[MediaPipeFaceLandmarker] FacialTransformationMatrix Face ").Append(faceIndex).AppendLine();
            sb.Append(data.ToString());
            return sb.ToString();
        }

        /// <summary>
        /// Draws one face worth of <see cref="FaceLandmarkerEstimationData"/>, called per face from <see cref="Visualize"/>.
        /// </summary>
        /// <param name="faceIndex">Face index used for labels.</param>
        /// <param name="printResult">When true, prints detailed output for face <paramref name="faceIndex"/> to the console, including all normalized landmarks.</param>
        internal static void VisualizeFaceLandmarkerEstimationData(Mat image, in FaceLandmarkerEstimationData data, int faceIndex,
            bool printResult, bool isRGB)
        {
            int w = image.cols();
            int h = image.rows();
            if (w <= 0 || h <= 0)
                return;

            Vec3f[] normLm = data.GetNormLandmarksArray();
            if (normLm == null || normLm.Length != FaceLandmarkerEstimationData.LANDMARK_VEC3F_COUNT)
                return;

            var pxLm = new Vec3f[normLm.Length];
            for (int i = 0; i < normLm.Length; i++)
            {
                ref readonly var n = ref normLm[i];
                pxLm[i] = new Vec3f(n.Item1 * w, n.Item2 * h, n.Item3);
            }

            float minX = float.MaxValue, minY = float.MaxValue;
            for (int i = 0; i < pxLm.Length; i++)
            {
                ref readonly var p = ref pxLm[i];
                if (p.Item1 < minX)
                    minX = p.Item1;
                if (p.Item2 < minY)
                    minY = p.Item2;
            }

            int left = (int)minX;
            int top = (int)Mathf.Max(0, minY - 30);

            var lineColor = kVisualizeScalarWhite.ToValueTuple();
            var pointColor = isRGB ? kVisualizeScalarBlue.ToValueTuple() : kVisualizeScalarRed.ToValueTuple();

            Imgproc.putText(image, "Face " + faceIndex, (left, top + 12), Imgproc.FONT_HERSHEY_DUPLEX, 0.5, pointColor);

            DrawFaceMeshContoursAndIris(image, pxLm, lineColor, pointColor, lineThickness: 1, pointRadius: 2);

            if (!printResult)
                return;

            var sb = new StringBuilder(Math.Max(16384, normLm.Length * 28));
            sb.Append("[MediaPipeFaceLandmarker] Face ").Append(faceIndex).AppendLine();
            sb.Append("NormLandmarks: ");
            for (int i = 0; i < normLm.Length; i++)
            {
                ref readonly var p = ref normLm[i];
                sb.AppendFormat("({0:F3},{1:F3},{2:F3}) ", p.Item1, p.Item2, p.Item3);
            }

            sb.AppendLine();
            sb.Append("Total ").Append(normLm.Length).Append(" points");
            sb.AppendLine();
            Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Draws contour and iris edges from the upstream MediaPipe Face Mesh topology, then draws a small circle at every landmark.
        /// </summary>
        static void DrawFaceMeshContoursAndIris(Mat image, Vec3f[] pixelLandmarks,
            (double, double, double, double) lineColor,
            (double, double, double, double) pointColor,
            int lineThickness,
            int pointRadius)
        {
            if (pixelLandmarks == null || pixelLandmarks.Length < kFaceMeshWithIrisLandmarksNum)
                return;

            ReadOnlySpan<(int from, int to)> edges = kFaceMeshContoursAndIrisConnections;
            for (int e = 0; e < edges.Length; e++)
            {
                int i = edges[e].from;
                int j = edges[e].to;
                if ((uint)i >= (uint)pixelLandmarks.Length || (uint)j >= (uint)pixelLandmarks.Length)
                    continue;

                ref readonly var a = ref pixelLandmarks[i];
                ref readonly var b = ref pixelLandmarks[j];
                Imgproc.line(image, (a.Item1, a.Item2), (b.Item1, b.Item2), lineColor, lineThickness, Imgproc.LINE_AA, 0);
            }

            for (int i = 0; i < pixelLandmarks.Length; i++)
            {
                ref readonly var p = ref pixelLandmarks[i];
                Imgproc.circle(image, (p.Item1, p.Item2), pointRadius, pointColor, -1, Imgproc.LINE_AA, 0);
            }
        }

        /// <summary>
        /// Converts a packed blendshape result matrix into a managed array of <see cref="FaceBlendshapeEstimationData"/>.
        /// </summary>
        /// <param name="blendshapeResult">
        /// Packed output matrix for face blendshapes.
        /// Each row corresponds to one face and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>face_blendshapes</c> output.
        /// </param>
        /// <returns>
        /// Managed array of face blendshape data.
        /// Returns an empty array when no blendshape rows are present.
        /// </returns>
        public virtual FaceBlendshapeEstimationData[] ToBlendshapeStructuredData(Mat blendshapeResult)
        {
            ThrowIfDisposed();
            if (blendshapeResult != null)
                blendshapeResult.ThrowIfDisposed();
            if (blendshapeResult == null || blendshapeResult.empty())
                return Array.Empty<FaceBlendshapeEstimationData>();

            int cols = FaceBlendshapeEstimationData.COEFFICIENT_COUNT;
            if (blendshapeResult.cols() < cols)
                throw new ArgumentException("The blendshape result Mat does not have enough columns.");
            if (!blendshapeResult.isContinuous())
                throw new ArgumentException("result is not continuous.");

            int n = blendshapeResult.rows();
            var dst = new FaceBlendshapeEstimationData[n];
            OpenCVMatUtils.CopyFromMat(blendshapeResult, dst);
            return dst;
        }

        /// <summary>
        /// Views a packed blendshape result matrix as a zero-allocation <see cref="Span{T}"/> of
        /// <see cref="FaceBlendshapeEstimationData"/>.
        /// </summary>
        /// <remarks>
        /// The returned span remains valid only while <paramref name="blendshapeResult"/> stays allocated
        /// and unchanged.
        /// </remarks>
        /// <param name="blendshapeResult">
        /// Packed output matrix for face blendshapes.
        /// Each row corresponds to one face and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>face_blendshapes</c> output.
        /// </param>
        /// <returns>
        /// Span whose elements correspond to faces in row order.
        /// Returns an empty span when the matrix is empty.
        /// </returns>
        public virtual Span<FaceBlendshapeEstimationData> ToBlendshapeStructuredDataAsSpan(Mat blendshapeResult)
        {
            ThrowIfDisposed();

            if (blendshapeResult != null)
                blendshapeResult.ThrowIfDisposed();
            if (blendshapeResult == null || blendshapeResult.empty())
                return Span<FaceBlendshapeEstimationData>.Empty;

            int cols = FaceBlendshapeEstimationData.COEFFICIENT_COUNT;
            if (blendshapeResult.cols() < cols)
                throw new ArgumentException("The blendshape result Mat does not have enough columns.");

            if (!blendshapeResult.isContinuous())
                throw new ArgumentException("result is not continuous.");

            return blendshapeResult.AsSpan<FaceBlendshapeEstimationData>();
        }

        /// <summary>
        /// Converts a packed facial transformation matrix result matrix into a managed array of <see cref="FacialTransformationMatrixEstimationData"/>.
        /// </summary>
        /// <param name="poseResult">
        /// Packed output matrix for facial transformation matrices.
        /// Each row corresponds to one face and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) facial transformation matrix output.
        /// </param>
        /// <returns>
        /// Managed array of facial transformation matrix data.
        /// Returns an empty array when no pose rows are present.
        /// </returns>
        public virtual FacialTransformationMatrixEstimationData[] ToFacialPoseStructuredData(Mat poseResult)
        {
            ThrowIfDisposed();
            if (poseResult != null)
                poseResult.ThrowIfDisposed();
            if (poseResult == null || poseResult.empty())
                return Array.Empty<FacialTransformationMatrixEstimationData>();

            const int cols = 16;
            if (poseResult.cols() < cols)
                throw new ArgumentException("The pose matrix result Mat does not have enough columns.");
            if (!poseResult.isContinuous())
                throw new ArgumentException("result is not continuous.");

            int n = poseResult.rows();
            var dst = new FacialTransformationMatrixEstimationData[n];
            OpenCVMatUtils.CopyFromMat(poseResult, dst);
            return dst;
        }

        /// <summary>
        /// Views a packed facial transformation matrix result matrix as a zero-allocation <see cref="Span{T}"/> of
        /// <see cref="FacialTransformationMatrixEstimationData"/>.
        /// </summary>
        /// <remarks>
        /// The returned span remains valid only while <paramref name="poseResult"/> stays allocated
        /// and unchanged.
        /// </remarks>
        /// <param name="poseResult">
        /// Packed output matrix for facial transformation matrices.
        /// Each row corresponds to one face and stores the packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) facial transformation matrix output.
        /// </param>
        /// <returns>
        /// Span whose elements correspond to faces in row order.
        /// Returns an empty span when the matrix is empty.
        /// </returns>
        public virtual Span<FacialTransformationMatrixEstimationData> ToFacialPoseStructuredDataAsSpan(Mat poseResult)
        {
            ThrowIfDisposed();

            if (poseResult != null)
                poseResult.ThrowIfDisposed();
            if (poseResult == null || poseResult.empty())
                return Span<FacialTransformationMatrixEstimationData>.Empty;

            int cols = FacialTransformationMatrixEstimationData.MATRIX_ELEMENT_COUNT;
            if (poseResult.cols() < cols)
                throw new ArgumentException("The pose matrix result Mat does not have enough columns.");

            if (!poseResult.isContinuous())
                throw new ArgumentException("result is not continuous.");

            return poseResult.AsSpan<FacialTransformationMatrixEstimationData>();
        }

        /// <inheritdoc cref="ProcessingWorkerBase"/>
        protected override Mat[] RunCoreProcessing(Mat[] inputs)
        {
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("MediaPipeFaceLandmarker accepts only a single input image at index 0.", nameof(inputs));

            Mat image = inputs[0];
            if (image != null) image.ThrowIfDisposed();
            if (image.channels() != 3)
                throw new ArgumentException("The input image must be a 3-channel BGR image.");

            List<FaceResult> faces = _runningMode == MediaPipeFaceRunningMode.IMAGE
                ? DetectPipeline(image)
                : DetectForVideoPipeline(image);

            return BuildPackedOutputMats(faces);
        }

        /// <inheritdoc cref="ProcessingWorkerBase"/>
        protected override async Task<Mat[]> RunCoreProcessingTaskAsync(Mat[] inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("MediaPipeFaceLandmarker accepts only a single input image at index 0.", nameof(inputs));
            var image = inputs[0];
            if (image != null) image.ThrowIfDisposed();
            if (image.channels() != 3)
                throw new ArgumentException("The input image must be a 3-channel BGR image.");

#if OPENCV_SENTIS_AVAILABLE
            if (_faceLandmarksNet.UsesSentis)
            {
                List<FaceResult> faces = _runningMode == MediaPipeFaceRunningMode.IMAGE
                    ? await ProcessImageDataAsync(image, cancellationToken)
                    : await ProcessVideoDataAsync(image, cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                return BuildPackedOutputMats(faces);
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
        /// Internal IMAGE-mode pipeline entry. Equivalent to <c>FaceLandmarker::Detect</c> and called from <see cref="RunCoreProcessing"/>.
        /// </summary>
        List<FaceResult> DetectPipeline(Mat image)
        {
            return ProcessImageData(image);
        }

        /// <summary>
        /// Internal VIDEO-mode pipeline entry. Equivalent to <c>FaceLandmarker::DetectForVideo</c>.
        /// </summary>
        List<FaceResult> DetectForVideoPipeline(Mat image)
        {
            return ProcessVideoData(image);
        }

        /// <summary>
        /// IMAGE-mode pipeline entry corresponding to Task API <c>ProcessImageData</c>, equivalent to <c>FaceLandmarkerGraph</c>.
        /// </summary>
        /// <remarks>
        /// Upstream correspondence in <c>face_landmarker_graph.cc</c>:
        /// - <c>FaceDetectorGraph</c> → <c>ClipNormalizedRectVectorSizeCalculator</c> → <c>MultiFaceLandmarksDetectorGraph</c>
        /// - Optional: <see cref="FaceGeometryFromLandmarksGraph"/> using image dimensions equivalent to <c>ImagePropertiesCalculator</c>
        /// - IMAGE mode does not include <c>PreviousLoopbackCalculator</c>, just like Pose.
        /// </remarks>
        List<FaceResult> ProcessImageData(Mat image)
        {
            // IMAGE graph has no PreviousLoopback, so keep VIDEO loopback state untouched, matching Pose's ProcessImageData behavior.
            _prevFaceRectsFromLandmarks.Clear();

            var det = FaceDetectorGraph(image, null);
            var clipped = ClipNormalizedRectVectorSizeCalculator(det.ExpandedFaceRects);
            List<FaceResult> faces = MultiFaceLandmarksDetectorGraph(image, clipped);

            faces.RemoveAll(f => !f.FacePresence);

            if (_outputFacialTransformationMatrixes && _faceGeometryCanonicalMetricLandmarks != null)
                FaceGeometryFromLandmarksGraph(image, faces);

            return faces;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="ProcessImageData"/> using the Sentis path with <see cref="MultiBackendNet.RunSentisForwardIntoListMatAsync"/> in the detector and landmark subgraphs.
        /// </summary>
        async Task<List<FaceResult>> ProcessImageDataAsync(Mat image, CancellationToken cancellationToken)
        {
            _prevFaceRectsFromLandmarks.Clear();

            var det = await FaceDetectorGraphAsync(image, null, cancellationToken);
            var clipped = ClipNormalizedRectVectorSizeCalculator(det.ExpandedFaceRects);
            List<FaceResult> faces = await MultiFaceLandmarksDetectorGraphAsync(image, clipped, cancellationToken);

            faces.RemoveAll(f => !f.FacePresence);

            if (_outputFacialTransformationMatrixes && _faceGeometryCanonicalMetricLandmarks != null)
                FaceGeometryFromLandmarksGraph(image, faces);

            return faces;
        }
#endif

        /// <summary>
        /// VIDEO-mode pipeline entry corresponding to <c>FaceLandmarker::DetectForVideo</c> and upstream <c>ProcessVideoData</c>.
        /// </summary>
        /// <remarks>
        /// Upstream correspondence in stream mode from <c>face_landmarker_graph.cc</c>:
        /// - <c>PreviousLoopbackCalculator</c> -> <c>NormalizedRectVectorHasMinSizeCalculator</c> -> (DisallowIf equivalent) <c>FaceDetectorGraph</c>
        ///   → <c>AssociationNormRectCalculator</c> → <c>ClipNormalizedRectVectorSizeCalculator</c> → <c>MultiFaceLandmarksDetectorGraph</c>
        /// - Optional: <see cref="FaceGeometryFromLandmarksGraph"/>
        /// - Writes output next-frame ROIs into <see cref="_prevFaceRectsFromLandmarks"/>, in the same order as Pose and Hand.
        /// </remarks>
        List<FaceResult> ProcessVideoData(Mat image)
        {
            // 1. PreviousLoopbackCalculator: get previous-frame FACE_RECTS_NEXT_FRAME as PREV_LOOP.
            var prevFaceRects = PreviousLoopbackCalculator(image, _prevFaceRectsFromLandmarks);

            // 2. NormalizedRectVectorHasMinSizeCalculator: detector execution can be skipped when the previous-frame rect count reaches num_faces.
            bool hasEnoughFaces = NormalizedRectVectorHasMinSizeCalculator(prevFaceRects, _numFaces);

            // 3. DisallowIf + FaceDetectorGraph: run FaceDetectorGraph only when tracking is insufficient; upstream uses an empty packet when skipped.
            List<NormalizedRect> expandedFaceRectsFromDetector = _processVideoExpandedDetectorScratch;
            expandedFaceRectsFromDetector.Clear();
            if (!hasEnoughFaces)
            {
                var det = FaceDetectorGraph(image, null);
                if (det.ExpandedFaceRects != null)
                    expandedFaceRectsFromDetector.AddRange(det.ExpandedFaceRects);
            }

            // 4. AssociationNormRectCalculator: input [0] is prev, input [1] is detector EXPANDED_FACE_RECTS, and min_similarity_threshold equals min_tracking_confidence.
            var associatedFaceRects = AssociationNormRectCalculator(prevFaceRects, expandedFaceRectsFromDetector);

            // 5. ClipNormalizedRectVectorSizeCalculator -> MultiFaceLandmarksDetectorGraph.
            var clipped = ClipNormalizedRectVectorSizeCalculator(associatedFaceRects);
            List<FaceResult> faces = MultiFaceLandmarksDetectorGraph(image, clipped);

            // 6. As in the upstream graph, absent faces are not included in the next-frame loopback, matching Pose and Hand ProcessVideoData.
            faces.RemoveAll(f => !f.FacePresence);

            if (_outputFacialTransformationMatrixes && _faceGeometryCanonicalMetricLandmarks != null)
                FaceGeometryFromLandmarksGraph(image, faces);

            // 7. Back edge into PreviousLoopbackCalculator: store FACE_RECTS_NEXT_FRAME for the next frame.
            _prevFaceRectsFromLandmarks.Clear();
            foreach (var f in faces)
                _prevFaceRectsFromLandmarks.Add(f.NextFrameRect);

            return faces;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="ProcessVideoData"/> using the Sentis path with <see cref="MultiBackendNet.RunSentisForwardIntoListMatAsync"/> in the detector and landmark subgraphs.
        /// </summary>
        async Task<List<FaceResult>> ProcessVideoDataAsync(Mat image, CancellationToken cancellationToken)
        {
            var prevFaceRects = PreviousLoopbackCalculator(image, _prevFaceRectsFromLandmarks);
            bool hasEnoughFaces = NormalizedRectVectorHasMinSizeCalculator(prevFaceRects, _numFaces);
            List<NormalizedRect> expandedFaceRectsFromDetector = _processVideoExpandedDetectorScratch;
            expandedFaceRectsFromDetector.Clear();
            if (!hasEnoughFaces)
            {
                var det = await FaceDetectorGraphAsync(image, null, cancellationToken);
                if (det.ExpandedFaceRects != null)
                    expandedFaceRectsFromDetector.AddRange(det.ExpandedFaceRects);
            }

            var associatedFaceRects = AssociationNormRectCalculator(prevFaceRects, expandedFaceRectsFromDetector);
            var clipped = ClipNormalizedRectVectorSizeCalculator(associatedFaceRects);
            List<FaceResult> faces = await MultiFaceLandmarksDetectorGraphAsync(image, clipped, cancellationToken);
            faces.RemoveAll(f => !f.FacePresence);

            if (_outputFacialTransformationMatrixes && _faceGeometryCanonicalMetricLandmarks != null)
                FaceGeometryFromLandmarksGraph(image, faces);

            _prevFaceRectsFromLandmarks.Clear();
            foreach (var f in faces)
                _prevFaceRectsFromLandmarks.Add(f.NextFrameRect);

            return faces;
        }
#endif

        /// <summary>
        /// Keeps detector-side <see cref="Mat"/> instances that persist across frames, such as transpose caches, and disposes the rest as transient headers.
        /// </summary>
        bool ShouldDisposeTransientFaceDetectorMat(Mat m)
        {
            if (m == null)
                return false;
            if (ReferenceEquals(m, _faceDetectorTransposeBuffer))
                return false;
            if (ReferenceEquals(m, _faceDetectorScoreColumnBuffer))
                return false;
            return true;
        }

        float[] RentFaceDetectorProjRow17()
        {
            return _poolFaceDetectorProjRow17.Count > 0
                ? _poolFaceDetectorProjRow17.Pop()
                : new float[FaceDetectorProjectedDetectionRowLength];
        }

        void ReleaseFaceDetectorProjRow17(float[] row)
        {
            if (row != null && row.Length == FaceDetectorProjectedDetectionRowLength)
                _poolFaceDetectorProjRow17.Push(row);
        }

        void ReleaseFaceDetectorProjRowList(IList<float[]> rows)
        {
            if (rows == null)
                return;
            for (int i = 0; i < rows.Count; i++)
                ReleaseFaceDetectorProjRow17(rows[i]);
        }

        float[] RentFaceDetectorNmsDec16()
        {
            return _poolFaceDetectorNmsDec16.Count > 0
                ? _poolFaceDetectorNmsDec16.Pop()
                : new float[kFaceDetectorTensorsToDetectionsNumCoords];
        }

        void ReleaseFaceDetectorNmsDec16(float[] row)
        {
            if (row != null && row.Length == kFaceDetectorTensorsToDetectionsNumCoords)
                _poolFaceDetectorNmsDec16.Push(row);
        }

        float[] RentFaceDetectorNmsBox4()
        {
            return _poolFaceDetectorNmsBox4.Count > 0 ? _poolFaceDetectorNmsBox4.Pop() : new float[4];
        }

        void ReleaseFaceDetectorNmsBox4(float[] row)
        {
            if (row != null && row.Length == 4)
                _poolFaceDetectorNmsBox4.Push(row);
        }

        void ReleaseFaceDetectorNmsMergedScratchLists()
        {
            for (int i = 0; i < _faceNmsMergedBoxScratch.Count; i++)
                ReleaseFaceDetectorNmsBox4(_faceNmsMergedBoxScratch[i]);
            for (int i = 0; i < _faceNmsMergedDecScratch.Count; i++)
                ReleaseFaceDetectorNmsDec16(_faceNmsMergedDecScratch[i]);
            _faceNmsMergedBoxScratch.Clear();
            _faceNmsMergedDecScratch.Clear();
            _faceNmsMergedScScratch.Clear();
        }

        /// <summary>
        /// Equivalent to <c>FaceDetectorGraph</c> from <c>mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc</c>.
        /// This method only invokes downstream calculators and subgraphs in upstream connection order.
        ///
        /// Correspondence to the upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>face_detector_graph.cc</c> implementation:
        /// - ImagePreprocessingGraph → <see cref="ImagePreprocessingGraph"/>
        /// - Inference subgraph (<c>AddInference</c>) -> <see cref="InferenceSubgraph_FaceDetection"/>
        /// - SsdAnchorsCalculator → <see cref="SsdAnchorsCalculator"/>
        /// - TensorsToDetectionsCalculator → <see cref="TensorsToDetectionsCalculator"/>
        /// - <c>min_score_thresh</c> (<c>min_detection_confidence</c>) -> <see cref="FaceDetectionsFilterByMinScoreThresh"/> (removed before NMS in the same stage as upstream <c>ConvertToDetection</c>; same role as Hand's <see cref="MediaPipeHandLandmarker.PalmDetectionsFilterByMinScoreThresh"/>)
        /// - NonMaxSuppressionCalculator → <see cref="NonMaxSuppressionCalculator"/>
        /// - DetectionProjectionCalculator → <see cref="DetectionProjectionCalculator"/>
        /// - <c>ClipDetectionVectorSizeCalculator</c> when <c>num_faces</c> is specified -> <see cref="ClipDetectionVectorSizeCalculator"/>
        /// - DetectionsToRectsCalculator → <see cref="DetectionsToRectsCalculator"/>
        /// - RectTransformationCalculator → <see cref="RectTransformationCalculator"/>
        /// - DetectionTransformationCalculator → <see cref="DetectionTransformationCalculator"/>
        /// </summary>
        FaceDetectorGraphResult FaceDetectorGraph(Mat image, NormalizedRect? normRect)
        {
            var empty = new FaceDetectorGraphResult
            {
                PixelDetections = new List<float[]>(),
                FaceRects = new List<NormalizedRect>(),
                ExpandedFaceRects = new List<NormalizedRect>(),
            };

            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;

            if (normRect.HasValue)
                throw new NotSupportedException(
                    "Non-null NORM_RECT is not implemented for FaceDetectorGraph yet (the upstream ImagePreprocessingGraph ROI path is planned to be wired later).");

            int imgW = image.cols();
            int imgH = image.rows();

            if (_faceDetectorLetterboxBgr == null
                || _faceDetectorLetterboxBgr.rows() != _faceDetectorTensorSize
                || _faceDetectorLetterboxBgr.cols() != _faceDetectorTensorSize)
            {
                _faceDetectorLetterboxBgr?.Dispose();
                _faceDetectorLetterboxBgr = new Mat(_faceDetectorTensorSize, _faceDetectorTensorSize, image.type());
            }

            Mat letter = _faceDetectorLetterboxBgr;
            List<Mat> outputBlobs = null;

            ImagePreprocessingGraph(image, letter, normRect, out _);
            outputBlobs = InferenceSubgraph_FaceDetection();
            if (outputBlobs == null || outputBlobs.Count < 2)
                return empty;

            Mat output0 = outputBlobs[1];
            Mat output1 = outputBlobs[0];
            if (output0 == null || output1 == null)
                return empty;

            Mat boxRows = FaceDetectorGraph_PrepareBoxMajorRows(output0);
            Mat scoreCol = FaceDetectorGraph_PrepareScoreColumn(output1);
            List<float[]> projectedRowsForPoolRelease = null;
            try
            {
                Mat anchors = SsdAnchorsCalculator();
                TensorsToDetectionsCalculator(boxRows, scoreCol, anchors);

                // BuildNmsBoxXywhFromDecoded returns the field _faceTensorsToDetectionsWorking.
                // Do not dispose it with a using block, or later frames would hit ObjectDisposedException.
                Mat nmsBoxXywh = FaceDetectorGraph_BuildNmsBoxXywhFromDecoded();
                FaceDetectionsFilterByMinScoreThresh(
                    nmsBoxXywh, scoreCol, _faceDetectorDecodedBoxesNx16, _minFaceDetectionConfidence,
                    out Mat nmsBoxFiltered, out Mat scoreFiltered, out Mat decodedFiltered);
                MatOfInt indices = NonMaxSuppressionCalculator(nmsBoxFiltered, scoreFiltered, decodedFiltered);
                var projectedRows = DetectionProjectionCalculator(
                    _faceWnmsMergedBoxXywh, _faceWnmsMergedScore, _faceWnmsMergedDecodedNx16, indices);
                projectedRows = ClipDetectionVectorSizeCalculator(projectedRows, _numFaces);
                projectedRowsForPoolRelease = projectedRows;
                List<NormalizedRect> faceRects = DetectionsToRectsCalculator(projectedRows, imgW, imgH);
                List<NormalizedRect> expanded = RectTransformationCalculator(faceRects, imgW, imgH);
                List<float[]> pixelDets = DetectionTransformationCalculator(projectedRows, imgW, imgH);

                return new FaceDetectorGraphResult
                {
                    PixelDetections = pixelDets,
                    FaceRects = faceRects,
                    ExpandedFaceRects = expanded,
                };
            }
            finally
            {
                ReleaseFaceDetectorProjRowList(projectedRowsForPoolRelease);
                if (ShouldDisposeTransientFaceDetectorMat(boxRows))
                    boxRows.Dispose();
                if (ShouldDisposeTransientFaceDetectorMat(scoreCol))
                    scoreCol.Dispose();
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="FaceDetectorGraph"/> using the Sentis path with <see cref="InferenceSubgraph_FaceDetectionAsync"/>.
        /// </summary>
        async Task<FaceDetectorGraphResult> FaceDetectorGraphAsync(Mat image, NormalizedRect? normRect, CancellationToken cancellationToken)
        {
            var empty = new FaceDetectorGraphResult
            {
                PixelDetections = new List<float[]>(),
                FaceRects = new List<NormalizedRect>(),
                ExpandedFaceRects = new List<NormalizedRect>(),
            };

            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;

            if (normRect.HasValue)
                throw new NotSupportedException(
                    "Non-null NORM_RECT is not implemented for FaceDetectorGraph yet (the upstream ImagePreprocessingGraph ROI path is planned to be wired later).");

            int imgW = image.cols();
            int imgH = image.rows();

            if (_faceDetectorLetterboxBgr == null
                || _faceDetectorLetterboxBgr.rows() != _faceDetectorTensorSize
                || _faceDetectorLetterboxBgr.cols() != _faceDetectorTensorSize)
            {
                _faceDetectorLetterboxBgr?.Dispose();
                _faceDetectorLetterboxBgr = new Mat(_faceDetectorTensorSize, _faceDetectorTensorSize, image.type());
            }

            Mat letter = _faceDetectorLetterboxBgr;
            List<Mat> outputBlobs = null;

            ImagePreprocessingGraph(image, letter, normRect, out _);
            outputBlobs = await InferenceSubgraph_FaceDetectionAsync(cancellationToken);
            if (outputBlobs == null || outputBlobs.Count < 2)
                return empty;

            Mat output0 = outputBlobs[1];
            Mat output1 = outputBlobs[0];
            if (output0 == null || output1 == null)
                return empty;

            Mat boxRows = FaceDetectorGraph_PrepareBoxMajorRows(output0);
            Mat scoreCol = FaceDetectorGraph_PrepareScoreColumn(output1);
            List<float[]> projectedRowsForPoolRelease = null;
            try
            {
                Mat anchors = SsdAnchorsCalculator();
                TensorsToDetectionsCalculator(boxRows, scoreCol, anchors);

                Mat nmsBoxXywh = FaceDetectorGraph_BuildNmsBoxXywhFromDecoded();
                FaceDetectionsFilterByMinScoreThresh(
                    nmsBoxXywh, scoreCol, _faceDetectorDecodedBoxesNx16, _minFaceDetectionConfidence,
                    out Mat nmsBoxFiltered, out Mat scoreFiltered, out Mat decodedFiltered);
                MatOfInt indices = NonMaxSuppressionCalculator(nmsBoxFiltered, scoreFiltered, decodedFiltered);
                var projectedRows = DetectionProjectionCalculator(
                    _faceWnmsMergedBoxXywh, _faceWnmsMergedScore, _faceWnmsMergedDecodedNx16, indices);
                projectedRows = ClipDetectionVectorSizeCalculator(projectedRows, _numFaces);
                projectedRowsForPoolRelease = projectedRows;
                List<NormalizedRect> faceRects = DetectionsToRectsCalculator(projectedRows, imgW, imgH);
                List<NormalizedRect> expanded = RectTransformationCalculator(faceRects, imgW, imgH);
                List<float[]> pixelDets = DetectionTransformationCalculator(projectedRows, imgW, imgH);

                return new FaceDetectorGraphResult
                {
                    PixelDetections = pixelDets,
                    FaceRects = faceRects,
                    ExpandedFaceRects = expanded,
                };
            }
            finally
            {
                ReleaseFaceDetectorProjRowList(projectedRowsForPoolRelease);
                if (ShouldDisposeTransientFaceDetectorMat(boxRows))
                    boxRows.Dispose();
                if (ShouldDisposeTransientFaceDetectorMat(scoreCol))
                    scoreCol.Dispose();
            }
        }

#endif

        /// <summary>
        /// Equivalent to the Tasks <c>ImagePreprocessingGraph</c> and <c>ImageToTensorCalculator</c>.
        /// Writes a letterboxed BGR image into <paramref name="letterboxTensorBgr"/> using <c>keep_aspect_ratio=true</c> and <c>BORDER_ZERO</c>, then outputs normalized padding values.
        /// The inference tensor is packed later inside <see cref="InferenceSubgraph_FaceDetection"/> from <see cref="_faceDetectorLetterboxBgr"/>, matching the Hand palm-detection path.
        /// Stores the <c>GetRotatedSubRectToRectTransformMatrix</c> result from <c>image_to_tensor_utils.cc</c> into <see cref="_faceDetectorProjectionMatrix16"/>.
        /// </summary>
        void ImagePreprocessingGraph(Mat image, Mat letterboxTensorBgr, NormalizedRect? normRect, out float[] letterboxPaddingNorm)
        {
            ImagePreprocessingGraph_FillLetterbox(image, letterboxTensorBgr, normRect, _faceDetectorTensorSize,
                out letterboxPaddingNorm);
        }

        /// <summary>
        /// Generates only the letterboxed BGR image for <see cref="_faceDetectorTensorSize"/>. No blob is created here.
        /// </summary>
        /// <remarks>
        /// In the full-frame letterbox path where <paramref name="normRect"/> is absent, the resized integer size is truncated with
        /// <c>(int)(width * ratio)</c> rather than <c>Mathf.RoundToInt</c>, matching the full-frame palm-detection path in <see cref="MediaPipeHandLandmarker"/>.
        /// </remarks>
        void ImagePreprocessingGraph_FillLetterbox(
            Mat image,
            Mat letterboxTensorBgr,
            NormalizedRect? normRect,
            int tensorSize,
            out float[] letterboxPaddingNorm)
        {
            int imageW = image.cols();
            int imageH = image.rows();

            FaceDetectorGetRoi(imageW, imageH, normRect, out float roiCx, out float roiCy, out float roiW, out float roiH,
                out float roiRot);
            FaceDetectorPadRoi(tensorSize, tensorSize, true, ref roiW, ref roiH);
            GetRotatedSubRectToRectTransformMatrix(roiCx, roiCy, roiW, roiH, roiRot, imageW, imageH, false,
                _faceDetectorProjectionMatrix16);

            if (normRect.HasValue)
            {
                FaceDetectorEnsureWarpMats(tensorSize);
                double angleDeg = roiRot * (180.0 / Math.PI);
                Imgproc.boxPoints((roiCx, roiCy, roiW, roiH, angleDeg), _faceDetectorWarpSrcPts);
                using (Mat projMat = Imgproc.getPerspectiveTransform(_faceDetectorWarpSrcPts, _faceDetectorWarpDstPts))
                {
                    Imgproc.warpPerspective(image, letterboxTensorBgr, projMat, (tensorSize, tensorSize),
                        Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
                }

                letterboxPaddingNorm = _faceDetectorLetterboxPaddingNormReuse;
                letterboxPaddingNorm[0] = 0f;
                letterboxPaddingNorm[1] = 0f;
                letterboxPaddingNorm[2] = 0f;
                letterboxPaddingNorm[3] = 0f;
            }
            else
            {
                double ratio = Math.Min((double)tensorSize / imageW, (double)tensorSize / imageH);
                int newW = Math.Max(1, (int)(imageW * ratio));
                int newH = Math.Max(1, (int)(imageH * ratio));

                int padX = (tensorSize - newW) / 2;
                int padY = (tensorSize - newH) / 2;

                letterboxTensorBgr.setTo((0d, 0d, 0d, 0d));
                if (_faceLetterboxResizeScratch == null)
                    _faceLetterboxResizeScratch = new Mat();
                Imgproc.resize(image, _faceLetterboxResizeScratch, (newW, newH));
                using (Mat roi = letterboxTensorBgr.submat(padY, padY + newH, padX, padX + newW))
                {
                    _faceLetterboxResizeScratch.copyTo(roi);
                }

                letterboxPaddingNorm = _faceDetectorLetterboxPaddingNormReuse;
                letterboxPaddingNorm[0] = padX / (float)tensorSize;
                letterboxPaddingNorm[1] = padY / (float)tensorSize;
                letterboxPaddingNorm[2] = (tensorSize - padX - newW) / (float)tensorSize;
                letterboxPaddingNorm[3] = (tensorSize - padY - newH) / (float)tensorSize;
            }
        }

        void FaceDetectorEnsureWarpMats(int tensorSize)
        {
            if (_faceDetectorWarpDstPts != null)
                return;

            float dw = tensorSize;
            float dh = tensorSize;
            _faceDetectorWarpDstPts = new Mat(4, 2, CvType.CV_32FC1);
            Span<float> dstPtsArr = stackalloc float[8];
            dstPtsArr[0] = 0f;
            dstPtsArr[1] = dh;
            dstPtsArr[2] = 0f;
            dstPtsArr[3] = 0f;
            dstPtsArr[4] = dw;
            dstPtsArr[5] = 0f;
            dstPtsArr[6] = dw;
            dstPtsArr[7] = dh;
            _faceDetectorWarpDstPts.put(0, 0, dstPtsArr);
            _faceDetectorWarpSrcPts = new Mat(4, 2, CvType.CV_32FC1);
        }

        /// <summary>Equivalent to <c>GetRoi</c> in <c>image_to_tensor_utils.cc</c>.</summary>
        static void FaceDetectorGetRoi(int inputWidth, int inputHeight, NormalizedRect? normRect, out float centerX,
            out float centerY, out float width, out float height, out float rotation)
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

        /// <summary>Equivalent to <c>PadRoi</c> in <c>image_to_tensor_utils.cc</c>.</summary>
        static void FaceDetectorPadRoi(int inputTensorWidth, int inputTensorHeight, bool keepAspectRatio, ref float roiWidth,
            ref float roiHeight)
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
        /// Same formula as <c>GetRotatedSubRectToRectTransformMatrix</c> in <c>image_to_tensor_utils.cc</c>, expressed as a row-major 4x4 matrix.
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
        /// Equivalent to the upstream inference subgraph (<c>AddInference</c> / <c>mediapipe.tasks.core.InferenceSubgraph</c>).
        /// Converts <see cref="_faceDetectorLetterboxBgr"/> into a reusable NHWC RGB blob with size <see cref="_faceDetectorTensorSize"/> and value range [-1,1],
        /// feeds it into <see cref="_faceDetectorNet"/>, and returns the resulting TENSORS.
        /// Allocation and normalization of the inference <see cref="Mat"/> happen inside this method, just as in <see cref="MediaPipeHandLandmarker.InferenceSubgraph_PalmDetection"/>.
        /// </summary>
        List<Mat> InferenceSubgraph_FaceDetection()
        {
            Mat letterboxBgr = _faceDetectorLetterboxBgr;
            int detH = _faceDetectorTensorSize;
            int detW = _faceDetectorTensorSize;
            const int detC = 3;
            const float imageToTensorDivisor = 127.5f;

            if (detH > 0 && detW > 0)
            {
                if (_faceDetectorInferenceBlob == null
                    || _faceDetectorInferenceRgb8u == null
                    || _faceDetectorInferenceRgb8u.rows() != detH
                    || _faceDetectorInferenceRgb8u.cols() != detW)
                {
                    _faceDetectorInferenceRgb8u?.Dispose();
                    _faceDetectorInferenceBlob?.Dispose();
                    _faceDetectorInferenceRgb8u = null;
                    _faceDetectorInferenceBlob = null;
                    _faceDetectorInferenceBlobHxW = null;

                    _faceDetectorInferenceRgb8u = new Mat(detH, detW, CvType.CV_8UC3);
                    _faceDetectorInferenceBlob = new Mat(new int[] { 1, detH, detW, detC }, CvType.CV_32FC1);
                    _faceDetectorInferenceBlobHxW =
                        _faceDetectorInferenceBlob.reshape(detC, new int[] { detH, detW });
                }

                if (letterboxBgr != null && !letterboxBgr.empty())
                {
                    Imgproc.cvtColor(letterboxBgr, _faceDetectorInferenceRgb8u, Imgproc.COLOR_BGR2RGB);
                    _faceDetectorInferenceRgb8u.convertTo(_faceDetectorInferenceBlobHxW, CvType.CV_32F,
                        1.0 / imageToTensorDivisor, -1.0);
                }
            }

            if (_faceDetectorOutLayerNames == null || _faceDetectorOutLayerNames.Count == 0)
                _faceDetectorOutLayerNames = _faceDetectorNet.getUnconnectedOutLayersNames();
            _faceDetectorForwardOutputList.Clear();
            _faceDetectorNet.setInput(_faceDetectorInferenceBlob);
            _faceDetectorNet.forward(_faceDetectorForwardOutputList, _faceDetectorOutLayerNames);
            return _faceDetectorForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="InferenceSubgraph_FaceDetection"/> (via <see cref="MultiBackendNet.forwardTaskAsync"/>). Invoked only from <see cref="RunCoreProcessingTaskAsync"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_FaceDetectionAsync(CancellationToken cancellationToken)
        {
            Mat letterboxBgr = _faceDetectorLetterboxBgr;
            int detH = _faceDetectorTensorSize;
            int detW = _faceDetectorTensorSize;
            const int detC = 3;
            const float imageToTensorDivisor = 127.5f;

            if (detH > 0 && detW > 0)
            {
                if (_faceDetectorInferenceBlob == null
                    || _faceDetectorInferenceRgb8u == null
                    || _faceDetectorInferenceRgb8u.rows() != detH
                    || _faceDetectorInferenceRgb8u.cols() != detW)
                {
                    _faceDetectorInferenceRgb8u?.Dispose();
                    _faceDetectorInferenceBlob?.Dispose();
                    _faceDetectorInferenceRgb8u = null;
                    _faceDetectorInferenceBlob = null;
                    _faceDetectorInferenceBlobHxW = null;

                    _faceDetectorInferenceRgb8u = new Mat(detH, detW, CvType.CV_8UC3);
                    _faceDetectorInferenceBlob = new Mat(new int[] { 1, detH, detW, detC }, CvType.CV_32FC1);
                    _faceDetectorInferenceBlobHxW =
                        _faceDetectorInferenceBlob.reshape(detC, new int[] { detH, detW });
                }

                if (letterboxBgr != null && !letterboxBgr.empty())
                {
                    Imgproc.cvtColor(letterboxBgr, _faceDetectorInferenceRgb8u, Imgproc.COLOR_BGR2RGB);
                    _faceDetectorInferenceRgb8u.convertTo(_faceDetectorInferenceBlobHxW, CvType.CV_32F,
                        1.0 / imageToTensorDivisor, -1.0);
                }
            }

            if (_faceDetectorOutLayerNames == null || _faceDetectorOutLayerNames.Count == 0)
                _faceDetectorOutLayerNames = _faceDetectorNet.getUnconnectedOutLayersNames();
            _faceDetectorForwardOutputList.Clear();
            _faceDetectorNet.setInput(_faceDetectorInferenceBlob);
            await _faceDetectorNet.forwardTaskAsync(_faceDetectorForwardOutputList, _faceDetectorOutLayerNames, cancellationToken);
            return _faceDetectorForwardOutputList;
        }
#endif

        /// <summary>
        /// Equivalent to <c>SsdAnchorsCalculator</c>.
        /// Follows <c>ConfigureSsdAnchorsCalculator</c> in <c>face_detector_graph.cc</c> (including the legacy 128/192 branches when metadata is absent)
        /// and <c>GenerateAnchors</c> in <c>ssd_anchors_calculator.cc</c> with <c>multiscale_anchor_generation</c> disabled and <c>fixed_anchor_size=true</c>.
        /// Returns an anchor matrix whose rows are <c>(x_center, y_center, w, h)</c>, with <c>w</c> and <c>h</c> equal to 1.
        /// </summary>
        Mat SsdAnchorsCalculator()
        {
            if (_faceDetectorIsLongRange)
            {
                _faceDetectorSsdAnchors192Cache ??= BuildFaceDetectorSsdAnchorsMat(
                    kFaceDetectorSsdLegacy192NumLayers,
                    kFaceDetectorSsdLegacyMinScale,
                    kFaceDetectorSsdLegacyMaxScale,
                    kFaceDetectorLongRangeImageSize,
                    kFaceDetectorLongRangeImageSize,
                    kFaceDetectorSsdLegacyAnchorOffset,
                    kFaceDetectorSsdLegacyAnchorOffset,
                    new[] { kFaceDetectorSsdLegacy192Stride },
                    kFaceDetectorSsdLegacyAspectRatio,
                    kFaceDetectorSsdLegacyFixedAnchorSize,
                    kFaceDetectorSsdLegacy192InterpolatedScaleAspectRatio,
                    kFaceDetectorLegacyLongRangeNumBoxes);
                return _faceDetectorSsdAnchors192Cache;
            }

            _faceDetectorSsdAnchors128Cache ??= BuildFaceDetectorSsdAnchorsMat(
                kFaceDetectorSsdLegacy128NumLayers,
                kFaceDetectorSsdLegacyMinScale,
                kFaceDetectorSsdLegacyMaxScale,
                kFaceDetectorShortRangeImageSize,
                kFaceDetectorShortRangeImageSize,
                kFaceDetectorSsdLegacyAnchorOffset,
                kFaceDetectorSsdLegacyAnchorOffset,
                kFaceDetectorSsdLegacy128Strides,
                kFaceDetectorSsdLegacyAspectRatio,
                kFaceDetectorSsdLegacyFixedAnchorSize,
                kFaceDetectorSsdLegacy128InterpolatedScaleAspectRatio,
                kFaceDetectorLegacyShortRangeNumBoxes);
            return _faceDetectorSsdAnchors128Cache;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToDetectionsCalculator</c> on the CPU path in <c>tensors_to_detections_calculator.cc</c>.
        /// Decodes <c>XYWH</c> using <c>reverse_output_order</c>, and applies <c>sigmoid_score</c> and <c>score_clipping_thresh</c>.
        /// As in the upstream graph, <c>min_score_thresh</c> is applied immediately before NMS by <see cref="FaceDetectionsFilterByMinScoreThresh"/>, matching the Hand palm-detection flow.
        /// Decoded rows are stored in <see cref="_faceDetectorDecodedBoxesNx16"/>.
        /// </summary>
        void TensorsToDetectionsCalculator(Mat boxRows, Mat scoreCol, Mat anchorsXywh)
        {
            int num = _faceDetectorNumBoxes;
            int numCoords = kFaceDetectorTensorsToDetectionsNumCoords;
            float xScale = _faceDetectorIsLongRange ? kFaceDetectorLongRangeImageSize : kFaceDetectorShortRangeImageSize;
            float yScale = xScale;
            float wScale = xScale;
            float hScale = xScale;

            if (_faceDetectorDecodedBoxesNx16 == null
                || _faceDetectorDecodedBoxesNx16.rows() != num
                || _faceDetectorDecodedBoxesNx16.cols() != numCoords)
            {
                _faceDetectorDecodedBoxesNx16?.Dispose();
                _faceDetectorDecodedBoxesNx16 = new Mat(num, numCoords, CvType.CV_32FC1);
            }

            NumpyClip(scoreCol, -kFaceDetectorTensorsToDetectionsScoreClippingThresh,
                kFaceDetectorTensorsToDetectionsScoreClippingThresh);
            Core.multiply(scoreCol, (-1.0, 0, 0, 0), scoreCol);
            Core.exp(scoreCol, scoreCol);
            Core.add(scoreCol, (1.0, 0, 0, 0), scoreCol);
            Core.divide(1.0, scoreCol, scoreCol);

            if (_faceDetectorDecodeRowSrc == null || _faceDetectorDecodeRowSrc.Length < numCoords)
                _faceDetectorDecodeRowSrc = new float[numCoords];
            if (_faceDetectorDecodeRowDst == null || _faceDetectorDecodeRowDst.Length < numCoords)
                _faceDetectorDecodeRowDst = new float[numCoords];
            if (_faceDetectorAnchorRow4 == null || _faceDetectorAnchorRow4.Length < 4)
                _faceDetectorAnchorRow4 = new float[4];

            float[] rowRaw = _faceDetectorDecodeRowSrc;
            float[] rowDecoded = _faceDetectorDecodeRowDst;
            float[] ar = _faceDetectorAnchorRow4;

            for (int i = 0; i < num; i++)
            {
                boxRows.get(i, 0, rowRaw.AsSpan(0, numCoords));
                anchorsXywh.get(i, 0, ar.AsSpan(0, 4));
                float ax = ar[0];
                float ay = ar[1];
                float aw = ar[2];
                float ah = ar[3];

                int boxOff = kFaceDetectorTensorsToDetectionsBoxCoordOffset;
                float xCenterRaw = rowRaw[boxOff + 0];
                float yCenterRaw = rowRaw[boxOff + 1];
                float wRaw = rowRaw[boxOff + 2];
                float hRaw = rowRaw[boxOff + 3];

                float xCenter = xCenterRaw / xScale * aw + ax;
                float yCenter = yCenterRaw / yScale * ah + ay;
                float boxH = hRaw / hScale * ah;
                float boxW = wRaw / wScale * aw;

                float ymin = yCenter - boxH * 0.5f;
                float xmin = xCenter - boxW * 0.5f;
                float ymax = yCenter + boxH * 0.5f;
                float xmax = xCenter + boxW * 0.5f;

                rowDecoded[0] = ymin;
                rowDecoded[1] = xmin;
                rowDecoded[2] = ymax;
                rowDecoded[3] = xmax;

                int kpOff = kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                for (int k = 0; k < kFaceDetectorTensorsToDetectionsNumKeypoints; k++)
                {
                    int o = kpOff + k * kFaceDetectorTensorsToDetectionsNumValuesPerKeypoint;
                    float kxRaw = rowRaw[o];
                    float kyRaw = rowRaw[o + 1];
                    rowDecoded[o] = kxRaw / xScale * aw + ax;
                    rowDecoded[o + 1] = kyRaw / yScale * ah + ay;
                }

                _faceDetectorDecodedBoxesNx16.put(i, 0, rowDecoded.AsSpan(0, numCoords));
            }
        }

        /// <summary>
        /// Equivalent to upstream <c>TensorsToDetectionsCalculator.ConvertToDetection.min_score_thresh</c>, which corresponds to Tasks <c>min_detection_confidence</c>.
        /// Returns NMS input matrices with rows below the threshold removed. If the threshold is 0 or lower, or if no rows are removed, the original buffers are returned directly.
        /// </summary>
        void FaceDetectionsFilterByMinScoreThresh(
            Mat boxXywh,
            Mat scoreNx1,
            Mat decodedNx16,
            float minScoreThresh,
            out Mat boxOut,
            out Mat scoreOut,
            out Mat decodedOut)
        {
            int num = boxXywh.rows();
            int nCoord = kFaceDetectorTensorsToDetectionsNumCoords;
            if (num <= 0 || minScoreThresh <= 0f)
            {
                boxOut = boxXywh;
                scoreOut = scoreNx1;
                decodedOut = decodedNx16;
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
                decodedOut = decodedNx16;
                return;
            }

            if (_faceScoreFilteredBoxXywh == null)
                _faceScoreFilteredBoxXywh = new Mat();
            if (_faceScoreFilteredScore == null)
                _faceScoreFilteredScore = new Mat();
            if (_faceScoreFilteredDecodedNx16 == null)
                _faceScoreFilteredDecodedNx16 = new Mat();

            _faceScoreFilteredBoxXywh.create(kept, 4, CvType.CV_32FC1);
            _faceScoreFilteredScore.create(kept, 1, CvType.CV_32FC1);
            _faceScoreFilteredDecodedNx16.create(kept, nCoord, CvType.CV_32FC1);

            int r = 0;
            for (int i = 0; i < num; i++)
            {
                if (scoreNx1.at<float>(i, 0)[0] < minScoreThresh)
                    continue;
                using (Mat srcRow = boxXywh.row(i))
                using (Mat dstRow = _faceScoreFilteredBoxXywh.row(r))
                    srcRow.copyTo(dstRow);
                using (Mat srcRow = scoreNx1.row(i))
                using (Mat dstRow = _faceScoreFilteredScore.row(r))
                    srcRow.copyTo(dstRow);
                using (Mat srcRow = decodedNx16.row(i))
                using (Mat dstRow = _faceScoreFilteredDecodedNx16.row(r))
                    srcRow.copyTo(dstRow);
                r++;
            }

            boxOut = _faceScoreFilteredBoxXywh;
            scoreOut = _faceScoreFilteredScore;
            decodedOut = _faceScoreFilteredDecodedNx16;
        }

        /// <summary>
        /// Equivalent to <c>NonMaxSuppressionCalculator</c>, specifically upstream <c>WeightedNonMaxSuppression</c> in <c>non_max_suppression_calculator.cc</c>.
        /// </summary>
        /// <remarks>
        /// Matches <c>ConfigureNonMaxSuppressionCalculator</c> in <c>face_detector_graph.cc</c> with
        /// <c>overlap_type=INTERSECTION_OVER_UNION</c>, <c>algorithm=WEIGHTED</c>, and a constructor-defined <c>min_suppression_threshold</c>.
        /// The upstream NMS option <c>min_score_threshold</c> is disabled by default, so this method assumes the score threshold has already been applied upstream by
        /// <see cref="FaceDetectionsFilterByMinScoreThresh"/>, which corresponds to <c>TensorsToDetectionsCalculatorOptions.min_score_thresh</c>, just like Hand's <see cref="MediaPipeHandLandmarker.NonMaxSuppressionCalculator"/>.
        /// Keypoints are aggregated by score-weighted averaging over <paramref name="decodedBoxesNx16"/>.
        /// The merged tensors are stored in <see cref="_faceWnmsMergedBoxXywh"/>, <see cref="_faceWnmsMergedDecodedNx16"/>, and <see cref="_faceWnmsMergedScore"/>,
        /// and <see cref="_faceNmsIndices"/> contains <c>0 .. K-1</c>.
        /// </remarks>
        MatOfInt NonMaxSuppressionCalculator(Mat boxXywhTensorNorm, Mat scoreCol, Mat decodedBoxesNx16)
        {
            const float kFaceMinSuppressionThreshold = 0.5f;

            if (_faceNmsIndices == null)
                _faceNmsIndices = new MatOfInt();
            if (_faceWnmsMergedBoxXywh == null)
                _faceWnmsMergedBoxXywh = new Mat();
            if (_faceWnmsMergedDecodedNx16 == null)
                _faceWnmsMergedDecodedNx16 = new Mat();
            if (_faceWnmsMergedScore == null)
                _faceWnmsMergedScore = new Mat();

            int num = boxXywhTensorNorm.rows();
            int numKpFloats = kFaceDetectorTensorsToDetectionsNumKeypoints * kFaceDetectorTensorsToDetectionsNumValuesPerKeypoint;
            if (num <= 0 || scoreCol == null || scoreCol.rows() < num
                         || decodedBoxesNx16 == null || decodedBoxesNx16.rows() < num)
            {
                _faceWnmsMergedBoxXywh.create(0, 4, CvType.CV_32FC1);
                _faceWnmsMergedDecodedNx16.create(0, kFaceDetectorTensorsToDetectionsNumCoords, CvType.CV_32FC1);
                _faceWnmsMergedScore.create(0, 1, CvType.CV_32FC1);
                _faceNmsIndices.create(0, 1, CvType.CV_32SC1);
                return _faceNmsIndices;
            }

            _faceWnmsIndexed.Clear();
            for (int i = 0; i < num; i++)
                _faceWnmsIndexed.Add((i, scoreCol.at<float>(i, 0)[0]));
            _faceWnmsIndexed.Sort((a, b) => b.sc.CompareTo(a.sc));

            _faceWnmsRemained.Clear();
            _faceWnmsRemained.AddRange(_faceWnmsIndexed);

            _faceNmsMergedBoxScratch.Clear();
            _faceNmsMergedDecScratch.Clear();
            _faceNmsMergedScScratch.Clear();

            if (_faceWnmsKpAccumulator == null || _faceWnmsKpAccumulator.Length < numKpFloats)
                _faceWnmsKpAccumulator = new float[numKpFloats];

            float[] decBuf = _faceDetectorDecodeRowDst;
            while (_faceWnmsRemained.Count > 0)
            {
                int originalSize = _faceWnmsRemained.Count;
                var anchor = _faceWnmsRemained[0];

                float ax = boxXywhTensorNorm.at<float>(anchor.idx, 0)[0];
                float ay = boxXywhTensorNorm.at<float>(anchor.idx, 1)[0];
                float aw = boxXywhTensorNorm.at<float>(anchor.idx, 2)[0];
                float ah = boxXywhTensorNorm.at<float>(anchor.idx, 3)[0];

                _faceWnmsNextRemained.Clear();
                for (int t = 0; t < _faceWnmsRemained.Count; t++)
                {
                    var item = _faceWnmsRemained[t];
                    float bx = boxXywhTensorNorm.at<float>(item.idx, 0)[0];
                    float by = boxXywhTensorNorm.at<float>(item.idx, 1)[0];
                    float bw = boxXywhTensorNorm.at<float>(item.idx, 2)[0];
                    float bh = boxXywhTensorNorm.at<float>(item.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) > kFaceMinSuppressionThreshold)
                        continue;
                    _faceWnmsNextRemained.Add(item);
                }

                float wXmin = 0f, wYmin = 0f, wXmax = 0f, wYmax = 0f;
                float totalScore = 0f;
                float[] kpAcc = _faceWnmsKpAccumulator;
                Array.Clear(kpAcc, 0, numKpFloats);
                for (int t = 0; t < _faceWnmsRemained.Count; t++)
                {
                    var c = _faceWnmsRemained[t];
                    float bx = boxXywhTensorNorm.at<float>(c.idx, 0)[0];
                    float by = boxXywhTensorNorm.at<float>(c.idx, 1)[0];
                    float bw = boxXywhTensorNorm.at<float>(c.idx, 2)[0];
                    float bh = boxXywhTensorNorm.at<float>(c.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) <= kFaceMinSuppressionThreshold)
                        continue;

                    float s = c.sc;
                    totalScore += s;
                    wXmin += bx * s;
                    wYmin += by * s;
                    wXmax += (bx + bw) * s;
                    wYmax += (by + bh) * s;
                    decodedBoxesNx16.get(c.idx, 0, decBuf.AsSpan(0, kFaceDetectorTensorsToDetectionsNumCoords));
                    int kpOff = kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                    for (int k = 0; k < numKpFloats; k++)
                        kpAcc[k] += decBuf[kpOff + k] * s;
                }

                if (totalScore <= 0f)
                    break;

                float outXmin = wXmin / totalScore;
                float outYmin = wYmin / totalScore;
                float outW = wXmax / totalScore - outXmin;
                float outH = wYmax / totalScore - outYmin;

                float[] outDec = RentFaceDetectorNmsDec16();
                outDec[0] = outYmin;
                outDec[1] = outXmin;
                outDec[2] = outYmin + outH;
                outDec[3] = outXmin + outW;
                int kOff = kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                for (int k = 0; k < numKpFloats; k++)
                    outDec[kOff + k] = kpAcc[k] / totalScore;

                float[] box4 = RentFaceDetectorNmsBox4();
                box4[0] = outXmin;
                box4[1] = outYmin;
                box4[2] = outW;
                box4[3] = outH;
                _faceNmsMergedBoxScratch.Add(box4);
                _faceNmsMergedDecScratch.Add(outDec);
                _faceNmsMergedScScratch.Add(anchor.sc);

                if (originalSize == _faceWnmsNextRemained.Count)
                    break;

                (_faceWnmsRemained, _faceWnmsNextRemained) = (_faceWnmsNextRemained, _faceWnmsRemained);
            }

            int kOut = _faceNmsMergedScScratch.Count;
            _faceWnmsMergedBoxXywh.create(kOut, 4, CvType.CV_32FC1);
            _faceWnmsMergedDecodedNx16.create(kOut, kFaceDetectorTensorsToDetectionsNumCoords, CvType.CV_32FC1);
            _faceWnmsMergedScore.create(kOut, 1, CvType.CV_32FC1);
            Span<float> putScore1 = stackalloc float[1];
            Span<int> putIdx1 = stackalloc int[1];
            for (int r = 0; r < kOut; r++)
            {
                _faceWnmsMergedBoxXywh.put(r, 0, _faceNmsMergedBoxScratch[r].AsSpan(0, 4));
                _faceWnmsMergedDecodedNx16.put(r, 0,
                    _faceNmsMergedDecScratch[r].AsSpan(0, kFaceDetectorTensorsToDetectionsNumCoords));
                putScore1[0] = _faceNmsMergedScScratch[r];
                _faceWnmsMergedScore.put(r, 0, putScore1);
            }

            _faceNmsIndices.create(kOut, 1, CvType.CV_32SC1);
            for (int r = 0; r < kOut; r++)
            {
                putIdx1[0] = r;
                _faceNmsIndices.put(r, 0, putIdx1);
            }

            ReleaseFaceDetectorNmsMergedScratchLists();

            return _faceNmsIndices;
        }

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
        /// Equivalent to <c>DetectionProjectionCalculator</c> from <c>detection_projection_calculator.cc</c>.
        /// Projects detections from tensor-normalized coordinates into input-image normalized coordinates.
        /// </summary>
        /// <param name="decodedBoxesNx16">
        /// Decoded 16-value rows from <see cref="TensorsToDetectionsCalculator"/>, or the K merged rows after <see cref="NonMaxSuppressionCalculator"/>.
        /// Used to project keypoints.
        /// </param>
        List<float[]> DetectionProjectionCalculator(Mat boxXywhTensorNorm, Mat scoreCol, Mat decodedBoxesNx16,
            MatOfInt indices)
        {
            var list = new List<float[]>();
            if (indices == null || indices.empty() || _faceDetectorProjectionMatrix16 == null)
                return list;

            ReadOnlySpan<float> m = _faceDetectorProjectionMatrix16;
            int selected = indices.rows();
            Span<float> dst = stackalloc float[FaceDetectorProjectedDetectionRowLength];
            float[] boxTn = _faceDetectorAnchorRow4;
            float[] allTn = _faceDetectorDecodeRowSrc;

            for (int i = 0; i < selected; i++)
            {
                int idx = indices.at<int>(i, 0)[0];
                boxXywhTensorNorm.get(idx, 0, boxTn.AsSpan(0, 4));
                float xminTn = boxTn[0];
                float yminTn = boxTn[1];
                float wTn = boxTn[2];
                float hTn = boxTn[3];

                float minNx = float.MaxValue;
                float minNy = float.MaxValue;
                float maxNx = float.MinValue;
                float maxNy = float.MinValue;
                FaceDetectorDetectionProjection_Project(m, xminTn, yminTn, out float p0x, out float p0y);
                FaceDetectorDetectionProjection_Project(m, xminTn + wTn, yminTn, out float p1x, out float p1y);
                FaceDetectorDetectionProjection_Project(m, xminTn + wTn, yminTn + hTn, out float p2x, out float p2y);
                FaceDetectorDetectionProjection_Project(m, xminTn, yminTn + hTn, out float p3x, out float p3y);
                minNx = Mathf.Min(Mathf.Min(p0x, p1x), Mathf.Min(p2x, p3x));
                minNy = Mathf.Min(Mathf.Min(p0y, p1y), Mathf.Min(p2y, p3y));
                maxNx = Mathf.Max(Mathf.Max(p0x, p1x), Mathf.Max(p2x, p3x));
                maxNy = Mathf.Max(Mathf.Max(p0y, p1y), Mathf.Max(p2y, p3y));

                float width = maxNx - minNx;
                float height = maxNy - minNy;
                dst[0] = minNx;
                dst[1] = minNy;
                dst[2] = width;
                dst[3] = height;

                decodedBoxesNx16.get(idx, 0, allTn.AsSpan(0, kFaceDetectorTensorsToDetectionsNumCoords));
                int kpOff = kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                for (int j = 0; j < kFaceDetectorTensorsToDetectionsNumKeypoints * 2; j += 2)
                {
                    float kx = allTn[kpOff + j];
                    float ky = allTn[kpOff + j + 1];
                    FaceDetectorDetectionProjection_Project(m, kx, ky, out float nx, out float ny);
                    dst[4 + j] = nx;
                    dst[4 + j + 1] = ny;
                }

                dst[16] = scoreCol.at<float>(idx, 0)[0];

                float[] row = RentFaceDetectorProjRow17();
                dst.CopyTo(row);
                list.Add(row);
            }

            return list;
        }

        static void FaceDetectorDetectionProjection_Project(ReadOnlySpan<float> m, float tx, float ty, out float nx,
            out float ny)
        {
            nx = tx * m[0] + ty * m[1] + m[3];
            ny = tx * m[4] + ty * m[5] + m[7];
        }

        /// <summary>
        /// Equivalent to <c>ClipDetectionVectorSizeCalculator</c> with <c>ClipVectorSizeCalculatorOptions.max_vec_size = num_faces</c>.
        /// </summary>
        List<float[]> ClipDetectionVectorSizeCalculator(List<float[]> detections, int maxVecSize)
        {
            if (detections == null)
                return new List<float[]>();

            if (detections.Count <= maxVecSize)
                return detections;

            for (int i = maxVecSize; i < detections.Count; i++)
                ReleaseFaceDetectorProjRow17(detections[i]);

            var clipped = new List<float[]>(maxVecSize);
            for (int i = 0; i < maxVecSize; i++)
                clipped.Add(detections[i]);
            return clipped;
        }

        /// <summary>
        /// Equivalent to <c>DetectionsToRectsCalculator</c> using the <c>DEFAULT</c> path in <c>detections_to_rects_calculator.cc</c>, with a bounding box and rotation keypoints.
        /// Builds <c>FACE_RECTS</c> from projected rows containing normalized <c>xmin,ymin,w,h</c>, normalized keypoints, and score.
        /// When there are no detections, returns an empty vector exactly as the upstream graph does, without placeholder rects.
        /// </summary>
        List<NormalizedRect> DetectionsToRectsCalculator(List<float[]> projectedRows, int imgW, int imgH)
        {
            if (imgW <= 0 || imgH <= 0)
                return new List<NormalizedRect>();

            if (projectedRows == null || projectedRows.Count == 0)
                return new List<NormalizedRect>();

            var rects = new List<NormalizedRect>(projectedRows.Count);
            foreach (var row in projectedRows)
            {
                if (row == null || row.Length < FaceDetectorProjectedDetectionRowLength)
                    continue;
                rects.Add(DetectionsToRectsCalculator_OneRow(row));
            }

            return rects;
        }

        NormalizedRect DetectionsToRectsCalculator_OneRow(ReadOnlySpan<float> row)
        {
            float xmin = row[0];
            float ymin = row[1];
            float wBox = row[2];
            float hBox = row[3];
            float centerX = xmin + wBox * 0.5f;
            float centerY = ymin + hBox * 0.5f;

            int k0 = kFaceDetectorDetectionsToRectsRotationStartKeypointIndex;
            int k1 = kFaceDetectorDetectionsToRectsRotationEndKeypointIndex;
            int o0 = 4 + k0 * 2;
            int o1 = 4 + k1 * 2;
            float x0 = row[o0];
            float y0 = row[o0 + 1];
            float x1 = row[o1];
            float y1 = row[o1 + 1];

            float targetRad = kFaceDetectorDetectionsToRectsTargetAngleDegrees * (Mathf.PI / 180f);
            float rotation = FaceDetectorNormalizeRadians(targetRad - Mathf.Atan2(-(y1 - y0), x1 - x0));

            return new NormalizedRect
            {
                XCenter = centerX,
                YCenter = centerY,
                Width = wBox,
                Height = hBox,
                Rotation = rotation,
            };
        }

        static float FaceDetectorNormalizeRadians(float angle)
        {
            const float twoPi = 2f * Mathf.PI;
            return angle - twoPi * Mathf.Floor((angle - (-Mathf.PI)) / twoPi);
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c>, specifically <c>TransformNormalizedRect</c> in <c>rect_transformation_calculator.cc</c>.
        /// Uses upstream <c>scale_x</c> and <c>scale_y</c> values of 1.5, with no shift or square-normalization options.
        /// </summary>
        List<NormalizedRect> RectTransformationCalculator(List<NormalizedRect> faceRects, int imageW, int imageH)
        {
            var list = new List<NormalizedRect>(faceRects.Count);
            foreach (var r in faceRects)
                list.Add(RectTransformationCalculator(r, imageW, imageH));
            return list;
        }

        NormalizedRect RectTransformationCalculator(NormalizedRect rect, int imageW, int imageH)
        {
            float width = rect.Width;
            float height = rect.Height;
            float rotation = rect.Rotation;
            float xCenter = rect.XCenter;
            float yCenter = rect.YCenter;

            const float shiftX = 0f;
            const float shiftY = 0f;
            float scaleX = kFaceDetectorExpandedRoiScale;
            float scaleY = kFaceDetectorExpandedRoiScale;

            float cosR = Mathf.Cos(rotation);
            float sinR = Mathf.Sin(rotation);
            float xShiftNorm = (imageW * width * shiftX * cosR - imageH * height * shiftY * sinR) / imageW;
            float yShiftNorm = (imageW * width * shiftX * sinR + imageH * height * shiftY * cosR) / imageH;
            xCenter += xShiftNorm;
            yCenter += yShiftNorm;

            return new NormalizedRect
            {
                XCenter = xCenter,
                YCenter = yCenter,
                Width = width * scaleX,
                Height = height * scaleY,
                Rotation = rotation,
                RectId = rect.RectId,
            };
        }

        /// <summary>
        /// Equivalent to <c>DetectionTransformationCalculator</c>: converts normalized bounding boxes into rows containing integer pixel bounding boxes, corresponding to task-external <c>PIXEL_DETECTIONS</c>.
        /// </summary>
        List<float[]> DetectionTransformationCalculator(List<float[]> projectedRows, int imageW, int imageH)
        {
            var list = new List<float[]>(projectedRows?.Count ?? 0);
            if (projectedRows == null)
                return list;

            foreach (var row in projectedRows)
            {
                if (row == null || row.Length < FaceDetectorProjectedDetectionRowLength)
                    continue;

                float xminN = row[0];
                float yminN = row[1];
                float wN = row[2];
                float hN = row[3];

                int xiMin = FaceDetectorBoundedInt(xminN * imageW, imageW);
                int yiMin = FaceDetectorBoundedInt(yminN * imageH, imageH);
                int wi = FaceDetectorBoundedInt(wN * imageW, imageW);
                int hi = FaceDetectorBoundedInt(hN * imageH, imageH);

                var copy = new float[FaceDetectorProjectedDetectionRowLength];
                row.AsSpan().CopyTo(copy);
                copy[0] = xiMin;
                copy[1] = yiMin;
                copy[2] = wi;
                copy[3] = hi;
                list.Add(copy);
            }

            return list;
        }

        static int FaceDetectorBoundedInt(float value, int upperBound)
        {
            int v = (int)Mathf.Floor(value);
            if (v < 0) return 0;
            if (v > upperBound) return upperBound;
            return v;
        }

        void NumpyClip(Mat a, double aMin, double aMax)
        {
            if (a == null || a.empty())
                return;
            if (_faceNumpyClipLo == null)
                _faceNumpyClipLo = new Mat();
            if (_faceNumpyClipHi == null)
                _faceNumpyClipHi = new Mat();
            _faceNumpyClipLo.create(a.rows(), a.cols(), a.type());
            _faceNumpyClipHi.create(a.rows(), a.cols(), a.type());
            _faceNumpyClipLo.setTo((aMin, aMin, aMin, aMin));
            _faceNumpyClipHi.setTo((aMax, aMax, aMax, aMax));
            Core.max(a, _faceNumpyClipLo, a);
            Core.min(a, _faceNumpyClipHi, a);
        }

        Mat FaceDetectorGraph_PrepareBoxMajorRows(Mat output0)
        {
            int n = _faceDetectorNumBoxes;
            int c = kFaceDetectorTensorsToDetectionsNumCoords;
            if (output0.size(1) == n && output0.size(2) == c)
                return output0.reshape(1, n);

            if (output0.size(1) == c && output0.size(2) == n)
            {
                using (Mat m16xN = output0.reshape(1, c))
                {
                    if (_faceDetectorTransposeBuffer == null
                        || _faceDetectorTransposeBuffer.rows() != n
                        || _faceDetectorTransposeBuffer.cols() != c)
                    {
                        _faceDetectorTransposeBuffer?.Dispose();
                        _faceDetectorTransposeBuffer = new Mat(n, c, CvType.CV_32FC1);
                    }

                    Core.transpose(m16xN, _faceDetectorTransposeBuffer);
                    return _faceDetectorTransposeBuffer;
                }
            }

            long total = output0.total();
            if (total == (long)n * c)
            {
                Mat reshaped = output0.reshape(1, n);
                if (reshaped.rows() == n && reshaped.cols() == c)
                    return reshaped;
            }

            throw new InvalidOperationException(
                $"Unsupported face_detector output tensor shape: dims={output0.dims()} size1={output0.size(1)} size2={output0.size(2)}");
        }

        Mat FaceDetectorGraph_PrepareScoreColumn(Mat output1)
        {
            int n = _faceDetectorNumBoxes;
            if (output1.size(1) == n && (output1.size(2) == 1 || output1.channels() * output1.size(2) == 1))
                return output1.reshape(1, n);

            if (output1.size(1) == 1 && output1.size(2) == n)
            {
                if (_faceDetectorScoreColumnBuffer == null
                    || _faceDetectorScoreColumnBuffer.rows() != n
                    || _faceDetectorScoreColumnBuffer.cols() != 1)
                {
                    _faceDetectorScoreColumnBuffer?.Dispose();
                    _faceDetectorScoreColumnBuffer = new Mat(n, 1, CvType.CV_32FC1);
                }

                using (Mat reshaped = output1.reshape(1, n))
                    Core.transpose(reshaped, _faceDetectorScoreColumnBuffer);
                return _faceDetectorScoreColumnBuffer;
            }

            long total = output1.total();
            if (total == n)
                return output1.reshape(1, n);

            throw new InvalidOperationException(
                $"Unsupported face_detector score tensor shape: size1={output1.size(1)} size2={output1.size(2)}");
        }

        Mat FaceDetectorGraph_BuildNmsBoxXywhFromDecoded()
        {
            int num = _faceDetectorNumBoxes;
            if (_faceTensorsToDetectionsWorking == null
                || _faceTensorsToDetectionsWorking.rows() != num
                || _faceTensorsToDetectionsWorking.cols() != 4)
            {
                _faceTensorsToDetectionsWorking?.Dispose();
                _faceTensorsToDetectionsWorking = new Mat(num, 4, CvType.CV_32FC1);
            }

            Mat dst = _faceTensorsToDetectionsWorking;
            float[] row = _faceDetectorDecodeRowSrc;
            Span<float> put4 = stackalloc float[4];
            for (int i = 0; i < num; i++)
            {
                _faceDetectorDecodedBoxesNx16.get(i, 0, row.AsSpan(0, kFaceDetectorTensorsToDetectionsNumCoords));
                float ymin = row[0];
                float xmin = row[1];
                float ymax = row[2];
                float xmax = row[3];
                float w = xmax - xmin;
                float h = ymax - ymin;
                put4[0] = xmin;
                put4[1] = ymin;
                put4[2] = w;
                put4[3] = h;
                dst.put(i, 0, put4);
            }

            return dst;
        }

        /// <summary>
        /// Same formula as <c>CalculateScale</c> in <c>ssd_anchors_calculator.cc</c>, used for face-detection anchors.
        /// </summary>
        static float FaceDetectorSsdAnchors_CalculateScale(float minScale, float maxScale, int strideIndex, int numStrides)
        {
            if (numStrides == 1)
                return (minScale + maxScale) * 0.5f;
            return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1.0f);
        }

        static Mat BuildFaceDetectorSsdAnchorsMat(
            int numLayers,
            float minScale,
            float maxScale,
            int inputSizeHeight,
            int inputSizeWidth,
            float anchorOffsetX,
            float anchorOffsetY,
            int[] strides,
            float aspectRatioOption,
            bool fixedAnchorSize,
            float interpolatedScaleAspectRatio,
            int expectedRows)
        {
            if (strides == null || strides.Length != numLayers)
                throw new InvalidOperationException("The SSD strides length does not match num_layers.");
            if (!fixedAnchorSize)
                throw new InvalidOperationException("Legacy face-detection SSD assumes fixed_anchor_size=true.");

            var aspectRatios = new List<float>(8);
            var scales = new List<float>(8);
            var anchorHeight = new List<float>(8);
            var anchorWidth = new List<float>(8);

            var xywh = new float[expectedRows * 4];
            int outIx = 0;

            int layerId = 0;
            int stridesLen = strides.Length;
            while (layerId < numLayers)
            {
                aspectRatios.Clear();
                scales.Clear();
                int lastSameStrideLayer = layerId;
                while (lastSameStrideLayer < stridesLen && strides[lastSameStrideLayer] == strides[layerId])
                {
                    float scale = FaceDetectorSsdAnchors_CalculateScale(minScale, maxScale, lastSameStrideLayer, stridesLen);
                    aspectRatios.Add(aspectRatioOption);
                    scales.Add(scale);
                    if (interpolatedScaleAspectRatio > 0f)
                    {
                        float scaleNext = lastSameStrideLayer == stridesLen - 1
                            ? 1.0f
                            : FaceDetectorSsdAnchors_CalculateScale(minScale, maxScale, lastSameStrideLayer + 1, stridesLen);
                        scales.Add(Mathf.Sqrt(scale * scaleNext));
                        aspectRatios.Add(interpolatedScaleAspectRatio);
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
                            xywh[outIx++] = xCenter;
                            xywh[outIx++] = yCenter;
                            xywh[outIx++] = 1f;
                            xywh[outIx++] = 1f;
                        }
                    }
                }

                layerId = lastSameStrideLayer;
            }

            if (outIx != expectedRows * 4)
                throw new InvalidOperationException(
                    $"Face SSD anchor count does not match the expected value: expected {expectedRows}, actual {outIx / 4}.");

            Mat anchors = new Mat(expectedRows, 4, CvType.CV_32FC1);
            anchors.put(0, 0, xywh.AsSpan(0, expectedRows * 4));
            return anchors;
        }

        /// <summary>
        /// Equivalent to <c>ClipNormalizedRectVectorSizeCalculator</c> with <c>max_vec_size = num_faces</c>.
        /// </summary>
        List<NormalizedRect> ClipNormalizedRectVectorSizeCalculator(List<NormalizedRect> rects, int maxVecSize)
        {
            if (rects == null)
                return new List<NormalizedRect>();
            if (rects.Count <= maxVecSize)
                return new List<NormalizedRect>(rects);
            var clipped = new List<NormalizedRect>(rects);
            clipped.RemoveRange(maxVecSize, clipped.Count - maxVecSize);
            return clipped;
        }

        /// <summary>Equivalent to <c>ClipNormalizedRectVectorSizeCalculator</c> using <c>_numFaces</c>.</summary>
        List<NormalizedRect> ClipNormalizedRectVectorSizeCalculator(List<NormalizedRect> rects)
        {
            return ClipNormalizedRectVectorSizeCalculator(rects, _numFaces);
        }

        /// <summary>
        /// Equivalent to <c>AssociationNormRectCalculator</c> from <c>association_norm_rect_calculator.cc</c>,
        /// using the base <c>GetNonOverlappingElements</c> behavior from <c>association_calculator.h</c>.
        /// </summary>
        /// <remarks>
        /// In upstream <c>face_landmarker_graph.cc</c> stream mode, input [0] receives previous-frame rects and input [1]
        /// receives <c>EXPANDED_FACE_RECTS</c> from <c>FaceDetectorGraph</c>, while <c>min_similarity_threshold</c> is the task option <c>min_tracking_confidence</c>.
        /// </remarks>
        /// <param name="prevFaceRects">Stream 0 input, corresponding to previous-frame <c>FACE_RECTS_NEXT_FRAME</c>.</param>
        /// <param name="expandedFaceRectsFromDetector">Stream 1 input, corresponding to detector-expanded rects. Empty when detection is skipped.</param>
        List<NormalizedRect> AssociationNormRectCalculator(
            List<NormalizedRect> prevFaceRects,
            List<NormalizedRect> expandedFaceRectsFromDetector)
        {
            float minSim = _minFaceTrackingConfidence;
            bool prevEmpty = prevFaceRects == null || prevFaceRects.Count == 0;
            bool detEmpty = expandedFaceRectsFromDetector == null || expandedFaceRectsFromDetector.Count == 0;
            List<NormalizedRect> result = _associationNormRectScratch;
            result.Clear();

            if (!prevEmpty)
            {
                result.Add(prevFaceRects[0]);
                for (int j = 1; j < prevFaceRects.Count; j++)
                    AssociationNormRectCalculator_AddElementToList(prevFaceRects[j], result, minSim);
                if (!detEmpty)
                {
                    foreach (var r in expandedFaceRectsFromDetector)
                        AssociationNormRectCalculator_AddElementToList(r, result, minSim);
                }
            }
            else if (!detEmpty)
            {
                result.Add(expandedFaceRectsFromDetector[0]);
                for (int j = 1; j < expandedFaceRectsFromDetector.Count; j++)
                    AssociationNormRectCalculator_AddElementToList(expandedFaceRectsFromDetector[j], result, minSim);
            }

            return result;
        }

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
        /// Equivalent to <c>MultiFaceLandmarksDetectorGraph</c>, specifically <c>BuildFaceLandmarksDetectorGraph</c> in <c>face_landmarks_detector_graph.cc</c>.
        /// Bundles BeginLoop, SingleFace, and EndLoop processing, plus optional smoothing and vector concatenation.
        ///
        /// Correspondence to the upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) implementation:
        /// - BeginLoopNormalizedRectCalculator → <see cref="BeginLoopNormalizedRectCalculator"/>
        /// - SingleFaceLandmarksDetectorGraph → <see cref="SingleFaceLandmarksDetectorGraph"/>
        /// - EndLoopBooleanCalculator → <see cref="EndLoopBooleanCalculator"/>
        /// - EndLoopFloatCalculator → <see cref="EndLoopFloatCalculator"/>
        /// - EndLoopNormalizedLandmarkListVectorCalculator → <see cref="EndLoopNormalizedLandmarkListVectorCalculator"/>
        /// - EndLoopNormalizedRectCalculator → <see cref="EndLoopNormalizedRectCalculator"/>
        /// - <c>smooth_landmarks</c> (only when VIDEO and <c>num_faces == 1</c>):
        ///   <see cref="GetNormalizedLandmarkListVectorItemCalculator"/> → <see cref="ImagePropertiesCalculator"/> →
        ///   <see cref="FaceLandmarksSmoothingPipeline"/> internal <c>LandmarksSmoothingCalculator</c> (One Euro) ->
        ///   <see cref="ConcatenateNormalizedLandmarkListVectorCalculator"/>
        /// - When <c>face_blendshapes_graph_options</c> is enabled: <see cref="MultiFaceLandmarksDetectorGraph_ApplyFaceBlendshapesClassifierLoop"/>
        /// </summary>
        List<FaceResult> MultiFaceLandmarksDetectorGraph(Mat image, List<NormalizedRect> faceRects)
        {
            List<bool> presences = _multiFacePresencesScratch;
            List<float> presenceScores = _multiFacePresenceScoresScratch;
            List<Vec3f[]> landmarkLists = _multiFaceLandmarkListsScratch;
            List<NormalizedRect> nextFrameRects = _multiFaceNextFrameRectsScratch;
            presences.Clear();
            presenceScores.Clear();
            landmarkLists.Clear();
            nextFrameRects.Clear();

            foreach (var loopItem in BeginLoopNormalizedRectCalculator(image, faceRects))
            {
                FaceResult? single = SingleFaceLandmarksDetectorGraph(loopItem.Image, loopItem.FaceRect);
                FaceResult fr = single ?? CreateAbsentFaceResultPlaceholder();
                EndLoopBooleanCalculator(presences, fr.FacePresence);
                EndLoopFloatCalculator(presenceScores, fr.FacePresenceScore);
                EndLoopNormalizedLandmarkListVectorCalculator(landmarkLists, fr.NormLandmarks);
                EndLoopNormalizedRectCalculator(nextFrameRects, fr.NextFrameRect);
            }

            var merged = MergeEndLoopFaceLandmarkOutputs(landmarkLists, nextFrameRects, presences, presenceScores);

            // Upstream behavior: apply outer-loop smoothing only in stream mode for a single face. IMAGE and multi-face cases are rejected earlier by the constructor.
            if (_faceLandmarksSmoothingPipeline != null
                && _runningMode == MediaPipeFaceRunningMode.VIDEO
                && _numFaces == 1)
            {
                if (merged.Count >= 1)
                    _faceLandmarksSmoothingPipeline.ApplyPostEndLoop(image, merged);
                else if (merged.Count == 0)
                    _faceLandmarksSmoothingPipeline.ResetAll();
            }

            if (_outputFaceBlendshapes && _hasFaceBlendshapesInference)
                MultiFaceLandmarksDetectorGraph_ApplyFaceBlendshapesClassifierLoop(image, merged);

            return merged;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="MultiFaceLandmarksDetectorGraph"/> using the Sentis path with <see cref="SingleFaceLandmarksDetectorGraphAsync"/>.
        /// </summary>
        async Task<List<FaceResult>> MultiFaceLandmarksDetectorGraphAsync(Mat image, List<NormalizedRect> faceRects, CancellationToken cancellationToken)
        {
            List<bool> presences = _multiFacePresencesScratch;
            List<float> presenceScores = _multiFacePresenceScoresScratch;
            List<Vec3f[]> landmarkLists = _multiFaceLandmarkListsScratch;
            List<NormalizedRect> nextFrameRects = _multiFaceNextFrameRectsScratch;
            presences.Clear();
            presenceScores.Clear();
            landmarkLists.Clear();
            nextFrameRects.Clear();

            foreach (var loopItem in BeginLoopNormalizedRectCalculator(image, faceRects))
            {
                FaceResult? single = await SingleFaceLandmarksDetectorGraphAsync(loopItem.Image, loopItem.FaceRect, cancellationToken);
                FaceResult fr = single ?? CreateAbsentFaceResultPlaceholder();
                EndLoopBooleanCalculator(presences, fr.FacePresence);
                EndLoopFloatCalculator(presenceScores, fr.FacePresenceScore);
                EndLoopNormalizedLandmarkListVectorCalculator(landmarkLists, fr.NormLandmarks);
                EndLoopNormalizedRectCalculator(nextFrameRects, fr.NextFrameRect);
            }

            var merged = MergeEndLoopFaceLandmarkOutputs(landmarkLists, nextFrameRects, presences, presenceScores);

            if (_faceLandmarksSmoothingPipeline != null
                && _runningMode == MediaPipeFaceRunningMode.VIDEO
                && _numFaces == 1)
            {
                if (merged.Count >= 1)
                    _faceLandmarksSmoothingPipeline.ApplyPostEndLoop(image, merged);
                else if (merged.Count == 0)
                    _faceLandmarksSmoothingPipeline.ResetAll();
            }

            if (_outputFaceBlendshapes && _hasFaceBlendshapesInference)
                await MultiFaceLandmarksDetectorGraph_ApplyFaceBlendshapesClassifierLoopAsync(image, merged, cancellationToken);

            return merged;
        }
#endif

        /// <summary>
        /// Equivalent to the <c>has_face_blendshapes_graph_options</c> block in <c>face_landmarks_detector_graph.cc</c>.
        /// <c>BeginLoopNormalizedLandmarkListVectorCalculator</c> -> each item ->
        /// <c>ImagePropertiesCalculator</c> + <c>FaceBlendshapesGraph</c> -> <c>EndLoopClassificationListCalculator</c>.
        /// </summary>
        void MultiFaceLandmarksDetectorGraph_ApplyFaceBlendshapesClassifierLoop(Mat image, List<FaceResult> merged)
        {
            if (image == null || merged == null)
                return;

            (int imgW, int imgH) = ImagePropertiesCalculator(image);
            for (int i = 0; i < merged.Count; i++)
            {
                Vec3f[] lm478 = BeginLoopNormalizedLandmarkListVectorCalculator_FaceBlendshapes_Item(merged, i);
                float[] coeffs = merged[i].FacePresence
                    ? FaceBlendshapesGraph(lm478, imgW, imgH)
                    : new float[kFaceBlendshapeCoefficientCount];
                EndLoopClassificationListCalculator_Append(merged, i, coeffs);
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary><see cref="MultiFaceLandmarksDetectorGraph_ApplyFaceBlendshapesClassifierLoop"/> with <see cref="FaceBlendshapesGraphAsync"/> for the Sentis path.</summary>
        async Task MultiFaceLandmarksDetectorGraph_ApplyFaceBlendshapesClassifierLoopAsync(Mat image, List<FaceResult> merged, CancellationToken cancellationToken)
        {
            if (image == null || merged == null)
                return;

            (int imgW, int imgH) = ImagePropertiesCalculator(image);
            for (int i = 0; i < merged.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Vec3f[] lm478 = BeginLoopNormalizedLandmarkListVectorCalculator_FaceBlendshapes_Item(merged, i);
                float[] coeffs = merged[i].FacePresence
                    ? await FaceBlendshapesGraphAsync(lm478, imgW, imgH, cancellationToken)
                    : new float[kFaceBlendshapeCoefficientCount];
                EndLoopClassificationListCalculator_Append(merged, i, coeffs);
            }
        }
#endif

        /// <summary>
        /// Equivalent to <c>BeginLoopNormalizedLandmarkListVectorCalculator</c>: returns the 478 normalized landmarks of the face at <paramref name="faceIndex"/>.
        /// </summary>
        static Vec3f[] BeginLoopNormalizedLandmarkListVectorCalculator_FaceBlendshapes_Item(List<FaceResult> merged,
            int faceIndex)
        {
            if (merged == null || faceIndex < 0 || faceIndex >= merged.Count)
                return s_emptyNormLandmarks478;
            Vec3f[] lm = merged[faceIndex].NormLandmarks;
            if (lm == null || lm.Length != kFaceMeshWithIrisLandmarksNum)
                return s_emptyNormLandmarks478;
            return lm;
        }

        /// <summary>
        /// Equivalent to <c>EndLoopClassificationListCalculator</c>: stores the coefficient vector into <see cref="FaceResult"/>.
        /// </summary>
        static void EndLoopClassificationListCalculator_Append(List<FaceResult> merged, int faceIndex, float[] coeffs52)
        {
            if (merged == null || faceIndex < 0 || faceIndex >= merged.Count)
                return;
            FaceResult fr = merged[faceIndex];
            fr.BlendshapeCoefficients = coeffs52 ?? new float[kFaceBlendshapeCoefficientCount];
            merged[faceIndex] = fr;
        }

        /// <summary>
        /// Equivalent to <c>FaceBlendshapesGraph</c> from <c>face_blendshapes_graph.cc</c>.
        /// This method only invokes child calculators and subgraphs in upstream order.
        ///
        /// Correspondence to the upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) implementation:
        /// - SplitNormalizedLandmarkListCalculator → <see cref="SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset"/>
        /// - LandmarksToTensorCalculator → <see cref="LandmarksToTensorCalculator_FaceBlendshapes"/>
        /// - Inference subgraph -> <see cref="InferenceSubgraph_FaceBlendshapes"/>
        /// - SplitTensorVectorCalculator → <see cref="SplitTensorVectorCalculator_FaceBlendshapesOutputTensor"/>
        /// - TensorsToClassificationCalculator → <see cref="TensorsToClassificationCalculator_FaceBlendshapes"/>
        /// </summary>
        float[] FaceBlendshapesGraph(Vec3f[] normLandmarks478, int imageWidth, int imageHeight)
        {
            if (normLandmarks478 == null || normLandmarks478.Length != kFaceMeshWithIrisLandmarksNum)
                return new float[kFaceBlendshapeCoefficientCount];

            Vec3f[] subset = SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset(normLandmarks478);
            LandmarksToTensorCalculator_FaceBlendshapes(subset, imageWidth, imageHeight);
            List<Mat> outs = InferenceSubgraph_FaceBlendshapes();
            Mat tensorVec = outs != null && outs.Count > 0 ? outs[0] : null;
            Mat coeffTensor = SplitTensorVectorCalculator_FaceBlendshapesOutputTensor(tensorVec);
            return TensorsToClassificationCalculator_FaceBlendshapes(coeffTensor);
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary><see cref="FaceBlendshapesGraph"/> with <see cref="InferenceSubgraph_FaceBlendshapesAsync"/> for the Unity Inference Engine async path.</summary>
        async Task<float[]> FaceBlendshapesGraphAsync(Vec3f[] normLandmarks478, int imageWidth, int imageHeight, CancellationToken cancellationToken)
        {
            if (normLandmarks478 == null || normLandmarks478.Length != kFaceMeshWithIrisLandmarksNum)
                return new float[kFaceBlendshapeCoefficientCount];

            Vec3f[] subset = SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset(normLandmarks478);
            LandmarksToTensorCalculator_FaceBlendshapes(subset, imageWidth, imageHeight);
            List<Mat> outs = await InferenceSubgraph_FaceBlendshapesAsync(cancellationToken);
            Mat tensorVec = outs != null && outs.Count > 0 ? outs[0] : null;
            Mat coeffTensor = SplitTensorVectorCalculator_FaceBlendshapesOutputTensor(tensorVec);
            return TensorsToClassificationCalculator_FaceBlendshapes(coeffTensor);
        }
#endif

        /// <summary>
        /// Equivalent to <c>SplitNormalizedLandmarkListCalculator</c> with <c>combine_outputs=true</c>, selecting the 146-landmark subset.
        /// </summary>
        static Vec3f[] SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset(Vec3f[] normLandmarks478)
        {
            var dst = new Vec3f[kFaceBlendshapesLandmarkSubsetIndices.Length];
            for (int i = 0; i < kFaceBlendshapesLandmarkSubsetIndices.Length; i++)
            {
                int idx = kFaceBlendshapesLandmarkSubsetIndices[i];
                dst[i] = normLandmarks478[idx];
            }

            return dst;
        }

        /// <summary>
        /// Equivalent to <c>LandmarksToTensorCalculator</c>: uses X/Y attributes, <c>flatten=false</c>, and scales normalized landmarks by image dimensions.
        /// </summary>
        void LandmarksToTensorCalculator_FaceBlendshapes(Vec3f[] subset146, int imageWidth, int imageHeight)
        {
            int n = kFaceBlendshapesLandmarkSubsetIndices.Length;
            if (_faceBlendshapesInputBlob == null
                || _faceBlendshapesInputBlob.dims() != 3
                || _faceBlendshapesInputBlob.size(0) != 1
                || _faceBlendshapesInputBlob.size(1) != n
                || _faceBlendshapesInputBlob.size(2) != 2)
            {
                _faceBlendshapesInputBlob?.Dispose();
                _faceBlendshapesInputBlob = new Mat(new int[] { 1, n, 2 }, CvType.CV_32FC1);
            }

            Span<float> buf = stackalloc float[n * 2];
            for (int i = 0; i < n; i++)
            {
                buf[i * 2] = subset146[i].Item1 * imageWidth;
                buf[i * 2 + 1] = subset146[i].Item2 * imageHeight;
            }

            // For 3D Mats (1 x N x 2), put(0,0,arr) can write only the first slice.
            // Copy the full contiguous buffer with OpenCVMatUtils.CopyToMat instead.
            OpenCVMatUtils.CopyToMat<float>(buf, _faceBlendshapesInputBlob);
        }

        /// <summary>Equivalent to <c>AddInference</c> inside <c>FaceBlendshapesGraph</c>.</summary>
        List<Mat> InferenceSubgraph_FaceBlendshapes()
        {
            if (_faceBlendshapesNet == null)
                return _faceBlendshapesForwardOutputList;
            if (_faceBlendshapesNetOutLayerNames == null || _faceBlendshapesNetOutLayerNames.Count == 0)
                _faceBlendshapesNetOutLayerNames = _faceBlendshapesNet.getUnconnectedOutLayersNames();
            _faceBlendshapesForwardOutputList.Clear();
            _faceBlendshapesNet.setInput(_faceBlendshapesInputBlob);
            _faceBlendshapesNet.forward(_faceBlendshapesForwardOutputList, _faceBlendshapesNetOutLayerNames);
            return _faceBlendshapesForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="InferenceSubgraph_FaceBlendshapes"/>. Invoked only from <see cref="RunCoreProcessingTaskAsync"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_FaceBlendshapesAsync(CancellationToken cancellationToken)
        {
            if (_faceBlendshapesNet == null)
                return _faceBlendshapesForwardOutputList;
            if (_faceBlendshapesNetOutLayerNames == null || _faceBlendshapesNetOutLayerNames.Count == 0)
                _faceBlendshapesNetOutLayerNames = _faceBlendshapesNet.getUnconnectedOutLayersNames();
            _faceBlendshapesForwardOutputList.Clear();
            _faceBlendshapesNet.setInput(_faceBlendshapesInputBlob);
            await _faceBlendshapesNet.forwardTaskAsync(_faceBlendshapesForwardOutputList, _faceBlendshapesNetOutLayerNames, cancellationToken);
            return _faceBlendshapesForwardOutputList;
        }
#endif

        /// <summary>
        /// Equivalent to <c>SplitTensorVectorCalculator</c>, taking the first tensor from the output tensor vector.
        /// </summary>
        static Mat SplitTensorVectorCalculator_FaceBlendshapesOutputTensor(Mat tensorsVectorHead)
        {
            return tensorsVectorHead;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToClassificationCalculator</c> with <c>top_k=0</c>, returning the coefficients as-is.
        /// </summary>
        static float[] TensorsToClassificationCalculator_FaceBlendshapes(Mat coefficients1x52)
        {
            var coeffs = new float[kFaceBlendshapeCoefficientCount];
            if (coefficients1x52 == null || coefficients1x52.empty())
                return coeffs;

            long t = coefficients1x52.total();
            int n = (int)Math.Min(t, kFaceBlendshapeCoefficientCount);
            if (n <= 0)
                return coeffs;

            using (Mat flat = coefficients1x52.reshape(1, (int)t))
            {
                flat.get(0, 0, coeffs.AsSpan(0, n));
            }

            return coeffs;
        }

        static IEnumerable<(Mat Image, NormalizedRect FaceRect)> BeginLoopNormalizedRectCalculator(
            Mat image, List<NormalizedRect> faceRects)
        {
            if (image == null || faceRects == null)
                yield break;
            foreach (var r in faceRects)
                yield return (image, r);
        }

        static void EndLoopBooleanCalculator(List<bool> iterable, bool item)
        {
            iterable.Add(item);
        }

        static void EndLoopFloatCalculator(List<float> iterable, float item)
        {
            iterable.Add(item);
        }

        static void EndLoopNormalizedLandmarkListVectorCalculator(List<Vec3f[]> iterable, Vec3f[] item)
        {
            iterable.Add(item ?? s_emptyNormLandmarks478);
        }

        static void EndLoopNormalizedRectCalculator(List<NormalizedRect> iterable, NormalizedRect item)
        {
            iterable.Add(item);
        }

        static List<FaceResult> MergeEndLoopFaceLandmarkOutputs(
            List<Vec3f[]> landmarkLists,
            List<NormalizedRect> nextFrameRects,
            List<bool> presences,
            List<float> presenceScores)
        {
            int n = landmarkLists.Count;
            var list = new List<FaceResult>(n);
            for (int i = 0; i < n; i++)
            {
                list.Add(new FaceResult
                {
                    FacePresence = presences[i],
                    FacePresenceScore = presenceScores[i],
                    NormLandmarks = landmarkLists[i],
                    NextFrameRect = nextFrameRects[i],
                });
            }

            return list;
        }

        static FaceResult CreateAbsentFaceResultPlaceholder()
        {
            return new FaceResult
            {
                FacePresence = false,
                FacePresenceScore = 0f,
                NormLandmarks = new Vec3f[kFaceMeshWithIrisLandmarksNum],
                NextFrameRect = new NormalizedRect(),
            };
        }

        /// <summary>
        /// Equivalent to <c>ImagePropertiesCalculator</c>: returns the input image pixel width and height used as <c>IMAGE_SIZE</c> for smoothing.
        /// </summary>
        static (int Width, int Height) ImagePropertiesCalculator(Mat image)
        {
            if (image == null || image.empty())
                return (0, 0);
            return (image.cols(), image.rows());
        }

        /// <summary>
        /// Equivalent to <c>GetNormalizedLandmarkListVectorItemCalculator</c> for <c>smooth_landmarks</c>, with <c>item_index = 0</c>.
        /// </summary>
        static Vec3f[] GetNormalizedLandmarkListVectorItemCalculator(List<FaceResult> merged)
        {
            if (merged == null || merged.Count <= kFaceSmoothLandmarksVectorItemIndex)
                return null;
            Vec3f[] src = merged[kFaceSmoothLandmarksVectorItemIndex].NormLandmarks;
            if (src == null || src.Length != kFaceMeshWithIrisLandmarksNum)
                return null;
            var copy = new Vec3f[src.Length];
            Array.Copy(src, copy, src.Length);
            return copy;
        }

        /// <summary>
        /// Equivalent to <c>ConcatenateNormalizedLandmarkListVectorCalculator</c>, replacing the first landmark list when the vector length is 1.
        /// </summary>
        static FaceResult ConcatenateNormalizedLandmarkListVectorCalculator(List<FaceResult> merged,
            Vec3f[] smoothedLandmarks)
        {
            FaceResult p = merged[kFaceSmoothLandmarksVectorItemIndex];
            p.NormLandmarks = smoothedLandmarks;
            return p;
        }

        /// <summary>
        /// Equivalent to <c>SingleFaceLandmarksDetectorGraph</c>, specifically <c>BuildSingleFaceLandmarksDetectorGraph</c> in <c>face_landmarks_detector_graph.cc</c>.
        /// Invokes downstream calculators and subgraphs only in upstream connection order.
        ///
        /// Correspondence to the upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) implementation:
        /// - ImagePreprocessingGraph → <see cref="ImagePreprocessingGraph_SingleFaceLandmarks"/>
        /// - Inference → <see cref="InferenceSubgraph_SingleFaceLandmarks"/>
        /// - SplitTensorVectorCalculator → <see cref="SplitTensorVectorCalculator_FaceLandmarks"/>
        /// - TensorsToFaceLandmarksGraph → <see cref="TensorsToFaceLandmarksGraph"/>
        /// - TensorsToFloatsCalculator → <see cref="TensorsToFloatsCalculator_FacePresence"/>
        /// - ThresholdingCalculator → <see cref="ThresholdingCalculator_FacePresence"/>
        /// - LandmarkLetterboxRemovalCalculator → <see cref="LandmarkLetterboxRemovalCalculator_Face"/>
        /// - LandmarkProjectionCalculator → <see cref="LandmarkProjectionCalculator_SingleFaceLandmarks"/>
        /// - AllowIf → <see cref="AllowIf_FaceNormLandmarks"/>
        /// - LandmarksToDetectionCalculator → <see cref="LandmarksToDetectionCalculator_Face"/>
        /// - DetectionsToRectsCalculator → <see cref="DetectionsToRectsCalculator_FaceLandmarksRoi"/>
        /// - RectTransformationCalculator → <see cref="RectTransformationCalculator_FaceLandmarksNextFrame"/>
        /// - AllowIf (next-frame rect) -> <see cref="AllowIf_FaceNextFrameRect"/>
        /// </summary>
        FaceResult? SingleFaceLandmarksDetectorGraph(Mat image, NormalizedRect faceRect)
        {
            if (!ImagePreprocessingGraph_SingleFaceLandmarks(image, faceRect, out SingleFaceLandmarkPreprocessOut pre))
                return null;

            Mat faceBlob = pre.FaceBlob;
            List<Mat> inferenceTensors = InferenceSubgraph_SingleFaceLandmarks(faceBlob);
            if (inferenceTensors == null || inferenceTensors.Count < kFaceLandmarksOutputTensorsNum)
                return null;

            if (!SplitTensorVectorCalculator_FaceLandmarks(inferenceTensors, out Mat landmarkTensor,
                out Mat presenceTensor))
                return null;

            float[] letterboxedNormLm = TensorsToFaceLandmarksGraph(landmarkTensor);
            float presenceScore = TensorsToFloatsCalculator_FacePresence(presenceTensor);
            bool facePresence = ThresholdingCalculator_FacePresence(presenceScore);

            float[] afterLetterbox = LandmarkLetterboxRemovalCalculator_Face(letterboxedNormLm,
                pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft, pre.LetterboxPaddingBottom,
                pre.LetterboxPaddingRight);

            Vec3f[] projectedRaw =
                LandmarkProjectionCalculator_SingleFaceLandmarks(afterLetterbox, faceRect, pre.ImageW, pre.ImageH);
            Vec3f[] projected = AllowIf_FaceNormLandmarks(facePresence, projectedRaw);

            NormalizedRect nextFrame;
            if (facePresence)
            {
                var det = LandmarksToDetectionCalculator_Face(projected);
                NormalizedRect faceLmRect =
                    DetectionsToRectsCalculator_FaceLandmarksRoi(det, pre.ImageW, pre.ImageH);
                nextFrame = RectTransformationCalculator_FaceLandmarksNextFrame(faceLmRect, pre.ImageW, pre.ImageH);
                if (faceRect.RectId.HasValue)
                    nextFrame.RectId = faceRect.RectId;
            }
            else
                nextFrame = new NormalizedRect();

            nextFrame = AllowIf_FaceNextFrameRect(facePresence, nextFrame);

            return new FaceResult
            {
                FacePresence = facePresence,
                FacePresenceScore = presenceScore,
                NormLandmarks = projected,
                NextFrameRect = nextFrame,
            };
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// <see cref="SingleFaceLandmarksDetectorGraph"/> using the Sentis path with <see cref="InferenceSubgraph_SingleFaceLandmarksAsync"/>.
        /// </summary>
        async Task<FaceResult?> SingleFaceLandmarksDetectorGraphAsync(Mat image, NormalizedRect faceRect, CancellationToken cancellationToken)
        {
            if (!ImagePreprocessingGraph_SingleFaceLandmarks(image, faceRect, out SingleFaceLandmarkPreprocessOut pre))
                return null;

            Mat faceBlob = pre.FaceBlob;
            List<Mat> inferenceTensors = await InferenceSubgraph_SingleFaceLandmarksAsync(faceBlob, cancellationToken);
            if (inferenceTensors == null || inferenceTensors.Count < kFaceLandmarksOutputTensorsNum)
                return null;

            if (!SplitTensorVectorCalculator_FaceLandmarks(inferenceTensors, out Mat landmarkTensor,
                out Mat presenceTensor))
                return null;

            float[] letterboxedNormLm = TensorsToFaceLandmarksGraph(landmarkTensor);
            float presenceScore = TensorsToFloatsCalculator_FacePresence(presenceTensor);
            bool facePresence = ThresholdingCalculator_FacePresence(presenceScore);

            float[] afterLetterbox = LandmarkLetterboxRemovalCalculator_Face(letterboxedNormLm,
                pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft, pre.LetterboxPaddingBottom,
                pre.LetterboxPaddingRight);

            Vec3f[] projectedRaw =
                LandmarkProjectionCalculator_SingleFaceLandmarks(afterLetterbox, faceRect, pre.ImageW, pre.ImageH);
            Vec3f[] projected = AllowIf_FaceNormLandmarks(facePresence, projectedRaw);

            NormalizedRect nextFrame;
            if (facePresence)
            {
                var det = LandmarksToDetectionCalculator_Face(projected);
                NormalizedRect faceLmRect =
                    DetectionsToRectsCalculator_FaceLandmarksRoi(det, pre.ImageW, pre.ImageH);
                nextFrame = RectTransformationCalculator_FaceLandmarksNextFrame(faceLmRect, pre.ImageW, pre.ImageH);
                if (faceRect.RectId.HasValue)
                    nextFrame.RectId = faceRect.RectId;
            }
            else
                nextFrame = new NormalizedRect();

            nextFrame = AllowIf_FaceNextFrameRect(facePresence, nextFrame);

            return new FaceResult
            {
                FacePresence = facePresence,
                FacePresenceScore = presenceScore,
                NormLandmarks = projected,
                NextFrameRect = nextFrame,
            };
        }
#endif

        /// <summary>Landmark preprocessing output for one face, corresponding to the Tasks CPU-side <c>ImagePreprocessingGraph</c> path.</summary>
        struct SingleFaceLandmarkPreprocessOut
        {
            public Mat FaceBlob;
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
        /// Equivalent to <c>ImagePreprocessingGraph</c>: warps the normalized-rect ROI into <see cref="_faceLandmarkTensorSize"/>
        /// and creates an NHWC RGB input blob with shape <c>[1,H,W,3]</c> normalized to 0-1.
        /// Allocation of the warp targets and inference blob happens inside this method, as in <see cref="MediaPipeHandLandmarker.ImagePreprocessingGraph_SingleHandLandmarks"/>.
        /// </summary>
        bool ImagePreprocessingGraph_SingleFaceLandmarks(Mat image, NormalizedRect faceRect,
            out SingleFaceLandmarkPreprocessOut pre)
        {
            pre = default;
            int imgW = image.cols();
            int imgH = image.rows();
            if (imgW <= 0 || imgH <= 0 || _faceLandmarkTensorSize <= 0)
                return false;

            int ts = _faceLandmarkTensorSize;
            const int lmC = 3;
            const float image01Divisor = 255f;

            if (_faceLmWarpDstPts == null)
            {
                _faceLmWarpDstPts = new Mat(4, 2, CvType.CV_32FC1);
                Span<float> dstPtsArr = stackalloc float[8];
                float dw = ts;
                float dh = ts;
                dstPtsArr[0] = 0f;
                dstPtsArr[1] = dh;
                dstPtsArr[2] = 0f;
                dstPtsArr[3] = 0f;
                dstPtsArr[4] = dw;
                dstPtsArr[5] = 0f;
                dstPtsArr[6] = dw;
                dstPtsArr[7] = dh;
                _faceLmWarpDstPts.put(0, 0, dstPtsArr);
                _faceLmWarpSrcPts = new Mat(4, 2, CvType.CV_32FC1);
            }

            if (_faceLmWarpedBgr == null || _faceLmWarpedBgr.rows() != ts || _faceLmWarpedBgr.cols() != ts)
            {
                _faceLmWarpedBgr?.Dispose();
                _faceLmWarpedRgb?.Dispose();
                _faceLandmarksInferenceBlob?.Dispose();
                _faceLmWarpedBgr = new Mat(ts, ts, CvType.CV_8UC3);
                _faceLmWarpedRgb = new Mat(ts, ts, CvType.CV_8UC3);
                _faceLandmarksInferenceBlob = new Mat(new int[] { 1, ts, ts, lmC }, CvType.CV_32FC1);
                _faceLandmarksInferenceBlobHxW =
                    _faceLandmarksInferenceBlob.reshape(lmC, new int[] { ts, ts });
            }

            float cx = faceRect.XCenter * imgW;
            float cy = faceRect.YCenter * imgH;
            float rw = faceRect.Width * imgW;
            float rh = faceRect.Height * imgH;
            if (rw <= 0f || rh <= 0f || float.IsNaN(rw) || float.IsNaN(rh))
                return false;

            PadRoiLikeImageToTensorCalculator(ts, ts, true, ref rw, ref rh,
                out float padL, out float padT, out float padR, out float padB);

            double angleDeg = faceRect.Rotation * 180.0 / Math.PI;

            Imgproc.boxPoints((cx, cy, rw, rh, angleDeg), _faceLmWarpSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_faceLmWarpSrcPts, _faceLmWarpDstPts))
            {
                Imgproc.warpPerspective(image, _faceLmWarpedBgr, projMat, (ts, ts),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
            }

            Imgproc.cvtColor(_faceLmWarpedBgr, _faceLmWarpedRgb, Imgproc.COLOR_BGR2RGB);
            _faceLmWarpedRgb.convertTo(_faceLandmarksInferenceBlobHxW, CvType.CV_32F,
                1.0 / image01Divisor);

            pre = new SingleFaceLandmarkPreprocessOut
            {
                FaceBlob = _faceLandmarksInferenceBlob,
                ImageW = imgW,
                ImageH = imgH,
                ModelW = ts,
                ModelH = ts,
                LetterboxPaddingTop = padT,
                LetterboxPaddingLeft = padL,
                LetterboxPaddingRight = padR,
                LetterboxPaddingBottom = padB,
            };
            return true;
        }

        /// <summary>Equivalent to <c>PadRoi</c> in <c>image_to_tensor_utils.cc</c>, returning normalized padding values.</summary>
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

        /// <summary>Equivalent to the inference subgraph: feeds the preprocessed blob into the face_landmarks model and returns the output tensor vector.</summary>
        List<Mat> InferenceSubgraph_SingleFaceLandmarks(Mat faceBlob)
        {
            if (_faceLandmarksNetOutLayerNames == null || _faceLandmarksNetOutLayerNames.Count == 0)
                _faceLandmarksNetOutLayerNames = _faceLandmarksNet.getUnconnectedOutLayersNames();
            _faceLandmarksForwardOutputList.Clear();
            _faceLandmarksNet.setInput(faceBlob);
            _faceLandmarksNet.forward(_faceLandmarksForwardOutputList, _faceLandmarksNetOutLayerNames);
            return _faceLandmarksForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="InferenceSubgraph_SingleFaceLandmarks"/>. Invoked only from <see cref="RunCoreProcessingTaskAsync"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_SingleFaceLandmarksAsync(Mat faceBlob, CancellationToken cancellationToken)
        {
            if (_faceLandmarksNetOutLayerNames == null || _faceLandmarksNetOutLayerNames.Count == 0)
                _faceLandmarksNetOutLayerNames = _faceLandmarksNet.getUnconnectedOutLayersNames();
            _faceLandmarksForwardOutputList.Clear();
            _faceLandmarksNet.setInput(faceBlob);
            await _faceLandmarksNet.forwardTaskAsync(_faceLandmarksForwardOutputList, _faceLandmarksNetOutLayerNames, cancellationToken);
            return _faceLandmarksForwardOutputList;
        }
#endif

        /// <summary>
        /// Equivalent to <c>SplitTensorVectorCalculator</c>: extracts tensors using the landmark and presence output indices fixed during bootstrap.
        /// </summary>
        bool SplitTensorVectorCalculator_FaceLandmarks(List<Mat> inferenceTensors, out Mat landmarkTensor,
            out Mat presenceTensor)
        {
            landmarkTensor = presenceTensor = null;
            if (inferenceTensors == null || inferenceTensors.Count < kFaceLandmarksOutputTensorsNum)
                return false;
            landmarkTensor = inferenceTensors[1];
            presenceTensor = inferenceTensors[0];
            return landmarkTensor != null && presenceTensor != null;
        }

        /// <summary>
        /// Equivalent to <c>TensorsToFaceLandmarksGraph</c> from <c>tensors_to_face_landmarks_graph.cc</c>. Internally this path consists only of <c>TensorsToLandmarksCalculator</c>.
        /// </summary>
        float[] TensorsToFaceLandmarksGraph(Mat landmarkTensor)
        {
            return TensorsToLandmarksCalculator_Face(landmarkTensor, _faceLandmarkTensorSize, _faceLandmarkTensorSize);
        }

        /// <summary>Equivalent to <c>TensorsToLandmarksCalculator</c>, producing normalized output with <c>normalize_z = 1</c>.</summary>
        float[] TensorsToLandmarksCalculator_Face(Mat tensor, int inputW, int inputH)
        {
            const float normalizeZ = 1f;
            int n = kFaceMeshWithIrisLandmarksNum;
            int need = n * 3;
            long tTotal = tensor.total();
            if (tTotal < need)
            {
                if (_faceTensorsToLmNorm == null || _faceTensorsToLmNorm.Length < need)
                    _faceTensorsToLmNorm = new float[need];
                Array.Clear(_faceTensorsToLmNorm, 0, need);
                return _faceTensorsToLmNorm;
            }

            if (_faceTensorsToLmRaw == null || _faceTensorsToLmRaw.Length < need)
                _faceTensorsToLmRaw = new float[need];
            if (_faceTensorsToLmNorm == null || _faceTensorsToLmNorm.Length < need)
                _faceTensorsToLmNorm = new float[need];

            using (var flat = tensor.reshape(1, (int)tTotal))
            {
                float[] raw = _faceTensorsToLmRaw;
                float[] norm = _faceTensorsToLmNorm;
                flat.get(0, 0, raw.AsSpan(0, need));
                float zDenom = inputW * normalizeZ;
                if (zDenom < 1e-8f)
                    zDenom = 1f;
                for (int i = 0; i < n; i++)
                {
                    int o = i * 3;
                    norm[o] = raw[o] / inputW;
                    norm[o + 1] = raw[o + 1] / inputH;
                    norm[o + 2] = raw[o + 2] / zDenom;
                }

                return norm;
            }
        }

        /// <summary>Equivalent to <c>TensorsToFloatsCalculator</c> with <c>SIGMOID</c>.</summary>
        static float TensorsToFloatsCalculator_FacePresence(Mat presenceTensor)
        {
            float v = presenceTensor.at<float>(0, 0)[0];
            return 1f / (1f + Mathf.Exp(-v));
        }

        /// <summary>Equivalent to <c>ThresholdingCalculator</c> using <c>min_detection_confidence</c>.</summary>
        bool ThresholdingCalculator_FacePresence(float score)
        {
            return score >= _minFacePresenceConfidence;
        }

        /// <summary>Equivalent to <c>LandmarkLetterboxRemovalCalculator</c> from <c>landmark_letterbox_removal_calculator.cc</c>.</summary>
        float[] LandmarkLetterboxRemovalCalculator_Face(float[] normLandmarks, float padTop, float padLeft,
            float padBottom, float padRight)
        {
            int el = kFaceMeshWithIrisLandmarksNum * 3;
            if (normLandmarks == null)
                return new float[el];
            if (normLandmarks.Length < el)
            {
                var tmpShort = new float[el];
                normLandmarks.AsSpan().CopyTo(tmpShort);
                return tmpShort;
            }

            if (padTop == 0f && padLeft == 0f && padBottom == 0f && padRight == 0f)
                return normLandmarks;

            float h = 1f - padTop - padBottom;
            float w = 1f - padLeft - padRight;
            if (h <= 1e-6f || w <= 1e-6f)
                return normLandmarks;

            float[] o = _faceLetterboxRemovedNormScratch;
            for (int i = 0; i < kFaceMeshWithIrisLandmarksNum; i++)
            {
                int k = i * 3;
                o[k] = (normLandmarks[k] - padLeft) / w;
                o[k + 1] = (normLandmarks[k + 1] - padTop) / h;
                o[k + 2] = normLandmarks[k + 2] / w;
            }

            return o;
        }

        /// <summary>
        /// Equivalent to <c>LandmarkProjectionCalculator</c> with <c>NORM_RECT</c> and <c>IMAGE_DIMENSIONS</c>, matching <c>landmark_projection_calculator.cc</c>.
        /// </summary>
        Vec3f[] LandmarkProjectionCalculator_SingleFaceLandmarks(float[] normLandmarksLetterboxRemoved,
            NormalizedRect faceRect, int imgW, int imgH)
        {
            int n = kFaceMeshWithIrisLandmarksNum;
            var screen = new Vec3f[n];
            if (normLandmarksLetterboxRemoved == null || normLandmarksLetterboxRemoved.Length < n * 3)
                return screen;

            float cx = faceRect.XCenter * imgW;
            float cy = faceRect.YCenter * imgH;
            float rw = faceRect.Width * imgW;
            float rh = faceRect.Height * imgH;
            float rot = faceRect.Rotation;
            GetRotatedSubRectToRectTransformMatrix(cx, cy, rw, rh, rot, imgW, imgH, false, _faceLmProjectionMatrix16);
            float zScale = FaceLandmarkProjection_CalculateZScale(_faceLmProjectionMatrix16);

            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float lx = normLandmarksLetterboxRemoved[o];
                float ly = normLandmarksLetterboxRemoved[o + 1];
                float lz = normLandmarksLetterboxRemoved[o + 2];
                FaceLandmarkProjection_ProjectXY(lx, ly, lz, _faceLmProjectionMatrix16, out float nx, out float ny);
                screen[i] = new Vec3f(nx, ny, zScale * lz);
            }

            return screen;
        }

        static void FaceLandmarkProjection_ProjectXY(float x, float y, float z, float[] m, out float nx, out float ny)
        {
            nx = x * m[0] + y * m[1] + z * m[2] + m[3];
            ny = x * m[4] + y * m[5] + z * m[6] + m[7];
        }

        static float FaceLandmarkProjection_CalculateZScale(float[] m)
        {
            FaceLandmarkProjection_ProjectXY(0f, 0f, 0f, m, out float ax, out float ay);
            FaceLandmarkProjection_ProjectXY(1f, 0f, 0f, m, out float bx, out float by);
            float dx = bx - ax;
            float dy = by - ay;
            return Mathf.Sqrt(dx * dx + dy * dy);
        }

        static Vec3f[] AllowIf_FaceNormLandmarks(bool facePresence, Vec3f[] landmarksWhenPresent)
        {
            if (!facePresence || landmarksWhenPresent == null)
                return new Vec3f[kFaceMeshWithIrisLandmarksNum];
            return landmarksWhenPresent;
        }

        /// <summary>
        /// Equivalent to <c>LandmarksToDetectionCalculator</c>: builds a <c>RELATIVE_BOUNDING_BOX</c> and full <c>relative_keypoints</c> from normalized landmarks.
        /// </summary>
        struct FaceLandmarkPseudoDetection
        {
            public float Xmin, Ymin, Width, Height;
            public Vec3f[] KeypointsNorm;
        }

        static FaceLandmarkPseudoDetection LandmarksToDetectionCalculator_Face(Vec3f[] normLandmarksFullImage)
        {
            int n = kFaceMeshWithIrisLandmarksNum;
            var d = new FaceLandmarkPseudoDetection { KeypointsNorm = normLandmarksFullImage };
            float xmin = float.MaxValue, ymin = float.MaxValue, xmax = float.MinValue, ymax = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                float x = normLandmarksFullImage[i].Item1;
                float y = normLandmarksFullImage[i].Item2;
                if (x < xmin) xmin = x;
                if (y < ymin) ymin = y;
                if (x > xmax) xmax = x;
                if (y > ymax) ymax = y;
            }

            d.Xmin = xmin;
            d.Ymin = ymin;
            d.Width = xmax - xmin;
            d.Height = ymax - ymin;
            return d;
        }

        /// <summary>
        /// Equivalent to <c>DetectionsToRectsCalculator</c> using <c>DEFAULT</c> with rotation keypoints 33 and 263, matching <c>ConfigureFaceDetectionsToRectsCalculator</c>.
        /// </summary>
        NormalizedRect DetectionsToRectsCalculator_FaceLandmarksRoi(FaceLandmarkPseudoDetection det, int imgW, int imgH)
        {
            float xmin = det.Xmin;
            float ymin = det.Ymin;
            float wBox = det.Width;
            float hBox = det.Height;
            float centerX = xmin + wBox * 0.5f;
            float centerY = ymin + hBox * 0.5f;

            int k0 = kFaceLandmarksDetectionsToRectsRotationStartKeypointIndex;
            int k1 = kFaceLandmarksDetectionsToRectsRotationEndKeypointIndex;
            Vec3f[] kp = det.KeypointsNorm;
            if (kp == null || kp.Length <= Mathf.Max(k0, k1))
                return new NormalizedRect();

            float x0 = kp[k0].Item1 * imgW;
            float y0 = kp[k0].Item2 * imgH;
            float x1 = kp[k1].Item1 * imgW;
            float y1 = kp[k1].Item2 * imgH;

            float targetRad = kFaceLandmarksDetectionsToRectsTargetAngleDegrees * (Mathf.PI / 180f);
            float rotation = FaceDetectorNormalizeRadians(targetRad - Mathf.Atan2(-(y1 - y0), x1 - x0));

            return new NormalizedRect
            {
                XCenter = centerX,
                YCenter = centerY,
                Width = wBox,
                Height = hBox,
                Rotation = rotation,
            };
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> for the next-frame ROI, using <c>scale_x</c> and <c>scale_y</c> of 1.5 with <c>square_long</c>.
        /// </summary>
        NormalizedRect RectTransformationCalculator_FaceLandmarksNextFrame(NormalizedRect rect, int imageW, int imageH)
        {
            if (imageW <= 0 || imageH <= 0)
                return new NormalizedRect();

            float width = rect.Width;
            float height = rect.Height;
            float rotation = rect.Rotation;
            float xCenter = rect.XCenter;
            float yCenter = rect.YCenter;

            float longSidePx = Mathf.Max(width * imageW, height * imageH);
            width = longSidePx / imageW;
            height = longSidePx / imageH;
            width *= kFaceLandmarksNextFrameRoiScale;
            height *= kFaceLandmarksNextFrameRoiScale;

            return new NormalizedRect
            {
                XCenter = xCenter,
                YCenter = yCenter,
                Width = width,
                Height = height,
                Rotation = rotation,
                RectId = rect.RectId,
            };
        }

        static NormalizedRect AllowIf_FaceNextFrameRect(bool facePresence, NormalizedRect rectWhenPresent)
        {
            return facePresence ? rectWhenPresent : new NormalizedRect();
        }

        /// <summary>
        /// Equivalent to <c>PreviousLoopbackCalculator</c>: returns the previous-frame rect vector as the LOOP input.
        /// </summary>
        List<NormalizedRect> PreviousLoopbackCalculator(Mat image, List<NormalizedRect> loopFaceRects)
        {
            // The image is unused; this returns only a copy of the looped-back rect vector, matching the Pose and Hand pattern.
            _ = image;
            List<NormalizedRect> copy = _previousLoopbackCopyScratch;
            copy.Clear();
            if (loopFaceRects != null)
                copy.AddRange(loopFaceRects);
            return copy;
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
        /// Equivalent to the <c>smooth_landmarks</c> block of <c>MultiFaceLandmarksDetectorGraph</c> in <c>face_landmarks_detector_graph.cc</c>.
        /// This covers the outer <see cref="GetNormalizedLandmarkListVectorItemCalculator"/> / <see cref="ImagePropertiesCalculator"/>,
        /// the in-class One Euro filter corresponding to <c>LandmarksSmoothingCalculator</c>, and the outer
        /// <see cref="ConcatenateNormalizedLandmarkListVectorCalculator"/>.
        /// </summary>
        sealed class FaceLandmarksSmoothingPipeline
        {
            const float kMinAllowedObjectScale = 1e-6f;

            readonly FaceLandmarksOneEuroFilter[] _fx;
            readonly FaceLandmarksOneEuroFilter[] _fy;
            readonly FaceLandmarksOneEuroFilter[] _fz;

            public FaceLandmarksSmoothingPipeline()
            {
                int n = kFaceMeshWithIrisLandmarksNum;
                _fx = new FaceLandmarksOneEuroFilter[n];
                _fy = new FaceLandmarksOneEuroFilter[n];
                _fz = new FaceLandmarksOneEuroFilter[n];
                for (int i = 0; i < n; i++)
                {
                    _fx[i] = FaceLandmarksOneEuroFilter.Create(kFaceLandmarksSmoothingDefaultFrequency,
                        kFaceLandmarksSmoothingOneEuroMinCutoff, kFaceLandmarksSmoothingOneEuroBeta,
                        kFaceLandmarksSmoothingOneEuroDerivateCutoff);
                    _fy[i] = FaceLandmarksOneEuroFilter.Create(kFaceLandmarksSmoothingDefaultFrequency,
                        kFaceLandmarksSmoothingOneEuroMinCutoff, kFaceLandmarksSmoothingOneEuroBeta,
                        kFaceLandmarksSmoothingOneEuroDerivateCutoff);
                    _fz[i] = FaceLandmarksOneEuroFilter.Create(kFaceLandmarksSmoothingDefaultFrequency,
                        kFaceLandmarksSmoothingOneEuroMinCutoff, kFaceLandmarksSmoothingOneEuroBeta,
                        kFaceLandmarksSmoothingOneEuroDerivateCutoff);
                }
            }

            public void ResetAll()
            {
                foreach (var f in _fx) f.Reset();
                foreach (var f in _fy) f.Reset();
                foreach (var f in _fz) f.Reset();
            }

            /// <summary>Post-loop step: smooths only the first vector element with One Euro filtering and replaces that first element.</summary>
            public void ApplyPostEndLoop(Mat image, List<FaceResult> merged)
            {
                if (image == null || merged == null || merged.Count < 1)
                    return;

                FaceResult p = merged[kFaceSmoothLandmarksVectorItemIndex];
                if (!p.FacePresence)
                    return;

                (int iw, int ih) = ImagePropertiesCalculator(image);
                if (iw <= 0 || ih <= 0)
                    return;

                Vec3f[] item = GetNormalizedLandmarkListVectorItemCalculator(merged);
                if (item == null || item.Length != kFaceMeshWithIrisLandmarksNum)
                    return;

                long timestampNs = (long)Environment.TickCount * 1_000_000L;
                Vec3f[] smoothed = LandmarksSmoothingCalculator_FaceNorm(item, timestampNs, iw, ih, p.NextFrameRect);
                merged[kFaceSmoothLandmarksVectorItemIndex] =
                    ConcatenateNormalizedLandmarkListVectorCalculator(merged, smoothed);
            }

            /// <summary>Equivalent to the normalized <c>LandmarksSmoothingCalculator</c> path in <c>landmarks_smoothing_calculator.cc</c>.</summary>
            Vec3f[] LandmarksSmoothingCalculator_FaceNorm(Vec3f[] normLm, long timestampNs, int imageWidth, int imageHeight,
                NormalizedRect roi)
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
                    xPx = _fx[i].Apply(timestampNs, xPx, valueScale, 1.0);
                    yPx = _fy[i].Apply(timestampNs, yPx, valueScale, 1.0);
                    zPx = _fz[i].Apply(timestampNs, zPx, valueScale, 1.0);
                    o[i] = new Vec3f(
                        (float)(xPx / imageWidth),
                        (float)(yPx / imageHeight),
                        (float)(zPx / imageWidth));
                }

                return o;
            }

            static float GetObjectScaleNormalizedRoi(NormalizedRect roi, int imageWidth, int imageHeight)
            {
                float w = roi.Width * imageWidth;
                float h = roi.Height * imageHeight;
                return (w + h) * 0.5f;
            }

            sealed class FaceLandmarksOneEuroFilter
            {
                const long kUninitializedTimestamp = -1;
                const double kEpsilon = 1e-6;

                double _frequency;
                readonly double _minCutoff;
                readonly double _beta;
                readonly double _derivateCutoff;
                long _lastTimeNs;
                FaceLandmarksLowPassFilter _x;
                FaceLandmarksLowPassFilter _dx;

                FaceLandmarksOneEuroFilter(double frequency, double minCutoff, double beta, double derivateCutoff)
                {
                    _frequency = frequency;
                    _minCutoff = minCutoff;
                    _beta = beta;
                    _derivateCutoff = derivateCutoff;
                    _lastTimeNs = kUninitializedTimestamp;
                    _x = FaceLandmarksLowPassFilter.Create();
                    _dx = FaceLandmarksLowPassFilter.Create();
                }

                public static FaceLandmarksOneEuroFilter Create(double frequency, double minCutoff, double beta,
                    double derivateCutoff)
                {
                    if (frequency <= kEpsilon || minCutoff <= kEpsilon || derivateCutoff <= kEpsilon)
                        throw new ArgumentException("OneEuroFilter: frequency / min_cutoff / derivate_cutoff must be positive.");
                    return new FaceLandmarksOneEuroFilter(frequency, minCutoff, beta, derivateCutoff);
                }

                public void Reset()
                {
                    _lastTimeNs = kUninitializedTimestamp;
                    _x = FaceLandmarksLowPassFilter.Create();
                    _dx = FaceLandmarksLowPassFilter.Create();
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

            sealed class FaceLandmarksLowPassFilter
            {
                bool _initialized;
                float _rawValue;
                float _storedValue;

                public static FaceLandmarksLowPassFilter Create()
                {
                    return new FaceLandmarksLowPassFilter();
                }

                public bool HasLastRawValue => _initialized;

                public float LastRawValue() => _rawValue;

                public float ApplyWithAlpha(float rawValue, float alpha)
                {
                    _rawValue = rawValue;
                    if (!_initialized)
                    {
                        _storedValue = rawValue;
                        _initialized = true;
                        return _storedValue;
                    }

                    _storedValue = alpha * rawValue + (1f - alpha) * _storedValue;
                    return _storedValue;
                }
            }
        }

        /// <summary>
        /// Facial geometry estimation corresponding to [MediaPipe](https://github.com/google-ai-edge/mediapipe) Tasks
        /// <c>CreateGeometryPipeline</c>, <c>ScreenToMetricSpaceConverter</c>, and <c>FloatPrecisionProcrustesSolver</c>,
        /// using 468 landmarks without iris points. Geometry metadata is loaded from the upstream <c>geometry_pipeline_metadata_landmarks.pbtxt</c> via <see cref="FaceGeometryLoadPbtxt"/>.
        /// </summary>
        void FaceGeometryEstimateFacialTransformationMatrixes(
            IReadOnlyList<Vec3f[]> multiFaceLandmarks468,
            int frameWidth,
            int frameHeight,
            IList<float[]> rowMajor16OutPerFace)
        {
            if (multiFaceLandmarks468 == null || rowMajor16OutPerFace == null)
                throw new ArgumentNullException();
            if (frameWidth <= 0 || frameHeight <= 0)
                throw new ArgumentException("Frame width and height must be positive.");

            var pcf = new FaceGeometryFrustum(_faceGeometryVerticalFovDegrees, _faceGeometryNearPlane, frameWidth,
                frameHeight);

            for (int i = 0; i < rowMajor16OutPerFace.Count; i++)
            {
                float[] row = rowMajor16OutPerFace[i];
                if (row == null || row.Length != 16)
                    continue;
                Array.Clear(row, 0, 16);

                if (i >= multiFaceLandmarks468.Count)
                    continue;

                Vec3f[] lm = multiFaceLandmarks468[i];
                if (lm == null || lm.Length < _faceGeometryNumVertices)
                    continue;

                if (FaceGeometryIsScreenLandmarkListTooCompact(lm, _faceGeometryNumVertices))
                    continue;

                FaceGeometryTryConvertLandmarksToPose(lm, pcf, row);
            }
        }

        static bool FaceGeometryIsScreenLandmarkListTooCompact(Vec3f[] landmarks, int n)
        {
            float meanX = 0f, meanY = 0f;
            for (int i = 0; i < n; i++)
            {
                meanX += (landmarks[i].Item1 - meanX) / (i + 1);
                meanY += (landmarks[i].Item2 - meanY) / (i + 1);
            }

            float maxSq = 0f;
            for (int i = 0; i < n; i++)
            {
                float dx = landmarks[i].Item1 - meanX;
                float dy = landmarks[i].Item2 - meanY;
                maxSq = Mathf.Max(maxSq, dx * dx + dy * dy);
            }

            const float threshold = 1e-3f;
            return Mathf.Sqrt(maxSq) <= threshold;
        }

        /// <summary>Equivalent to <c>ScreenToMetricSpaceConverter.Convert</c> for the <c>FACE_LANDMARK_PIPELINE</c> path only.</summary>
        void FaceGeometryTryConvertLandmarksToPose(Vec3f[] screenLm, FaceGeometryFrustum pcf, float[] poseRowMajor16)
        {
            int n = _faceGeometryNumVertices;
            var screen = new float[3 * n];
            for (int i = 0; i < n; i++)
            {
                screen[i * 3 + 0] = screenLm[i].Item1;
                screen[i * 3 + 1] = screenLm[i].Item2;
                screen[i * 3 + 2] = screenLm[i].Item3;
            }

            FaceGeometryProjectXYTopLeftOrigin(pcf, screen, n);

            float depthOffset = 0f;
            for (int i = 0; i < n; i++)
                depthOffset += (screen[i * 3 + 2] - depthOffset) / (i + 1);

            var work = new float[3 * n];
            Array.Copy(screen, work, screen.Length);
            FaceGeometryChangeHandedness(work, n);

            if (!FaceGeometryEstimateScaleFromProcrustes(work, n, out float firstScale))
                return;

            Array.Copy(screen, work, screen.Length);
            FaceGeometryMoveAndRescaleZ(pcf, depthOffset, firstScale, work, n);
            FaceGeometryUnprojectXY(pcf, work, n);
            FaceGeometryChangeHandedness(work, n);

            if (!FaceGeometryEstimateScaleFromProcrustes(work, n, out float secondScale))
                return;

            float totalScale = firstScale * secondScale;
            Array.Copy(screen, work, screen.Length);
            FaceGeometryMoveAndRescaleZ(pcf, depthOffset, totalScale, work, n);
            FaceGeometryUnprojectXY(pcf, work, n);
            FaceGeometryChangeHandedness(work, n);

            if (!FaceGeometrySolveWeightedOrthogonalProblemTo4x4(_faceGeometryCanonicalMetricLandmarks, work,
                    _faceGeometryLandmarkWeights, n, poseRowMajor16))
                return;

            FaceGeometryInvertPose4x4InPlace(poseRowMajor16, out float[] inv16);
            FaceGeometryApplyInversePoseToMetricLandmarks(inv16, work, n);
        }

        static void FaceGeometryProjectXYTopLeftOrigin(FaceGeometryFrustum pcf, float[] lm, int n)
        {
            float xScale = pcf.Right - pcf.Left;
            float yScale = pcf.Top - pcf.Bottom;
            float xTrans = pcf.Left;
            float yTrans = pcf.Bottom;

            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float y = 1f - lm[o + 1];
                lm[o + 0] = lm[o + 0] * xScale + xTrans;
                lm[o + 1] = y * yScale + yTrans;
                lm[o + 2] = lm[o + 2] * xScale;
            }
        }

        static void FaceGeometryChangeHandedness(float[] lm, int n)
        {
            for (int i = 0; i < n; i++)
                lm[i * 3 + 2] *= -1f;
        }

        static void FaceGeometryMoveAndRescaleZ(FaceGeometryFrustum pcf, float depthOffset, float scale, float[] lm,
            int n)
        {
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                lm[o + 2] = (lm[o + 2] - depthOffset + pcf.NearPlane) / scale;
            }
        }

        static void FaceGeometryUnprojectXY(FaceGeometryFrustum pcf, float[] lm, int n)
        {
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float z = lm[o + 2];
                lm[o + 0] = lm[o + 0] * z / pcf.NearPlane;
                lm[o + 1] = lm[o + 1] * z / pcf.NearPlane;
            }
        }

        bool FaceGeometryEstimateScaleFromProcrustes(float[] landmarks, int n, out float scale)
        {
            scale = 0f;
            if (!FaceGeometrySolveWeightedOrthogonalProblemTo4x4(_faceGeometryCanonicalMetricLandmarks, landmarks,
                    _faceGeometryLandmarkWeights, n, out float[] tmp16))
                return false;
            float c0 = tmp16[0], c1 = tmp16[4], c2 = tmp16[8];
            scale = Mathf.Sqrt(c0 * c0 + c1 * c1 + c2 * c2);
            return scale > 1e-9f;
        }

        static void FaceGeometryApplyInversePoseToMetricLandmarks(float[] inv16, float[] metricLm, int n)
        {
            Span<float> tmp = stackalloc float[3];
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float x = metricLm[o + 0];
                float y = metricLm[o + 1];
                float z = metricLm[o + 2];
                tmp[0] = inv16[0] * x + inv16[1] * y + inv16[2] * z + inv16[3];
                tmp[1] = inv16[4] * x + inv16[5] * y + inv16[6] * z + inv16[7];
                tmp[2] = inv16[8] * x + inv16[9] * y + inv16[10] * z + inv16[11];
                metricLm[o + 0] = tmp[0];
                metricLm[o + 1] = tmp[1];
                metricLm[o + 2] = tmp[2];
            }
        }

        static void FaceGeometryInvertPose4x4InPlace(float[] m, out float[] inv)
        {
            inv = new float[16];
            using (var a = new Mat(4, 4, CvType.CV_32FC1))
            using (var b = new Mat(4, 4, CvType.CV_32FC1))
            {
                a.put(0, 0, m.AsSpan(0, 16));
                // DECOMP_LU: non-zero on success, zero when singular, matching OpenCVForUnity Core.invert behavior.
                if (Core.invert(a, b, Core.DECOMP_LU) == 0.0)
                {
                    Array.Copy(m, inv, 16);
                    return;
                }

                b.get(0, 0, inv.AsSpan(0, 16));
            }
        }

        static bool FaceGeometrySolveWeightedOrthogonalProblemTo4x4(float[] source3N, float[] target3N,
            float[] weights, int n, float[] transformRowMajor16)
        {
            if (!FaceGeometrySolveWeightedOrthogonalProblemTo4x4(source3N, target3N, weights, n, out float[] tmp))
                return false;
            Array.Copy(tmp, transformRowMajor16, 16);
            return true;
        }

        static bool FaceGeometrySolveWeightedOrthogonalProblemTo4x4(float[] source3N, float[] target3N,
            float[] weights, int n, out float[] transformRowMajor16)
        {
            transformRowMajor16 = new float[16];
            var sqrtW = new float[n];
            float totalW = 0f;
            for (int i = 0; i < n; i++)
            {
                if (weights[i] < 0f)
                    return false;
                sqrtW[i] = Mathf.Sqrt(weights[i]);
                totalW += weights[i];
            }

            if (totalW <= 1e-9f)
                return false;

            var wSrc = new float[3 * n];
            var wTgt = new float[3 * n];
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float sw = sqrtW[i];
                wSrc[o + 0] = source3N[o + 0] * sw;
                wSrc[o + 1] = source3N[o + 1] * sw;
                wSrc[o + 2] = source3N[o + 2] * sw;
                wTgt[o + 0] = target3N[o + 0] * sw;
                wTgt[o + 1] = target3N[o + 1] * sw;
                wTgt[o + 2] = target3N[o + 2] * sw;
            }

            var twice = new float[3 * n];
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float s = sqrtW[i];
                twice[o + 0] = wSrc[o + 0] * s;
                twice[o + 1] = wSrc[o + 1] * s;
                twice[o + 2] = wSrc[o + 2] * s;
            }

            float cx = 0f, cy = 0f, cz = 0f;
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                cx += twice[o + 0];
                cy += twice[o + 1];
                cz += twice[o + 2];
            }

            cx /= totalW;
            cy /= totalW;
            cz /= totalW;

            var centeredWSrc = new float[3 * n];
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                centeredWSrc[o + 0] = wSrc[o + 0] - cx * sqrtW[i];
                centeredWSrc[o + 1] = wSrc[o + 1] - cy * sqrtW[i];
                centeredWSrc[o + 2] = wSrc[o + 2] - cz * sqrtW[i];
            }

            Span<float> design = stackalloc float[9];
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    float s = 0f;
                    for (int k = 0; k < n; k++)
                    {
                        int o = k * 3;
                        s += wTgt[o + r] * centeredWSrc[o + c];
                    }

                    design[r * 3 + c] = s;
                }
            }

            if (!FaceGeometryComputeOptimalRotation3x3(design, out float[] rot3))
                return false;

            if (!FaceGeometryComputeOptimalScale(centeredWSrc, wSrc, wTgt, rot3, n, out float sc))
                return false;

            Span<float> rAndS = stackalloc float[9];
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                    rAndS[r * 3 + c] = sc * rot3[r * 3 + c];
            }

            Span<float> diffSum = stackalloc float[3];
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float px = rAndS[0] * wSrc[o + 0] + rAndS[1] * wSrc[o + 1] + rAndS[2] * wSrc[o + 2];
                float py = rAndS[3] * wSrc[o + 0] + rAndS[4] * wSrc[o + 1] + rAndS[5] * wSrc[o + 2];
                float pz = rAndS[6] * wSrc[o + 0] + rAndS[7] * wSrc[o + 1] + rAndS[8] * wSrc[o + 2];
                diffSum[0] += (wTgt[o + 0] - px) * sqrtW[i];
                diffSum[1] += (wTgt[o + 1] - py) * sqrtW[i];
                diffSum[2] += (wTgt[o + 2] - pz) * sqrtW[i];
            }

            float tx = diffSum[0] / totalW;
            float ty = diffSum[1] / totalW;
            float tz = diffSum[2] / totalW;

            transformRowMajor16[0] = rAndS[0];
            transformRowMajor16[1] = rAndS[1];
            transformRowMajor16[2] = rAndS[2];
            transformRowMajor16[3] = tx;
            transformRowMajor16[4] = rAndS[3];
            transformRowMajor16[5] = rAndS[4];
            transformRowMajor16[6] = rAndS[5];
            transformRowMajor16[7] = ty;
            transformRowMajor16[8] = rAndS[6];
            transformRowMajor16[9] = rAndS[7];
            transformRowMajor16[10] = rAndS[8];
            transformRowMajor16[11] = tz;
            transformRowMajor16[12] = 0f;
            transformRowMajor16[13] = 0f;
            transformRowMajor16[14] = 0f;
            transformRowMajor16[15] = 1f;
            return true;
        }

        static bool FaceGeometryComputeOptimalScale(float[] centeredWSrc, float[] wSrc, float[] wTgt, float[] rot3,
            int n, out float scale)
        {
            scale = 0f;
            float num = 0f, den = 0f;
            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float rsx = rot3[0] * centeredWSrc[o + 0] + rot3[1] * centeredWSrc[o + 1] +
                    rot3[2] * centeredWSrc[o + 2];
                float rsy = rot3[3] * centeredWSrc[o + 0] + rot3[4] * centeredWSrc[o + 1] +
                    rot3[5] * centeredWSrc[o + 2];
                float rsz = rot3[6] * centeredWSrc[o + 0] + rot3[7] * centeredWSrc[o + 1] +
                    rot3[8] * centeredWSrc[o + 2];
                num += rsx * wTgt[o + 0] + rsy * wTgt[o + 1] + rsz * wTgt[o + 2];
                den += centeredWSrc[o + 0] * wSrc[o + 0] + centeredWSrc[o + 1] * wSrc[o + 1] +
                    centeredWSrc[o + 2] * wSrc[o + 2];
            }

            if (den <= 1e-9f || num / den <= 1e-9f)
                return false;
            scale = num / den;
            return true;
        }

        static bool FaceGeometryComputeOptimalRotation3x3(ReadOnlySpan<float> designRowMajor3x3,
            out float[] rotationRowMajor3x3)
        {
            rotationRowMajor3x3 = new float[9];
            float norm = 0f;
            for (int i = 0; i < 9; i++)
                norm += designRowMajor3x3[i] * designRowMajor3x3[i];
            if (norm <= 1e-18f)
                return false;

            using (var src = new Mat(3, 3, CvType.CV_32FC1))
            using (var w = new Mat())
            using (var u = new Mat(3, 3, CvType.CV_32FC1))
            using (var vt = new Mat(3, 3, CvType.CV_32FC1))
            {
                src.put(0, 0, designRowMajor3x3);
                Core.SVDecomp(src, w, u, vt);

                Span<float> uArr = stackalloc float[9];
                Span<float> vtArr = stackalloc float[9];
                u.get(0, 0, uArr);
                vt.get(0, 0, vtArr);

                if (FaceGeometryDeterminant3x3(uArr) * FaceGeometryDeterminant3x3(vtArr) < 0f)
                {
                    uArr[2] *= -1f;
                    uArr[5] *= -1f;
                    uArr[8] *= -1f;
                }

                FaceGeometryMultiply3x3RowMajor(uArr, vtArr, rotationRowMajor3x3.AsSpan());
            }

            return true;
        }

        static float FaceGeometryDeterminant3x3(ReadOnlySpan<float> m)
        {
            return m[0] * (m[4] * m[8] - m[5] * m[7])
               - m[1] * (m[3] * m[8] - m[5] * m[6])
               + m[2] * (m[3] * m[7] - m[4] * m[6]);
        }

        static void FaceGeometryMultiply3x3RowMajor(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
        {
            for (int r = 0; r < 3; r++)
            {
                for (int col = 0; col < 3; col++)
                {
                    float s = 0f;
                    for (int k = 0; k < 3; k++)
                        s += a[r * 3 + k] * b[k * 3 + col];
                    c[r * 3 + col] = s;
                }
            }
        }

        /// <summary>
        /// Equivalent to <c>FaceGeometryEnvGeneratorCalculator</c> from <c>face_geometry_env_generator_calculator.cc</c>.
        /// Returns the same default perspective used when the Tasks side <c>ENVIRONMENT</c> input is not connected: vertical FOV 63 degrees and near plane 1.
        /// Returns zeros when geometry output is disabled, assuming <see cref="FaceGeometryFrustum"/> will not be used.
        /// </summary>
        /// <param name="geometryOutputEnabled">When true, generates the default environment corresponding to enabled <c>output_facial_transformation_matrixes</c>.</param>
        static void FaceGeometryEnvGeneratorCalculator(bool geometryOutputEnabled, out float verticalFovDegrees,
            out float nearPlane)
        {
            if (geometryOutputEnabled)
            {
                verticalFovDegrees = 63f;
                nearPlane = 1f;
            }
            else
            {
                verticalFovDegrees = 0f;
                nearPlane = 0f;
            }
        }

        /// <summary>
        /// Loads the canonical mesh and Procrustes weights from the upstream <c>geometry_pipeline_metadata_landmarks.pbtxt</c>.
        /// </summary>
        static void FaceGeometryLoadPbtxt(string path, out float[] canonical3N, out float[] weights, out int numVertices)
        {
            string text = File.ReadAllText(path);
            int idxMesh = text.IndexOf("canonical_mesh:", StringComparison.Ordinal);
            if (idxMesh < 0)
                throw new InvalidDataException("canonical_mesh was not found.");

            int idxIndex = text.IndexOf("index_buffer:", idxMesh, StringComparison.Ordinal);
            if (idxIndex < 0)
                throw new InvalidDataException("index_buffer was not found.");

            string meshChunk = text.Substring(idxMesh, idxIndex - idxMesh);
            var vbMatch = Regex.Matches(meshChunk, @"vertex_buffer:\s*([-0-9.eE+]+)", RegexOptions.Multiline);
            if (vbMatch.Count < 5 || vbMatch.Count % 5 != 0)
                throw new InvalidDataException(
                    "The vertex_buffer count is invalid. Specify the geometry_pipeline_metadata_landmarks.pbtxt file for Tasks.");

            int vbCount = vbMatch.Count;
            numVertices = vbCount / 5;
            var vb = new float[vbCount];
            for (int i = 0; i < vbCount; i++)
                vb[i] = float.Parse(vbMatch[i].Groups[1].Value, CultureInfo.InvariantCulture);

            canonical3N = new float[3 * numVertices];
            for (int vi = 0; vi < numVertices; vi++)
            {
                int b = vi * 5;
                canonical3N[vi * 3 + 0] = vb[b + 0];
                canonical3N[vi * 3 + 1] = vb[b + 1];
                canonical3N[vi * 3 + 2] = vb[b + 2];
            }

            weights = new float[numVertices];
            foreach (Match m in Regex.Matches(text,
                         @"procrustes_landmark_basis\s*\{\s*landmark_id:\s*(\d+)\s+weight:\s*([-0-9.eE+]+)\s*\}"))
            {
                int id = int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
                float w = float.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
                if (id >= 0 && id < numVertices)
                    weights[id] = w;
            }
        }

        /// <summary>Perspective camera frustum used by the geometry pipeline.</summary>
        readonly struct FaceGeometryFrustum
        {
            public readonly float Left;
            public readonly float Right;
            public readonly float Bottom;
            public readonly float Top;
            public readonly float NearPlane;

            public FaceGeometryFrustum(float verticalFovDeg, float nearPlane, int frameWidth, int frameHeight)
            {
                float heightAtNear = 2f * nearPlane * Mathf.Tan(0.5f * kFaceGeometryDegreesToRadians * verticalFovDeg);
                float widthAtNear = frameWidth * heightAtNear / frameHeight;
                Left = -0.5f * widthAtNear;
                Right = 0.5f * widthAtNear;
                Bottom = -0.5f * heightAtNear;
                Top = 0.5f * heightAtNear;
                NearPlane = nearPlane;
            }
        }

        /// <summary>
        /// Equivalent to <c>FaceGeometryFromLandmarksGraph</c> from <c>face_geometry_from_landmarks_graph.cc</c>.
        /// This method only invokes downstream calculators in upstream order.
        ///
        /// Correspondence to the upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) implementation:
        /// - <c>FaceGeometryEnvGeneratorCalculator</c> when the side input is disconnected -> <see cref="FaceGeometryEnvGeneratorCalculator"/> and <see cref="FaceGeometryLoadPbtxt"/>
        /// - BeginLoopNormalizedLandmarkListVectorCalculator → <see cref="BeginLoopNormalizedLandmarkListVectorCalculator_FaceGeometry"/>
        /// - <c>SplitNormalizedLandmarkListCalculator</c> (slices 468 or 478 landmarks to match the metadata vertex count) -> <see cref="SplitNormalizedLandmarkListCalculator_FaceGeometry"/>
        /// - <c>EndLoopNormalizedLandmarkListVectorCalculator</c> -> aggregated into <c>_faceGeomMultiNoIrisScratch</c> inside <see cref="FaceGeometryFromLandmarksGraph"/>
        /// - FaceGeometryPipelineCalculator → <see cref="FaceGeometryPipelineCalculator_Process"/>
        /// </summary>
        void FaceGeometryFromLandmarksGraph(Mat image, List<FaceResult> faces)
        {
            if (image == null || faces == null || _faceGeometryCanonicalMetricLandmarks == null)
                return;

            (int w, int h) = ImagePropertiesCalculator(image);
            int requiredVertexCount = _faceGeometryNumVertices > 0 ? _faceGeometryNumVertices : kFaceMeshLandmarksNum;
            List<Vec3f[]> multiNoIris = _faceGeomMultiNoIrisScratch;
            multiNoIris.Clear();
            if (faces != null)
            {
                foreach (var f in BeginLoopNormalizedLandmarkListVectorCalculator_FaceGeometry(faces))
                    multiNoIris.Add(SplitNormalizedLandmarkListCalculator_FaceGeometry(f.NormLandmarks, requiredVertexCount));
            }

            List<float[]> poseRows = _faceGeometryPoseRowsScratch;
            poseRows.Clear();
            for (int i = 0; i < faces.Count; i++)
                poseRows.Add(new float[16]);
            FaceGeometryPipelineCalculator_Process(multiNoIris, w, h, poseRows);
            for (int i = 0; i < faces.Count; i++)
            {
                FaceResult fr = faces[i];
                fr.FacialPoseTransformRowMajor16 = poseRows[i];
                faces[i] = fr;
            }
        }

        /// <summary>Equivalent to <c>BeginLoopNormalizedLandmarkListVectorCalculator</c>, serving as the geometry iteration entry point.</summary>
        static IEnumerable<FaceResult> BeginLoopNormalizedLandmarkListVectorCalculator_FaceGeometry(List<FaceResult> faces)
        {
            if (faces == null)
                yield break;
            foreach (var f in faces)
                yield return f;
        }

        /// <summary>
        /// Equivalent to <c>SplitNormalizedLandmarkListCalculator</c>.
        /// Slices the leading landmark sequence to match the FaceGeometry metadata vertex count of 468 or 478.
        /// </summary>
        static Vec3f[] SplitNormalizedLandmarkListCalculator_FaceGeometry(Vec3f[] normLandmarks478, int requiredVertexCount)
        {
            int dstCount = requiredVertexCount > 0 ? requiredVertexCount : kFaceMeshLandmarksNum;
            var dst = new Vec3f[dstCount];
            if (normLandmarks478 == null)
                return dst;
            int n = Math.Min(dstCount, normLandmarks478.Length);
            for (int i = 0; i < n; i++)
                dst[i] = normLandmarks478[i];
            return dst;
        }

        /// <summary>Equivalent to <c>FaceGeometryPipelineCalculator</c>, transforming <c>MULTI_FACE_LANDMARKS</c> into <c>MULTI_FACE_GEOMETRY</c>.</summary>
        void FaceGeometryPipelineCalculator_Process(
            List<Vec3f[]> multiFaceLandmarksNoIris,
            int frameWidth,
            int frameHeight,
            IList<float[]> rowMajor16PerFace)
        {
            FaceGeometryEstimateFacialTransformationMatrixes(multiFaceLandmarksNoIris, frameWidth, frameHeight,
                rowMajor16PerFace);
        }

        /// <summary>
        /// Bundles the main landmark <see cref="Mat"/> together with any enabled optional output <see cref="Mat"/> values.
        /// Index 0 is the landmark output. When optional outputs are enabled, the result always has 3 elements and slots (1) and (2) remain fixed.
        /// </summary>
        Mat[] BuildPackedOutputMats(List<FaceResult> faces)
        {
            int faceCount = faces?.Count ?? 0;
            bool fixedTriple = _outputFaceBlendshapes || _outputFacialTransformationMatrixes;

            lock (_lockObject)
            {
                if (faceCount == 0)
                {
                    if (!fixedTriple)
                        return new[] { new Mat() };
                    return new[] { new Mat(), new Mat(), new Mat() };
                }

                Mat main = PackMainLandmarksMatToBufferUnlocked(faces, faceCount);
                if (!fixedTriple)
                    return new[] { main };

                Mat blendSlot = _outputFaceBlendshapes
                    ? PackBlendshapesMatToBufferUnlocked(faces, faceCount)
                    : new Mat();
                Mat poseSlot = _outputFacialTransformationMatrixes
                    ? PackFacialPoseMatToBufferUnlocked(faces, faceCount)
                    : new Mat();
                return new[] { main, blendSlot, poseSlot };
            }
        }

        /// <summary>
        /// Writes the main per-face result rows into <see cref="_outputBuffer"/> and returns the leading <paramref name="faceCount"/> rows via <see cref="Mat.rowRange"/>.
        /// </summary>
        Mat PackMainLandmarksMatToBufferUnlocked(List<FaceResult> faces, int faceCount)
        {
            int L = FaceLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            int R = FaceLandmarkerEstimationData.ELEMENT_COUNT;

            if (_outputBuffer == null
                || _outputBuffer.rows() < faceCount
                || _outputBuffer.cols() != R
                || _outputBuffer.type() != CvType.CV_32FC1)
            {
                _outputBuffer?.Dispose();
                int rows = Math.Max(faceCount, _numFaces);
                _outputBuffer = new Mat(rows, R, CvType.CV_32FC1);
            }

            Mat packed = _outputBuffer;
            Span<float> row = stackalloc float[FaceLandmarkerEstimationData.ELEMENT_COUNT];

            for (int i = 0; i < faceCount; i++)
            {
                row.Clear();

                FaceResult f = faces[i];
                Vec3f[] lm = f.NormLandmarks;

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

                packed.put(i, 0, row);
            }

            return packed.rowRange(0, faceCount);
        }

        Mat PackBlendshapesMatToBufferUnlocked(List<FaceResult> faces, int faceCount)
        {
            int C = kFaceBlendshapeCoefficientCount;
            if (_outputBlendshapesBuffer == null
                || _outputBlendshapesBuffer.rows() < faceCount
                || _outputBlendshapesBuffer.cols() != C
                || _outputBlendshapesBuffer.type() != CvType.CV_32FC1)
            {
                _outputBlendshapesBuffer?.Dispose();
                _outputBlendshapesBuffer = new Mat(Math.Max(faceCount, _numFaces), C, CvType.CV_32FC1);
            }

            Span<float> row = stackalloc float[C];
            for (int i = 0; i < faceCount; i++)
            {
                row.Clear();
                float[] c = faces[i].BlendshapeCoefficients;
                if (c != null)
                {
                    int n = Math.Min(c.Length, C);
                    for (int j = 0; j < n; j++)
                        row[j] = c[j];
                }

                _outputBlendshapesBuffer.put(i, 0, row);
            }

            return _outputBlendshapesBuffer.rowRange(0, faceCount);
        }

        Mat PackFacialPoseMatToBufferUnlocked(List<FaceResult> faces, int faceCount)
        {
            const int C = 16;
            if (_outputFacialPoseBuffer == null
                || _outputFacialPoseBuffer.rows() < faceCount
                || _outputFacialPoseBuffer.cols() != C
                || _outputFacialPoseBuffer.type() != CvType.CV_32FC1)
            {
                _outputFacialPoseBuffer?.Dispose();
                _outputFacialPoseBuffer = new Mat(Math.Max(faceCount, _numFaces), C, CvType.CV_32FC1);
            }

            Span<float> row = stackalloc float[C];
            for (int i = 0; i < faceCount; i++)
            {
                row.Clear();
                float[] p = faces[i].FacialPoseTransformRowMajor16;
                if (p != null)
                {
                    int n = Math.Min(p.Length, C);
                    for (int j = 0; j < n; j++)
                        row[j] = p[j];
                }

                _outputFacialPoseBuffer.put(i, 0, row);
            }

            return _outputFacialPoseBuffer.rowRange(0, faceCount);
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                _faceDetectorNet?.Dispose();
                _faceLandmarksNet?.Dispose();
                _faceBlendshapesNet?.Dispose();
                _faceDetectorForwardOutputList.Clear();
                _faceLandmarksForwardOutputList.Clear();
                _faceBlendshapesForwardOutputList.Clear();
                _faceBlendshapesInputBlob?.Dispose();
                _faceBlendshapesInputBlob = null;
                _outputBlendshapesBuffer?.Dispose();
                _outputBlendshapesBuffer = null;
                _outputFacialPoseBuffer?.Dispose();
                _outputFacialPoseBuffer = null;
                _faceDetectorLetterboxBgr?.Dispose();
                _faceDetectorLetterboxBgr = null;
                _faceDetectorInferenceRgb8u?.Dispose();
                _faceDetectorInferenceRgb8u = null;
                _faceDetectorInferenceBlob?.Dispose();
                _faceDetectorInferenceBlob = null;
                _faceDetectorInferenceBlobHxW = null;
                _faceDetectorAnchorsBuffer?.Dispose();
                _faceDetectorAnchorsBuffer = null;
                _faceDetectorDecodedBoxesNx16?.Dispose();
                _faceDetectorDecodedBoxesNx16 = null;
                _faceDetectorWarpSrcPts?.Dispose();
                _faceDetectorWarpSrcPts = null;
                _faceDetectorWarpDstPts?.Dispose();
                _faceDetectorWarpDstPts = null;
                _faceTensorsToDetectionsWorking?.Dispose();
                _faceTensorsToDetectionsWorking = null;
                _faceNmsIndices?.Dispose();
                _faceNmsIndices = null;
                _faceWnmsMergedBoxXywh?.Dispose();
                _faceWnmsMergedBoxXywh = null;
                _faceWnmsMergedDecodedNx16?.Dispose();
                _faceWnmsMergedDecodedNx16 = null;
                _faceWnmsMergedScore?.Dispose();
                _faceWnmsMergedScore = null;
                _faceLmWarpSrcPts?.Dispose();
                _faceLmWarpSrcPts = null;
                _faceLmWarpDstPts?.Dispose();
                _faceLmWarpDstPts = null;
                _faceLmWarpedBgr?.Dispose();
                _faceLmWarpedBgr = null;
                _faceLmWarpedRgb?.Dispose();
                _faceLmWarpedRgb = null;
                _faceLandmarksInferenceBlob?.Dispose();
                _faceLandmarksInferenceBlob = null;
                _faceLandmarksInferenceBlobHxW = null;
                _faceLandmarksSmoothingPipeline?.ResetAll();
                _outputBuffer?.Dispose();
                _outputBuffer = null;
                _faceNumpyClipLo?.Dispose();
                _faceNumpyClipLo = null;
                _faceNumpyClipHi?.Dispose();
                _faceNumpyClipHi = null;
                _faceDetectorTransposeBuffer?.Dispose();
                _faceDetectorTransposeBuffer = null;
                _faceDetectorScoreColumnBuffer?.Dispose();
                _faceDetectorScoreColumnBuffer = null;
                _faceLetterboxResizeScratch?.Dispose();
                _faceLetterboxResizeScratch = null;
            }

            base.Dispose(disposing);
        }
    }
}
#endif
