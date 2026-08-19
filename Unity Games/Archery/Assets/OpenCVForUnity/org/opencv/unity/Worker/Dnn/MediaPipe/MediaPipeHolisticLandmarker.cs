#if !UNITY_WSA_10_0
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Worker;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using OpenCVForUnity.UnityIntegration.Worker.Utils;
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe
{
    /// <summary>
    /// Processing worker that reproduces the holistic landmarking graph logic of
    /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HolisticLandmarker
    /// on top of the OpenCV for Unity Dnn module.
    /// </summary>
    public partial class MediaPipeHolisticLandmarker : DnnInferenceWorkerBase
    {
        /// <summary>
        /// Fixed number of <see cref="Mat"/> output slots returned by <see cref="RunCoreProcessing"/> /
        /// <see cref="Detect"/>.
        /// Disabled outputs are represented by empty <see cref="Mat"/> instances (<see cref="Mat.empty"/>),
        /// matching the empty-slot convention used by <see cref="MediaPipeFaceLandmarker"/>.
        /// </summary>
        /// <remarks>
        /// The public <see cref="Mat"/>[] aligns with upstream <c>holistic_landmarker_graph.cc</c> outward streams.
        /// For each stream type, normalized / world data are folded into a <b>single packed row</b>, matching
        /// <see cref="MediaPipePoseLandmarker.PoseLandmarkerEstimationData"/> /
        /// <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/> where applicable.
        /// <list type="table">
        /// <listheader><term>Index</term><description>Contents</description></listheader>
        /// <item><term>0</term><description><c>POSE_LANDMARKS</c> (one row of <see cref="MediaPipePoseLandmarker.PoseLandmarkerEstimationData"/>)</description></item>
        /// <item><term>1</term><description><c>LEFT_HAND_LANDMARKS</c> (one row of <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/>)</description></item>
        /// <item><term>2</term><description><c>RIGHT_HAND_LANDMARKS</c> (one row of <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/>)</description></item>
        /// <item><term>3</term><description><c>FACE_LANDMARKS</c> (one row of <see cref="MediaPipeFaceLandmarker.FaceLandmarkerEstimationData"/>)</description></item>
        /// <item><term>4</term><description><c>POSE_SEGMENTATION_MASK</c>. When enabled, this slot is a <see cref="Mat.rowRange"/> view over the internal packing buffer (valid until the next <see cref="Execute"/>, per <see cref="ProcessingWorkerBase"/> contract). See <see cref="BuildHolisticPackedOutputs"/> remarks.</description></item>
        /// <item><term>5</term><description><c>FACE_BLENDSHAPES</c> (one row with the same column count as <see cref="MediaPipeFaceLandmarker.FaceBlendshapeEstimationData"/>)</description></item>
        /// </list>
        /// </remarks>
        const int HolisticPackedOutputSlotCount = 6;

        /// <summary>
        /// Execution modes compatible with the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HolisticLandmarker task.
        /// This enum corresponds to the task running mode that switches between
        /// per-image processing and stateful video processing.
        /// </summary>
        public enum MediaPipeHolisticRunningMode : byte
        {
            /// <summary>
            /// IMAGE mode.
            /// Runs holistic landmarking for each input image without
            /// reusing loopback tracking state from previous frames.
            /// </summary>
            IMAGE = 0,

            /// <summary>
            /// VIDEO mode.
            /// Assumes a frame sequence and reuses holistic tracking state from the previous
            /// frame so detector work can be skipped on frames where tracking remains valid.
            /// </summary>
            VIDEO = 1,
        }

        /// <summary>
        /// Equivalent to upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>HolisticPoseTrackingRequest</c>, which <c>holistic_landmarker_graph.cc</c> passes to
        /// <c>TrackHolisticPose</c>.
        /// </summary>
        /// <remarks>
        /// <see cref="NeedLandmarks"/> becomes true whenever any of the corresponding outputs is enabled,
        /// because pose 2D landmarks are still required internally even for hand-only or face-only requests.
        /// </remarks>
        readonly struct HolisticPoseTrackingRequest
        {
            /// <summary>Whether pose normalized landmarks are required by the internal pipeline.</summary>
            public bool NeedLandmarks { get; }

            /// <summary>Whether the pose-world or hand-world path is required.</summary>
            public bool NeedWorldLandmarks { get; }

            /// <summary>Whether the pose segmentation mask is required.</summary>
            public bool NeedSegmentationMask { get; }

            /// <summary>
            /// Builds the request from output flags using the same conditions as upstream
            /// [MediaPipe](https://github.com/google-ai-edge/mediapipe).
            /// </summary>
            public static HolisticPoseTrackingRequest FromOutputFlags(
                bool outputPoseLandmarks,
                bool outputPoseWorldLandmarks,
                bool outputLeftHandLandmarks,
                bool outputRightHandLandmarks,
                bool outputLeftHandWorldLandmarks,
                bool outputRightHandWorldLandmarks,
                bool outputPoseSegmentationMasks,
                bool outputFaceLandmarks,
                bool outputFaceBlendshapes)
            {
                bool anyHand = outputLeftHandLandmarks || outputRightHandLandmarks
                    || outputLeftHandWorldLandmarks || outputRightHandWorldLandmarks;
                bool anyFace = outputFaceLandmarks || outputFaceBlendshapes;

                bool needLandmarks = outputPoseLandmarks || anyHand || anyFace;
                // Upstream holistic_landmarker_graph.cc sets pose_request.world_landmarks = hands_requested,
                // so POSE_WORLD is still required even for hand 2D-only output.
                bool needWorld = outputPoseWorldLandmarks
                    || outputLeftHandWorldLandmarks
                    || outputRightHandWorldLandmarks
                    || anyHand;
                bool needSeg = outputPoseSegmentationMasks;

                return new HolisticPoseTrackingRequest(needLandmarks, needWorld, needSeg);
            }

            HolisticPoseTrackingRequest(bool needLandmarks, bool needWorldLandmarks, bool needSegmentationMask)
            {
                NeedLandmarks = needLandmarks;
                NeedWorldLandmarks = needWorldLandmarks;
                NeedSegmentationMask = needSegmentationMask;
            }
        }

        /// <summary>Per-hand result from <c>TrackHolisticHand</c> before <see cref="BuildHolisticPackedOutputs"/>.</summary>
        struct HolisticHandTrackFrameResult
        {
            public bool HandPresence;
            /// <summary>The 21 points corresponding to upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>NormalizedLandmark</c>, with the same meaning as <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData.NormLandmarks"/>.</summary>
            public Vec3f[] NormLandmarks;
            public Vec3f[] WorldLandmarks;
            public float Handedness;
            public float PresenceConfidence;
        }

        /// <summary>Per-frame result from <c>TrackHolisticFace</c> before <see cref="BuildHolisticPackedOutputs"/>.</summary>
        struct HolisticFaceTrackFrameResult
        {
            public bool FacePresence;
            public float FacePresenceScore;
            public Vec3f[] NormLandmarks478;
            /// <summary><c>null</c> when the single-face detection path is not reached; otherwise corresponds to the upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>FACE_BLENDSHAPES</c> stream and may be produced by <see cref="MediaPipeHolisticLandmarker.FaceTracking"/> independently of <c>FacePresence</c>.</summary>
            public float[] BlendshapeCoefficients52;
        }

        /// <summary>
        /// One-frame inference result from <see cref="HolisticLandmarkerGraph"/> before <see cref="PackResultsToMats"/>.
        /// This is analogous to <see cref="MediaPipePoseLandmarker"/>'s <c>List&lt;PoseResult&gt;</c>.
        /// </summary>
        /// <remarks>
        /// <see cref="Pose"/>'s <see cref="HolisticPoseTrackFrameResult.SegmentationMaskFull"/> remains intermediate-owned until it is passed to <see cref="BuildHolisticPackedOutputs"/>; after packing it is released or reused according to the worker's internal policy. See <see cref="HolisticPoseTrackFrameResult"/> remarks for details.
        /// </remarks>
        readonly struct HolisticLandmarkerFrameResult
        {
            public HolisticPoseTrackFrameResult Pose { get; }
            public HolisticHandTrackFrameResult LeftHand { get; }
            public HolisticHandTrackFrameResult RightHand { get; }
            public HolisticFaceTrackFrameResult Face { get; }

            public HolisticLandmarkerFrameResult(
                HolisticPoseTrackFrameResult pose,
                HolisticHandTrackFrameResult leftHand,
                HolisticHandTrackFrameResult rightHand,
                HolisticFaceTrackFrameResult face)
            {
                Pose = pose;
                LeftHand = leftHand;
                RightHand = rightHand;
                Face = face;
            }
        }

        readonly MediaPipeHolisticRunningMode _runningMode;

        readonly bool _outputPoseLandmarks;
        readonly bool _outputPoseWorldLandmarks;
        readonly bool _outputLeftHandLandmarks;
        readonly bool _outputRightHandLandmarks;
        readonly bool _outputLeftHandWorldLandmarks;
        readonly bool _outputRightHandWorldLandmarks;
        readonly bool _outputPoseSegmentationMasks;
        readonly bool _outputFaceLandmarks;
        readonly bool _outputFaceBlendshapes;

        readonly float _minFaceDetectionConfidence;
        readonly float _minFaceSuppressionThreshold;
        readonly float _minFacePresenceConfidence;
        readonly float _minHandLandmarksConfidence;
        readonly float _minPoseDetectionConfidence;
        readonly float _minPoseSuppressionThreshold;
        readonly float _minPosePresenceConfidence;

        readonly HolisticPoseTrackingRequest _poseTrackingRequest;

        readonly MultiBackendNet _poseDetectorNet;
        readonly MultiBackendNet _poseLandmarksNet;
        /// <summary>Hand-landmark network. <c>null</c> when all hand outputs are disabled.</summary>
        readonly MultiBackendNet _handLandmarksNet;
        /// <summary>Hand ROI refinement network. <c>null</c> when all hand outputs are disabled.</summary>
        readonly MultiBackendNet _handRoiRefinementNet;
        /// <summary>Face detector network. <c>null</c> when all face-related outputs are disabled.</summary>
        readonly MultiBackendNet _faceDetectorNet;
        /// <summary>Face landmark network. <c>null</c> when all face-related outputs are disabled.</summary>
        readonly MultiBackendNet _faceLandmarksNet;
        readonly MultiBackendNet _faceBlendshapesNet; // Null when outputFaceBlendshapes is false.

        /// <summary>Cached <c>forward</c> output layer names for hand landmark inference, avoiding per-frame <c>getUnconnectedOutLayersNames</c>. <c>null</c> when the hand model is not loaded.</summary>
        readonly List<string> _handLandmarksNetOutLayerNames;

        /// <summary>Cached <c>forward</c> output layer names for the hand ROI refinement model. <c>null</c> when not loaded.</summary>
        readonly List<string> _handRoiRefinementNetOutLayerNames;

        /// <summary>Cached <c>forward</c> output layer names for the Holistic face-detector path, matching <see cref="MediaPipeFaceLandmarker"/>. <c>null</c> when not loaded.</summary>
        readonly List<string> _hfFaceDetectorOutLayerNames;

        /// <summary>Cached <c>forward</c> output layer names for the Holistic face-landmarks path. <c>null</c> when not loaded.</summary>
        readonly List<string> _hfFaceLandmarksNetOutLayerNames;

        /// <summary>Cached <c>forward</c> output layer names for the Holistic face-blendshapes path. <c>null</c> when disabled.</summary>
        readonly List<string> _hfFaceBlendshapesNetOutLayerNames;

        /// <summary>Public slot 0: one packed row of <c>POSE_LANDMARKS</c> (same role as <see cref="MediaPipePoseLandmarker._outputBuffer"/>).</summary>
        Mat _holisticPackBufferPose;

        /// <summary>Public slot 1: one row of left-hand <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/>.</summary>
        Mat _holisticPackBufferLeftHand;

        /// <summary>Public slot 2: one row of right-hand <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/>.</summary>
        Mat _holisticPackBufferRightHand;

        /// <summary>Public slot 3: one row of face landmarks.</summary>
        Mat _holisticPackBufferFaceLandmarks;

        /// <summary>Public slot 5: one row of blendshape coefficients.</summary>
        Mat _holisticPackBufferFaceBlendshapes;

        /// <summary>Public slot 4: full-frame pose segmentation. Like <see cref="MediaPipePoseLandmarker._segmentationStackOutput"/>, the packed slot is a <see cref="Mat.rowRange"/> view over this buffer.</summary>
        Mat _holisticPackBufferSegmentation;

        /// <summary>
        /// Reusable intermediate <see cref="Mat"/> buffers for pose-segmentation visualization.
        /// Non-null only when <c>output_pose_segmentation_masks</c> is enabled and passed to
        /// <see cref="MediaPipePoseLandmarker.VisualizePackedPoseOutputs"/>.
        /// </summary>
        readonly MediaPipePoseLandmarker.PoseSegmentationVisualizationBuffers _poseSegmentationVisualizationBuffers;

        static readonly Vec3f[] HolisticPackedOutputZeroVec3f33 =
            new Vec3f[MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT];
        static readonly Vec3f[] HolisticPackedOutputZeroVec3f21 =
            new Vec3f[MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT];

        /// <summary>
        /// Creates a holistic landmarker worker backed by pose, hand, and face models.
        /// This public API maps to the model assets and runtime options used by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) holistic landmarker graph.
        /// The pose detector and pose landmark models are always required.
        /// Hand and face models are optional and are enabled only when the corresponding
        /// output paths are requested.
        /// </summary>
        /// <param name="poseDetectorModelFilepath">
        /// File path to the pose detector model.
        /// Corresponds to the detector model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>PoseDetectorGraph</c> path.
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, pass a path loadable by <c>Unity.InferenceEngine.ModelLoader.Load</c> (e.g. <c>.sentis</c>).
        /// </param>
        /// <param name="poseLandmarksModelFilepath">
        /// File path to the pose landmark model.
        /// Corresponds to the landmark model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>SinglePoseLandmarksDetectorGraph</c> path.
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, same as the detector path convention.
        /// </param>
        /// <param name="handLandmarksModelFilepath">
        /// File path to the hand landmark model.
        /// Corresponds to the landmark model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>SingleHandLandmarksDetectorGraph</c> path.
        /// <c>null</c> or empty disables hand-related outputs.
        /// </param>
        /// <param name="handRoiRefinementModelFilepath">
        /// File path to the hand ROI refinement model.
        /// Corresponds to the refinement model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>HandRoiRefinementGraph</c> path.
        /// <c>null</c> or empty disables hand-related outputs.
        /// </param>
        /// <param name="faceDetectorModelFilepath">
        /// File path to the face detector model.
        /// Corresponds to the detector model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>FaceDetectorGraph</c> path.
        /// <c>null</c> or empty disables face-related outputs.
        /// </param>
        /// <param name="faceLandmarksModelFilepath">
        /// File path to the face landmark model.
        /// Corresponds to the landmark model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>SingleFaceLandmarksDetectorGraph</c> path.
        /// <c>null</c> or empty disables face-related outputs.
        /// </param>
        /// <param name="runningMode">
        /// Task running mode.
        /// Corresponds to whether the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) task behaves like single-image processing
        /// or stateful video processing with loopback tracking state.
        /// </param>
        /// <param name="outputPoseSegmentationMasks">
        /// Whether pose segmentation masks are exposed in slot 4 of the packed output array.
        /// Corresponds to enabling the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>POSE_SEGMENTATION_MASK</c> output stream in the public result contract.
        /// </param>
        /// <param name="outputFaceBlendshapes">
        /// Whether face blendshapes are exposed in slot 5 of the packed output array.
        /// Corresponds to enabling the [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>FACE_BLENDSHAPES</c> output stream in the public result contract.
        /// </param>
        /// <param name="faceBlendshapesModelFilepath">
        /// File path to the face blendshapes model.
        /// Corresponds to the model asset consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>FaceBlendshapesGraph</c> path
        /// when <paramref name="outputFaceBlendshapes"/> is enabled.
        /// </param>
        /// <param name="minFaceDetectionConfidence">
        /// Minimum confidence for face detections to be kept before later stages.
        /// Corresponds to [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>min_face_detection_confidence</c>.
        /// </param>
        /// <param name="minFaceSuppressionThreshold">
        /// Suppression threshold for face detection filtering.
        /// Corresponds to [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>min_face_suppression_threshold</c>.
        /// </param>
        /// <param name="minFacePresenceConfidence">
        /// Minimum presence confidence required for face landmark results to be treated as present.
        /// Corresponds to [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>min_face_presence_confidence</c>.
        /// </param>
        /// <param name="minHandLandmarksConfidence">
        /// Minimum confidence required for hand landmark results to be treated as present.
        /// Corresponds to [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>min_hand_landmarks_confidence</c>.
        /// </param>
        /// <param name="minPoseDetectionConfidence">
        /// Minimum confidence for pose detections to be kept before later stages.
        /// Corresponds to [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>min_pose_detection_confidence</c>.
        /// </param>
        /// <param name="minPoseSuppressionThreshold">
        /// Suppression threshold for pose detection filtering.
        /// Corresponds to [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>min_pose_suppression_threshold</c>.
        /// </param>
        /// <param name="minPosePresenceConfidence">
        /// Minimum presence confidence required for pose landmark results to be treated as present.
        /// Corresponds to [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>min_pose_presence_confidence</c>.
        /// </param>
        /// <param name="dnnBackend">
        /// Inference backend: an OpenCV <see cref="Dnn"/> <c>DNN_BACKEND_*</c> constant, or <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>.
#if OPENCV_SENTIS_AVAILABLE
        /// When <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, all loaded models use Unity Inference Engine; <paramref name="dnnTarget"/> is interpreted as an integer <see cref="Unity.InferenceEngine.BackendType"/> value. Assumes Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer.
#else
        /// <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/> is only usable when the project includes Unity Inference Engine (com.unity.ai.inference) 2.6.1 or newer and the <c>OPENCV_SENTIS_AVAILABLE</c> define.
#endif
        /// </param>
        /// <param name="dnnTarget">
#if OPENCV_SENTIS_AVAILABLE
        /// OpenCV <see cref="Dnn"/> <c>DNN_TARGET_*</c> constant; when the backend is <see cref="MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS"/>, cast to Unity Inference Engine <see cref="Unity.InferenceEngine.BackendType"/>.
#else
        /// An OpenCV <see cref="Dnn"/> <c>DNN_TARGET_*</c> constant.
#endif
        /// </param>
        /// <exception cref="ArgumentException">Thrown when the combination of model paths and options is invalid.</exception>
        public MediaPipeHolisticLandmarker(
            string poseDetectorModelFilepath,
            string poseLandmarksModelFilepath,
            string handLandmarksModelFilepath,
            string handRoiRefinementModelFilepath,
            string faceDetectorModelFilepath,
            string faceLandmarksModelFilepath,
            MediaPipeHolisticRunningMode runningMode = MediaPipeHolisticRunningMode.IMAGE,
            bool outputPoseSegmentationMasks = false,
            bool outputFaceBlendshapes = false,
            string faceBlendshapesModelFilepath = null,
            float minFaceDetectionConfidence = 0.5f,
            float minFaceSuppressionThreshold = 0.3f,
            float minFacePresenceConfidence = 0.5f,
            float minHandLandmarksConfidence = 0.5f,
            float minPoseDetectionConfidence = 0.5f,
            float minPoseSuppressionThreshold = 0.3f,
            float minPosePresenceConfidence = 0.5f,
            int dnnBackend = Dnn.DNN_BACKEND_OPENCV,
            int dnnTarget = Dnn.DNN_TARGET_CPU)
            : base(dnnBackend, dnnTarget)
        {
            if (string.IsNullOrEmpty(poseDetectorModelFilepath))
                throw new ArgumentException("The pose detector model filepath is not specified.", nameof(poseDetectorModelFilepath));
            if (string.IsNullOrEmpty(poseLandmarksModelFilepath))
                throw new ArgumentException("The pose landmarks model filepath is not specified.", nameof(poseLandmarksModelFilepath));

            bool handsNeeded = !string.IsNullOrWhiteSpace(handLandmarksModelFilepath)
                && !string.IsNullOrWhiteSpace(handRoiRefinementModelFilepath);
            bool faceNeeded = !string.IsNullOrWhiteSpace(faceDetectorModelFilepath)
                && !string.IsNullOrWhiteSpace(faceLandmarksModelFilepath);
            bool blendshapesNeeded = outputFaceBlendshapes
                && faceNeeded
                && !string.IsNullOrWhiteSpace(faceBlendshapesModelFilepath);

            _runningMode = runningMode;

            _outputPoseLandmarks = true;
            _outputPoseWorldLandmarks = true;
            _outputLeftHandLandmarks = handsNeeded;
            _outputRightHandLandmarks = handsNeeded;
            _outputLeftHandWorldLandmarks = handsNeeded;
            _outputRightHandWorldLandmarks = handsNeeded;
            _outputPoseSegmentationMasks = outputPoseSegmentationMasks;
            _outputFaceLandmarks = faceNeeded;
            _outputFaceBlendshapes = blendshapesNeeded;

            _minFaceDetectionConfidence = Mathf.Clamp01(minFaceDetectionConfidence);
            _minFaceSuppressionThreshold = Mathf.Clamp01(minFaceSuppressionThreshold);
            _minFacePresenceConfidence = Mathf.Clamp01(minFacePresenceConfidence);
            _minHandLandmarksConfidence = Mathf.Clamp01(minHandLandmarksConfidence);
            _minPoseDetectionConfidence = Mathf.Clamp01(minPoseDetectionConfidence);
            _minPoseSuppressionThreshold = Mathf.Clamp01(minPoseSuppressionThreshold);
            _minPosePresenceConfidence = Mathf.Clamp01(minPosePresenceConfidence);

            _poseTrackingRequest = HolisticPoseTrackingRequest.FromOutputFlags(
                _outputPoseLandmarks,
                _outputPoseWorldLandmarks,
                _outputLeftHandLandmarks,
                _outputRightHandLandmarks,
                _outputLeftHandWorldLandmarks,
                _outputRightHandWorldLandmarks,
                _outputPoseSegmentationMasks,
                _outputFaceLandmarks,
                _outputFaceBlendshapes);

            _poseSegmentationVisualizationBuffers = outputPoseSegmentationMasks
                ? new MediaPipePoseLandmarker.PoseSegmentationVisualizationBuffers()
                : null;

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

                _poseLandmarksNet = MultiBackendDnn.readNet(poseLandmarksModelFilepath);
                _poseLandmarksNet.setPreferableBackend(DnnBackend);
                _poseLandmarksNet.setPreferableTarget(DnnTarget);

                if (handsNeeded)
                {
                    _handLandmarksNet = MultiBackendDnn.readNet(handLandmarksModelFilepath);
                    _handLandmarksNet.setPreferableBackend(DnnBackend);
                    _handLandmarksNet.setPreferableTarget(DnnTarget);
                    _handLandmarksNetOutLayerNames = _handLandmarksNet.getUnconnectedOutLayersNames();

                    _handRoiRefinementNet = MultiBackendDnn.readNet(handRoiRefinementModelFilepath);
                    _handRoiRefinementNet.setPreferableBackend(DnnBackend);
                    _handRoiRefinementNet.setPreferableTarget(DnnTarget);
                    _handRoiRefinementNetOutLayerNames = _handRoiRefinementNet.getUnconnectedOutLayersNames();
                }
                else
                {
                    _handLandmarksNet = null;
                    _handRoiRefinementNet = null;
                    _handLandmarksNetOutLayerNames = null;
                    _handRoiRefinementNetOutLayerNames = null;
                }

                if (faceNeeded)
                {
                    _faceDetectorNet = MultiBackendDnn.readNet(faceDetectorModelFilepath);
                    _faceDetectorNet.setPreferableBackend(DnnBackend);
                    _faceDetectorNet.setPreferableTarget(DnnTarget);

                    _faceLandmarksNet = MultiBackendDnn.readNet(faceLandmarksModelFilepath);
                    _faceLandmarksNet.setPreferableBackend(DnnBackend);
                    _faceLandmarksNet.setPreferableTarget(DnnTarget);

                    _hfFaceDetectorOutLayerNames = _faceDetectorNet.getUnconnectedOutLayersNames();
                    _hfFaceLandmarksNetOutLayerNames = _faceLandmarksNet.getUnconnectedOutLayersNames();
                }
                else
                {
                    _faceDetectorNet = null;
                    _faceLandmarksNet = null;
                    _hfFaceDetectorOutLayerNames = null;
                    _hfFaceLandmarksNetOutLayerNames = null;
                }

                _faceBlendshapesNet = null;
                _hfFaceBlendshapesNetOutLayerNames = null;
                if (_outputFaceBlendshapes)
                {
                    _faceBlendshapesNet = MultiBackendDnn.readNet(faceBlendshapesModelFilepath);
                    _faceBlendshapesNet.setPreferableBackend(DnnBackend);
                    _faceBlendshapesNet.setPreferableTarget(DnnTarget);
                    _hfFaceBlendshapesNetOutLayerNames = _faceBlendshapesNet.getUnconnectedOutLayersNames();
                }
            }
            catch (Exception e)
            {
                throw new ArgumentException("Failed to initialize the DNN models for Holistic Landmarker. Check the model paths and file contents.", e);
            }
        }

        /// <summary>
        /// High-level inference API equivalent to the synchronous detect entry points of
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HolisticLandmarker.
        /// Returns a packed output array whose slots correspond to the public holistic result streams.
        /// </summary>
        /// <param name="image">
        /// Input image in BGR 3-channel format.
        /// Corresponds to the input image consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) holistic landmarker graph.
        /// </param>
        /// <param name="useCopyOutput">
        /// If true, returns a copied output matrix.
        /// If false, returns a view backed by the worker's reusable output buffer.
        /// </param>
        /// <returns>
        /// Fixed-length array of <see cref="HolisticPackedOutputSlotCount"/> entries following the public holistic contract.
        /// Filled slots are either one packed <c>CV_32FC1</c> row or a 2D <see cref="Mat"/> for segmentation.
        /// Slot 0: <see cref="MediaPipePoseLandmarker.PoseLandmarkerEstimationData"/>.
        /// Slots 1–2: left/right <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/>.
        /// Slot 3: <see cref="MediaPipeFaceLandmarker.FaceLandmarkerEstimationData"/>.
        /// Slot 4: full-resolution pose segmentation mask.
        /// Slot 5: one row of 52 coefficients equivalent to <see cref="MediaPipeFaceLandmarker.FaceBlendshapeEstimationData"/>.
        /// Disabled slots are empty <see cref="Mat"/> instances.
        /// </returns>
        public Mat[] Detect(Mat image, bool useCopyOutput = false)
        {
            if (image != null) image.ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.channels() != 3) throw new ArgumentException("The input image must be a 3-channel BGR image.");

            Execute(image);
            return BuildDetectReturnArray(useCopyOutput);
        }

        /// <summary>
        /// Asynchronous inference API equivalent to the async detect entry points of
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HolisticLandmarker.
        /// The effective behavior still depends on the configured <c>runningMode</c>.
        /// </summary>
        /// <param name="image">
        /// Input image in BGR 3-channel format.
        /// Corresponds to the input image consumed by the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) holistic landmarker graph.
        /// </param>
        /// <param name="cancellationToken">
        /// Cancellation token for the C# async operation.
        /// </param>
        /// <returns>
        /// Same slot contract as <see cref="Detect(Mat, bool)"/>; returns only copied <see cref="Mat"/> instances.
        /// </returns>
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
        /// Asynchronous inference API equivalent to the async detect entry points of
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) HolisticLandmarker.
        /// </summary>
        /// <remarks>
        /// <c>@deprecated</c> Use <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. In a future version, this member will return Unity <c>Awaitable</c> instead of <see cref="Task{TResult}"/>.
        /// See <see cref="DetectTaskAsync(Mat, CancellationToken)"/>. Web synchronous fallback applies only to the OpenCV Dnn backend; Sentis remains asynchronous on every platform, including Web.
        /// </remarks>
        [Obsolete("Use DetectTaskAsync(). DetectAsync() will return Awaitable in a future version.")]
        public Task<Mat[]> DetectAsync(Mat image, CancellationToken cancellationToken = default) =>
            DetectTaskAsync(image, cancellationToken);

        Mat[] BuildDetectReturnArray(bool useCopyOutput)
        {
            int n = HolisticPackedOutputSlotCount;
            var arr = new Mat[n];
            for (int i = 0; i < n; i++)
                arr[i] = useCopyOutput ? CopyOutput(i) : PeekOutput(i);
            return arr;
        }

        /// <inheritdoc cref="ProcessingWorkerBase"/>
        protected override Mat[] RunCoreProcessing(Mat[] inputs)
        {
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("HolisticLandmarker accepts only one input image at index 0.", nameof(inputs));

            Mat image = inputs[0];
            image.ThrowIfDisposed();
            if (image.channels() != 3)
                throw new ArgumentException("The input image must be a 3-channel BGR image.");

            HolisticLandmarkerFrameResult frame = _runningMode == MediaPipeHolisticRunningMode.IMAGE
                ? DetectPipeline(image)
                : DetectForVideoPipeline(image);

            return PackResultsToMats(frame, image);
        }

        /// <inheritdoc cref="ProcessingWorkerBase"/>
        protected override async Task<Mat[]> RunCoreProcessingTaskAsync(Mat[] inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (inputs == null || inputs.Length != 1 || inputs[0] == null)
                throw new ArgumentException("HolisticLandmarker accepts only one input image at index 0.", nameof(inputs));

            Mat image = inputs[0];
            image.ThrowIfDisposed();
            if (image.channels() != 3)
                throw new ArgumentException("The input image must be a 3-channel BGR image.");

#if OPENCV_SENTIS_AVAILABLE
            if (_poseLandmarksNet.UsesSentis)
            {
                HolisticLandmarkerFrameResult frame = _runningMode == MediaPipeHolisticRunningMode.IMAGE
                    ? await ProcessImageDataAsync(image, cancellationToken)
                    : await ProcessVideoDataAsync(image, cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                return PackResultsToMats(frame, image);
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
        /// Private IMAGE-mode entry point, corresponding to <c>HolisticLandmarker::Detect</c> and called
        /// from <see cref="RunCoreProcessing"/>.
        /// </summary>
        HolisticLandmarkerFrameResult DetectPipeline(Mat image)
        {
            return ProcessImageData(image);
        }

        /// <summary>
        /// Private VIDEO-mode entry point, corresponding to <c>HolisticLandmarker::DetectForVideo</c>.
        /// </summary>
        HolisticLandmarkerFrameResult DetectForVideoPipeline(Mat image)
        {
            return ProcessVideoData(image);
        }

        /// <summary>
        /// IMAGE-mode pipeline entry point corresponding to the Task API <c>ProcessImageData</c> path
        /// and to <c>HolisticLandmarkerGraph</c>.
        /// As in upstream <c>BaseVisionTaskApi::ProcessImageData</c>, internal loopback state is not
        /// automatically cleared between calls.
        /// </summary>
        HolisticLandmarkerFrameResult ProcessImageData(Mat image)
        {
            return HolisticLandmarkerGraph(image);
        }

        /// <summary>
        /// VIDEO-mode pipeline entry point corresponding to the Task API <c>ProcessVideoData</c> path.
        /// </summary>
        HolisticLandmarkerFrameResult ProcessVideoData(Mat image)
        {
            return HolisticLandmarkerGraph(image);
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="ProcessImageData"/> (via <see cref="HolisticLandmarkerGraphAsync"/>).
        /// </summary>
        async Task<HolisticLandmarkerFrameResult> ProcessImageDataAsync(Mat image, CancellationToken cancellationToken)
        {
            return await HolisticLandmarkerGraphAsync(image, cancellationToken);
        }

        /// <summary>
        /// Async Sentis variant of <see cref="ProcessVideoData"/>. The current implementation calls the same <see cref="HolisticLandmarkerGraphAsync"/> as IMAGE.
        /// </summary>
        async Task<HolisticLandmarkerFrameResult> ProcessVideoDataAsync(Mat image, CancellationToken cancellationToken)
        {
            return await HolisticLandmarkerGraphAsync(image, cancellationToken);
        }
#endif

        /// <summary>
        /// Packs inference results into <see cref="HolisticPackedOutputSlotCount"/> <see cref="Mat"/> slots.
        /// Equivalent to <see cref="MediaPipePoseLandmarker.PackResultsToMats"/>.
        /// </summary>
        /// <remarks>
        /// Shared packing buffers (<c>_holisticPackBuffer*</c>) are synchronized inside
        /// <see cref="BuildHolisticPackedOutputs"/> via <c>lock (_lockObject)</c>.
        /// </remarks>
        /// <param name="frame">Return value from <see cref="DetectPipeline"/> or <see cref="DetectForVideoPipeline"/>.</param>
        /// <param name="image">Input image, kept for API consistency. It is currently unused during packing.</param>
        Mat[] PackResultsToMats(HolisticLandmarkerFrameResult frame, Mat image)
        {
            return BuildHolisticPackedOutputs(frame.Pose, frame.LeftHand, frame.RightHand, frame.Face);
        }

        /// <summary>
        /// Packs one frame of Holistic results into <see cref="HolisticPackedOutputSlotCount"/> <see cref="Mat"/> instances.
        /// </summary>
        /// <remarks>
        /// <para><c>POSE_SEGMENTATION_MASK</c> (public slot 4):</para>
        /// <list type="bullet">
        /// <item><description>If <paramref name="pose"/>.<see cref="HolisticPoseTrackFrameResult.SegmentationMaskFull"/> points to a worker-owned reuse buffer (<c>_hpSegmentationFullPlaneReuse</c> or <c>_hpSegmentationSmoothedReuse</c>), it is not disposed.</description></item>
        /// <item><description>Any other temporary <see cref="Mat"/> is disposed after <see cref="Mat.copyTo"/>, and public slot 4 is always normalized to a view backed by the internal packing buffer <c>_holisticPackBufferSegmentation</c>.</description></item>
        /// </list>
        /// <para>For every slot, the returned array is retained by <see cref="ProcessingWorkerBase"/> as the result of <see cref="RunCoreProcessing"/>, so the same lifetime rule applies: the references are valid only until the next execution.</para>
        /// <para>As in <see cref="MediaPipeHandLandmarker.PackResultsToMats"/>, reallocation and writes to <c>_holisticPackBuffer*</c> and use of the packed <see cref="Mat"/> outputs are serialized by <see cref="ProcessingWorkerBase"/>'s <c>_lockObject</c>; reentrancy during <see cref="Execute"/> is allowed.</para>
        /// </remarks>
        Mat[] BuildHolisticPackedOutputs(
            HolisticPoseTrackFrameResult pose,
            HolisticHandTrackFrameResult leftHand = default,
            HolisticHandTrackFrameResult rightHand = default,
            HolisticFaceTrackFrameResult face = default)
        {
            lock (_lockObject)
            {
                var outputs = new Mat[HolisticPackedOutputSlotCount];
                for (int i = 0; i < outputs.Length; i++)
                    outputs[i] = new Mat();

                Mat seg = pose.SegmentationMaskFull;
                // Dispose seg in finally only when this flag remains true, avoiding double-disposal
                // after buffer copy or early return.
                bool disposeSegInFinally = true;
                try
                {
                    if (!pose.PosePresence)
                    {
                        if (seg != null && seg != _hpSegmentationFullPlaneReuse && seg != _hpSegmentationSmoothedReuse)
                            seg.Dispose();
                        disposeSegInFinally = false;
                        return outputs;
                    }

                    if (_outputPoseLandmarks)
                    {
                        outputs[0].Dispose();
                        outputs[0] = HolisticPackPoseLandmarkerOutputRow(
                            pose.NormLandmarks,
                            _outputPoseWorldLandmarks ? pose.WorldLandmarks : HolisticPackedOutputZeroVec3f33,
                            pose.LandmarkVisibility,
                            pose.LandmarkVisibilityWorld,
                            pose.LandmarkPresence);
                    }

                    if (_outputLeftHandLandmarks && leftHand.HandPresence)
                    {
                        outputs[1].Dispose();
                        outputs[1] = HolisticPackLeftHandLandmarkerOutputRow(
                            leftHand.NormLandmarks,
                            _outputLeftHandWorldLandmarks ? leftHand.WorldLandmarks : HolisticPackedOutputZeroVec3f21,
                            leftHand.Handedness);
                    }

                    if (_outputRightHandLandmarks && rightHand.HandPresence)
                    {
                        outputs[2].Dispose();
                        outputs[2] = HolisticPackRightHandLandmarkerOutputRow(
                            rightHand.NormLandmarks,
                            _outputRightHandWorldLandmarks ? rightHand.WorldLandmarks : HolisticPackedOutputZeroVec3f21,
                            rightHand.Handedness);
                    }

                    if (_outputPoseSegmentationMasks && seg != null && !seg.empty())
                    {
                        outputs[4].Dispose();
                        int sh = seg.rows();
                        int sw = seg.cols();
                        int segType = seg.type();
                        if (_holisticPackBufferSegmentation == null
                            || _holisticPackBufferSegmentation.rows() != sh
                            || _holisticPackBufferSegmentation.cols() != sw
                            || _holisticPackBufferSegmentation.type() != segType)
                        {
                            _holisticPackBufferSegmentation?.Dispose();
                            _holisticPackBufferSegmentation = new Mat(sh, sw, segType);
                        }

                        seg.copyTo(_holisticPackBufferSegmentation);
                        if (seg != _hpSegmentationFullPlaneReuse && seg != _hpSegmentationSmoothedReuse)
                            seg.Dispose();
                        disposeSegInFinally = false;
                        outputs[4] = _holisticPackBufferSegmentation.rowRange(0, sh);
                    }

                    if (_outputFaceLandmarks && face.FacePresence && face.NormLandmarks478 != null)
                    {
                        outputs[3].Dispose();
                        outputs[3] = HolisticPackFaceLandmarkerOutputRow(face.NormLandmarks478);
                    }

                    // Upstream holistic_landmarker.cc emits face blendshapes when output_face_blendshapes
                    // is enabled and the packet/classifications are non-empty; FacePresence is not required.
                    if (_outputFaceBlendshapes && face.BlendshapeCoefficients52 != null)
                    {
                        outputs[5].Dispose();
                        outputs[5] = HolisticPackFaceBlendshapeCoefficientsOutputRow(face.BlendshapeCoefficients52);
                    }
                }
                finally
                {
                    if (disposeSegInFinally && seg != null && seg != _hpSegmentationFullPlaneReuse && seg != _hpSegmentationSmoothedReuse)
                        seg.Dispose();
                }

                return outputs;
            }
        }

        void DisposeHolisticPackedOutputBuffers()
        {
            _holisticPackBufferPose?.Dispose();
            _holisticPackBufferPose = null;
            _holisticPackBufferLeftHand?.Dispose();
            _holisticPackBufferLeftHand = null;
            _holisticPackBufferRightHand?.Dispose();
            _holisticPackBufferRightHand = null;
            _holisticPackBufferFaceLandmarks?.Dispose();
            _holisticPackBufferFaceLandmarks = null;
            _holisticPackBufferFaceBlendshapes?.Dispose();
            _holisticPackBufferFaceBlendshapes = null;
            _holisticPackBufferSegmentation?.Dispose();
            _holisticPackBufferSegmentation = null;
            _poseSegmentationVisualizationBuffers?.Dispose();
        }

        /// <summary>Public slot 0: writes one row equivalent to <see cref="MediaPipePoseLandmarker.PoseLandmarkerEstimationData"/> into <see cref="_holisticPackBufferPose"/> and returns a <see cref="Mat.rowRange"/> view. Callers must pass non-null <paramref name="landmarksScreen"/>, <paramref name="landmarksWorld"/>, visibility, and presence arrays of length <see cref="MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT"/> (as <see cref="BuildHolisticPackedOutputs"/> does when <c>PosePresence</c> is true).</summary>
        Mat HolisticPackPoseLandmarkerOutputRow(Vec3f[] landmarksScreen, Vec3f[] landmarksWorld,
            float[] landmarkVisibilityNorm, float[] landmarkVisibilityWorld, float[] landmarkPresence)
        {
            int L = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            int R = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.ELEMENT_COUNT;
            if (_holisticPackBufferPose == null
                || _holisticPackBufferPose.rows() < 1
                || _holisticPackBufferPose.cols() != R
                || _holisticPackBufferPose.type() != CvType.CV_32FC1)
            {
                _holisticPackBufferPose?.Dispose();
                _holisticPackBufferPose = new Mat(1, R, CvType.CV_32FC1);
            }

            HolisticPutPoseLandmarkerRowToBuffer(_holisticPackBufferPose, L, R, landmarksScreen, landmarksWorld,
                landmarkVisibilityNorm, landmarkVisibilityWorld, landmarkPresence);
            return _holisticPackBufferPose.rowRange(0, 1);
        }

        /// <summary>
        /// Writes the first row of <paramref name="dst"/> with two blocks of 33 joints × 5 floats each, matching upstream
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>NormalizedLandmark</c> / <c>Landmark</c> layout.
        /// Fills a <c>stackalloc</c> scratch span of length <paramref name="R"/> and copies it into <paramref name="dst"/> via <see cref="Mat.put"/> before returning.
        /// </summary>
        void HolisticPutPoseLandmarkerRowToBuffer(Mat dst, int L, int R, Vec3f[] landmarksScreen, Vec3f[] landmarksWorld,
            float[] landmarkVisibilityNorm, float[] landmarkVisibilityWorld, float[] landmarkPresence)
        {
            int stride = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_FLOAT_STRIDE;
            int offWorld = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.NORM_LANDMARKS_FLOAT_COUNT;
            Span<float> row = stackalloc float[R];

            for (int j = 0; j < L; j++)
            {
                int oN = j * stride;
                int oW = offWorld + j * stride;
                ref readonly Vec3f sn = ref landmarksScreen[j];
                ref readonly Vec3f wn = ref landmarksWorld[j];
                float pr = landmarkPresence[j];
                row[oN + 0] = sn.Item1;
                row[oN + 1] = sn.Item2;
                row[oN + 2] = sn.Item3;
                row[oN + 3] = landmarkVisibilityNorm[j];
                row[oN + 4] = pr;
                row[oW + 0] = wn.Item1;
                row[oW + 1] = wn.Item2;
                row[oW + 2] = wn.Item3;
                row[oW + 3] = landmarkVisibilityWorld[j];
                row[oW + 4] = pr;
            }

            dst.put(0, 0, row);
        }

        /// <summary>Public slot 1: packs one row equivalent to left-hand <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/> (normalized + world + handedness) and returns a <see cref="Mat.rowRange"/> view. <paramref name="landmarksScreen"/> and <paramref name="landmarksWorld"/> must be non-null arrays of length <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT"/>.</summary>
        Mat HolisticPackLeftHandLandmarkerOutputRow(Vec3f[] landmarksScreen, Vec3f[] landmarksWorld, float handedness)
        {
            int R = MediaPipeHandLandmarker.HandLandmarkerEstimationData.ELEMENT_COUNT;
            if (_holisticPackBufferLeftHand == null
                || _holisticPackBufferLeftHand.rows() < 1
                || _holisticPackBufferLeftHand.cols() != R
                || _holisticPackBufferLeftHand.type() != CvType.CV_32FC1)
            {
                _holisticPackBufferLeftHand?.Dispose();
                _holisticPackBufferLeftHand = new Mat(1, R, CvType.CV_32FC1);
            }

            HolisticPutHandLandmarkerRowToBuffer(_holisticPackBufferLeftHand, landmarksScreen, landmarksWorld, handedness);
            return _holisticPackBufferLeftHand.rowRange(0, 1);
        }

        /// <summary>Public slot 2: packs one row equivalent to right-hand <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/> and returns a <see cref="Mat.rowRange"/> view. <paramref name="landmarksScreen"/> and <paramref name="landmarksWorld"/> must be non-null arrays of length <see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT"/>.</summary>
        Mat HolisticPackRightHandLandmarkerOutputRow(Vec3f[] landmarksScreen, Vec3f[] landmarksWorld, float handedness)
        {
            int R = MediaPipeHandLandmarker.HandLandmarkerEstimationData.ELEMENT_COUNT;
            if (_holisticPackBufferRightHand == null
                || _holisticPackBufferRightHand.rows() < 1
                || _holisticPackBufferRightHand.cols() != R
                || _holisticPackBufferRightHand.type() != CvType.CV_32FC1)
            {
                _holisticPackBufferRightHand?.Dispose();
                _holisticPackBufferRightHand = new Mat(1, R, CvType.CV_32FC1);
            }

            HolisticPutHandLandmarkerRowToBuffer(_holisticPackBufferRightHand, landmarksScreen, landmarksWorld, handedness);
            return _holisticPackBufferRightHand.rowRange(0, 1);
        }

        void HolisticPutHandLandmarkerRowToBuffer(Mat dst, Vec3f[] landmarksScreen, Vec3f[] landmarksWorld, float handedness)
        {
            int L = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            int R = MediaPipeHandLandmarker.HandLandmarkerEstimationData.ELEMENT_COUNT;
            int offWorld = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT;
            ReadOnlySpan<float> norm = MemoryMarshal.Cast<Vec3f, float>(landmarksScreen.AsSpan(0, L));
            ReadOnlySpan<float> world = MemoryMarshal.Cast<Vec3f, float>(landmarksWorld.AsSpan(0, L));
            dst.put(0, 0, norm);
            dst.put(0, offWorld, world);
            Span<float> handednessCell = stackalloc float[1] { handedness };
            dst.put(0, R - 1, handednessCell);
        }

        /// <summary>Public slot 3: packs one row equivalent to <see cref="MediaPipeFaceLandmarker.FaceLandmarkerEstimationData"/> (478 normalized xyz values) and returns a <see cref="Mat.rowRange"/> view. Callers must pass a non-null array of length <see cref="MediaPipeFaceLandmarker.FaceLandmarkerEstimationData.LANDMARK_VEC3F_COUNT"/>.</summary>
        Mat HolisticPackFaceLandmarkerOutputRow(Vec3f[] normLandmarks478)
        {
            int L = MediaPipeFaceLandmarker.FaceLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            int R = MediaPipeFaceLandmarker.FaceLandmarkerEstimationData.ELEMENT_COUNT;
            if (_holisticPackBufferFaceLandmarks == null
                || _holisticPackBufferFaceLandmarks.rows() < 1
                || _holisticPackBufferFaceLandmarks.cols() != R
                || _holisticPackBufferFaceLandmarks.type() != CvType.CV_32FC1)
            {
                _holisticPackBufferFaceLandmarks?.Dispose();
                _holisticPackBufferFaceLandmarks = new Mat(1, R, CvType.CV_32FC1);
            }

            Mat dst = _holisticPackBufferFaceLandmarks;
            ReadOnlySpan<float> row = MemoryMarshal.Cast<Vec3f, float>(normLandmarks478.AsSpan(0, L));
            dst.put(0, 0, row);
            return dst.rowRange(0, 1);
        }

        /// <summary>Public slot 5: packs one row of blendshape coefficients with the same column count as <see cref="MediaPipeFaceLandmarker.FaceBlendshapeEstimationData"/> and returns a <see cref="Mat.rowRange"/> view. Callers must pass a non-null array of length <see cref="MediaPipeFaceLandmarker.kFaceBlendshapeCoefficientCount"/>.</summary>
        Mat HolisticPackFaceBlendshapeCoefficientsOutputRow(float[] coeffs52)
        {
            int C = MediaPipeFaceLandmarker.kFaceBlendshapeCoefficientCount;
            if (_holisticPackBufferFaceBlendshapes == null
                || _holisticPackBufferFaceBlendshapes.rows() < 1
                || _holisticPackBufferFaceBlendshapes.cols() != C
                || _holisticPackBufferFaceBlendshapes.type() != CvType.CV_32FC1)
            {
                _holisticPackBufferFaceBlendshapes?.Dispose();
                _holisticPackBufferFaceBlendshapes = new Mat(1, C, CvType.CV_32FC1);
            }

            Mat dst = _holisticPackBufferFaceBlendshapes;
            dst.put(0, 0, coeffs52.AsSpan(0, C));
            return dst.rowRange(0, 1);
        }

        /// <summary>
        /// Draws holistic landmarks from a <see cref="Mat"/> array whose layout matches
        /// <see cref="Detect(Mat, bool)"/>.
        /// The array is interpreted using the public holistic slot order and drawn in
        /// pose -> hands -> face order.
        /// </summary>
        /// <param name="image">Destination image.</param>
        /// <param name="results">
        /// Array of output matrices.
        /// Slot 0: pose (<see cref="MediaPipePoseLandmarker.PoseLandmarkerEstimationData"/>);
        /// slots 1–2: left/right hand (<see cref="MediaPipeHandLandmarker.HandLandmarkerEstimationData"/>);
        /// slot 3: face; slot 4: pose segmentation; slot 5: face blendshape coefficients.
        /// Empty slots are skipped.
        /// </param>
        /// <param name="printResult">If true, prints decoded results to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public void Visualize(Mat image, Mat[] results, bool printResult = false, bool isRGB = false)
        {
            ThrowIfDisposed();
            if (image != null)
                image.ThrowIfDisposed();
            if (results == null || results.Length == 0)
                return;

            int n = results.Length;

            Mat poseMain = n > 0 ? results[0] : null;
            if (poseMain != null && !poseMain.empty() && poseMain.rows() > 0)
            {
                Mat seg = (n > 4 && results[4] != null && !results[4].empty()) ? results[4] : null;
                Mat[] posePack = seg != null
                    ? new[] { poseMain, seg }
                    : new[] { poseMain };
                MediaPipePoseLandmarker.VisualizePackedPoseOutputs(image, posePack, printResult, isRGB, _poseSegmentationVisualizationBuffers);
            }

            if (n > 1 && results[1] != null && !results[1].empty())
                MediaPipeHandLandmarker.VisualizePackedHandOutputMat(image, results[1], printResult, isRGB);
            if (n > 2 && results[2] != null && !results[2].empty())
                MediaPipeHandLandmarker.VisualizePackedHandOutputMat(image, results[2], printResult, isRGB);

            if (n > 3 && results[3] != null && !results[3].empty())
            {
                Mat blend = (n > 5 && results[5] != null && !results[5].empty()) ? results[5] : null;
                Mat[] facePack = blend != null
                    ? new[] { results[3], blend }
                    : new[] { results[3] };
                MediaPipeFaceLandmarker.VisualizePackedFaceOutputs(image, facePack, printResult, isRGB);
            }
        }

        /// <summary>
        /// Visualizes the packed holistic output returned by <see cref="Detect(Mat, bool)"/>.
        /// The matrix is interpreted as slot 0 of the public holistic packed output layout.
        /// </summary>
        /// <param name="image">Destination image for visualization.</param>
        /// <param name="results">
        /// Packed result matrix corresponding to slot 0 of the holistic output array.
        /// This matrix stores the public packed representation of the
        /// [MediaPipe](https://github.com/google-ai-edge/mediapipe) pose landmark output.
        /// </param>
        /// <param name="printResult">If true, prints the decoded result to the console.</param>
        /// <param name="isRGB">If true, treats <paramref name="image"/> as RGB instead of BGR.</param>
        public override void Visualize(Mat image, Mat results, bool printResult = false, bool isRGB = false)
        {
            Visualize(image, results == null ? null : new[] { results }, printResult, isRGB);
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                _poseDetectorNet?.Dispose();
                _poseLandmarksNet?.Dispose();
                _handLandmarksNet?.Dispose();
                _handRoiRefinementNet?.Dispose();
                _faceDetectorNet?.Dispose();
                _faceLandmarksNet?.Dispose();
                _faceBlendshapesNet?.Dispose();

                DisposeHolisticPoseTrackingScratch();
                DisposeHolisticHandTrackingScratch();
                DisposeHolisticFaceTrackingScratch();
                DisposeHolisticPackedOutputBuffers();
            }

            base.Dispose(disposing);
        }
    }
}
#endif
#endif
