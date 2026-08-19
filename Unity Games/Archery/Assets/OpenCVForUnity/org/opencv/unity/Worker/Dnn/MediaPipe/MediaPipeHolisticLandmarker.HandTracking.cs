#if !UNITY_WSA_10_0
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe
{

    public partial class MediaPipeHolisticLandmarker
    {
        const float kHolisticHandLandmarksNormalizeZ = 0.4f;
        const int kHolisticHandRoiRefineInputSize = 256;

        /// <summary>Equivalent to <c>PoseIndices</c> in <c>holistic_hand_tracking.cc</c>.</summary>
        readonly struct HolisticPoseHandIndices
        {
            public readonly int WristIdx;
            public readonly int PinkyIdx;
            public readonly int IndexIdx;

            public HolisticPoseHandIndices(int wristIdx, int pinkyIdx, int indexIdx)
            {
                WristIdx = wristIdx;
                PinkyIdx = pinkyIdx;
                IndexIdx = indexIdx;
            }
        }

        /// <summary>Equivalent to upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>HolisticHandTrackingRequest</c>.</summary>
        readonly struct HolisticHandTrackingRequest
        {
            /// <summary>Whether normalized hand landmarks should be packed into <see cref="PeekOutput"/>.</summary>
            public bool PackLandmarks { get; }

            /// <summary>Whether world hand landmarks should be packed into <see cref="PeekOutput"/>.</summary>
            public bool PackWorldLandmarks { get; }

            public HolisticHandTrackingRequest(bool packLandmarks, bool packWorldLandmarks)
            {
                PackLandmarks = packLandmarks;
                PackWorldLandmarks = packWorldLandmarks;
            }
        }

        static readonly HolisticPoseHandIndices HolisticLeftHandPoseIndices = new HolisticPoseHandIndices(15, 17, 19);
        static readonly HolisticPoseHandIndices HolisticRightHandPoseIndices = new HolisticPoseHandIndices(16, 18, 20);

        // --- Scratch buffers for hand ROI refinement (256) and single-hand landmarks (224) ---
        Mat _hhRoiRefineSrcPts;
        Mat _hhRoiRefineDstPts;
        Mat _hhRoiRefineWarpedBgr;
        Mat _hhRoiRefineWarpedRgb;
        Mat _hhRoiRefineBlob;
        Mat _hhRoiRefineBlobHxW;
        readonly float[] _hhRoiRefineMatrix16 = new float[16];

        Mat _hhSingleHandSrcPts;
        Mat _hhSingleHandDstPts;
        Mat _hhSingleHandWarpedBgr;
        Mat _hhSingleHandWarpedRgb;
        Mat _hhSingleHandBlob;
        Mat _hhSingleHandBlobHxW;

        readonly List<Mat> _hhHandRoiRefinementForwardOutputList = new List<Mat>();
        readonly List<Mat> _hhHandLandmarksForwardOutputList = new List<Mat>();

        readonly float[] _hhLandmarksToDetKp6 = new float[6];
        readonly float[] _hhRoiRefineRaw4 = new float[4];
        readonly float[] _hhRoiRefineKp4 = new float[4];

        Vec3f[] _hhPoseNorm33Scratch;
        readonly Vec3f[] _hhHolisticHandPalmThreeScratch = new Vec3f[3];
        readonly Vec3f[] _hhRoiRefineNormTwoScratch = new Vec3f[2];
        readonly Vec3f[] _hhRoiRefineProjectedTwoScratch = new Vec3f[2];

        float[] _hhSingleHandTensorNorm;
        float[] _hhSingleHandTensorWorld;
        readonly float[] _hhSingleHandLetterboxRemovedScratch =
            new float[MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT];

        static readonly Vec3f[] HolisticHandEmptyNormOrWorld21 =
            new Vec3f[MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT];

        Vec3f[] _holisticLeftHandPrevNormLandmarks;
        Vec3f[] _holisticRightHandPrevNormLandmarks;

        /// <summary>
        /// Equivalent to <c>HolisticLandmarkerGraph</c>, limited to child-graph invocation.
        /// Child graph to method mapping: <see cref="TrackHolisticPose"/>, left/right
        /// <see cref="TrackHolisticHand"/> when enabled by output flags, and
        /// <see cref="TrackHolisticFace"/> when face or blendshape output is enabled, with
        /// <c>SplitToRanges(0-11)</c> at the start.
        /// </summary>
        HolisticLandmarkerFrameResult HolisticLandmarkerGraph(Mat image)
        {
            HolisticPoseTrackFrameResult poseFrame = TrackHolisticPose(image);
            var left = default(HolisticHandTrackFrameResult);
            var right = default(HolisticHandTrackFrameResult);
            var face = default(HolisticFaceTrackFrameResult);
            int iw = image.cols();
            int ih = image.rows();
            if (!poseFrame.PosePresence || iw <= 0 || ih <= 0)
                return new HolisticLandmarkerFrameResult(poseFrame, left, right, face);

            Vec3f[] normPose33 = HolisticCopyPoseNormLandmarks33(poseFrame.NormLandmarks);

            if (_outputLeftHandLandmarks || _outputLeftHandWorldLandmarks)
            {
                var req = new HolisticHandTrackingRequest(_outputLeftHandLandmarks, _outputLeftHandWorldLandmarks);
                left = TrackHolisticHand(image, normPose33, poseFrame.LandmarkVisibility, poseFrame.WorldLandmarks, HolisticLeftHandPoseIndices, req, isLeftHand: true);
            }

            if (_outputRightHandLandmarks || _outputRightHandWorldLandmarks)
            {
                var req = new HolisticHandTrackingRequest(_outputRightHandLandmarks, _outputRightHandWorldLandmarks);
                right = TrackHolisticHand(image, normPose33, poseFrame.LandmarkVisibility, poseFrame.WorldLandmarks, HolisticRightHandPoseIndices, req, isLeftHand: false);
            }

            if (_outputFaceLandmarks || _outputFaceBlendshapes)
                face = TrackHolisticFace(image, normPose33);

            return new HolisticLandmarkerFrameResult(poseFrame, left, right, face);
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="HolisticLandmarkerGraph"/> (via the <see cref="MultiBackendNet.forwardTaskAsync"/> path).
        /// Invoked only from <see cref="MediaPipeHolisticLandmarker.RunCoreProcessingTaskAsync"/>.
        /// </summary>
        async Task<HolisticLandmarkerFrameResult> HolisticLandmarkerGraphAsync(Mat image, CancellationToken cancellationToken)
        {
            HolisticPoseTrackFrameResult poseFrame = await TrackHolisticPoseAsync(image, cancellationToken);
            var left = default(HolisticHandTrackFrameResult);
            var right = default(HolisticHandTrackFrameResult);
            var face = default(HolisticFaceTrackFrameResult);
            int iw = image.cols();
            int ih = image.rows();
            if (!poseFrame.PosePresence || iw <= 0 || ih <= 0)
                return new HolisticLandmarkerFrameResult(poseFrame, left, right, face);

            Vec3f[] normPose33 = HolisticCopyPoseNormLandmarks33(poseFrame.NormLandmarks);

            if (_outputLeftHandLandmarks || _outputLeftHandWorldLandmarks)
            {
                var req = new HolisticHandTrackingRequest(_outputLeftHandLandmarks, _outputLeftHandWorldLandmarks);
                left = await TrackHolisticHandAsync(image, normPose33, poseFrame.LandmarkVisibility, poseFrame.WorldLandmarks, HolisticLeftHandPoseIndices, req, isLeftHand: true, cancellationToken);
            }

            if (_outputRightHandLandmarks || _outputRightHandWorldLandmarks)
            {
                var req = new HolisticHandTrackingRequest(_outputRightHandLandmarks, _outputRightHandWorldLandmarks);
                right = await TrackHolisticHandAsync(image, normPose33, poseFrame.LandmarkVisibility, poseFrame.WorldLandmarks, HolisticRightHandPoseIndices, req, isLeftHand: false, cancellationToken);
            }

            if (_outputFaceLandmarks || _outputFaceBlendshapes)
                face = await TrackHolisticFaceAsync(image, normPose33, cancellationToken);

            return new HolisticLandmarkerFrameResult(poseFrame, left, right, face);
        }
#endif

        /// <summary>
        /// Equivalent to <c>TrackHolisticHand</c> in
        /// <c>mediapipe/tasks/cc/vision/holistic_landmarker/holistic_hand_tracking.cc</c>.
        /// Section mapping: <see cref="HolisticHandTracking_PipelineSection5A_PosePalmLandmarksVisibilityAndAllowIf"/> ->
        /// <see cref="HolisticHandTracking_PipelineSection5B_HandRoiFromPosePalmLandmarks"/> →
        /// <see cref="HolisticHandTracking_PipelineSection5C_RefineHandRoi"/> →
        /// <see cref="HolisticHandTracking_PipelineSection5D_TrackHandRoi"/> →
        /// <see cref="HolisticHandTracking_PipelineSection5E_SingleHandLandmarksDetection"/> →
        /// <see cref="HolisticHandTracking_PipelineSection5F_BuildOutputAlignWorldSetPreviousLoopback"/>。
        /// Calculator mapping inside each section:
        /// <list type="bullet">
        /// <item><description>§5-A: <c>SplitAndCombine</c>, wrist extraction for visibility, <c>LandmarkVisibilityCalculator</c>, <c>ThresholdingCalculator</c>, and <c>AllowIf</c></description></item>
        /// <item><description>§5-B: <c>ImagePropertiesCalculator</c>、<c>LandmarksToDetectionCalculator</c>、<c>HandDetectionsFromPoseToRectsCalculator</c>、<c>RectTransformationCalculator</c>（2.7）</description></item>
        /// <item><description>§5-C: <c>HandRoiRefinementGraph</c></description></item>
        /// <item><description>§5-D: <c>PreviousLoopbackCalculator</c>、<c>HandLandmarksToRectCalculator</c>、<c>RectTransformationCalculator</c>（2.0）、<c>RoiTrackingCalculator</c></description></item>
        /// <item><description>§5-E: <c>SingleHandLandmarksDetectorGraph</c></description></item>
        /// <item><description>§5-F: <c>set_prev_landmarks_fn</c>、<c>AlignHandToPoseInWorldCalculator</c></description></item>
        /// </list>
        /// </summary>
        HolisticHandTrackFrameResult TrackHolisticHand(
            Mat image,
            Vec3f[] poseLandmarksNorm33,
            float[] poseLandmarkVisibility33,
            Vec3f[] poseWorld33,
            HolisticPoseHandIndices poseIndices,
            HolisticHandTrackingRequest request,
            bool isLeftHand)
        {
            int Lm = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            var empty = new HolisticHandTrackFrameResult
            {
                NormLandmarks = new Vec3f[Lm],
                WorldLandmarks = new Vec3f[Lm],
            };

            if (image == null || image.empty() || poseLandmarksNorm33 == null || poseLandmarksNorm33.Length < 33)
                return empty;

            if (!HolisticHandTracking_PipelineSection5A_PosePalmLandmarksVisibilityAndAllowIf(
                    poseLandmarksNorm33, poseLandmarkVisibility33, poseIndices, out Vec3f[] gatedPalm))
                return empty;

            (int w, int h) = ImagePropertiesCalculator_GetImageSize(image);
            if (w <= 0 || h <= 0)
                return empty;

            if (!HolisticHandTracking_PipelineSection5B_HandRoiFromPosePalmLandmarks(gatedPalm, w, h, out HolisticNormalizedRect roiFromPose))
                return empty;

            HolisticNormalizedRect roiFromRecrop = HolisticHandTracking_PipelineSection5C_RefineHandRoi(image, roiFromPose, w, h);

            HolisticNormalizedRect trackingRoi = HolisticHandTracking_PipelineSection5D_TrackHandRoi(isLeftHand, w, h, roiFromRecrop);
            if (!HolisticNormalizedRectIsPresent(trackingRoi))
                return empty;

            HolisticSingleHandGraphResult? single = HolisticHandTracking_PipelineSection5E_SingleHandLandmarksDetection(image, trackingRoi);
            if (!single.HasValue)
                return empty;

            var r = single.Value;
            // In upstream holistic_hand_tracking.cc, set_prev_landmarks_fn is always invoked immediately
            // after GetHandLandmarksDetection, even when the subgraph yields an empty-style output.
            if (!r.HandPresence)
            {
                SetPreviousHandLandmarksLoopback_Holistic(isLeftHand, r.NormLandmarks);
                return empty;
            }

            return HolisticHandTracking_PipelineSection5F_BuildOutputAlignWorldSetPreviousLoopback(
                r, request, poseWorld33, poseIndices, isLeftHand, Lm);
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<HolisticHandTrackFrameResult> TrackHolisticHandAsync(
            Mat image,
            Vec3f[] poseLandmarksNorm33,
            float[] poseLandmarkVisibility33,
            Vec3f[] poseWorld33,
            HolisticPoseHandIndices poseIndices,
            HolisticHandTrackingRequest request,
            bool isLeftHand,
            CancellationToken cancellationToken)
        {
            int Lm = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            var empty = new HolisticHandTrackFrameResult
            {
                NormLandmarks = new Vec3f[Lm],
                WorldLandmarks = new Vec3f[Lm],
            };

            if (image == null || image.empty() || poseLandmarksNorm33 == null || poseLandmarksNorm33.Length < 33)
                return empty;

            if (!HolisticHandTracking_PipelineSection5A_PosePalmLandmarksVisibilityAndAllowIf(
                    poseLandmarksNorm33, poseLandmarkVisibility33, poseIndices, out Vec3f[] gatedPalm))
                return empty;

            (int w, int h) = ImagePropertiesCalculator_GetImageSize(image);
            if (w <= 0 || h <= 0)
                return empty;

            if (!HolisticHandTracking_PipelineSection5B_HandRoiFromPosePalmLandmarks(gatedPalm, w, h, out HolisticNormalizedRect roiFromPose))
                return empty;

            HolisticNormalizedRect roiFromRecrop = await HolisticHandTracking_PipelineSection5C_RefineHandRoiAsync(image, roiFromPose, w, h, cancellationToken);

            HolisticNormalizedRect trackingRoi = HolisticHandTracking_PipelineSection5D_TrackHandRoi(isLeftHand, w, h, roiFromRecrop);
            if (!HolisticNormalizedRectIsPresent(trackingRoi))
                return empty;

            HolisticSingleHandGraphResult? single = await HolisticHandTracking_PipelineSection5E_SingleHandLandmarksDetectionAsync(image, trackingRoi, cancellationToken);
            if (!single.HasValue)
                return empty;

            var r = single.Value;
            if (!r.HandPresence)
            {
                SetPreviousHandLandmarksLoopback_Holistic(isLeftHand, r.NormLandmarks);
                return empty;
            }

            return HolisticHandTracking_PipelineSection5F_BuildOutputAlignWorldSetPreviousLoopback(
                r, request, poseWorld33, poseIndices, isLeftHand, Lm);
        }
#endif

        /// <summary>
        /// Equivalent to the <c>SplitAndCombine</c> -> <c>GetPosePalmVisibility</c> -> <c>AllowIf</c>
        /// path in <c>holistic_hand_tracking.cc</c>, acting as the gate for the three-point palm stream.
        /// Assumes <paramref name="poseLandmarksNorm33"/> has at least 33 elements.
        /// </summary>
        bool HolisticHandTracking_PipelineSection5A_PosePalmLandmarksVisibilityAndAllowIf(
            Vec3f[] poseLandmarksNorm33,
            float[] poseLandmarkVisibility33,
            HolisticPoseHandIndices poseIndices,
            out Vec3f[] gatedPalmThree)
        {
            Vec3f[] palm3 = SplitNormalizedLandmarkListCalculator_SplitAndCombine_HolisticHandPosePalm(poseLandmarksNorm33, poseIndices);
            _ = SplitNormalizedLandmarkListCalculator_SplitToWristOnly_HolisticHandPosePalm(palm3);
            float visScore = LandmarkVisibilityCalculator_HolisticPosePalmWrist(poseLandmarkVisibility33, poseIndices.WristIdx);
            bool palmVisible = ThresholdingCalculator_HolisticPosePalmVisibility(visScore);
            gatedPalmThree = GateCalculator_AllowIf_HolisticPosePalmLandmarks(palm3, palmVisible);
            return gatedPalmThree != null;
        }

        /// <summary>
        /// Equivalent to <c>GetHandRoiFromPosePalmLandmarks</c>:
        /// detection conversion -> rect conversion -> <c>ScaleAndShiftAndMakeSquareLong</c> with
        /// <c>scale=2.7</c> and <c>shift_y=-0.1</c>.
        /// </summary>
        bool HolisticHandTracking_PipelineSection5B_HandRoiFromPosePalmLandmarks(
            Vec3f[] gatedPalmThree,
            int imageWidth,
            int imageHeight,
            out HolisticNormalizedRect roiFromPose)
        {
            HolisticPoseDetectionData detPosePalm = LandmarksToDetectionCalculator_ConvertLandmarksToDetection_HolisticPosePalm(gatedPalmThree);
            HolisticNormalizedRect rectFromPoseDetection = HandDetectionsFromPoseToRectsCalculator(detPosePalm, imageWidth, imageHeight);
            roiFromPose = RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HolisticHandRoiFromPose(rectFromPoseDetection, imageWidth, imageHeight);
            return HolisticNormalizedRectIsPresent(roiFromPose);
        }

        /// <summary>Equivalent to <c>RefineHandRoi</c>; returns the input ROI unchanged on failure.</summary>
        HolisticNormalizedRect HolisticHandTracking_PipelineSection5C_RefineHandRoi(
            Mat image,
            HolisticNormalizedRect roiFromPose,
            int imageWidth,
            int imageHeight)
        {
            if (!HandRoiRefinementGraph(image, roiFromPose, imageWidth, imageHeight, out HolisticNormalizedRect roiFromRecrop))
                roiFromRecrop = roiFromPose;
            return roiFromRecrop;
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<HolisticNormalizedRect> HolisticHandTracking_PipelineSection5C_RefineHandRoiAsync(
            Mat image,
            HolisticNormalizedRect roiFromPose,
            int imageWidth,
            int imageHeight,
            CancellationToken cancellationToken)
        {
            var (ok, roiFromRecrop) = await HandRoiRefinementGraphAsync(image, roiFromPose, imageWidth, imageHeight, cancellationToken);
            if (!ok)
                roiFromRecrop = roiFromPose;
            return roiFromRecrop;
        }
#endif

        /// <summary>Equivalent to <c>TrackHandRoi</c>, converting previous-landmark loopback into a tracking ROI.</summary>
        HolisticNormalizedRect HolisticHandTracking_PipelineSection5D_TrackHandRoi(
            bool isLeftHand,
            int imageWidth,
            int imageHeight,
            HolisticNormalizedRect roiFromRecrop)
        {
            Vec3f[] prevNormLm = PreviousLoopbackCalculator_HolisticHandPrevLandmarks(isLeftHand, imageWidth, imageHeight);
            HolisticNormalizedRect? prevRoiNullable = null;
            if (prevNormLm != null)
            {
                var tightNorm = HandLandmarksToRectCalculator_HolisticNormalized(prevNormLm, imageWidth, imageHeight);
                prevRoiNullable = RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HolisticPrevHandLandmarksRoi(tightNorm, imageWidth, imageHeight);
            }

            return RoiTrackingCalculator_HolisticHand(prevNormLm, prevRoiNullable, roiFromRecrop, imageWidth, imageHeight);
        }

        /// <summary>Equivalent to <c>GetHandLandmarksDetection</c> for the single-hand landmark subgraph.</summary>
        HolisticSingleHandGraphResult? HolisticHandTracking_PipelineSection5E_SingleHandLandmarksDetection(
            Mat image,
            HolisticNormalizedRect trackingRoi)
        {
            return SingleHandLandmarksDetectorGraph(image, trackingRoi);
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<HolisticSingleHandGraphResult?> HolisticHandTracking_PipelineSection5E_SingleHandLandmarksDetectionAsync(
            Mat image,
            HolisticNormalizedRect trackingRoi,
            CancellationToken cancellationToken)
        {
            return await SingleHandLandmarksDetectorGraphAsync(image, trackingRoi, cancellationToken);
        }
#endif

        /// <summary>
        /// Builds the result after successful detection, aligns world landmarks, and updates the loopback
        /// state equivalent to <c>set_prev_landmarks_fn</c>.
        /// </summary>
        HolisticHandTrackFrameResult HolisticHandTracking_PipelineSection5F_BuildOutputAlignWorldSetPreviousLoopback(
            HolisticSingleHandGraphResult r,
            HolisticHandTrackingRequest request,
            Vec3f[] poseWorld33,
            HolisticPoseHandIndices poseIndices,
            bool isLeftHand,
            int landmarkCount)
        {
            Vec3f[] worldOut = r.WorldLandmarks;
            if (request.PackWorldLandmarks && poseWorld33 != null && poseWorld33.Length >= 33)
                worldOut = AlignHandToPoseInWorldCalculator(r.WorldLandmarks, poseWorld33, poseIndices.WristIdx);

            SetPreviousHandLandmarksLoopback_Holistic(isLeftHand, r.NormLandmarks);

            return new HolisticHandTrackFrameResult
            {
                HandPresence = true,
                NormLandmarks = r.NormLandmarks,
                WorldLandmarks = worldOut ?? new Vec3f[landmarkCount],
                Handedness = r.Handedness,
                PresenceConfidence = r.PresenceConfidence,
            };
        }

        /// <summary>
        /// Snapshot of the main 33 pose landmarks. The upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// equivalent <c>NormalizedLandmark</c> data comes from <see cref="TrackHolisticPose"/> output
        /// via <see cref="HolisticPoseTrackFrameResult.NormLandmarks"/>.
        /// </summary>
        Vec3f[] HolisticCopyPoseNormLandmarks33(Vec3f[] normLandmarks33)
        {
            if (_hhPoseNorm33Scratch == null)
                _hhPoseNorm33Scratch = new Vec3f[33];
            for (int i = 0; i < 33; i++)
                _hhPoseNorm33Scratch[i] = default;
            if (normLandmarks33 != null)
            {
                for (int i = 0; i < 33 && i < normLandmarks33.Length; i++)
                    _hhPoseNorm33Scratch[i] = normLandmarks33[i];
            }
            return _hhPoseNorm33Scratch;
        }

        /// <summary>Equivalent to <c>SplitAndCombine</c> for wrist, pinky, and index landmarks.</summary>
        Vec3f[] SplitNormalizedLandmarkListCalculator_SplitAndCombine_HolisticHandPosePalm(Vec3f[] poseNorm33, HolisticPoseHandIndices idx)
        {
            _hhHolisticHandPalmThreeScratch[0] = poseNorm33[idx.WristIdx];
            _hhHolisticHandPalmThreeScratch[1] = poseNorm33[idx.PinkyIdx];
            _hhHolisticHandPalmThreeScratch[2] = poseNorm33[idx.IndexIdx];
            return _hhHolisticHandPalmThreeScratch;
        }

        /// <summary>Returns only the wrist landmark from the palm triplet, for visibility evaluation.</summary>
        static Vec3f SplitNormalizedLandmarkListCalculator_SplitToWristOnly_HolisticHandPosePalm(Vec3f[] palmThree)
        {
            return palmThree != null && palmThree.Length > 0 ? palmThree[0] : default;
        }

        /// <summary>Equivalent to <c>LandmarkVisibilityCalculator</c> from NORM_LANDMARKS to VISIBILITY.</summary>
        static float LandmarkVisibilityCalculator_HolisticPosePalmWrist(float[] poseVisibility33, int wristIdx)
        {
            if (poseVisibility33 == null || wristIdx < 0 || wristIdx >= poseVisibility33.Length)
                return 0f;
            return poseVisibility33[wristIdx];
        }

        /// <summary>Equivalent to <c>ThresholdingCalculator</c> / <c>IsOverThreshold</c> with threshold 0.1, using <c>value &gt; threshold</c> exactly as in upstream <c>thresholding_calculator.cc</c>.</summary>
        static bool ThresholdingCalculator_HolisticPosePalmVisibility(float visibilityScore)
        {
            return visibilityScore > 0.1f;
        }

        /// <summary>Equivalent to <c>GateCalculator</c> in AllowIf mode; disables the palm stream when false.</summary>
        static Vec3f[] GateCalculator_AllowIf_HolisticPosePalmLandmarks(Vec3f[] palmThree, bool allow)
        {
            return allow ? palmThree : null;
        }

        /// <summary>Equivalent to <c>LandmarksToDetectionCalculator</c> with <c>ConvertLandmarksToDetection</c> for the three palm landmarks.</summary>
        HolisticPoseDetectionData LandmarksToDetectionCalculator_ConvertLandmarksToDetection_HolisticPosePalm(Vec3f[] palmNormThree)
        {
            float xmin = float.MaxValue, ymin = float.MaxValue, xmax = float.MinValue, ymax = float.MinValue;
            for (int i = 0; i < 3; i++)
            {
                float rx = palmNormThree[i].Item1;
                float ry = palmNormThree[i].Item2;
                xmin = Mathf.Min(xmin, rx);
                xmax = Mathf.Max(xmax, rx);
                ymin = Mathf.Min(ymin, ry);
                ymax = Mathf.Max(ymax, ry);
                _hhLandmarksToDetKp6[i * 2] = rx;
                _hhLandmarksToDetKp6[i * 2 + 1] = ry;
            }
            return new HolisticPoseDetectionData
            {
                RelXmin = xmin,
                RelYmin = ymin,
                RelWidth = xmax - xmin,
                RelHeight = ymax - ymin,
                RelKeypointsXy = _hhLandmarksToDetKp6,
                Score = 1f,
            };
        }

        /// <summary>Equivalent to <c>HandDetectionsFromPoseToRectsCalculator</c> in <c>hand_detections_from_pose_to_rects_calculator.cc</c>.</summary>
        static HolisticNormalizedRect HandDetectionsFromPoseToRectsCalculator(HolisticPoseDetectionData d, int imageWidth, int imageHeight)
        {
            float[] kp = d.RelKeypointsXy;
            if (kp == null || kp.Length < 6 || imageWidth <= 0 || imageHeight <= 0)
                return default;

            const int kWrist = 0, kPinky = 1, kIndex = 2;
            float xW = kp[kWrist * 2] * imageWidth;
            float yW = kp[kWrist * 2 + 1] * imageHeight;
            float xI = kp[kIndex * 2] * imageWidth;
            float yI = kp[kIndex * 2 + 1] * imageHeight;
            float xP = kp[kPinky * 2] * imageWidth;
            float yP = kp[kPinky * 2 + 1] * imageHeight;
            float xMiddle = (2f * xI + xP) / 3f;
            float yMiddle = (2f * yI + yP) / 3f;
            float boxSize = Mathf.Sqrt((xMiddle - xW) * (xMiddle - xW) + (yMiddle - yW) * (yMiddle - yW)) * 2f;
            float rot = HolisticNormalizeRadiansHand(
                Mathf.PI * 0.5f - Mathf.Atan2(-(yMiddle - yW), xMiddle - xW));
            return new HolisticNormalizedRect
            {
                XCenter = xMiddle / imageWidth,
                YCenter = yMiddle / imageHeight,
                Width = boxSize / imageWidth,
                Height = boxSize / imageHeight,
                Rotation = rot,
            };
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> with <c>ScaleAndShiftAndMakeSquareLong</c>,
        /// using <c>scale=2.7</c> and <c>shift_y=-0.1</c> for
        /// <c>GetHandRoiFromPosePalmLandmarks</c>.
        /// </summary>
        static HolisticNormalizedRect RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HolisticHandRoiFromPose(
            HolisticNormalizedRect r, int imgW, int imgH)
        {
            return RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_Internal(r, imgW, imgH, 2.7f, 2.7f, 0f, -0.1f);
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> with <c>scale=2.0</c> and
        /// <c>shift_y=-0.1</c> for the previous ROI used inside <c>TrackHandRoi</c>.
        /// </summary>
        static HolisticNormalizedRect RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HolisticPrevHandLandmarksRoi(
            HolisticNormalizedRect r, int imgW, int imgH)
        {
            return RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_Internal(r, imgW, imgH, 2f, 2f, 0f, -0.1f);
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> with <c>scale=1.0</c> and
        /// <c>shift_y=-0.1</c> at the end of <c>HandRoiRefinementGraph</c>.
        /// </summary>
        static HolisticNormalizedRect RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HandRoiRefinementOutput(
            HolisticNormalizedRect r, int imgW, int imgH)
        {
            return RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_Internal(r, imgW, imgH, 1f, 1f, 0f, -0.1f);
        }

        /// <summary>
        /// Uses the same formula as <see cref="MediaPipeHandLandmarker"/>'s
        /// <c>RectTransformationCalculator_SingleHandLandmarks</c>, with configurable coefficients.
        /// </summary>
        static HolisticNormalizedRect RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_Internal(
            HolisticNormalizedRect handRect, int imgW, int imgH,
            float scaleX, float scaleY, float shiftX, float shiftY)
        {
            if (imgW <= 0 || imgH <= 0)
                return default;

            float rotation = handRect.Rotation;
            float cosR = Mathf.Cos(rotation);
            float sinR = Mathf.Sin(rotation);
            float widthPx = handRect.Width * imgW;
            float heightPx = handRect.Height * imgH;
            float xCenterNorm = handRect.XCenter;
            float yCenterNorm = handRect.YCenter;
            float widthNorm = handRect.Width;
            float heightNorm = handRect.Height;
            float xShiftNorm = (imgW * widthNorm * shiftX * cosR - imgH * heightNorm * shiftY * sinR) / imgW;
            float yShiftNorm = (imgW * widthNorm * shiftX * sinR + imgH * heightNorm * shiftY * cosR) / imgH;
            xCenterNorm += xShiftNorm;
            yCenterNorm += yShiftNorm;
            float longSidePx = Mathf.Max(widthPx, heightPx);
            widthNorm = longSidePx / imgW;
            heightNorm = longSidePx / imgH;
            widthNorm *= scaleX;
            heightNorm *= scaleY;
            return new HolisticNormalizedRect
            {
                XCenter = xCenterNorm,
                YCenter = yCenterNorm,
                Width = widthNorm,
                Height = heightNorm,
                Rotation = rotation,
            };
        }

        /// <summary>
        /// Equivalent to <c>HandRoiRefinementGraph</c> in
        /// <c>mediapipe/tasks/cc/vision/hand_landmarker/hand_roi_refinement_graph.cc</c>.
        /// Child path: <see cref="ImagePreprocessingGraph_HandRoiRefinement"/> -> <see cref="InferenceSubgraph_HandRoiRefinement"/> ->
        /// <see cref="TensorsToLandmarksCalculator_HandRoiRefinement"/> → <see cref="LandmarkProjectionCalculator_HandRoiRefinement"/> →
        /// <see cref="LandmarksToDetectionCalculator_HandRoiRefinementTwoPoints"/> → <see cref="AlignmentPointsRectsCalculator_HandRoiRefinement"/> →
        /// <see cref="RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HandRoiRefinementOutput"/>。
        /// </summary>
        bool HandRoiRefinementGraph(Mat image, HolisticNormalizedRect roi, int imgW, int imgH, out HolisticNormalizedRect refined)
        {
            refined = default;
            if (_handRoiRefinementNet == null || _handRoiRefinementNetOutLayerNames == null)
                return false;
            if (!ImagePreprocessingGraph_HandRoiRefinement(image, roi, imgW, imgH, out HolisticHandRoiRefinePreprocessOut pre))
                return false;

            List<Mat> tensors = InferenceSubgraph_HandRoiRefinement(pre.RoiBlob);
            if (tensors == null || tensors.Count == 0 || tensors[0] == null || tensors[0].empty())
                return false;

            using (Mat flat = tensors[0].reshape(1, 1))
            {
                if (flat.cols() < 4 || flat.rows() < 1)
                    return false;
                flat.get(0, 0, _hhRoiRefineRaw4.AsSpan(0, 4));
                Vec3f[] normTwo = TensorsToLandmarksCalculator_HandRoiRefinement(_hhRoiRefineRaw4, kHolisticHandRoiRefineInputSize, kHolisticHandRoiRefineInputSize);
                Vec3f[] projected = LandmarkProjectionCalculator_HandRoiRefinement(normTwo, _hhRoiRefineMatrix16);
                HolisticPoseDetectionData det = LandmarksToDetectionCalculator_HandRoiRefinementTwoPoints(projected);
                HolisticNormalizedRect alignRect = AlignmentPointsRectsCalculator_HandRoiRefinement(det, imgW, imgH);
                refined = RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HandRoiRefinementOutput(alignRect, imgW, imgH);
                return HolisticNormalizedRectIsPresent(refined);
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<(bool ok, HolisticNormalizedRect refined)> HandRoiRefinementGraphAsync(
            Mat image, HolisticNormalizedRect roi, int imgW, int imgH, CancellationToken cancellationToken)
        {
            var fail = (false, default(HolisticNormalizedRect));
            if (_handRoiRefinementNet == null || _handRoiRefinementNetOutLayerNames == null)
                return fail;
            if (!ImagePreprocessingGraph_HandRoiRefinement(image, roi, imgW, imgH, out HolisticHandRoiRefinePreprocessOut pre))
                return fail;

            var tensors = await InferenceSubgraph_HandRoiRefinementAsync(pre.RoiBlob, cancellationToken);
            if (tensors == null || tensors.Count == 0 || tensors[0] == null || tensors[0].empty())
                return fail;

            using (Mat flat = tensors[0].reshape(1, 1))
            {
                if (flat.cols() < 4 || flat.rows() < 1)
                    return fail;
                flat.get(0, 0, _hhRoiRefineRaw4.AsSpan(0, 4));
                Vec3f[] normTwo = TensorsToLandmarksCalculator_HandRoiRefinement(_hhRoiRefineRaw4, kHolisticHandRoiRefineInputSize, kHolisticHandRoiRefineInputSize);
                Vec3f[] projected = LandmarkProjectionCalculator_HandRoiRefinement(normTwo, _hhRoiRefineMatrix16);
                HolisticPoseDetectionData det = LandmarksToDetectionCalculator_HandRoiRefinementTwoPoints(projected);
                HolisticNormalizedRect alignRect = AlignmentPointsRectsCalculator_HandRoiRefinement(det, imgW, imgH);
                HolisticNormalizedRect refined = RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_HandRoiRefinementOutput(alignRect, imgW, imgH);
                return (HolisticNormalizedRectIsPresent(refined), refined);
            }
        }
#endif

        struct HolisticHandRoiRefinePreprocessOut
        {
            public Mat RoiBlob;
        }

        /// <summary>Equivalent to <c>ImagePreprocessingGraph</c> with <c>256</c>, <c>BORDER_REPLICATE</c>, and <c>keep_aspect_ratio</c>.</summary>
        bool ImagePreprocessingGraph_HandRoiRefinement(Mat image, HolisticNormalizedRect normRect, int imgW, int imgH, out HolisticHandRoiRefinePreprocessOut pre)
        {
            pre = default;
            const int inputSize = kHolisticHandRoiRefineInputSize;
            HolisticHandDetectorGetRoi(imgW, imgH, normRect, out float roiCx, out float roiCy, out float roiW, out float roiH, out float roiRot);
            HolisticHandDetectorPadRoi(inputSize, inputSize, true, ref roiW, ref roiH);
            HolisticGetRotatedSubRectToRectTransformMatrix(roiCx, roiCy, roiW, roiH, roiRot, imgW, imgH, false, _hhRoiRefineMatrix16);

            if (_hhRoiRefineDstPts == null)
            {
                _hhRoiRefineDstPts = new Mat(4, 2, CvType.CV_32FC1);
                Span<float> dstPtsArr = stackalloc float[8];
                float dw = inputSize, dh = inputSize;
                dstPtsArr[0] = 0f; dstPtsArr[1] = dh;
                dstPtsArr[2] = 0f; dstPtsArr[3] = 0f;
                dstPtsArr[4] = dw; dstPtsArr[5] = 0f;
                dstPtsArr[6] = dw; dstPtsArr[7] = dh;
                _hhRoiRefineDstPts.put(0, 0, dstPtsArr);
                _hhRoiRefineSrcPts = new Mat(4, 2, CvType.CV_32FC1);
                _hhRoiRefineWarpedBgr = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _hhRoiRefineWarpedRgb = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _hhRoiRefineBlob = new Mat(new int[] { 1, inputSize, inputSize, 3 }, CvType.CV_32FC1);
                _hhRoiRefineBlobHxW = _hhRoiRefineBlob.reshape(3, new int[] { inputSize, inputSize });
            }

            if (roiW <= 0f || roiH <= 0f || float.IsNaN(roiCx))
                return false;

            double angleDeg = roiRot * (180.0 / Math.PI);
            Imgproc.boxPoints((roiCx, roiCy, roiW, roiH, angleDeg), _hhRoiRefineSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_hhRoiRefineSrcPts, _hhRoiRefineDstPts))
            {
                Imgproc.warpPerspective(image, _hhRoiRefineWarpedBgr, projMat, (inputSize, inputSize),
                    Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE, (0d, 0d, 0d, 0d));
            }
            Imgproc.cvtColor(_hhRoiRefineWarpedBgr, _hhRoiRefineWarpedRgb, Imgproc.COLOR_BGR2RGB);
            _hhRoiRefineWarpedRgb.convertTo(_hhRoiRefineBlobHxW, CvType.CV_32F, 1.0 / 255.0);

            pre = new HolisticHandRoiRefinePreprocessOut { RoiBlob = _hhRoiRefineBlob };
            return true;
        }

        /// <summary>Equivalent to the <c>InferenceSubgraph</c> used by <c>hand_roi_refinement</c>.</summary>
        List<Mat> InferenceSubgraph_HandRoiRefinement(Mat roiBlob)
        {
            if (_handRoiRefinementNet == null || _handRoiRefinementNetOutLayerNames == null)
            {
                _hhHandRoiRefinementForwardOutputList.Clear();
                return _hhHandRoiRefinementForwardOutputList;
            }

            _handRoiRefinementNet.setInput(roiBlob);
            _hhHandRoiRefinementForwardOutputList.Clear();
            _handRoiRefinementNet.forward(_hhHandRoiRefinementForwardOutputList, _handRoiRefinementNetOutLayerNames);
            return _hhHandRoiRefinementForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<List<Mat>> InferenceSubgraph_HandRoiRefinementAsync(Mat roiBlob, CancellationToken cancellationToken)
        {
            if (_handRoiRefinementNet == null || _handRoiRefinementNetOutLayerNames == null)
            {
                _hhHandRoiRefinementForwardOutputList.Clear();
                return _hhHandRoiRefinementForwardOutputList;
            }

            _hhHandRoiRefinementForwardOutputList.Clear();
            _handRoiRefinementNet.setInput(roiBlob);
            await _handRoiRefinementNet.forwardTaskAsync(_hhHandRoiRefinementForwardOutputList, _handRoiRefinementNetOutLayerNames, cancellationToken);
            return _hhHandRoiRefinementForwardOutputList;
        }
#endif

        /// <summary>Equivalent to <c>TensorsToLandmarksCalculator</c> with <c>num_landmarks=2</c> and <c>normalize_z=1</c>.</summary>
        Vec3f[] TensorsToLandmarksCalculator_HandRoiRefinement(float[] raw4, int inputW, int inputH)
        {
            float zDenom = inputW * 1f;
            if (zDenom < 1e-8f) zDenom = 1f;
            _hhRoiRefineNormTwoScratch[0] = new Vec3f(raw4[0] / inputW, raw4[1] / inputH, 0f);
            _hhRoiRefineNormTwoScratch[1] = new Vec3f(raw4[2] / inputW, raw4[3] / inputH, 0f);
            return _hhRoiRefineNormTwoScratch;
        }

        /// <summary>Equivalent to <c>LandmarkProjectionCalculator</c> with <c>PROJECTION_MATRIX</c>, following <c>ProjectXY</c> in <c>landmark_projection_calculator.cc</c>.</summary>
        Vec3f[] LandmarkProjectionCalculator_HandRoiRefinement(Vec3f[] normLandmarks, float[] m16)
        {
            float zs = HolisticLandmarkProjectionCalculateZScale(m16);
            for (int i = 0; i < normLandmarks.Length; i++)
            {
                var lm = normLandmarks[i];
                float ox = lm.Item1 * m16[0] + lm.Item2 * m16[1] + lm.Item3 * m16[2] + m16[3];
                float oy = lm.Item1 * m16[4] + lm.Item2 * m16[5] + lm.Item3 * m16[6] + m16[7];
                _hhRoiRefineProjectedTwoScratch[i] = new Vec3f(ox, oy, zs * lm.Item3);
            }
            return _hhRoiRefineProjectedTwoScratch;
        }

        static float HolisticLandmarkProjectionCalculateZScale(float[] m16)
        {
            void Proj(float nx, float ny, out float ox, out float oy)
            {
                ox = nx * m16[0] + ny * m16[1] + m16[3];
                oy = nx * m16[4] + ny * m16[5] + m16[7];
            }
            Proj(0f, 0f, out float ax, out float ay);
            Proj(1f, 0f, out float bx, out float by);
            float dx = bx - ax, dy = by - ay;
            return Mathf.Sqrt(dx * dx + dy * dy);
        }

        HolisticPoseDetectionData LandmarksToDetectionCalculator_HandRoiRefinementTwoPoints(Vec3f[] projectedNormTwo)
        {
            _hhRoiRefineKp4[0] = projectedNormTwo[0].Item1;
            _hhRoiRefineKp4[1] = projectedNormTwo[0].Item2;
            _hhRoiRefineKp4[2] = projectedNormTwo[1].Item1;
            _hhRoiRefineKp4[3] = projectedNormTwo[1].Item2;
            float xmin = Mathf.Min(_hhRoiRefineKp4[0], _hhRoiRefineKp4[2]);
            float xmax = Mathf.Max(_hhRoiRefineKp4[0], _hhRoiRefineKp4[2]);
            float ymin = Mathf.Min(_hhRoiRefineKp4[1], _hhRoiRefineKp4[3]);
            float ymax = Mathf.Max(_hhRoiRefineKp4[1], _hhRoiRefineKp4[3]);
            return new HolisticPoseDetectionData
            {
                RelXmin = xmin,
                RelYmin = ymin,
                RelWidth = xmax - xmin,
                RelHeight = ymax - ymin,
                RelKeypointsXy = _hhRoiRefineKp4,
                Score = 1f,
            };
        }

        /// <summary>Equivalent to <c>AlignmentPointsRectsCalculator</c> with <c>start=0</c>, <c>end=1</c>, and <c>target_angle=-90°</c>.</summary>
        static HolisticNormalizedRect AlignmentPointsRectsCalculator_HandRoiRefinement(HolisticPoseDetectionData d, int iw, int ih)
        {
            float[] kp = d.RelKeypointsXy;
            if (kp == null || kp.Length < 4 || iw <= 0 || ih <= 0)
                return default;
            float x0 = kp[0] * iw;
            float y0 = kp[1] * ih;
            float x1 = kp[2] * iw;
            float y1 = kp[3] * ih;
            float boxSize = Mathf.Sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)) * 2f;
            float rot = HolisticNormalizeRadiansHand(-Mathf.PI * 0.5f - Mathf.Atan2(-(y1 - y0), x1 - x0));
            float xc = kp[0];
            float yc = kp[1];
            return new HolisticNormalizedRect
            {
                XCenter = xc,
                YCenter = yc,
                Width = boxSize / iw,
                Height = boxSize / ih,
                Rotation = rot,
            };
        }

        /// <summary>Equivalent to <c>PreviousLoopbackCalculator</c> with <c>GetLoopbackData&lt;NormalizedLandmarkList&gt;</c>.</summary>
        Vec3f[] PreviousLoopbackCalculator_HolisticHandPrevLandmarks(bool isLeftHand, int imageW, int imageH)
        {
            _ = imageW;
            _ = imageH;
            return isLeftHand ? _holisticLeftHandPrevNormLandmarks : _holisticRightHandPrevNormLandmarks;
        }

        void SetPreviousHandLandmarksLoopback_Holistic(bool isLeftHand, Vec3f[] normLandmarks21)
        {
            if (isLeftHand)
                _holisticLeftHandPrevNormLandmarks = normLandmarks21 != null ? (Vec3f[])normLandmarks21.Clone() : null;
            else
                _holisticRightHandPrevNormLandmarks = normLandmarks21 != null ? (Vec3f[])normLandmarks21.Clone() : null;
        }

        /// <summary>
        /// Normalized-landmark version of
        /// <c>modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc</c> for all 21
        /// points, where x/y use image-normalized coordinates.
        /// </summary>
        static HolisticNormalizedRect HandLandmarksToRectCalculator_HolisticNormalized(Vec3f[] normLm, int imgW, int imgH)
        {
            int L = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            if (normLm == null || normLm.Length < L || imgW <= 0 || imgH <= 0)
                return default;

            int[] partialIndices = { 0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18 };
            const int kWrist = 0;
            const int kIndexFingerMcp = 5;
            const int kMiddleFingerMcp = 9;
            const int kRingFingerMcp = 13;

            float x0 = normLm[kWrist].Item1 * imgW;
            float y0 = normLm[kWrist].Item2 * imgH;
            float x1 = (normLm[kIndexFingerMcp].Item1 * imgW + normLm[kRingFingerMcp].Item1 * imgW) * 0.5f;
            float y1 = (normLm[kIndexFingerMcp].Item2 * imgH + normLm[kRingFingerMcp].Item2 * imgH) * 0.5f;
            x1 = (x1 + normLm[kMiddleFingerMcp].Item1 * imgW) * 0.5f;
            y1 = (y1 + normLm[kMiddleFingerMcp].Item2 * imgH) * 0.5f;
            float rotation = HolisticNormalizeRadiansHand(Mathf.PI * 0.5f - Mathf.Atan2(-(y1 - y0), x1 - x0));
            float reverseAngle = HolisticNormalizeRadiansHand(-rotation);
            float cosR = Mathf.Cos(rotation);
            float sinR = Mathf.Sin(rotation);
            float cosRev = Mathf.Cos(reverseAngle);
            float sinRev = Mathf.Sin(reverseAngle);

            float minAx = float.MaxValue, minAy = float.MaxValue, maxAx = float.MinValue, maxAy = float.MinValue;
            foreach (int i in partialIndices)
            {
                float px = normLm[i].Item1 * imgW;
                float py = normLm[i].Item2 * imgH;
                minAx = Mathf.Min(minAx, px);
                minAy = Mathf.Min(minAy, py);
                maxAx = Mathf.Max(maxAx, px);
                maxAy = Mathf.Max(maxAy, py);
            }
            float axisCenterX = (minAx + maxAx) * 0.5f;
            float axisCenterY = (minAy + maxAy) * 0.5f;

            float minPx = float.MaxValue, minPy = float.MaxValue, maxPx = float.MinValue, maxPy = float.MinValue;
            foreach (int i in partialIndices)
            {
                float origX = normLm[i].Item1 * imgW - axisCenterX;
                float origY = normLm[i].Item2 * imgH - axisCenterY;
                float projX = origX * cosRev - origY * sinRev;
                float projY = origX * sinRev + origY * cosRev;
                minPx = Mathf.Min(minPx, projX);
                minPy = Mathf.Min(minPy, projY);
                maxPx = Mathf.Max(maxPx, projX);
                maxPy = Mathf.Max(maxPy, projY);
            }
            float widthPx = maxPx - minPx;
            float heightPx = maxPy - minPy;
            float projCenterX = (minPx + maxPx) * 0.5f;
            float projCenterY = (minPy + maxPy) * 0.5f;
            float centerX = projCenterX * cosR - projCenterY * sinR + axisCenterX;
            float centerY = projCenterX * sinR + projCenterY * cosR + axisCenterY;

            return new HolisticNormalizedRect
            {
                XCenter = centerX / imgW,
                YCenter = centerY / imgH,
                Width = widthPx / imgW,
                Height = heightPx / imgH,
                Rotation = rotation,
            };
        }

        /// <summary>Equivalent to <c>RoiTrackingCalculator</c> in <c>roi_tracking_calculator.cc</c>, using the Holistic-hand options.</summary>
        static HolisticNormalizedRect RoiTrackingCalculator_HolisticHand(
            Vec3f[] prevNormLandmarks21,
            HolisticNormalizedRect? prevRoiFromLandmarks,
            HolisticNormalizedRect recropRect,
            int imageWidth,
            int imageHeight)
        {
            if (!HolisticNormalizedRectIsPresent(recropRect))
                return default;
            if (prevRoiFromLandmarks == null || !HolisticNormalizedRectIsPresent(prevRoiFromLandmarks.Value))
                return recropRect;

            var prevRoi = prevRoiFromLandmarks.Value;
            if (!RectRequirementsSatisfied_HolisticRoiTracking(prevRoi, recropRect, imageWidth, imageHeight,
                    rotationDegrees: 40f, translation: 0.2f, scale: 0.4f))
                return recropRect;
            if (prevNormLandmarks21 == null ||
                !LandmarksRequirementsSatisfied_HolisticRoiTracking(prevNormLandmarks21, recropRect, imageWidth, imageHeight, recropRectMargin: -0.1f))
                return recropRect;
            return prevRoi;
        }

        static bool RectRequirementsSatisfied_HolisticRoiTracking(
            HolisticNormalizedRect prevRect, HolisticNormalizedRect recropRect,
            int imageWidth, int imageHeight,
            float rotationDegrees, float translation, float scale)
        {
            float rotation = -recropRect.Rotation;
            float cosa = Mathf.Cos(rotation);
            float sina = Mathf.Sin(rotation);

            float prev_rect_x = prevRect.XCenter * imageWidth * cosa - prevRect.YCenter * imageHeight * sina;
            float prev_rect_y = prevRect.XCenter * imageWidth * sina + prevRect.YCenter * imageHeight * cosa;
            float prev_rect_width = prevRect.Width * imageWidth;
            float prev_rect_height = prevRect.Height * imageHeight;
            float prev_rect_rotation = prevRect.Rotation * Mathf.Rad2Deg;

            float recrop_rect_x = recropRect.XCenter * imageWidth * cosa - recropRect.YCenter * imageHeight * sina;
            float recrop_rect_y = recropRect.XCenter * imageWidth * sina + recropRect.YCenter * imageHeight * cosa;
            float recrop_rect_width = recropRect.Width * imageWidth;
            float recrop_rect_height = recropRect.Height * imageHeight;
            float recrop_rect_rotation = recropRect.Rotation * Mathf.Rad2Deg;

            float rotationDiff = prev_rect_rotation - recrop_rect_rotation;
            if (rotationDiff > 180f) rotationDiff -= 360f;
            if (rotationDiff < -180f) rotationDiff += 360f;
            rotationDiff = Mathf.Abs(rotationDiff);
            if (rotationDiff > rotationDegrees)
                return false;

            if (Mathf.Abs(prev_rect_x - recrop_rect_x) > recrop_rect_width * translation)
                return false;
            if (Mathf.Abs(prev_rect_y - recrop_rect_y) > recrop_rect_height * translation)
                return false;
            if (Mathf.Abs(prev_rect_width - recrop_rect_width) > recrop_rect_width * scale)
                return false;
            if (Mathf.Abs(prev_rect_height - recrop_rect_height) > recrop_rect_height * scale)
                return false;
            return true;
        }

        static bool LandmarksRequirementsSatisfied_HolisticRoiTracking(
            Vec3f[] landmarksNorm21, HolisticNormalizedRect recropRect, int imageWidth, int imageHeight, float recropRectMargin)
        {
            float rotation = -recropRect.Rotation;
            float cosa = Mathf.Cos(rotation);
            float sina = Mathf.Sin(rotation);

            float rect_x = recropRect.XCenter * imageWidth * cosa - recropRect.YCenter * imageHeight * sina;
            float rect_y = recropRect.XCenter * imageWidth * sina + recropRect.YCenter * imageHeight * cosa;
            float rect_width = recropRect.Width * imageWidth * (1f + recropRectMargin);
            float rect_height = recropRect.Height * imageHeight * (1f + recropRectMargin);
            float rect_left = rect_x - rect_width * 0.5f;
            float rect_right = rect_x + rect_width * 0.5f;
            float rect_top = rect_y - rect_height * 0.5f;
            float rect_bottom = rect_y + rect_height * 0.5f;

            int L = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            for (int i = 0; i < L && i < landmarksNorm21.Length; i++)
            {
                float nx = landmarksNorm21[i].Item1 * imageWidth * cosa - landmarksNorm21[i].Item2 * imageHeight * sina;
                float ny = landmarksNorm21[i].Item1 * imageWidth * sina + landmarksNorm21[i].Item2 * imageHeight * cosa;
                if (!(rect_left < nx && nx < rect_right && rect_top < ny && ny < rect_bottom))
                    return false;
            }
            return true;
        }

        struct HolisticSingleHandGraphResult
        {
            public bool HandPresence;
            /// <summary>The 21 full-image normalized landmarks after <c>LandmarkProjectionCalculator</c>, corresponding to upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>NormalizedLandmark</c> and using the same formula as <see cref="MediaPipeHandLandmarker"/>.</summary>
            public Vec3f[] NormLandmarks;
            public Vec3f[] WorldLandmarks;
            public float Handedness;
            public float PresenceConfidence;
        }

        /// <summary>
        /// Equivalent to <c>SingleHandLandmarksDetectorGraph</c> in <c>hand_landmarks_detector_graph.cc</c>.
        /// Child steps are local implementations named after the breakdown of
        /// <see cref="MediaPipeHandLandmarker.SingleHandLandmarksDetectorGraph"/>.
        /// </summary>
        HolisticSingleHandGraphResult? SingleHandLandmarksDetectorGraph(Mat image, HolisticNormalizedRect handRect)
        {
            if (_handLandmarksNet == null || _handLandmarksNetOutLayerNames == null)
                return null;
            if (!ImagePreprocessingGraph_SingleHandLandmarks(image, handRect, out HolisticSingleHandLmPreprocessOut pre))
                return null;

            List<Mat> inferenceTensors = InferenceSubgraph_SingleHandLandmarks(pre.HandBlob);
            if (inferenceTensors == null || inferenceTensors.Count < 4)
                return null;
            if (!SplitTensorVectorCalculator_SingleHandLandmarks(inferenceTensors,
                    out Mat landmarkTensors, out Mat handFlagTensors,
                    out Mat handednessTensors, out Mat worldLandmarkTensors))
                return null;

            float[] normLetterboxed = TensorsToLandmarksCalculator_NormalizedLandmarks_SingleHand(landmarkTensors, pre.ModelW, pre.ModelH);
            float[] worldRaw = TensorsToLandmarksCalculator_WorldLandmarks_SingleHand(worldLandmarkTensors);
            float presenceScore = TensorsToFloatsCalculator_HandPresence_SingleHand(handFlagTensors);
            bool handPresence = ThresholdingCalculator_HandPresence_SingleHand(presenceScore);
            float handednessRaw = TensorsToClassificationCalculator_Handedness_SingleHand(handednessTensors);
            float handednessGated = AllowIf_ClassificationListByHandPresence_SingleHand(handPresence, handednessRaw);
            float[] afterLetterbox = LandmarkLetterboxRemovalCalculator_SingleHand(
                normLetterboxed, pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft,
                pre.LetterboxPaddingBottom, pre.LetterboxPaddingRight);

            var handRectStruct = new HolisticHandNormRectBridge
            {
                XCenter = handRect.XCenter,
                YCenter = handRect.YCenter,
                Width = handRect.Width,
                Height = handRect.Height,
                Rotation = handRect.Rotation,
            };

            Vec3f[] projectedLandmarksRaw = LandmarkProjectionCalculator_SingleHand(afterLetterbox, handRectStruct);
            Vec3f[] projectedLandmarks = AllowIf_NormalizedLandmarkListByHandPresence_SingleHand(handPresence, projectedLandmarksRaw);
            Vec3f[] projectedWorldRaw = WorldLandmarkProjectionCalculator_SingleHand(worldRaw, handRectStruct);
            Vec3f[] projectedWorld = AllowIf_LandmarkListByHandPresence_SingleHand(handPresence, projectedWorldRaw);

            return new HolisticSingleHandGraphResult
            {
                HandPresence = handPresence,
                NormLandmarks = projectedLandmarks,
                WorldLandmarks = projectedWorld,
                Handedness = handednessGated,
                PresenceConfidence = presenceScore,
            };
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<HolisticSingleHandGraphResult?> SingleHandLandmarksDetectorGraphAsync(Mat image, HolisticNormalizedRect handRect, CancellationToken cancellationToken)
        {
            if (_handLandmarksNet == null || _handLandmarksNetOutLayerNames == null)
                return null;
            if (!ImagePreprocessingGraph_SingleHandLandmarks(image, handRect, out HolisticSingleHandLmPreprocessOut pre))
                return null;

            var inferenceTensors = await InferenceSubgraph_SingleHandLandmarksAsync(pre.HandBlob, cancellationToken);
            if (inferenceTensors == null || inferenceTensors.Count < 4)
                return null;
            if (!SplitTensorVectorCalculator_SingleHandLandmarks(inferenceTensors,
                    out Mat landmarkTensors, out Mat handFlagTensors,
                    out Mat handednessTensors, out Mat worldLandmarkTensors))
                return null;

            float[] normLetterboxed = TensorsToLandmarksCalculator_NormalizedLandmarks_SingleHand(landmarkTensors, pre.ModelW, pre.ModelH);
            float[] worldRaw = TensorsToLandmarksCalculator_WorldLandmarks_SingleHand(worldLandmarkTensors);
            float presenceScore = TensorsToFloatsCalculator_HandPresence_SingleHand(handFlagTensors);
            bool handPresence = ThresholdingCalculator_HandPresence_SingleHand(presenceScore);
            float handednessRaw = TensorsToClassificationCalculator_Handedness_SingleHand(handednessTensors);
            float handednessGated = AllowIf_ClassificationListByHandPresence_SingleHand(handPresence, handednessRaw);
            float[] afterLetterbox = LandmarkLetterboxRemovalCalculator_SingleHand(
                normLetterboxed, pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft,
                pre.LetterboxPaddingBottom, pre.LetterboxPaddingRight);

            var handRectStruct = new HolisticHandNormRectBridge
            {
                XCenter = handRect.XCenter,
                YCenter = handRect.YCenter,
                Width = handRect.Width,
                Height = handRect.Height,
                Rotation = handRect.Rotation,
            };

            Vec3f[] projectedLandmarksRaw = LandmarkProjectionCalculator_SingleHand(afterLetterbox, handRectStruct);
            Vec3f[] projectedLandmarks = AllowIf_NormalizedLandmarkListByHandPresence_SingleHand(handPresence, projectedLandmarksRaw);
            Vec3f[] projectedWorldRaw = WorldLandmarkProjectionCalculator_SingleHand(worldRaw, handRectStruct);
            Vec3f[] projectedWorld = AllowIf_LandmarkListByHandPresence_SingleHand(handPresence, projectedWorldRaw);

            return new HolisticSingleHandGraphResult
            {
                HandPresence = handPresence,
                NormLandmarks = projectedLandmarks,
                WorldLandmarks = projectedWorld,
                Handedness = handednessGated,
                PresenceConfidence = presenceScore,
            };
        }
#endif

        struct HolisticHandNormRectBridge
        {
            public float XCenter, YCenter, Width, Height, Rotation;
        }

        struct HolisticSingleHandLmPreprocessOut
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

        bool ImagePreprocessingGraph_SingleHandLandmarks(Mat image, HolisticNormalizedRect handRect, out HolisticSingleHandLmPreprocessOut pre)
        {
            pre = default;
            int imgW = image.cols();
            int imgH = image.rows();
            if (imgW <= 0 || imgH <= 0)
                return false;
            const int inputSize = 224;

            if (_hhSingleHandDstPts == null)
            {
                _hhSingleHandSrcPts = new Mat(4, 2, CvType.CV_32FC1);
                _hhSingleHandDstPts = new Mat(4, 2, CvType.CV_32FC1);
                float dw = inputSize, dh = inputSize;
                Span<float> dstPtsArr = stackalloc float[8];
                dstPtsArr[0] = 0f; dstPtsArr[1] = dh;
                dstPtsArr[2] = 0f; dstPtsArr[3] = 0f;
                dstPtsArr[4] = dw; dstPtsArr[5] = 0f;
                dstPtsArr[6] = dw; dstPtsArr[7] = dh;
                _hhSingleHandDstPts.put(0, 0, dstPtsArr);
                _hhSingleHandWarpedBgr = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _hhSingleHandWarpedRgb = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _hhSingleHandBlob = new Mat(new int[] { 1, inputSize, inputSize, 3 }, CvType.CV_32FC1);
                _hhSingleHandBlobHxW = _hhSingleHandBlob.reshape(3, new int[] { inputSize, inputSize });
            }

            float cx = handRect.XCenter * imgW;
            float cy = handRect.YCenter * imgH;
            float rw = handRect.Width * imgW;
            float rh = handRect.Height * imgH;
            if (rw <= 0f || rh <= 0f || float.IsNaN(rw) || float.IsNaN(rh))
                return false;

            PadRoiLikeImageToTensorCalculator_SingleHand(inputSize, inputSize, true, ref rw, ref rh,
                out float padL, out float padT, out float padR, out float padB);

            double angleDeg = handRect.Rotation * 180.0 / Math.PI;

            Imgproc.boxPoints((cx, cy, rw, rh, angleDeg), _hhSingleHandSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_hhSingleHandSrcPts, _hhSingleHandDstPts))
            {
                Imgproc.warpPerspective(image, _hhSingleHandWarpedBgr, projMat, (inputSize, inputSize),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
            }
            Imgproc.cvtColor(_hhSingleHandWarpedBgr, _hhSingleHandWarpedRgb, Imgproc.COLOR_BGR2RGB);
            _hhSingleHandWarpedRgb.convertTo(_hhSingleHandBlobHxW, CvType.CV_32F, 1.0 / 255.0);

            pre = new HolisticSingleHandLmPreprocessOut
            {
                HandBlob = _hhSingleHandBlob,
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

        static void PadRoiLikeImageToTensorCalculator_SingleHand(int tensorW, int tensorH, bool keepAspectRatio,
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

        List<Mat> InferenceSubgraph_SingleHandLandmarks(Mat handBlob)
        {
            if (_handLandmarksNet == null || _handLandmarksNetOutLayerNames == null)
            {
                _hhHandLandmarksForwardOutputList.Clear();
                return _hhHandLandmarksForwardOutputList;
            }

            _handLandmarksNet.setInput(handBlob);
            _hhHandLandmarksForwardOutputList.Clear();
            _handLandmarksNet.forward(_hhHandLandmarksForwardOutputList, _handLandmarksNetOutLayerNames);
            return _hhHandLandmarksForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<List<Mat>> InferenceSubgraph_SingleHandLandmarksAsync(Mat handBlob, CancellationToken cancellationToken)
        {
            if (_handLandmarksNet == null || _handLandmarksNetOutLayerNames == null)
            {
                _hhHandLandmarksForwardOutputList.Clear();
                return _hhHandLandmarksForwardOutputList;
            }

            _hhHandLandmarksForwardOutputList.Clear();
            _handLandmarksNet.setInput(handBlob);
            await _handLandmarksNet.forwardTaskAsync(_hhHandLandmarksForwardOutputList, _handLandmarksNetOutLayerNames, cancellationToken);
            return _hhHandLandmarksForwardOutputList;
        }
#endif

        static bool SplitTensorVectorCalculator_SingleHandLandmarks(List<Mat> inferenceTensors,
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

        float[] TensorsToLandmarksCalculator_NormalizedLandmarks_SingleHand(Mat tensor, int inputW, int inputH)
        {
            float zDenom = inputW * kHolisticHandLandmarksNormalizeZ;
            if (zDenom < 1e-8f)
                zDenom = 1f;
            int n = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            int n3 = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT;
            if (_hhSingleHandTensorNorm == null || _hhSingleHandTensorNorm.Length < n3)
                _hhSingleHandTensorNorm = new float[n3];
            using (var reshaped = tensor.reshape(1, n))
            {
                reshaped.get(0, 0, _hhSingleHandTensorNorm.AsSpan(0, n3));
                for (int i = 0; i < n; i++)
                {
                    _hhSingleHandTensorNorm[i * 3 + 0] /= inputW;
                    _hhSingleHandTensorNorm[i * 3 + 1] /= inputH;
                    _hhSingleHandTensorNorm[i * 3 + 2] /= zDenom;
                }
                return _hhSingleHandTensorNorm;
            }
        }

        float[] TensorsToLandmarksCalculator_WorldLandmarks_SingleHand(Mat tensor)
        {
            int n = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            int n3 = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_ELEMENT_COUNT;
            if (_hhSingleHandTensorWorld == null || _hhSingleHandTensorWorld.Length < n3)
                _hhSingleHandTensorWorld = new float[n3];
            using (var reshaped = tensor.reshape(1, n))
            {
                reshaped.get(0, 0, _hhSingleHandTensorWorld.AsSpan(0, n3));
                return _hhSingleHandTensorWorld;
            }
        }

        static float TensorsToFloatsCalculator_HandPresence_SingleHand(Mat handFlagTensors) =>
            handFlagTensors.at<float>(0, 0)[0];

        bool ThresholdingCalculator_HandPresence_SingleHand(float handPresenceScore) =>
            handPresenceScore >= _minHandLandmarksConfidence;

        static float TensorsToClassificationCalculator_Handedness_SingleHand(Mat handednessTensors) =>
            MediaPipeHandLandmarker.PackHandednessBinaryTop1(handednessTensors.at<float>(0, 0)[0]);

        static float AllowIf_ClassificationListByHandPresence_SingleHand(bool handPresence, float handednessWhenPresent) =>
            handPresence ? handednessWhenPresent : 0f;

        float[] LandmarkLetterboxRemovalCalculator_SingleHand(float[] normLandmarks, float padTop, float padLeft, float padBottom, float padRight)
        {
            if (padTop == 0f && padLeft == 0f && padBottom == 0f && padRight == 0f)
                return normLandmarks;
            float h = 1f - padTop - padBottom;
            float w = 1f - padLeft - padRight;
            if (h <= 1e-6f || w <= 1e-6f)
                return normLandmarks;
            int n = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            for (int i = 0; i < n; i++)
            {
                _hhSingleHandLetterboxRemovedScratch[i * 3 + 0] = (normLandmarks[i * 3 + 0] - padLeft) / w;
                _hhSingleHandLetterboxRemovedScratch[i * 3 + 1] = (normLandmarks[i * 3 + 1] - padTop) / h;
                _hhSingleHandLetterboxRemovedScratch[i * 3 + 2] = normLandmarks[i * 3 + 2] / w;
            }
            return _hhSingleHandLetterboxRemovedScratch;
        }

        /// <summary>
        /// Equivalent to <c>LandmarkProjectionCalculator</c>, using the same formula as the
        /// implementation in <see cref="MediaPipeHandLandmarker"/>. Output coordinates are normalized to
        /// the full image, with <c>new_z = landmark.z * NORM_RECT.width</c>.
        /// </summary>
        static Vec3f[] LandmarkProjectionCalculator_SingleHand(float[] normLandmarksAfterLetterbox, HolisticHandNormRectBridge handRect)
        {
            float angle = handRect.Rotation;
            float ca = Mathf.Cos(angle);
            float sa = Mathf.Sin(angle);
            float cx = handRect.XCenter;
            float cy = handRect.YCenter;
            float nw = handRect.Width;
            float nh = handRect.Height;
            int n = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            var projected = new Vec3f[n];
            for (int i = 0; i < n; i++)
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

        static Vec3f[] WorldLandmarkProjectionCalculator_SingleHand(float[] worldLandmarksRaw, HolisticHandNormRectBridge handRect)
        {
            float ca = Mathf.Cos(handRect.Rotation);
            float sa = Mathf.Sin(handRect.Rotation);
            int n = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            var v = new Vec3f[n];
            for (int i = 0; i < n; i++)
            {
                int k = i * 3;
                float x = worldLandmarksRaw[k];
                float y = worldLandmarksRaw[k + 1];
                float z = worldLandmarksRaw[k + 2];
                v[i] = new Vec3f(ca * x - sa * y, sa * x + ca * y, z);
            }
            return v;
        }

        static Vec3f[] AllowIf_NormalizedLandmarkListByHandPresence_SingleHand(bool handPresence, Vec3f[] landmarksWhenPresent)
        {
            if (!handPresence || landmarksWhenPresent == null)
                return HolisticHandEmptyNormOrWorld21;
            return landmarksWhenPresent;
        }

        static Vec3f[] AllowIf_LandmarkListByHandPresence_SingleHand(bool handPresence, Vec3f[] worldWhenPresent)
        {
            if (!handPresence || worldWhenPresent == null)
                return HolisticHandEmptyNormOrWorld21;
            return worldWhenPresent;
        }

        /// <summary>Equivalent to <c>AlignHandToPoseInWorldCalculator</c> in <c>align_hand_to_pose_in_world_calculator.cc</c>.</summary>
        static Vec3f[] AlignHandToPoseInWorldCalculator(Vec3f[] handWorld, Vec3f[] poseWorld33, int poseWristIdx)
        {
            int n = MediaPipeHandLandmarker.HandLandmarkerEstimationData.LANDMARK_VEC3F_COUNT;
            var o = new Vec3f[n];
            if (handWorld == null || poseWorld33 == null || poseWristIdx < 0 || poseWristIdx >= poseWorld33.Length)
                return o;
            var hw = handWorld[0];
            var pw = poseWorld33[poseWristIdx];
            for (int i = 0; i < n && i < handWorld.Length; i++)
            {
                var l = handWorld[i];
                o[i] = new Vec3f(
                    l.Item1 - hw.Item1 + pw.Item1,
                    l.Item2 - hw.Item2 + pw.Item2,
                    l.Item3 - hw.Item3 + pw.Item3);
            }
            return o;
        }

        static void HolisticHandDetectorGetRoi(int inputWidth, int inputHeight, HolisticNormalizedRect normRect,
            out float centerX, out float centerY, out float width, out float height, out float rotation)
        {
            centerX = normRect.XCenter * inputWidth;
            centerY = normRect.YCenter * inputHeight;
            width = normRect.Width * inputWidth;
            height = normRect.Height * inputHeight;
            rotation = normRect.Rotation;
        }

        static void HolisticHandDetectorPadRoi(int inputTensorWidth, int inputTensorHeight, bool keepAspectRatio, ref float roiWidth, ref float roiHeight)
        {
            if (!keepAspectRatio)
                return;
            float tensorAspectRatio = (float)inputTensorHeight / inputTensorWidth;
            float roiAspectRatio = roiHeight / roiWidth;
            if (tensorAspectRatio > roiAspectRatio)
            {
                roiHeight = roiWidth * tensorAspectRatio;
            }
            else
            {
                roiWidth = roiHeight / tensorAspectRatio;
            }
        }

        /// <summary>
        /// Clone of <c>GetRotatedSubRectToRectTransformMatrix</c> from <see cref="MediaPipeHandLandmarker"/>.
        /// </summary>
        static void HolisticGetRotatedSubRectToRectTransformMatrix(
            float centerX, float centerY, float subWidth, float subHeight, float rotation,
            int rectWidth, int rectHeight, bool flipHorizontally, float[] matrix16)
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

        static float HolisticNormalizeRadiansHand(float angle)
        {
            float twoPi = Mathf.PI * 2f;
            return angle - twoPi * Mathf.Floor((angle - (-Mathf.PI)) / twoPi);
        }

        void DisposeHolisticHandTrackingScratch()
        {
            _hhRoiRefineSrcPts?.Dispose();
            _hhRoiRefineDstPts?.Dispose();
            _hhRoiRefineWarpedBgr?.Dispose();
            _hhRoiRefineWarpedRgb?.Dispose();
            _hhRoiRefineBlob?.Dispose();
            _hhRoiRefineBlobHxW = null;
            _hhSingleHandSrcPts?.Dispose();
            _hhSingleHandDstPts?.Dispose();
            _hhSingleHandWarpedBgr?.Dispose();
            _hhSingleHandWarpedRgb?.Dispose();
            _hhSingleHandBlob?.Dispose();
            _hhSingleHandBlobHxW = null;
            foreach (var m in _hhHandRoiRefinementForwardOutputList)
                m?.Dispose();
            _hhHandRoiRefinementForwardOutputList.Clear();
            foreach (var m in _hhHandLandmarksForwardOutputList)
                m?.Dispose();
            _hhHandLandmarksForwardOutputList.Clear();
        }
    }
}
#endif
#endif
