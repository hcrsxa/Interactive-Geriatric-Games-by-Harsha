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
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe
{

    public partial class MediaPipeHolisticLandmarker
    {
        // --- Holistic pose state (loopback and smoothing). As in upstream MediaPipe, there is no dedicated clear-state API. ---
        HolisticNormalizedRect? _holisticPreviousRoiLoopback;
        Mat _holisticPrevSegmentationMaskSmoothed;

        readonly HolisticAuxiliaryLandmarkSmoothingPipeline _holisticAuxiliarySmoothing = new HolisticAuxiliaryLandmarkSmoothingPipeline();
        readonly HolisticPoseLandmarkOutputSmoothingPipeline _holisticPoseOutputSmoothing = new HolisticPoseLandmarkOutputSmoothingPipeline();
        readonly HolisticWorldLandmarkSmoothingPipeline _holisticWorldSmoothing = new HolisticWorldLandmarkSmoothingPipeline();

        // --- pose_detector scratch buffers, matching MediaPipePoseLandmarker ---
        Mat _hpPoseDetectorLetterbox224;
        List<string> _hpPoseDetectorOutLayerNames;
        Mat _hpPoseDetectorAnchorsNx8;
        Mat _hpPoseTensorsToDetectionsBoxXywh;
        Mat _hpPoseTensorsToDetectionsNmsBoxXywh;
        Mat _hpPoseTensorsToDetectionsNmsScore;
        Mat _hpPoseTensorsToDetectionsNmsBoxLm;

        readonly List<(int idx, float sc)> _hpPoseWnmsIndexed = new List<(int, float)>();
        List<(int idx, float sc)> _hpPoseWnmsRemained = new List<(int, float)>();
        List<(int idx, float sc)> _hpPoseWnmsNextRemained = new List<(int, float)>();

        // --- pose_landmarks scratch buffers ---
        List<string> _hpPoseLandmarksNetOutLayerNames;
        Mat _hpSinglePoseLandmarkWarpedBgr;
        Mat _hpSinglePoseLandmarkWarpedRgb;
        Mat _hpSinglePoseLandmarkBlob;
        Mat _hpSinglePoseLandmarkBlobHxW;
        Mat _hpSinglePoseLandmarkSrcPts;
        Mat _hpSinglePoseLandmarkDstPts;
        Mat _hpSinglePoseLandmarkProjMat3x3;
        Mat _hpSegmentationFullWarpInvMat3x3;
        Mat _hpSegmentationScratchSmall;

        /// <summary>
        /// Full-image pose-segmentation mask written by <c>SegmentationMaskFromTensorToFullImage</c>.
        /// Because upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) uses
        /// <c>num_poses = 1</c>, the slot array used by <see cref="MediaPipePoseLandmarker"/> is
        /// unnecessary here; a single <see cref="Mat"/> is reused until the resolution changes.
        /// </summary>
        Mat _hpSegmentationFullPlaneReuse;

        /// <summary>
        /// Reuse target for <see cref="SegmentationSmoothingCalculator"/> output.
        /// A single <see cref="Mat"/> is reused until the resolution changes.
        /// </summary>
        Mat _hpSegmentationSmoothedReuse;

        const int kHpPoseLandmarkTensorSplitCount = 5;
        const int kHpPoseLandmarkModelLandmarkCount = 39;

        const int kHpPoseLandmarkInputSize = 256;
        const int kHpPoseLandmarkHeatmapKernelSize = 7;
        const int kHolisticNumPoses = 1;

        readonly List<Mat> _hpPoseDetectorForwardOutputList = new List<Mat>();
        readonly List<Mat> _hpPoseLandmarksForwardOutputList = new List<Mat>();
        readonly float[] _hpPoseDetectorLetterboxPadding4 = new float[4];
        readonly float[] _hpPoseWnmsRowBuf12 = new float[12];
        readonly float[] _hpPoseWnmsKpAcc8 = new float[8];
        float[] _hpLandmarksTensorFlatScratch;
        HolisticPoseLandmarkDecoded[] _hpDecodedLandmarkScratch;
        readonly HolisticPoseLandmarkDecoded[] _hpHeatmapRefineDecodedScratch =
            new HolisticPoseLandmarkDecoded[kHpPoseLandmarkModelLandmarkCount];
        readonly HolisticPoseLandmarkDecoded[] _hpWorldDecodedLandmarkScratch =
            new HolisticPoseLandmarkDecoded[kHpPoseLandmarkModelLandmarkCount];
        float[] _hpHeatmapReadScratch;
        readonly float[] _hpHolisticAuxLandmarksToDetKp8 = new float[8];

        /// <summary>
        /// Equivalent to upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        /// <c>NormalizedRect</c>, used for Holistic pose ROI loopback.
        /// </summary>
        struct HolisticNormalizedRect
        {
            public float XCenter;
            public float YCenter;
            public float Width;
            public float Height;
            public float Rotation;
        }

        struct HolisticPoseDetectionData
        {
            public float RelXmin;
            public float RelYmin;
            public float RelWidth;
            public float RelHeight;
            public float[] RelKeypointsXy;
            public float Score;
        }

        struct HolisticPoseLandmarkDecoded
        {
            public float X, Y, Z, Visibility, Presence;
        }

        struct HolisticSinglePoseLandmarkPreprocessOut
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
        /// Intermediate one-frame result from <c>TrackHolisticPose</c> before
        /// <see cref="MediaPipeHolisticLandmarker.BuildHolisticPackedOutputs"/>.
        /// </summary>
        /// <remarks>
        /// <para>Ownership rules for <see cref="SegmentationMaskFull"/>:</para>
        /// <list type="bullet">
        /// <item><description>When no pose is detected and only a zero mask is returned, it may point to the worker-owned full-image buffer <c>_hpSegmentationFullPlaneReuse</c>. <see cref="MediaPipeHolisticLandmarker.BuildHolisticPackedOutputs"/> does not dispose that reference; the worker keeps it until the next frame.</description></item>
        /// <item><description>After detection, the smoothed segmentation may point to the worker-owned reuse buffer <c>_hpSegmentationSmoothedReuse</c>. <see cref="MediaPipeHolisticLandmarker.BuildHolisticPackedOutputs"/> does not dispose that buffer and instead copies it into the packing buffer.</description></item>
        /// <item><description>After <see cref="MediaPipeHolisticLandmarker.RunCoreProcessing"/> completes, <see cref="OpenCVForUnity.UnityIntegration.Worker.ProcessingWorkerBase.PeekOutput"/>(4) references a different buffer even when enabled: a <see cref="Mat.rowRange"/> view of the packing buffer <c>_holisticPackBufferSegmentation</c>. As with the base class contract, callers should treat it as a reference valid only until the next execution and use <see cref="OpenCVForUnity.UnityIntegration.Worker.ProcessingWorkerBase.CopyOutput"/> for long-term retention.</description></item>
        /// </list>
        /// </remarks>
        struct HolisticPoseTrackFrameResult
        {
            public bool PosePresence;
            public float PosePresenceScore;
            /// <summary>The 33 points corresponding to upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>NormalizedLandmark</c>, normalized to the full image.</summary>
            public Vec3f[] NormLandmarks;
            public Vec3f[] WorldLandmarks;
            public float[] LandmarkVisibility;
            public float[] LandmarkVisibilityWorld;
            public float[] LandmarkPresence;
            public Vec3f[] AuxiliaryLandmarksSmoothedNorm;
            public Mat SegmentationMaskFull;
        }

        /// <summary>
        /// Equivalent to <c>TrackHolisticPose</c> in
        /// <c>mediapipe/tasks/cc/vision/holistic_landmarker/holistic_pose_tracking.cc</c>.
        /// Child calculator to method mapping:
        /// <list type="bullet">
        /// <item><description>Entire §2-A: <see cref="HolisticPoseTracking_PipelineSection2A_RoiLoopbackDetectionGateMerge"/></description></item>
        /// <item><description><see cref="PreviousLoopbackCalculator_HolisticPoseNormRect"/> (<c>PreviousLoopbackCalculator</c>) -> previous-frame <c>roi_from_auxiliary_landmarks</c></description></item>
        /// <item><description><see cref="PacketPresenceCalculator_IsPresentHolisticPosePreviousRoi"/>（<c>PacketPresenceCalculator</c> / IsPresent）</description></item>
        /// <item><description><see cref="GateCalculator_DisallowIf_HolisticPoseImageForDetection"/>（<c>GateCalculator</c> DisallowIf）</description></item>
        /// <item><description><see cref="PoseDetectorGraph"/>（§3-1-3-1）</description></item>
        /// <item><description><see cref="ImagePropertiesCalculator_GetImageSize"/> / <see cref="AlignmentPointsRectsCalculator_ConvertAlignmentPointsDetectionsToRect_HolisticPose"/> / <see cref="RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromDetectionsList"/> → <c>roi_from_detections</c></description></item>
        /// <item><description><see cref="MergeCalculator_HolisticPoseRoiFromDetectionsAndPrevious"/> (<c>MergeCalculator</c>) -> final <c>roi</c>, or the full-image rectangle when invalid</description></item>
        /// <item><description><see cref="SinglePoseLandmarksDetectorGraph"/> (§3-1-4 and its child calculators)</description></item>
        /// <item><description>Entire §2-C: <see cref="HolisticPoseTracking_PipelineSection2C_AuxiliaryLandmarkSmoothingAndNextFrameRoi"/></description></item>
        /// <item><description><see cref="LandmarksToDetectionCalculator_ConvertLandmarksToDetection_HolisticPoseAuxiliary"/> / <see cref="AlignmentPointsRectsCalculator_ConvertAlignmentPointsDetectionsToRect_HolisticPose"/> / <see cref="RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromAuxiliaryLandmarks"/> / <see cref="LandmarksSmoothingCalculator_HolisticPoseAuxiliaryLandmarks"/> / <see cref="SetPreviousRoiLoopback_HolisticPose"/></description></item>
        /// <item><description>Entire §2-D: <see cref="HolisticPoseTracking_PipelineSection2D_PoseNormalized2DSmoothing"/>, executed only when <c>NeedLandmarks</c> is true, matching upstream <c>if (request.landmarks)</c> with <see cref="VisibilitySmoothingCalculator_HolisticPoseNormalizedLandmarks2D"/> and <see cref="LandmarksSmoothingCalculator_HolisticPoseNormalizedLandmarks2D"/></description></item>
        /// <item><description>Entire §2-E: <see cref="HolisticPoseTracking_PipelineSection2E_PoseWorldLandmarksSmoothing"/>, executed when <c>NeedWorldLandmarks</c> is true using <see cref="SplitLandmarkListCalculator_SplitToRanges_0_33"/>, <see cref="VisibilitySmoothingCalculator_HolisticPoseWorldLandmarks"/>, and <see cref="LandmarksSmoothingCalculator_HolisticPoseWorldLandmarks"/></description></item>
        /// <item><description>Entire §2-F: <see cref="HolisticPoseTracking_PipelineSection2F_SegmentationMaskSmoothing"/> using <see cref="PreviousLoopbackCalculator_HolisticPoseSegmentationMask"/>, <see cref="SegmentationSmoothingCalculator_HolisticPose"/>, and <see cref="SetPreviousSegmentationMaskLoopback"/></description></item>
        /// </list>
        /// </summary>
        HolisticPoseTrackFrameResult TrackHolisticPose(Mat image)
        {
            int L0 = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            var empty = new HolisticPoseTrackFrameResult
            {
                NormLandmarks = new Vec3f[L0],
                WorldLandmarks = new Vec3f[L0],
                LandmarkVisibility = new float[L0],
                LandmarkVisibilityWorld = new float[L0],
                LandmarkPresence = new float[L0],
                AuxiliaryLandmarksSmoothedNorm = new Vec3f[2],
            };

            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;

            bool runPipeline = _poseTrackingRequest.NeedLandmarks
                || _poseTrackingRequest.NeedWorldLandmarks
                || _poseTrackingRequest.NeedSegmentationMask;
            if (!runPipeline)
                return empty;

            int iw = image.cols();
            int ih = image.rows();

            // A. ROI loopback and detection-input gate (§2-A, corresponding to holistic_pose_tracking.cc 160-170)
            HolisticNormalizedRect mergedRoi = HolisticPoseTracking_PipelineSection2A_RoiLoopbackDetectionGateMerge(image);

            // B. SinglePoseLandmarksDetectorGraph (the full-image segmentation buffer is reused while resolution is unchanged)
            Mat segPlane = null;
            try
            {
                if (_poseTrackingRequest.NeedSegmentationMask && iw > 0 && ih > 0)
                {
                    EnsureHolisticSegmentationFullPlaneReuse(ih, iw);
                    segPlane = _hpSegmentationFullPlaneReuse;
                }

                var lmResult = SinglePoseLandmarksDetectorGraph(image, mergedRoi, segPlane);
                if (!lmResult.HasValue || !lmResult.Value.PosePresence)
                {
                    segPlane?.setTo((0d, 0d, 0d, 0d));
                    if (_poseTrackingRequest.NeedSegmentationMask && segPlane != null)
                    {
                        var r = empty;
                        r.SegmentationMaskFull = segPlane;
                        segPlane = null; // Transfer ownership to the result.
                        return r;
                    }
                    return empty;
                }

                var pr = lmResult.Value;
                if (pr.SegmentationMaskFull != null && pr.SegmentationMaskFull == segPlane)
                    segPlane = null;

                // C. Auxiliary landmark smoothing and next-frame ROI (§2-C, corresponding to holistic_pose_tracking.cc 190-214)
                Vec3f[] auxSmoothed = HolisticPoseTracking_PipelineSection2C_AuxiliaryLandmarkSmoothingAndNextFrameRoi(
                    image, pr.AuxiliaryLandmarksNorm, out HolisticNormalizedRect scaleRoiForSmoothing);

                // D-F. §2-D to §2-F (upstream holistic_pose_tracking.cc 216 onward: 2D smoothing, world smoothing, and segmentation smoothing)
                float[] visWorld = pr.LandmarkVisibilityWorld ?? pr.LandmarkVisibility;
                (Vec3f[] lmNorm, float[] visNorm) = HolisticPoseTracking_PipelineSection2D_PoseNormalized2DSmoothing(
                    pr.NormLandmarks, pr.LandmarkVisibility, scaleRoiForSmoothing, iw, ih);
                (Vec3f[] world, float[] visWorldSm) = HolisticPoseTracking_PipelineSection2E_PoseWorldLandmarksSmoothing(
                    pr.WorldLandmarks, visWorld, _poseTrackingRequest.NeedWorldLandmarks);
                visWorld = visWorldSm;

                Mat segOut = HolisticPoseTracking_PipelineSection2F_SegmentationMaskSmoothing(pr.SegmentationMaskFull);

                return new HolisticPoseTrackFrameResult
                {
                    PosePresence = true,
                    PosePresenceScore = pr.PosePresenceScore,
                    NormLandmarks = lmNorm,
                    WorldLandmarks = world,
                    LandmarkVisibility = visNorm,
                    LandmarkVisibilityWorld = visWorld,
                    LandmarkPresence = pr.LandmarkPresence,
                    AuxiliaryLandmarksSmoothedNorm = auxSmoothed,
                    SegmentationMaskFull = segOut,
                };
            }
            finally
            {
                // Reuse buffers are kept for the worker lifetime and are not disposed here, even when not moved into the returned result.
                if (segPlane != null && segPlane != _hpSegmentationFullPlaneReuse)
                    segPlane.Dispose();
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="TrackHolisticPose"/>.
        /// </summary>
        async Task<HolisticPoseTrackFrameResult> TrackHolisticPoseAsync(Mat image, CancellationToken cancellationToken)
        {
            int L0 = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            var empty = new HolisticPoseTrackFrameResult
            {
                NormLandmarks = new Vec3f[L0],
                WorldLandmarks = new Vec3f[L0],
                LandmarkVisibility = new float[L0],
                LandmarkVisibilityWorld = new float[L0],
                LandmarkPresence = new float[L0],
                AuxiliaryLandmarksSmoothedNorm = new Vec3f[2],
            };

            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;

            bool runPipeline = _poseTrackingRequest.NeedLandmarks
                || _poseTrackingRequest.NeedWorldLandmarks
                || _poseTrackingRequest.NeedSegmentationMask;
            if (!runPipeline)
                return empty;

            int iw = image.cols();
            int ih = image.rows();

            HolisticNormalizedRect mergedRoi = await HolisticPoseTracking_PipelineSection2A_RoiLoopbackDetectionGateMergeAsync(image, cancellationToken);

            Mat segPlane = null;
            try
            {
                if (_poseTrackingRequest.NeedSegmentationMask && iw > 0 && ih > 0)
                {
                    EnsureHolisticSegmentationFullPlaneReuse(ih, iw);
                    segPlane = _hpSegmentationFullPlaneReuse;
                }

                var lmResult = await SinglePoseLandmarksDetectorGraphAsync(image, mergedRoi, segPlane, cancellationToken);
                if (!lmResult.HasValue || !lmResult.Value.PosePresence)
                {
                    segPlane?.setTo((0d, 0d, 0d, 0d));
                    if (_poseTrackingRequest.NeedSegmentationMask && segPlane != null)
                    {
                        var r = empty;
                        r.SegmentationMaskFull = segPlane;
                        segPlane = null;
                        return r;
                    }
                    return empty;
                }

                var pr = lmResult.Value;
                if (pr.SegmentationMaskFull != null && pr.SegmentationMaskFull == segPlane)
                    segPlane = null;

                Vec3f[] auxSmoothed = HolisticPoseTracking_PipelineSection2C_AuxiliaryLandmarkSmoothingAndNextFrameRoi(
                    image, pr.AuxiliaryLandmarksNorm, out HolisticNormalizedRect scaleRoiForSmoothing);

                float[] visWorld = pr.LandmarkVisibilityWorld ?? pr.LandmarkVisibility;
                (Vec3f[] lmNorm, float[] visNorm) = HolisticPoseTracking_PipelineSection2D_PoseNormalized2DSmoothing(
                    pr.NormLandmarks, pr.LandmarkVisibility, scaleRoiForSmoothing, iw, ih);
                (Vec3f[] world, float[] visWorldSm) = HolisticPoseTracking_PipelineSection2E_PoseWorldLandmarksSmoothing(
                    pr.WorldLandmarks, visWorld, _poseTrackingRequest.NeedWorldLandmarks);
                visWorld = visWorldSm;

                Mat segOut = HolisticPoseTracking_PipelineSection2F_SegmentationMaskSmoothing(pr.SegmentationMaskFull);

                return new HolisticPoseTrackFrameResult
                {
                    PosePresence = true,
                    PosePresenceScore = pr.PosePresenceScore,
                    NormLandmarks = lmNorm,
                    WorldLandmarks = world,
                    LandmarkVisibility = visNorm,
                    LandmarkVisibilityWorld = visWorld,
                    LandmarkPresence = pr.LandmarkPresence,
                    AuxiliaryLandmarksSmoothedNorm = auxSmoothed,
                    SegmentationMaskFull = segOut,
                };
            }
            finally
            {
                if (segPlane != null && segPlane != _hpSegmentationFullPlaneReuse)
                    segPlane.Dispose();
            }
        }
#endif

        /// <summary>
        /// Corresponds to the portion of <c>TrackHolisticPoseUsingCustomPoseDetection</c> in
        /// <c>holistic_pose_tracking.cc</c> that performs
        /// Covers ROI loopback, detection input gating, detection ROI generation, and merge.
        /// </summary>
        /// <remarks>
        /// Upstream connection order: <c>GetLoopbackData</c> -> <c>IsPresent</c> -> <c>DisallowIf</c> -> <c>PoseDetectorGraph</c> ->
        /// <c>GetImageSize(image_for_detection)</c> → <c>ConvertAlignmentPointsDetectionsToRect</c> → <c>ScaleAndMakeSquare(1.25)</c> → <c>Merge</c>。
        /// Returns <see cref="HolisticFullImageNormalizedRect"/> when the merged ROI is invalid.
        /// </remarks>
        HolisticNormalizedRect HolisticPoseTracking_PipelineSection2A_RoiLoopbackDetectionGateMerge(Mat image)
        {
            HolisticNormalizedRect? previousRoi = PreviousLoopbackCalculator_HolisticPoseNormRect(image, _holisticPreviousRoiLoopback);
            bool isPreviousRoiAvailable = PacketPresenceCalculator_IsPresentHolisticPosePreviousRoi(previousRoi);
            Mat imageForDetection = GateCalculator_DisallowIf_HolisticPoseImageForDetection(
                image, isPreviousRoiAvailable, out bool ranPoseDetector);

            HolisticNormalizedRect roiFromDetections = default;
            bool hasRoiFromDetections = false;
            if (ranPoseDetector)
            {
                var detResult = PoseDetectorGraph(imageForDetection, normRect: null);
                var imageSizeDet = ImagePropertiesCalculator_GetImageSize(imageForDetection);
                var rectsFromDet = AlignmentPointsRectsCalculator_ConvertAlignmentPointsDetectionsToRect_HolisticPose(
                    detResult.PoseDetections, imageSizeDet.width, imageSizeDet.height);
                var expanded = RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromDetectionsList(
                    rectsFromDet, imageSizeDet.width, imageSizeDet.height);
                if (expanded.Count > 0)
                {
                    roiFromDetections = expanded[0];
                    hasRoiFromDetections = HolisticNormalizedRectIsPresent(roiFromDetections);
                }
            }

            HolisticNormalizedRect mergedRoi = MergeCalculator_HolisticPoseRoiFromDetectionsAndPrevious(
                roiFromDetections, hasRoiFromDetections, previousRoi);
            if (!HolisticNormalizedRectIsPresent(mergedRoi))
                mergedRoi = HolisticFullImageNormalizedRect();
            return mergedRoi;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="HolisticPoseTracking_PipelineSection2A_RoiLoopbackDetectionGateMerge"/> (via <see cref="PoseDetectorGraphAsync"/>).
        /// Invoked only from <see cref="MediaPipeHolisticLandmarker.RunCoreProcessingTaskAsync"/>.
        /// </summary>
        async Task<HolisticNormalizedRect> HolisticPoseTracking_PipelineSection2A_RoiLoopbackDetectionGateMergeAsync(Mat image, CancellationToken cancellationToken)
        {
            HolisticNormalizedRect? previousRoi = PreviousLoopbackCalculator_HolisticPoseNormRect(image, _holisticPreviousRoiLoopback);
            bool isPreviousRoiAvailable = PacketPresenceCalculator_IsPresentHolisticPosePreviousRoi(previousRoi);
            Mat imageForDetection = GateCalculator_DisallowIf_HolisticPoseImageForDetection(
                image, isPreviousRoiAvailable, out bool ranPoseDetector);

            HolisticNormalizedRect roiFromDetections = default;
            bool hasRoiFromDetections = false;
            if (ranPoseDetector)
            {
                var detResult = await PoseDetectorGraphAsync(imageForDetection, normRect: null, cancellationToken);
                var imageSizeDet = ImagePropertiesCalculator_GetImageSize(imageForDetection);
                var rectsFromDet = AlignmentPointsRectsCalculator_ConvertAlignmentPointsDetectionsToRect_HolisticPose(
                    detResult.PoseDetections, imageSizeDet.width, imageSizeDet.height);
                var expanded = RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromDetectionsList(
                    rectsFromDet, imageSizeDet.width, imageSizeDet.height);
                if (expanded.Count > 0)
                {
                    roiFromDetections = expanded[0];
                    hasRoiFromDetections = HolisticNormalizedRectIsPresent(roiFromDetections);
                }
            }

            HolisticNormalizedRect mergedRoi = MergeCalculator_HolisticPoseRoiFromDetectionsAndPrevious(
                roiFromDetections, hasRoiFromDetections, previousRoi);
            if (!HolisticNormalizedRectIsPresent(mergedRoi))
                mergedRoi = HolisticFullImageNormalizedRect();
            return mergedRoi;
        }
#endif

        /// <summary>
        /// Equivalent to <c>holistic_pose_tracking.cc</c> L190-214: computes auxiliary-landmark
        /// <c>scale_roi</c>, auxiliary <c>SmoothLandmarks</c>,
        /// Calculates <c>roi_from_auxiliary_landmarks</c> and runs <c>set_previous_roi_fn</c>.
        /// </summary>
        /// <remarks>
        /// Upstream connection order: <c>GetImageSize(image)</c> -> <c>CalculateScaleRoiFromAuxiliaryLandmarks</c> -> <c>SmoothLandmarks</c> (0.01/10/1) ->
        /// <c>CalculateRoiFromAuxiliaryLandmarks</c> → <c>set_previous_roi_fn</c>。
        /// </remarks>
        /// <param name="scaleRoiForSmoothing">The <c>scale_roi</c> also passed to main-landmark smoothing in §2-D, matching the upstream shared reference.</param>
        Vec3f[] HolisticPoseTracking_PipelineSection2C_AuxiliaryLandmarkSmoothingAndNextFrameRoi(
            Mat image,
            Vec3f[] auxiliaryLandmarksNorm,
            out HolisticNormalizedRect scaleRoiForSmoothing)
        {
            var imageSizeMain = ImagePropertiesCalculator_GetImageSize(image);
            scaleRoiForSmoothing = CalculateScaleRoiFromAuxiliaryLandmarks_HolisticPose(
                auxiliaryLandmarksNorm, imageSizeMain.width, imageSizeMain.height);
            Vec3f[] auxSmoothed = LandmarksSmoothingCalculator_HolisticPoseAuxiliaryLandmarks(
                auxiliaryLandmarksNorm, imageSizeMain.width, imageSizeMain.height, scaleRoiForSmoothing);
            HolisticNormalizedRect roiFromAux = CalculateRoiFromAuxiliaryLandmarks_HolisticPose(
                auxSmoothed, imageSizeMain.width, imageSizeMain.height);
            SetPreviousRoiLoopback_HolisticPose(roiFromAux);
            return auxSmoothed;
        }

        /// <summary>
        /// Equivalent to <c>holistic_pose_tracking.cc</c> L216-234: smooths visibility and coordinates for normalized main outputs. When upstream <c>request.landmarks</c> is false this step is skipped; the C# path mirrors that with <see cref="HolisticPoseTrackingRequest.NeedLandmarks"/>.
        /// </summary>
        (Vec3f[] normLandmarks, float[] visibilityNorm) HolisticPoseTracking_PipelineSection2D_PoseNormalized2DSmoothing(
            Vec3f[] normLandmarksRaw,
            float[] landmarkVisibilityRaw,
            HolisticNormalizedRect scaleRoi,
            int imageWidth,
            int imageHeight)
        {
            if (!_poseTrackingRequest.NeedLandmarks)
                return (normLandmarksRaw, landmarkVisibilityRaw);
            float[] visNorm = VisibilitySmoothingCalculator_HolisticPoseNormalizedLandmarks2D(landmarkVisibilityRaw);
            Vec3f[] lmNorm = LandmarksSmoothingCalculator_HolisticPoseNormalizedLandmarks2D(
                normLandmarksRaw, imageWidth, imageHeight, scaleRoi);
            return (lmNorm, visNorm);
        }

        /// <summary>
        /// Equivalent to <c>holistic_pose_tracking.cc</c> L236-255: applies world-landmark <c>SplitToRanges {0,33}</c>, visibility smoothing, and coordinate smoothing without <c>scale_roi</c>. This is independent of §2-D <c>NeedLandmarks</c>.
        /// </summary>
        (Vec3f[] worldLandmarks, float[] visibilityWorld) HolisticPoseTracking_PipelineSection2E_PoseWorldLandmarksSmoothing(
            Vec3f[] worldFromDetector,
            float[] visibilityWorldInput,
            bool needWorldLandmarks)
        {
            int l = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            if (!needWorldLandmarks)
                return (new Vec3f[l], visibilityWorldInput);

            if (worldFromDetector == null || worldFromDetector.Length < 33)
                return (worldFromDetector, visibilityWorldInput);

            Vec3f[] world33 = SplitLandmarkListCalculator_SplitToRanges_0_33(worldFromDetector);
            float[] visWorld = VisibilitySmoothingCalculator_HolisticPoseWorldLandmarks(visibilityWorldInput);
            Vec3f[] world = LandmarksSmoothingCalculator_HolisticPoseWorldLandmarks(world33);
            return (world, visWorld);
        }

        /// <summary>
        /// Segmentation smoothing and loopback update via <c>PreviousLoopbackCalculator</c>,
        /// <c>SegmentationSmoothingCalculator</c>, and <c>SetPreviousSegmentationMaskLoopback</c>.
        /// </summary>
        Mat HolisticPoseTracking_PipelineSection2F_SegmentationMaskSmoothing(Mat segmentationMaskFull)
        {
            if (segmentationMaskFull == null || segmentationMaskFull.empty())
                return segmentationMaskFull;

            Mat prev = PreviousLoopbackCalculator_HolisticPoseSegmentationMask(
                segmentationMaskFull, _holisticPrevSegmentationMaskSmoothed);
            Mat smoothed = SegmentationSmoothingCalculator_HolisticPose(
                segmentationMaskFull, prev, combineWithPreviousRatio: 0.7f);
            if (segmentationMaskFull != _hpSegmentationFullPlaneReuse)
                segmentationMaskFull.Dispose();
            SetPreviousSegmentationMaskLoopback(smoothed);
            return smoothed;
        }

        /// <summary>
        /// Reallocates the full-image pose-segmentation <see cref="Mat"/> only when the image size changes.
        /// </summary>
        void EnsureHolisticSegmentationFullPlaneReuse(int ih, int iw)
        {
            if (ih <= 0 || iw <= 0)
                return;
            if (_hpSegmentationFullPlaneReuse == null
                || _hpSegmentationFullPlaneReuse.rows() != ih
                || _hpSegmentationFullPlaneReuse.cols() != iw
                || _hpSegmentationFullPlaneReuse.type() != CvType.CV_32FC1)
            {
                _hpSegmentationFullPlaneReuse?.Dispose();
                _hpSegmentationFullPlaneReuse = new Mat(ih, iw, CvType.CV_32FC1);
            }
        }

        /// <summary>
        /// Reallocates the segmentation-smoothing output <see cref="Mat"/> only when the mask resolution changes.
        /// </summary>
        void EnsureHolisticSegmentationSmoothedReuse(int rows, int cols)
        {
            if (rows <= 0 || cols <= 0)
                return;
            if (_hpSegmentationSmoothedReuse == null
                || _hpSegmentationSmoothedReuse.rows() != rows
                || _hpSegmentationSmoothedReuse.cols() != cols
                || _hpSegmentationSmoothedReuse.type() != CvType.CV_32FC1)
            {
                _hpSegmentationSmoothedReuse?.Dispose();
                _hpSegmentationSmoothedReuse = new Mat(rows, cols, CvType.CV_32FC1);
            }
        }

        /// <summary>
        /// Equivalent to <c>PreviousLoopbackCalculator</c> with <c>GetLoopbackData&lt;NormalizedRect&gt;</c>
        /// and tick stream <c>image</c>.
        /// </summary>
        static HolisticNormalizedRect? PreviousLoopbackCalculator_HolisticPoseNormRect(Mat imageTick, HolisticNormalizedRect? loopRoi)
        {
            return loopRoi;
        }

        /// <summary>
        /// Equivalent to <c>PacketPresenceCalculator</c> with <c>IsPresent(previous_roi)</c>.
        /// </summary>
        static bool PacketPresenceCalculator_IsPresentHolisticPosePreviousRoi(HolisticNormalizedRect? previousRoi)
        {
            return previousRoi.HasValue && HolisticNormalizedRectIsPresent(previousRoi.Value);
        }

        /// <summary>
        /// Equivalent to <c>GateCalculator</c> with <c>DisallowIf(image, is_previous_roi_available)</c>.
        /// The return value is the image stream for detection; in this implementation it is the same input
        /// <see cref="Mat"/> by reference only. <paramref name="runPoseDetector"/> indicates whether
        /// <c>PoseDetectorGraph</c> should run.
        /// </summary>
        Mat GateCalculator_DisallowIf_HolisticPoseImageForDetection(Mat image, bool isPreviousRoiAvailable, out bool runPoseDetector)
        {
            runPoseDetector = !isPreviousRoiAvailable;
            return image;
        }

        /// <summary>
        /// Equivalent to <c>ImagePropertiesCalculator</c> with <c>GetImageSize</c>.
        /// Shared image-size accessor for the Holistic partials.
        /// </summary>
        static (int width, int height) ImagePropertiesCalculator_GetImageSize(Mat image)
        {
            return (image.cols(), image.rows());
        }

        /// <summary>
        /// Equivalent to <c>MergeCalculator</c> with <c>Merge(roi_from_detections, previous_roi)</c>:
        /// use the first ROI when valid, otherwise fall back to the loopback ROI.
        /// </summary>
        static HolisticNormalizedRect MergeCalculator_HolisticPoseRoiFromDetectionsAndPrevious(
            HolisticNormalizedRect roiFromDetections,
            bool hasRoiFromDetections,
            HolisticNormalizedRect? previousRoi)
        {
            if (hasRoiFromDetections && HolisticNormalizedRectIsPresent(roiFromDetections))
                return roiFromDetections;
            if (previousRoi.HasValue && HolisticNormalizedRectIsPresent(previousRoi.Value))
                return previousRoi.Value;
            return default;
        }

        /// <summary>
        /// Equivalent to upstream <c>set_previous_roi_fn</c>, feeding the next frame into
        /// <see cref="PreviousLoopbackCalculator_HolisticPoseNormRect"/>.
        /// </summary>
        void SetPreviousRoiLoopback_HolisticPose(HolisticNormalizedRect roiFromAuxiliaryLandmarks)
        {
            if (HolisticNormalizedRectIsPresent(roiFromAuxiliaryLandmarks))
                _holisticPreviousRoiLoopback = roiFromAuxiliaryLandmarks;
            else
                _holisticPreviousRoiLoopback = null;
        }

        static bool HolisticNormalizedRectIsPresent(HolisticNormalizedRect r)
        {
            return r.Width > 1e-5f && r.Height > 1e-5f && !float.IsNaN(r.Width) && !float.IsNaN(r.Height);
        }

        static HolisticNormalizedRect HolisticFullImageNormalizedRect()
        {
            return new HolisticNormalizedRect
            {
                XCenter = 0.5f,
                YCenter = 0.5f,
                Width = 1f,
                Height = 1f,
                Rotation = 0f,
            };
        }

        /// <summary>
        /// Disposes pose-tracking reuse <see cref="Mat"/> instances and segmentation loopback state.
        /// Called from <see cref="Dispose(bool)"/>.
        /// </summary>
        void DisposeHolisticPoseTrackingScratch()
        {
            _hpPoseDetectorLetterbox224?.Dispose();
            _hpPoseDetectorAnchorsNx8?.Dispose();
            _hpPoseTensorsToDetectionsBoxXywh?.Dispose();
            _hpPoseTensorsToDetectionsNmsBoxXywh?.Dispose();
            _hpPoseTensorsToDetectionsNmsScore?.Dispose();
            _hpPoseTensorsToDetectionsNmsBoxLm?.Dispose();
            foreach (var m in _hpPoseDetectorForwardOutputList)
                m?.Dispose();
            _hpPoseDetectorForwardOutputList.Clear();
            foreach (var m in _hpPoseLandmarksForwardOutputList)
                m?.Dispose();
            _hpPoseLandmarksForwardOutputList.Clear();
            _hpSinglePoseLandmarkWarpedBgr?.Dispose();
            _hpSinglePoseLandmarkWarpedRgb?.Dispose();
            _hpSinglePoseLandmarkBlob?.Dispose();
            _hpSinglePoseLandmarkSrcPts?.Dispose();
            _hpSinglePoseLandmarkDstPts?.Dispose();
            _hpSinglePoseLandmarkProjMat3x3?.Dispose();
            _hpSegmentationFullWarpInvMat3x3?.Dispose();
            _hpSegmentationScratchSmall?.Dispose();
            _hpSegmentationFullPlaneReuse?.Dispose();
            _hpSegmentationFullPlaneReuse = null;
            _hpSegmentationSmoothedReuse?.Dispose();
            _hpSegmentationSmoothedReuse = null;
            _holisticPrevSegmentationMaskSmoothed?.Dispose();
            _holisticPrevSegmentationMaskSmoothed = null;
        }

        /// <summary>
        /// Equivalent to <c>PreviousLoopbackCalculator</c> for the segmentation <c>Image</c> stream
        /// loopback, returning the previous frame's smoothed mask.
        /// </summary>
        static Mat PreviousLoopbackCalculator_HolisticPoseSegmentationMask(Mat currentMask, Mat previousMask)
        {
            return previousMask;
        }

        /// <summary>
        /// Keeps the previous-frame mask for segmentation-smoothing loopback. Instead of <c>clone()</c>,
        /// it uses <see cref="Mat.copyTo"/> to copy into <see cref="_holisticPrevSegmentationMaskSmoothed"/>
        /// and reuses the same <see cref="Mat"/> until the resolution changes.
        /// </summary>
        void SetPreviousSegmentationMaskLoopback(Mat maskSmoothed)
        {
            if (maskSmoothed == null || maskSmoothed.empty())
            {
                _holisticPrevSegmentationMaskSmoothed?.Dispose();
                _holisticPrevSegmentationMaskSmoothed = null;
                return;
            }

            int rows = maskSmoothed.rows();
            int cols = maskSmoothed.cols();
            int matType = maskSmoothed.type();
            if (_holisticPrevSegmentationMaskSmoothed == null
                || _holisticPrevSegmentationMaskSmoothed.rows() != rows
                || _holisticPrevSegmentationMaskSmoothed.cols() != cols
                || _holisticPrevSegmentationMaskSmoothed.type() != matType)
            {
                _holisticPrevSegmentationMaskSmoothed?.Dispose();
                _holisticPrevSegmentationMaskSmoothed = new Mat(rows, cols, matType);
            }

            maskSmoothed.copyTo(_holisticPrevSegmentationMaskSmoothed);
        }

    }
}
#endif
#endif
