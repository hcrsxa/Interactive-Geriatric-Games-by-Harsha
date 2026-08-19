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
        const int kHolisticFaceDetectorProjectedRowLength = 17;
        const int kHolisticNumFacesFaceDetector = 1;

        struct HolisticSingleFaceLmPreprocessOut
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

        struct HolisticSingleFaceGraphFaceResult
        {
            public bool FacePresence;
            public float FacePresenceScore;
            public Vec3f[] NormLandmarks;
            public HolisticNormalizedRect NextFrameRect;
        }

        /// <summary>Pseudo detection for <c>LandmarksToDetectionCalculator</c>, including 478 keypoints.</summary>
        struct HolisticFaceLandmarkPseudoDetection
        {
            public float Xmin, Ymin, Width, Height;
            public Vec3f[] KeypointsNorm;
        }

        Vec3f[] _holisticPrevFaceNormLandmarks478;

        Mat _hfFaceDetectorLetterboxBgr;
        Mat _hfFaceDetectorInferenceBlob;
        Mat _hfFaceDetectorInferenceBlobHxW;
        Mat _hfFaceDetectorInferenceRgb8u;
        readonly float[] _hfFaceDetectorProjectionMatrix16 = new float[16];
        Mat _hfFaceDetectorWarpSrcPts;
        Mat _hfFaceDetectorWarpDstPts;
        Mat _hfFaceDetectorDecodedBoxesNx16;
        Mat _hfFaceTensorsToDetectionsWorking;
        Mat _hfFaceScoreFilteredBoxXywh;
        Mat _hfFaceScoreFilteredScore;
        Mat _hfFaceScoreFilteredDecodedNx16;
        MatOfInt _hfFaceNmsIndices;
        Mat _hfWnmsMergedBoxXywh;
        Mat _hfWnmsMergedDecodedNx16;
        Mat _hfWnmsMergedScore;
        readonly List<(int idx, float sc)> _hfWnmsIndexed = new List<(int, float)>();
        List<(int idx, float sc)> _hfWnmsRemained = new List<(int, float)>();
        List<(int idx, float sc)> _hfWnmsNextRemained = new List<(int, float)>();
        float[] _hfFaceDetectorDecodeRowSrc;
        float[] _hfFaceDetectorDecodeRowDst;
        float[] _hfFaceDetectorAnchorRow4;

        Mat _hfFaceLmWarpSrcPts;
        Mat _hfFaceLmWarpDstPts;
        Mat _hfFaceLmWarpedBgr;
        Mat _hfFaceLmWarpedRgb;
        Mat _hfFaceLandmarksInferenceBlob;
        Mat _hfFaceLandmarksInferenceBlobHxW;
        readonly float[] _hfFaceLmProjectionMatrix16 = new float[16];

        Mat _hfFaceBlendshapesInputBlob;

        /// <summary>Reusable list for <c>face_detector</c> <c>forward</c> outputs, serving the same role as <see cref="MediaPipeFaceLandmarker._faceDetectorForwardOutputList"/>.</summary>
        readonly List<Mat> _hfFaceDetectorForwardOutputList = new List<Mat>();

        /// <summary>Reusable list for <c>face_landmarks</c> <c>forward</c> outputs.</summary>
        readonly List<Mat> _hfFaceLandmarksForwardOutputList = new List<Mat>();

        /// <summary>Reusable list for <c>FaceBlendshapes</c> <c>forward</c> outputs.</summary>
        readonly List<Mat> _hfFaceBlendshapesForwardOutputList = new List<Mat>();

        /// <summary>Temporary list of merged boxes and decoded rows used by <see cref="HolisticFaceNonMaxSuppressionCalculator"/>.</summary>
        readonly List<float[]> _hfNmsMergedBoxScratch = new List<float[]>();

        readonly List<float[]> _hfNmsMergedDecScratch = new List<float[]>();
        readonly List<float> _hfNmsMergedScScratch = new List<float>();
        float[] _hfWnmsKpAccumulator;

        readonly Stack<float[]> _hfPoolFaceDetectorProjRow17 = new Stack<float[]>();
        readonly Stack<float[]> _hfPoolFaceDetectorNmsDec16 = new Stack<float[]>();
        readonly Stack<float[]> _hfPoolFaceDetectorNmsBox4 = new Stack<float[]>();

        Mat _hfNumpyClipLo;
        Mat _hfNumpyClipHi;

        float[] _hfFaceTensorsToLmRaw;
        float[] _hfFaceTensorsToLmNorm;

        /// <summary>Output buffer for <see cref="LandmarkLetterboxRemovalCalculator_Face"/>.</summary>
        readonly float[] _hfFaceLetterboxRemovedNormScratch =
            new float[MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum * 3];

        Vec3f[] _hfFaceBlendshapesSubsetScratch;
        float[] _hfFaceBlendshapesLandmarkFlattenBuf;

        Vec3f[] _holisticSplitPoseFace11Scratch;

        static readonly List<float[]> HolisticFaceDetectorGraphEmptyDetections = new List<float[]>();

        static Mat _hfHolisticFaceSsdAnchors128Cache;

        float[] RentHolisticFaceDetectorProjRow17()
        {
            return _hfPoolFaceDetectorProjRow17.Count > 0
                ? _hfPoolFaceDetectorProjRow17.Pop()
                : new float[kHolisticFaceDetectorProjectedRowLength];
        }

        void ReleaseHolisticFaceDetectorProjRow17(float[] row)
        {
            if (row != null && row.Length == kHolisticFaceDetectorProjectedRowLength)
                _hfPoolFaceDetectorProjRow17.Push(row);
        }

        void ReleaseHolisticFaceDetectorProjRowList(IList<float[]> rows)
        {
            if (rows == null)
                return;
            for (int i = 0; i < rows.Count; i++)
                ReleaseHolisticFaceDetectorProjRow17(rows[i]);
        }

        float[] RentHolisticFaceDetectorNmsDec16()
        {
            return _hfPoolFaceDetectorNmsDec16.Count > 0
                ? _hfPoolFaceDetectorNmsDec16.Pop()
                : new float[MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords];
        }

        void ReleaseHolisticFaceDetectorNmsDec16(float[] row)
        {
            if (row != null && row.Length == MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords)
                _hfPoolFaceDetectorNmsDec16.Push(row);
        }

        float[] RentHolisticFaceDetectorNmsBox4()
        {
            return _hfPoolFaceDetectorNmsBox4.Count > 0 ? _hfPoolFaceDetectorNmsBox4.Pop() : new float[4];
        }

        void ReleaseHolisticFaceDetectorNmsBox4(float[] row)
        {
            if (row != null && row.Length == 4)
                _hfPoolFaceDetectorNmsBox4.Push(row);
        }

        void ReleaseHolisticFaceDetectorNmsMergedScratchLists()
        {
            for (int i = 0; i < _hfNmsMergedBoxScratch.Count; i++)
                ReleaseHolisticFaceDetectorNmsBox4(_hfNmsMergedBoxScratch[i]);
            for (int i = 0; i < _hfNmsMergedDecScratch.Count; i++)
                ReleaseHolisticFaceDetectorNmsDec16(_hfNmsMergedDecScratch[i]);
            _hfNmsMergedBoxScratch.Clear();
            _hfNmsMergedDecScratch.Clear();
            _hfNmsMergedScScratch.Clear();
        }

        /// <summary>
        /// Equivalent to <c>TrackHolisticFace</c> in
        /// <c>mediapipe/tasks/cc/vision/holistic_landmarker/holistic_face_tracking.cc</c>.
        /// Section mapping: <see cref="HolisticFaceTracking_PipelineSection7A_FaceRoiFromPoseLandmarks"/> ->
        /// <see cref="HolisticFaceTracking_PipelineSection7B_FaceRoiFromFaceDetections"/> →
        /// <see cref="HolisticFaceTracking_PipelineSection7C_TrackFaceRoi"/> →
        /// <see cref="HolisticFaceTracking_PipelineSection7D_SingleFaceLandmarksDetection"/> →
        /// <see cref="HolisticFaceTracking_PipelineSection7E_FaceBlendshapesIfRequested"/> →
        /// <see cref="HolisticFaceTracking_PipelineSection7F_BuildHolisticFaceTrackOutput"/>, with
        /// <c>set_prev_landmarks_fn</c> executed inside <c>TrackHolisticFace</c> immediately after 7E in
        /// the same order as upstream line 246.
        /// Calculator mapping inside each section:
        /// <list type="bullet">
        /// <item><description>§7-A: <c>SplitToRanges</c>（0–10）／<c>GetFaceRoiFromPoseFaceLandmarks</c>（<c>ConvertLandmarksToDetection</c> → <c>ConvertDetectionToRect</c> 5→2 → <c>ScaleAndMakeSquare</c> 3.0）</description></item>
        /// <item><description>§7-B: <c>GetFaceDetections</c>（<c>FaceDetectorGraph</c>）／<c>GetFaceRoiFromFaceDetections</c></description></item>
        /// <item><description>§7-C: previous-frame landmarks from a <c>GetLoopbackData</c>-equivalent path and <c>TrackFaceRoi</c> (<c>GetFaceRoiFromFaceLandmarks</c> + <c>RoiTrackingCalculator</c>)</description></item>
        /// <item><description>§7-D: <c>SingleFaceLandmarksDetectorGraph</c> inside <c>GetFaceLandmarksDetection</c></description></item>
        /// <item><description>§7-E: <c>FaceBlendshapesGraph</c> when requested</description></item>
        /// <item><description>§7-F: result packing, with <c>set_prev_landmarks_fn</c> executed unconditionally in <c>TrackHolisticFace</c> immediately after 7E to match upstream line 246</description></item>
        /// </list>
        /// </summary>
        HolisticFaceTrackFrameResult TrackHolisticFace(Mat image, Vec3f[] poseLandmarksNorm33)
        {
            int nLm = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            int nBs = MediaPipeFaceLandmarker.kFaceBlendshapeCoefficientCount;
            var empty = new HolisticFaceTrackFrameResult
            {
                NormLandmarks478 = new Vec3f[nLm],
                // Upstream holistic_landmarker.cc emits blendshapes whenever the stream is non-empty.
                // In the non-detected path this remains null, which matches omitting packed output.
                BlendshapeCoefficients52 = null,
            };

            if (image == null || image.empty() || poseLandmarksNorm33 == null || poseLandmarksNorm33.Length < 33)
                return empty;

            (int iw, int ih) = ImagePropertiesCalculator_GetImageSize(image);
            if (iw <= 0 || ih <= 0)
                return empty;

            if (!HolisticFaceTracking_PipelineSection7A_FaceRoiFromPoseLandmarks(poseLandmarksNorm33, iw, ih, out HolisticNormalizedRect roiFromPose))
            {
                HolisticFaceTracking_ClearFaceLandmarksLoopback_EmptyLoopPacketEquivalent();
                return empty;
            }

            if (!HolisticFaceTracking_PipelineSection7B_FaceRoiFromFaceDetections(image, roiFromPose, iw, ih, out HolisticNormalizedRect roiFromDetection))
            {
                HolisticFaceTracking_ClearFaceLandmarksLoopback_EmptyLoopPacketEquivalent();
                return empty;
            }

            if (!HolisticFaceTracking_PipelineSection7C_TrackFaceRoi(iw, ih, roiFromDetection, out HolisticNormalizedRect trackingRoi))
            {
                HolisticFaceTracking_ClearFaceLandmarksLoopback_EmptyLoopPacketEquivalent();
                return empty;
            }

            HolisticSingleFaceGraphFaceResult? single = HolisticFaceTracking_PipelineSection7D_SingleFaceLandmarksDetection(image, trackingRoi, iw, ih);
            if (!single.HasValue)
                return empty;

            // In upstream GetFaceLandmarksDetection, FaceBlendshapesGraph remains connected to
            // NORM_LANDMARKS whenever request.classifications is true, independently of FacePresence.
            float[] blend = HolisticFaceTracking_PipelineSection7E_FaceBlendshapesIfRequested(single.Value.NormLandmarks, iw, ih, nBs);

            // Upstream holistic_face_tracking.cc L245-246 calls set_prev_landmarks_fn(landmarks.value())
            // unconditionally immediately after GetFaceLandmarksDetection, regardless of FacePresence.
            SetPreviousFaceLandmarksLoopback_Holistic(
                HolisticScreenLandmarksToNormalizedFace478(single.Value.NormLandmarks, iw, ih));

            if (!single.Value.FacePresence)
            {
                return new HolisticFaceTrackFrameResult
                {
                    FacePresence = false,
                    FacePresenceScore = single.Value.FacePresenceScore,
                    NormLandmarks478 = single.Value.NormLandmarks,
                    BlendshapeCoefficients52 = blend,
                };
            }

            return HolisticFaceTracking_PipelineSection7F_BuildHolisticFaceTrackOutput(single.Value, blend);
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<HolisticFaceTrackFrameResult> TrackHolisticFaceAsync(Mat image, Vec3f[] poseLandmarksNorm33, CancellationToken cancellationToken)
        {
            int nLm = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            int nBs = MediaPipeFaceLandmarker.kFaceBlendshapeCoefficientCount;
            var empty = new HolisticFaceTrackFrameResult
            {
                NormLandmarks478 = new Vec3f[nLm],
                BlendshapeCoefficients52 = null,
            };

            if (image == null || image.empty() || poseLandmarksNorm33 == null || poseLandmarksNorm33.Length < 33)
                return empty;

            (int iw, int ih) = ImagePropertiesCalculator_GetImageSize(image);
            if (iw <= 0 || ih <= 0)
                return empty;

            if (!HolisticFaceTracking_PipelineSection7A_FaceRoiFromPoseLandmarks(poseLandmarksNorm33, iw, ih, out HolisticNormalizedRect roiFromPose))
            {
                HolisticFaceTracking_ClearFaceLandmarksLoopback_EmptyLoopPacketEquivalent();
                return empty;
            }

            var (okDet, roiFromDetection) = await HolisticFaceTracking_PipelineSection7B_FaceRoiFromFaceDetectionsAsync(image, roiFromPose, iw, ih, cancellationToken);
            if (!okDet)
            {
                HolisticFaceTracking_ClearFaceLandmarksLoopback_EmptyLoopPacketEquivalent();
                return empty;
            }

            if (!HolisticFaceTracking_PipelineSection7C_TrackFaceRoi(iw, ih, roiFromDetection, out HolisticNormalizedRect trackingRoi))
            {
                HolisticFaceTracking_ClearFaceLandmarksLoopback_EmptyLoopPacketEquivalent();
                return empty;
            }

            HolisticSingleFaceGraphFaceResult? single = await HolisticFaceTracking_PipelineSection7D_SingleFaceLandmarksDetectionAsync(image, trackingRoi, iw, ih, cancellationToken);
            if (!single.HasValue)
                return empty;

            float[] blend = await HolisticFaceTracking_PipelineSection7E_FaceBlendshapesIfRequestedAsync(single.Value.NormLandmarks, iw, ih, nBs, cancellationToken);

            SetPreviousFaceLandmarksLoopback_Holistic(
                HolisticScreenLandmarksToNormalizedFace478(single.Value.NormLandmarks, iw, ih));

            if (!single.Value.FacePresence)
            {
                return new HolisticFaceTrackFrameResult
                {
                    FacePresence = false,
                    FacePresenceScore = single.Value.FacePresenceScore,
                    NormLandmarks478 = single.Value.NormLandmarks,
                    BlendshapeCoefficients52 = blend,
                };
            }

            return HolisticFaceTracking_PipelineSection7F_BuildHolisticFaceTrackOutput(single.Value, blend);
        }
#endif

        /// <summary>
        /// Equivalent to <c>GetFaceRoiFromPoseFaceLandmarks</c> in <c>holistic_face_tracking.cc</c>,
        /// including the leading <c>SplitToRanges {{0,11}}</c>.
        /// Assumes <paramref name="poseLandmarksNorm33"/> has at least 33 elements and that the image
        /// size is valid.
        /// </summary>
        bool HolisticFaceTracking_PipelineSection7A_FaceRoiFromPoseLandmarks(
            Vec3f[] poseLandmarksNorm33,
            int imageWidth,
            int imageHeight,
            out HolisticNormalizedRect roiFromPose)
        {
            Vec3f[] poseFace11 = SplitNormalizedLandmarkListCalculator_SplitToRanges_0_11(poseLandmarksNorm33);
            LandmarksToDetectionCalculator_HolisticPoseFaceLandmarks(poseFace11, out float xmin, out float ymin, out float wBox, out float hBox);
            HolisticNormalizedRect rectFromPoseLm = DetectionsToRectsCalculator_ConvertDetectionToRect_HolisticPoseFace(
                poseFace11, xmin, ymin, wBox, hBox);
            roiFromPose = RectTransformationCalculator_ScaleAndMakeSquare_HolisticFaceRoiFromPose(
                rectFromPoseLm, imageWidth, imageHeight, 3f, 3f);
            return HolisticNormalizedRectIsPresent(roiFromPose);
        }

        /// <summary>
        /// Equivalent to <c>GetFaceDetections</c> and <c>GetFaceRoiFromFaceDetections</c>.
        /// </summary>
        bool HolisticFaceTracking_PipelineSection7B_FaceRoiFromFaceDetections(
            Mat image,
            HolisticNormalizedRect roiFromPose,
            int imageWidth,
            int imageHeight,
            out HolisticNormalizedRect roiFromDetection)
        {
            roiFromDetection = default;
            List<float[]> detRows = null;
            try
            {
                detRows = FaceDetectorGraph(image, roiFromPose);
                roiFromDetection = RectTransformationCalculator_ScaleAndMakeSquare_HolisticFaceRoiFromDetection(
                    DetectionsToRectsCalculator_ConvertDetectionsToRectUsingKeypoints_HolisticFaceDetector(detRows), imageWidth, imageHeight, 2f, 2f);
                return HolisticNormalizedRectIsPresent(roiFromDetection);
            }
            finally
            {
                if (detRows != null && !ReferenceEquals(detRows, HolisticFaceDetectorGraphEmptyDetections))
                    ReleaseHolisticFaceDetectorProjRowList(detRows);
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<(bool ok, HolisticNormalizedRect roiFromDetection)> HolisticFaceTracking_PipelineSection7B_FaceRoiFromFaceDetectionsAsync(
            Mat image,
            HolisticNormalizedRect roiFromPose,
            int imageWidth,
            int imageHeight,
            CancellationToken cancellationToken)
        {
            List<float[]> detRows = null;
            try
            {
                detRows = await FaceDetectorGraphAsync(image, roiFromPose, cancellationToken);
                HolisticNormalizedRect roiFromDetection = RectTransformationCalculator_ScaleAndMakeSquare_HolisticFaceRoiFromDetection(
                    DetectionsToRectsCalculator_ConvertDetectionsToRectUsingKeypoints_HolisticFaceDetector(detRows), imageWidth, imageHeight, 2f, 2f);
                return (HolisticNormalizedRectIsPresent(roiFromDetection), roiFromDetection);
            }
            finally
            {
                if (detRows != null && !ReferenceEquals(detRows, HolisticFaceDetectorGraphEmptyDetections))
                    ReleaseHolisticFaceDetectorProjRowList(detRows);
            }
        }
#endif

        /// <summary>Equivalent to <c>TrackFaceRoi</c>: previous-frame loopback -> <c>GetFaceRoiFromFaceLandmarks</c> -> <c>RoiTrackingCalculator</c>.</summary>
        bool HolisticFaceTracking_PipelineSection7C_TrackFaceRoi(
            int imageWidth,
            int imageHeight,
            HolisticNormalizedRect roiFromDetection,
            out HolisticNormalizedRect trackingRoi)
        {
            Vec3f[] prevLm = PreviousLoopbackCalculator_HolisticFacePrevLandmarks(imageWidth, imageHeight);
            HolisticNormalizedRect? prevRoiNullable = null;
            if (prevLm != null)
            {
                HolisticNormalizedRect prevTight = DetectionsToRectsCalculator_FaceLandmarksRoi_Holistic(prevLm, imageWidth, imageHeight);
                prevRoiNullable = RectTransformationCalculator_Scale_HolisticFaceRoiFromLandmarks(prevTight, imageWidth, imageHeight, 1.5f, 1.5f);
            }

            trackingRoi = RoiTrackingCalculator_HolisticFace(prevLm, prevRoiNullable, roiFromDetection, imageWidth, imageHeight);
            return HolisticNormalizedRectIsPresent(trackingRoi);
        }

        /// <summary>Equivalent to the single-face subgraph portion of <c>GetFaceLandmarksDetection</c>.</summary>
        HolisticSingleFaceGraphFaceResult? HolisticFaceTracking_PipelineSection7D_SingleFaceLandmarksDetection(
            Mat image,
            HolisticNormalizedRect trackingRoi,
            int imageWidth,
            int imageHeight)
        {
            return SingleFaceLandmarksDetectorGraph(image, trackingRoi, imageWidth, imageHeight);
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<HolisticSingleFaceGraphFaceResult?> HolisticFaceTracking_PipelineSection7D_SingleFaceLandmarksDetectionAsync(
            Mat image,
            HolisticNormalizedRect trackingRoi,
            int imageWidth,
            int imageHeight,
            CancellationToken cancellationToken)
        {
            return await SingleFaceLandmarksDetectorGraphAsync(image, trackingRoi, imageWidth, imageHeight, cancellationToken);
        }
#endif

        /// <summary>
        /// Equivalent to <c>FaceBlendshapesGraph</c> inside <c>GetFaceLandmarksDetection</c>, where
        /// <c>request.classifications</c> maps to <c>_outputFaceBlendshapes</c>.
        /// As in upstream, it is invoked independently of <c>FacePresence</c>, matching the always-on
        /// connection from <c>SingleFaceLandmarksDetectorGraph</c> landmarks into
        /// <c>FaceBlendshapesGraph</c>.
        /// </summary>
        float[] HolisticFaceTracking_PipelineSection7E_FaceBlendshapesIfRequested(
            Vec3f[] normLandmarks478,
            int imageWidth,
            int imageHeight,
            int blendshapeCoefficientCount)
        {
            if (!_outputFaceBlendshapes || _faceBlendshapesNet == null)
                return null;

            var blend = new float[blendshapeCoefficientCount];
            float[] fromGraph = FaceBlendshapesGraph(normLandmarks478, imageWidth, imageHeight);
            if (fromGraph != null)
                Array.Copy(fromGraph, blend, Math.Min(fromGraph.Length, blendshapeCoefficientCount));
            return blend;
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<float[]> HolisticFaceTracking_PipelineSection7E_FaceBlendshapesIfRequestedAsync(
            Vec3f[] normLandmarks478,
            int imageWidth,
            int imageHeight,
            int blendshapeCoefficientCount,
            CancellationToken cancellationToken)
        {
            if (!_outputFaceBlendshapes || _faceBlendshapesNet == null)
                return null;

            var blend = new float[blendshapeCoefficientCount];
            float[] fromGraph = await FaceBlendshapesGraphAsync(normLandmarks478, imageWidth, imageHeight, cancellationToken);
            if (fromGraph != null)
                Array.Copy(fromGraph, blend, Math.Min(fromGraph.Length, blendshapeCoefficientCount));
            return blend;
        }
#endif

        /// <summary>
        /// Builds the Holistic-specific <see cref="HolisticFaceTrackFrameResult"/> when
        /// <c>FacePresence</c> is true.
        /// <c>set_prev_landmarks_fn</c> has already been executed unconditionally inside
        /// <see cref="TrackHolisticFace"/> immediately after §7-E, matching the upstream order.
        /// </summary>
        HolisticFaceTrackFrameResult HolisticFaceTracking_PipelineSection7F_BuildHolisticFaceTrackOutput(
            HolisticSingleFaceGraphFaceResult r,
            float[] blendshapeCoefficients52)
        {
            return new HolisticFaceTrackFrameResult
            {
                FacePresence = true,
                FacePresenceScore = r.FacePresenceScore,
                NormLandmarks478 = r.NormLandmarks,
                BlendshapeCoefficients52 = blendshapeCoefficients52,
            };
        }

        /// <summary>
        /// Equivalent to <c>SplitNormalizedLandmarkListCalculator</c> with
        /// <c>SplitToRanges(pose_landmarks, {{0,11}})</c>, yielding the 11 landmarks at indices 0 through 10.
        /// </summary>
        Vec3f[] SplitNormalizedLandmarkListCalculator_SplitToRanges_0_11(Vec3f[] poseNorm33)
        {
            _holisticSplitPoseFace11Scratch ??= new Vec3f[11];
            Vec3f[] o = _holisticSplitPoseFace11Scratch;
            if (poseNorm33 == null)
            {
                for (int i = 0; i < 11; i++)
                    o[i] = default;
                return o;
            }

            for (int i = 0; i < 11 && i < poseNorm33.Length; i++)
                o[i] = poseNorm33[i];
            for (int i = poseNorm33.Length; i < 11; i++)
                o[i] = default;
            return o;
        }

        /// <summary>Equivalent to <c>ConvertLandmarksToDetection</c> for the bounding box of the 11 pose-face landmarks.</summary>
        static void LandmarksToDetectionCalculator_HolisticPoseFaceLandmarks(Vec3f[] lm11, out float xmin, out float ymin, out float wBox, out float hBox)
        {
            xmin = float.MaxValue;
            ymin = float.MaxValue;
            float xmax = float.MinValue;
            float ymax = float.MinValue;
            if (lm11 == null)
            {
                xmin = ymin = wBox = hBox = 0f;
                return;
            }

            for (int i = 0; i < lm11.Length; i++)
            {
                float x = lm11[i].Item1;
                float y = lm11[i].Item2;
                xmin = Mathf.Min(xmin, x);
                ymin = Mathf.Min(ymin, y);
                xmax = Mathf.Max(xmax, x);
                ymax = Mathf.Max(ymax, y);
            }

            wBox = xmax - xmin;
            hBox = ymax - ymin;
        }

        /// <summary>Equivalent to <c>ConvertDetectionToRect</c> with keypoint pair 5 -> 2 and <c>target_angle=0</c>, used by <c>GetFaceRoiFromPoseFaceLandmarks</c> in <c>holistic_face_tracking.cc</c>.</summary>
        static HolisticNormalizedRect DetectionsToRectsCalculator_ConvertDetectionToRect_HolisticPoseFace(
            Vec3f[] lm11, float xmin, float ymin, float wBox, float hBox)
        {
            if (lm11 == null || lm11.Length < 11 || wBox <= 1e-8f || hBox <= 1e-8f)
                return default;

            float centerX = xmin + wBox * 0.5f;
            float centerY = ymin + hBox * 0.5f;
            const int k0 = 5;
            const int k1 = 2;
            float x0 = lm11[k0].Item1;
            float y0 = lm11[k0].Item2;
            float x1 = lm11[k1].Item1;
            float y1 = lm11[k1].Item2;
            float rotation = HolisticFaceNormalizeRadians(0f - Mathf.Atan2(-(y1 - y0), x1 - x0));
            return new HolisticNormalizedRect
            {
                XCenter = centerX,
                YCenter = centerY,
                Width = wBox,
                Height = hBox,
                Rotation = rotation,
            };
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> with <c>ScaleAndMakeSquare</c>,
        /// following <c>InternalScaleAndShift</c> in <c>rect_transformation.cc</c> with
        /// <c>square_long=true</c>.
        /// Uses the same rectangle-transform pattern as <see cref="MediaPipeHandLandmarker"/>'s hand
        /// tracking path <c>RectTransformationCalculator_ScaleAndShiftAndMakeSquareLong_Internal</c>.
        /// </summary>
        static HolisticNormalizedRect RectTransformationCalculator_ScaleAndMakeSquare_HolisticFaceRoiFromPose(
            HolisticNormalizedRect r, int imgW, int imgH, float scaleX, float scaleY) =>
            RectTransformationCalculator_ScaleAndMakeSquare_Internal(r, imgW, imgH, scaleX, scaleY, 0f, 0f);

        static HolisticNormalizedRect RectTransformationCalculator_ScaleAndMakeSquare_HolisticFaceRoiFromDetection(
            HolisticNormalizedRect r, int imgW, int imgH, float scaleX, float scaleY) =>
            RectTransformationCalculator_ScaleAndMakeSquare_Internal(r, imgW, imgH, scaleX, scaleY, 0f, 0f);

        static HolisticNormalizedRect RectTransformationCalculator_ScaleAndMakeSquare_Internal(
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

        /// <summary>Equivalent to <c>ConvertDetectionsToRectUsingKeypoints</c> with keypoints 0 -> 1 inside <c>GetFaceRoiFromFaceDetections</c>.</summary>
        static HolisticNormalizedRect DetectionsToRectsCalculator_ConvertDetectionsToRectUsingKeypoints_HolisticFaceDetector(
            List<float[]> projectedRows)
        {
            if (projectedRows == null || projectedRows.Count == 0)
                return default;
            float[] row = projectedRows[0];
            if (row == null || row.Length < kHolisticFaceDetectorProjectedRowLength)
                return default;
            return HolisticFace_DetectionRowToNormalizedRect(row.AsSpan());
        }

        /// <summary>
        /// Converts a projected detection row (normalized bbox + keypoints + score) into
        /// <c>FACE_RECTS</c>, using the same formula as
        /// <see cref="MediaPipeFaceLandmarker"/>'s <c>DetectionsToRectsCalculator_OneRow</c>.
        /// </summary>
        static HolisticNormalizedRect HolisticFace_DetectionRowToNormalizedRect(ReadOnlySpan<float> row)
        {
            float xmin = row[0];
            float ymin = row[1];
            float wBox = row[2];
            float hBox = row[3];
            float centerX = xmin + wBox * 0.5f;
            float centerY = ymin + hBox * 0.5f;

            int k0 = MediaPipeFaceLandmarker.kFaceDetectorDetectionsToRectsRotationStartKeypointIndex;
            int k1 = MediaPipeFaceLandmarker.kFaceDetectorDetectionsToRectsRotationEndKeypointIndex;
            int o0 = 4 + k0 * 2;
            int o1 = 4 + k1 * 2;
            float x0 = row[o0];
            float y0 = row[o0 + 1];
            float x1 = row[o1];
            float y1 = row[o1 + 1];
            float targetRad = MediaPipeFaceLandmarker.kFaceDetectorDetectionsToRectsTargetAngleDegrees * (Mathf.PI / 180f);
            float rotation = HolisticFaceNormalizeRadians(targetRad - Mathf.Atan2(-(y1 - y0), x1 - x0));
            return new HolisticNormalizedRect
            {
                XCenter = centerX,
                YCenter = centerY,
                Width = wBox,
                Height = hBox,
                Rotation = rotation,
            };
        }

        /// <summary>Equivalent to <c>ConvertDetectionToRect</c> with keypoints 33 -> 263 inside <c>GetFaceRoiFromFaceLandmarks</c>.</summary>
        static HolisticNormalizedRect DetectionsToRectsCalculator_FaceLandmarksRoi_Holistic(Vec3f[] normLm478, int imgW, int imgH)
        {
            int n = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            if (normLm478 == null || normLm478.Length < n || imgW <= 0 || imgH <= 0)
                return default;

            float xmin = float.MaxValue, ymin = float.MaxValue, xmax = float.MinValue, ymax = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                float x = normLm478[i].Item1;
                float y = normLm478[i].Item2;
                xmin = Mathf.Min(xmin, x);
                ymin = Mathf.Min(ymin, y);
                xmax = Mathf.Max(xmax, x);
                ymax = Mathf.Max(ymax, y);
            }

            float wBox = xmax - xmin;
            float hBox = ymax - ymin;
            float centerX = xmin + wBox * 0.5f;
            float centerY = ymin + hBox * 0.5f;

            int k0 = MediaPipeFaceLandmarker.kFaceLandmarksDetectionsToRectsRotationStartKeypointIndex;
            int k1 = MediaPipeFaceLandmarker.kFaceLandmarksDetectionsToRectsRotationEndKeypointIndex;
            float x0 = normLm478[k0].Item1 * imgW;
            float y0 = normLm478[k0].Item2 * imgH;
            float x1 = normLm478[k1].Item1 * imgW;
            float y1 = normLm478[k1].Item2 * imgH;
            float targetRad = MediaPipeFaceLandmarker.kFaceLandmarksDetectionsToRectsTargetAngleDegrees * (Mathf.PI / 180f);
            float rotation = HolisticFaceNormalizeRadians(targetRad - Mathf.Atan2(-(y1 - y0), x1 - x0));

            return new HolisticNormalizedRect
            {
                XCenter = centerX,
                YCenter = centerY,
                Width = wBox,
                Height = hBox,
                Rotation = rotation,
            };
        }

        /// <summary>Equivalent to <c>Scale</c> with factor 1.5 in <c>GetFaceRoiFromFaceLandmarks</c>.</summary>
        static HolisticNormalizedRect RectTransformationCalculator_Scale_HolisticFaceRoiFromLandmarks(
            HolisticNormalizedRect rect, int imageW, int imageH, float scaleX, float scaleY)
        {
            if (imageW <= 0 || imageH <= 0)
                return default;

            float width = rect.Width;
            float height = rect.Height;
            float rotation = rect.Rotation;
            float xCenter = rect.XCenter;
            float yCenter = rect.YCenter;
            float cosR = Mathf.Cos(rotation);
            float sinR = Mathf.Sin(rotation);
            float xShiftNorm = (imageW * width * 0f * cosR - imageH * height * 0f * sinR) / imageW;
            float yShiftNorm = (imageW * width * 0f * sinR + imageH * height * 0f * cosR) / imageH;
            xCenter += xShiftNorm;
            yCenter += yShiftNorm;
            return new HolisticNormalizedRect
            {
                XCenter = xCenter,
                YCenter = yCenter,
                Width = width * scaleX,
                Height = height * scaleY,
                Rotation = rotation,
            };
        }

        /// <summary>
        /// Equivalent to <c>LandmarksToDetectionCalculator</c> for the enclosing box of all 478 points,
        /// assuming the previous-frame ROI path used by <c>GetFaceRoiFromFaceLandmarks</c>.
        /// </summary>
        static HolisticFaceLandmarkPseudoDetection LandmarksToDetectionCalculator_Face478(Vec3f[] normLandmarksFullImage)
        {
            int n = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            var d = new HolisticFaceLandmarkPseudoDetection { KeypointsNorm = normLandmarksFullImage };
            float xmin = float.MaxValue, ymin = float.MaxValue, xmax = float.MinValue, ymax = float.MinValue;
            if (normLandmarksFullImage == null)
                return d;
            for (int i = 0; i < n && i < normLandmarksFullImage.Length; i++)
            {
                float x = normLandmarksFullImage[i].Item1;
                float y = normLandmarksFullImage[i].Item2;
                xmin = Mathf.Min(xmin, x);
                ymin = Mathf.Min(ymin, y);
                xmax = Mathf.Max(xmax, x);
                ymax = Mathf.Max(ymax, y);
            }

            d.Xmin = xmin;
            d.Ymin = ymin;
            d.Width = xmax - xmin;
            d.Height = ymax - ymin;
            return d;
        }

        /// <summary>
        /// Equivalent to <c>RoiTrackingCalculator</c> for <c>TrackFaceRoi</c> in
        /// <c>holistic_face_tracking.cc</c>.
        /// </summary>
        static HolisticNormalizedRect RoiTrackingCalculator_HolisticFace(
            Vec3f[] prevNormLandmarks478,
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
            if (!HolisticFaceRectRequirementsSatisfied(prevRoi, recropRect, imageWidth, imageHeight,
                    rotationDegrees: 15f, translation: 0.1f, scale: 0.3f))
                return recropRect;
            if (prevNormLandmarks478 == null ||
                !HolisticFaceLandmarksRequirementsSatisfied(prevNormLandmarks478, recropRect, imageWidth, imageHeight, recropRectMargin: -0.2f))
                return recropRect;
            return prevRoi;
        }

        static bool HolisticFaceRectRequirementsSatisfied(
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

        static bool HolisticFaceLandmarksRequirementsSatisfied(
            Vec3f[] landmarksNorm478, HolisticNormalizedRect recropRect, int imageWidth, int imageHeight, float recropRectMargin)
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

            int L = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            for (int i = 0; i < L && i < landmarksNorm478.Length; i++)
            {
                float nx = landmarksNorm478[i].Item1 * imageWidth * cosa - landmarksNorm478[i].Item2 * imageHeight * sina;
                float ny = landmarksNorm478[i].Item1 * imageWidth * sina + landmarksNorm478[i].Item2 * imageHeight * cosa;
                if (!(rect_left < nx && nx < rect_right && rect_top < ny && ny < rect_bottom))
                    return false;
            }

            return true;
        }

        static float HolisticFaceNormalizeRadians(float angle)
        {
            float twoPi = Mathf.PI * 2f;
            return angle - twoPi * Mathf.Floor((angle - (-Mathf.PI)) / twoPi);
        }

        Vec3f[] PreviousLoopbackCalculator_HolisticFacePrevLandmarks(int imageW, int imageH)
        {
            _ = imageW;
            _ = imageH;
            return _holisticPrevFaceNormLandmarks478;
        }

        void SetPreviousFaceLandmarksLoopback_Holistic(Vec3f[] normLandmarks478)
        {
            _holisticPrevFaceNormLandmarks478 = normLandmarks478 != null ? (Vec3f[])normLandmarks478.Clone() : null;
        }

        /// <summary>
        /// In upstream <c>PreviousLoopbackCalculator</c>, when LOOP is an <strong>empty packet</strong>,
        /// execution advances the boundary without sending to <c>PREV_LOOP</c>
        /// (<c>previous_loopback_calculator.cc</c> L122-124).
        /// Failures in §7-A through §7-C never reach <c>GetFaceLandmarksDetection</c>, so this method
        /// clears the retained previous-frame face landmarks to <c>null</c> to mirror the absence of a
        /// loopback update at that timestamp.
        /// </summary>
        void HolisticFaceTracking_ClearFaceLandmarksLoopback_EmptyLoopPacketEquivalent()
        {
            SetPreviousFaceLandmarksLoopback_Holistic(null);
        }

        static Vec3f[] HolisticScreenLandmarksToNormalizedFace478(Vec3f[] screenLm, int iw, int ih)
        {
            int L = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            var n = new Vec3f[L];
            if (screenLm == null || iw <= 0 || ih <= 0)
                return n;
            float invW = 1f / iw;
            float invH = 1f / ih;
            for (int i = 0; i < L && i < screenLm.Length; i++)
            {
                var p = screenLm[i];
                n[i] = new Vec3f(p.Item1 * invW, p.Item2 * invH, p.Item3 * invW);
            }

            return n;
        }

        /// <summary>
        /// Equivalent to <c>FaceDetectorGraph</c> in <c>face_detector_graph.cc</c>.
        /// In the Holistic path, <c>NORM_RECT</c> is required because the ROI comes from pose output.
        /// Child calculator to method mapping follows the same XML order as
        /// <see cref="MediaPipeFaceLandmarker.FaceDetectorGraph"/>.
        /// </summary>
        List<float[]> FaceDetectorGraph(Mat image, HolisticNormalizedRect normRoi)
        {
            if (_faceDetectorNet == null || _hfFaceDetectorOutLayerNames == null)
                return HolisticFaceDetectorGraphEmptyDetections;
            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return HolisticFaceDetectorGraphEmptyDetections;
            if (!HolisticNormalizedRectIsPresent(normRoi))
                return HolisticFaceDetectorGraphEmptyDetections;

            int imgW = image.cols();
            int imgH = image.rows();
            int tensorSize = MediaPipeFaceLandmarker.kFaceDetectorShortRangeImageSize;
            int numBoxes = MediaPipeFaceLandmarker.kFaceDetectorLegacyShortRangeNumBoxes;

            if (_hfFaceDetectorLetterboxBgr == null
                || _hfFaceDetectorLetterboxBgr.rows() != tensorSize
                || _hfFaceDetectorLetterboxBgr.cols() != tensorSize)
            {
                _hfFaceDetectorLetterboxBgr?.Dispose();
                _hfFaceDetectorLetterboxBgr = new Mat(tensorSize, tensorSize, image.type());
            }

            Mat letter = _hfFaceDetectorLetterboxBgr;

            HolisticFaceImagePreprocessingGraph_FillLetterboxRoi(image, letter, normRoi, tensorSize, _hfFaceDetectorProjectionMatrix16);
            List<Mat> outputBlobs = HolisticFaceInferenceSubgraph_FaceDetection(tensorSize);
            if (outputBlobs == null || outputBlobs.Count < 2)
                return HolisticFaceDetectorGraphEmptyDetections;

            Mat output0 = outputBlobs[1];
            Mat output1 = outputBlobs[0];
            if (output0 == null || output1 == null)
                return HolisticFaceDetectorGraphEmptyDetections;

            using (Mat boxRows = HolisticFaceDetectorGraph_PrepareBoxMajorRows(output0, numBoxes))
            using (Mat scoreCol = HolisticFaceDetectorGraph_PrepareScoreColumn(output1, numBoxes))
            {
                Mat anchors = HolisticFaceSsdAnchorsCalculator();
                HolisticFaceTensorsToDetectionsCalculator(boxRows, scoreCol, anchors, numBoxes);
                Mat nmsBoxXywh = HolisticFaceDetectorGraph_BuildNmsBoxXywhFromDecoded(numBoxes);
                HolisticFaceDetectionsFilterByMinScoreThresh(
                    nmsBoxXywh, scoreCol, _hfFaceDetectorDecodedBoxesNx16, _minFaceDetectionConfidence,
                    out Mat nmsFiltered, out Mat scoreFiltered, out Mat decodedFiltered);
                MatOfInt indices = HolisticFaceNonMaxSuppressionCalculator(nmsFiltered, scoreFiltered, decodedFiltered);
                return HolisticFaceDetectionProjectionCalculator(
                    _hfWnmsMergedBoxXywh, _hfWnmsMergedScore, _hfWnmsMergedDecodedNx16, indices, imgW, imgH);
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<List<float[]>> FaceDetectorGraphAsync(Mat image, HolisticNormalizedRect normRoi, CancellationToken cancellationToken)
        {
            if (_faceDetectorNet == null || _hfFaceDetectorOutLayerNames == null)
                return HolisticFaceDetectorGraphEmptyDetections;
            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return HolisticFaceDetectorGraphEmptyDetections;
            if (!HolisticNormalizedRectIsPresent(normRoi))
                return HolisticFaceDetectorGraphEmptyDetections;

            int imgW = image.cols();
            int imgH = image.rows();
            int tensorSize = MediaPipeFaceLandmarker.kFaceDetectorShortRangeImageSize;
            int numBoxes = MediaPipeFaceLandmarker.kFaceDetectorLegacyShortRangeNumBoxes;

            if (_hfFaceDetectorLetterboxBgr == null
                || _hfFaceDetectorLetterboxBgr.rows() != tensorSize
                || _hfFaceDetectorLetterboxBgr.cols() != tensorSize)
            {
                _hfFaceDetectorLetterboxBgr?.Dispose();
                _hfFaceDetectorLetterboxBgr = new Mat(tensorSize, tensorSize, image.type());
            }

            Mat letter = _hfFaceDetectorLetterboxBgr;

            HolisticFaceImagePreprocessingGraph_FillLetterboxRoi(image, letter, normRoi, tensorSize, _hfFaceDetectorProjectionMatrix16);
            var outputBlobs = await HolisticFaceInferenceSubgraph_FaceDetectionAsync(tensorSize, cancellationToken);
            if (outputBlobs == null || outputBlobs.Count < 2)
                return HolisticFaceDetectorGraphEmptyDetections;

            Mat output0 = outputBlobs[1];
            Mat output1 = outputBlobs[0];
            if (output0 == null || output1 == null)
                return HolisticFaceDetectorGraphEmptyDetections;

            using (Mat boxRows = HolisticFaceDetectorGraph_PrepareBoxMajorRows(output0, numBoxes))
            using (Mat scoreCol = HolisticFaceDetectorGraph_PrepareScoreColumn(output1, numBoxes))
            {
                Mat anchors = HolisticFaceSsdAnchorsCalculator();
                HolisticFaceTensorsToDetectionsCalculator(boxRows, scoreCol, anchors, numBoxes);
                Mat nmsBoxXywh = HolisticFaceDetectorGraph_BuildNmsBoxXywhFromDecoded(numBoxes);
                HolisticFaceDetectionsFilterByMinScoreThresh(
                    nmsBoxXywh, scoreCol, _hfFaceDetectorDecodedBoxesNx16, _minFaceDetectionConfidence,
                    out Mat nmsFiltered, out Mat scoreFiltered, out Mat decodedFiltered);
                MatOfInt indices = HolisticFaceNonMaxSuppressionCalculator(nmsFiltered, scoreFiltered, decodedFiltered);
                return HolisticFaceDetectionProjectionCalculator(
                    _hfWnmsMergedBoxXywh, _hfWnmsMergedScore, _hfWnmsMergedDecodedNx16, indices, imgW, imgH);
            }
        }
#endif

        void HolisticFaceImagePreprocessingGraph_FillLetterboxRoi(
            Mat image, Mat letterboxTensorBgr, HolisticNormalizedRect normRect, int tensorSize, float[] projectionMatrix16)
        {
            int imageW = image.cols();
            int imageH = image.rows();
            HolisticFaceDetectorGetRoi(imageW, imageH, normRect, out float roiCx, out float roiCy, out float roiW, out float roiH,
                out float roiRot);
            HolisticHandDetectorPadRoi(tensorSize, tensorSize, true, ref roiW, ref roiH);
            HolisticGetRotatedSubRectToRectTransformMatrix(roiCx, roiCy, roiW, roiH, roiRot, imageW, imageH, false,
                projectionMatrix16);

            HolisticFaceDetectorEnsureWarpMats(tensorSize);
            double angleDeg = roiRot * (180.0 / Math.PI);
            Imgproc.boxPoints((roiCx, roiCy, roiW, roiH, angleDeg), _hfFaceDetectorWarpSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_hfFaceDetectorWarpSrcPts, _hfFaceDetectorWarpDstPts))
            {
                Imgproc.warpPerspective(image, letterboxTensorBgr, projMat, (tensorSize, tensorSize),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
            }
        }

        static void HolisticFaceDetectorGetRoi(int inputWidth, int inputHeight, HolisticNormalizedRect normRect,
            out float centerX, out float centerY, out float width, out float height, out float rotation)
        {
            centerX = normRect.XCenter * inputWidth;
            centerY = normRect.YCenter * inputHeight;
            width = normRect.Width * inputWidth;
            height = normRect.Height * inputHeight;
            rotation = normRect.Rotation;
        }

        void HolisticFaceDetectorEnsureWarpMats(int tensorSize)
        {
            if (_hfFaceDetectorWarpDstPts != null)
                return;

            float dw = tensorSize;
            float dh = tensorSize;
            _hfFaceDetectorWarpDstPts = new Mat(4, 2, CvType.CV_32FC1);
            Span<float> dstPtsArr = stackalloc float[8];
            dstPtsArr[0] = 0f;
            dstPtsArr[1] = dh;
            dstPtsArr[2] = 0f;
            dstPtsArr[3] = 0f;
            dstPtsArr[4] = dw;
            dstPtsArr[5] = 0f;
            dstPtsArr[6] = dw;
            dstPtsArr[7] = dh;
            _hfFaceDetectorWarpDstPts.put(0, 0, dstPtsArr);
            _hfFaceDetectorWarpSrcPts = new Mat(4, 2, CvType.CV_32FC1);
        }

        List<Mat> HolisticFaceInferenceSubgraph_FaceDetection(int detH)
        {
            if (_faceDetectorNet == null || _hfFaceDetectorOutLayerNames == null)
            {
                _hfFaceDetectorForwardOutputList.Clear();
                return _hfFaceDetectorForwardOutputList;
            }

            const int detC = 3;
            const float imageToTensorDivisor = 127.5f;
            Mat letterboxBgr = _hfFaceDetectorLetterboxBgr;

            if (detH > 0)
            {
                if (_hfFaceDetectorInferenceBlob == null
                    || _hfFaceDetectorInferenceRgb8u == null
                    || _hfFaceDetectorInferenceRgb8u.rows() != detH
                    || _hfFaceDetectorInferenceRgb8u.cols() != detH)
                {
                    _hfFaceDetectorInferenceRgb8u?.Dispose();
                    _hfFaceDetectorInferenceBlob?.Dispose();
                    _hfFaceDetectorInferenceRgb8u = null;
                    _hfFaceDetectorInferenceBlob = null;
                    _hfFaceDetectorInferenceBlobHxW = null;

                    _hfFaceDetectorInferenceRgb8u = new Mat(detH, detH, CvType.CV_8UC3);
                    _hfFaceDetectorInferenceBlob = new Mat(new int[] { 1, detH, detH, detC }, CvType.CV_32FC1);
                    _hfFaceDetectorInferenceBlobHxW =
                        _hfFaceDetectorInferenceBlob.reshape(detC, new int[] { detH, detH });
                }

                if (letterboxBgr != null && !letterboxBgr.empty())
                {
                    Imgproc.cvtColor(letterboxBgr, _hfFaceDetectorInferenceRgb8u, Imgproc.COLOR_BGR2RGB);
                    _hfFaceDetectorInferenceRgb8u.convertTo(_hfFaceDetectorInferenceBlobHxW, CvType.CV_32F,
                        1.0 / imageToTensorDivisor, -1.0);
                }
            }

            _faceDetectorNet.setInput(_hfFaceDetectorInferenceBlob);
            _hfFaceDetectorForwardOutputList.Clear();
            _faceDetectorNet.forward(_hfFaceDetectorForwardOutputList, _hfFaceDetectorOutLayerNames);
            return _hfFaceDetectorForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<List<Mat>> HolisticFaceInferenceSubgraph_FaceDetectionAsync(int detH, CancellationToken cancellationToken)
        {
            if (_faceDetectorNet == null || _hfFaceDetectorOutLayerNames == null)
            {
                _hfFaceDetectorForwardOutputList.Clear();
                return _hfFaceDetectorForwardOutputList;
            }

            const int detC = 3;
            const float imageToTensorDivisor = 127.5f;
            Mat letterboxBgr = _hfFaceDetectorLetterboxBgr;

            if (detH > 0)
            {
                if (_hfFaceDetectorInferenceBlob == null
                    || _hfFaceDetectorInferenceRgb8u == null
                    || _hfFaceDetectorInferenceRgb8u.rows() != detH
                    || _hfFaceDetectorInferenceRgb8u.cols() != detH)
                {
                    _hfFaceDetectorInferenceRgb8u?.Dispose();
                    _hfFaceDetectorInferenceBlob?.Dispose();
                    _hfFaceDetectorInferenceRgb8u = null;
                    _hfFaceDetectorInferenceBlob = null;
                    _hfFaceDetectorInferenceBlobHxW = null;

                    _hfFaceDetectorInferenceRgb8u = new Mat(detH, detH, CvType.CV_8UC3);
                    _hfFaceDetectorInferenceBlob = new Mat(new int[] { 1, detH, detH, detC }, CvType.CV_32FC1);
                    _hfFaceDetectorInferenceBlobHxW =
                        _hfFaceDetectorInferenceBlob.reshape(detC, new int[] { detH, detH });
                }

                if (letterboxBgr != null && !letterboxBgr.empty())
                {
                    Imgproc.cvtColor(letterboxBgr, _hfFaceDetectorInferenceRgb8u, Imgproc.COLOR_BGR2RGB);
                    _hfFaceDetectorInferenceRgb8u.convertTo(_hfFaceDetectorInferenceBlobHxW, CvType.CV_32F,
                        1.0 / imageToTensorDivisor, -1.0);
                }
            }

            _hfFaceDetectorForwardOutputList.Clear();
            _faceDetectorNet.setInput(_hfFaceDetectorInferenceBlob);
            await _faceDetectorNet.forwardTaskAsync(_hfFaceDetectorForwardOutputList, _hfFaceDetectorOutLayerNames, cancellationToken);
            return _hfFaceDetectorForwardOutputList;
        }
#endif

        Mat HolisticFaceSsdAnchorsCalculator()
        {
            _hfHolisticFaceSsdAnchors128Cache ??= HolisticFaceBuildSsdAnchorsMat(
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacy128NumLayers,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacyMinScale,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacyMaxScale,
                MediaPipeFaceLandmarker.kFaceDetectorShortRangeImageSize,
                MediaPipeFaceLandmarker.kFaceDetectorShortRangeImageSize,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacyAnchorOffset,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacyAnchorOffset,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacy128Strides,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacyAspectRatio,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacyFixedAnchorSize,
                MediaPipeFaceLandmarker.kFaceDetectorSsdLegacy128InterpolatedScaleAspectRatio,
                MediaPipeFaceLandmarker.kFaceDetectorLegacyShortRangeNumBoxes);
            return _hfHolisticFaceSsdAnchors128Cache;
        }

        static float HolisticFaceSsdAnchors_CalculateScale(float minScale, float maxScale, int strideIndex, int numStrides)
        {
            if (numStrides == 1)
                return (minScale + maxScale) * 0.5f;
            return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1.0f);
        }

        static Mat HolisticFaceBuildSsdAnchorsMat(
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
                throw new InvalidOperationException("The lengths of SSD strides and num_layers do not match.");
            if (!fixedAnchorSize)
                throw new InvalidOperationException("The legacy SSD face detector requires fixed_anchor_size=true.");

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
                    float scale = HolisticFaceSsdAnchors_CalculateScale(minScale, maxScale, lastSameStrideLayer, stridesLen);
                    aspectRatios.Add(aspectRatioOption);
                    scales.Add(scale);
                    if (interpolatedScaleAspectRatio > 0f)
                    {
                        float scaleNext = lastSameStrideLayer == stridesLen - 1
                            ? 1.0f
                            : HolisticFaceSsdAnchors_CalculateScale(minScale, maxScale, lastSameStrideLayer + 1, stridesLen);
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
                    $"Face SSD anchor count mismatch: expected {expectedRows}, actual {outIx / 4}.");

            Mat anchors = new Mat(expectedRows, 4, CvType.CV_32FC1);
            anchors.put(0, 0, xywh.AsSpan(0, expectedRows * 4));
            return anchors;
        }

        void HolisticFaceTensorsToDetectionsCalculator(Mat boxRows, Mat scoreCol, Mat anchorsXywh, int num)
        {
            int numCoords = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords;
            float xScale = MediaPipeFaceLandmarker.kFaceDetectorShortRangeImageSize;
            float yScale = xScale;
            float wScale = xScale;
            float hScale = xScale;

            if (_hfFaceDetectorDecodedBoxesNx16 == null
                || _hfFaceDetectorDecodedBoxesNx16.rows() != num
                || _hfFaceDetectorDecodedBoxesNx16.cols() != numCoords)
            {
                _hfFaceDetectorDecodedBoxesNx16?.Dispose();
                _hfFaceDetectorDecodedBoxesNx16 = new Mat(num, numCoords, CvType.CV_32FC1);
            }

            HolisticFaceNumpyClip(scoreCol, -MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsScoreClippingThresh,
                MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsScoreClippingThresh);
            Core.multiply(scoreCol, (-1.0, 0, 0, 0), scoreCol);
            Core.exp(scoreCol, scoreCol);
            Core.add(scoreCol, (1.0, 0, 0, 0), scoreCol);
            Core.divide(1.0, scoreCol, scoreCol);

            if (_hfFaceDetectorDecodeRowSrc == null || _hfFaceDetectorDecodeRowSrc.Length < numCoords)
                _hfFaceDetectorDecodeRowSrc = new float[numCoords];
            if (_hfFaceDetectorDecodeRowDst == null || _hfFaceDetectorDecodeRowDst.Length < numCoords)
                _hfFaceDetectorDecodeRowDst = new float[numCoords];
            if (_hfFaceDetectorAnchorRow4 == null || _hfFaceDetectorAnchorRow4.Length < 4)
                _hfFaceDetectorAnchorRow4 = new float[4];

            float[] rowRaw = _hfFaceDetectorDecodeRowSrc;
            float[] rowDecoded = _hfFaceDetectorDecodeRowDst;
            float[] ar = _hfFaceDetectorAnchorRow4;

            for (int i = 0; i < num; i++)
            {
                boxRows.get(i, 0, rowRaw.AsSpan(0, numCoords));
                anchorsXywh.get(i, 0, ar.AsSpan(0, 4));
                float ax = ar[0];
                float ay = ar[1];
                float aw = ar[2];
                float ah = ar[3];

                int boxOff = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsBoxCoordOffset;
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

                int kpOff = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                int nKp = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumKeypoints;
                for (int k = 0; k < nKp; k++)
                {
                    int o = kpOff + k * MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumValuesPerKeypoint;
                    float kxRaw = rowRaw[o];
                    float kyRaw = rowRaw[o + 1];
                    rowDecoded[o] = kxRaw / xScale * aw + ax;
                    rowDecoded[o + 1] = kyRaw / yScale * ah + ay;
                }

                _hfFaceDetectorDecodedBoxesNx16.put(i, 0, rowDecoded.AsSpan(0, numCoords));
            }
        }

        void HolisticFaceNumpyClip(Mat a, double aMin, double aMax)
        {
            if (a == null || a.empty())
                return;
            if (_hfNumpyClipLo == null)
                _hfNumpyClipLo = new Mat();
            if (_hfNumpyClipHi == null)
                _hfNumpyClipHi = new Mat();
            _hfNumpyClipLo.create(a.rows(), a.cols(), a.type());
            _hfNumpyClipHi.create(a.rows(), a.cols(), a.type());
            _hfNumpyClipLo.setTo((aMin, aMin, aMin, aMin));
            _hfNumpyClipHi.setTo((aMax, aMax, aMax, aMax));
            Core.max(a, _hfNumpyClipLo, a);
            Core.min(a, _hfNumpyClipHi, a);
        }

        /// <summary>
        /// Follows the same order as <see cref="MediaPipeFaceLandmarker.FaceDetectionsFilterByMinScoreThresh"/>
        /// for Holistic face detection.
        /// </summary>
        void HolisticFaceDetectionsFilterByMinScoreThresh(
            Mat boxXywh,
            Mat scoreNx1,
            Mat decodedNx16,
            float minScoreThresh,
            out Mat boxOut,
            out Mat scoreOut,
            out Mat decodedOut)
        {
            int num = boxXywh.rows();
            int nCoord = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords;
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

            if (_hfFaceScoreFilteredBoxXywh == null)
                _hfFaceScoreFilteredBoxXywh = new Mat();
            if (_hfFaceScoreFilteredScore == null)
                _hfFaceScoreFilteredScore = new Mat();
            if (_hfFaceScoreFilteredDecodedNx16 == null)
                _hfFaceScoreFilteredDecodedNx16 = new Mat();

            _hfFaceScoreFilteredBoxXywh.create(kept, 4, CvType.CV_32FC1);
            _hfFaceScoreFilteredScore.create(kept, 1, CvType.CV_32FC1);
            _hfFaceScoreFilteredDecodedNx16.create(kept, nCoord, CvType.CV_32FC1);

            int r = 0;
            for (int i = 0; i < num; i++)
            {
                if (scoreNx1.at<float>(i, 0)[0] < minScoreThresh)
                    continue;
                using (Mat srcRow = boxXywh.row(i))
                using (Mat dstRow = _hfFaceScoreFilteredBoxXywh.row(r))
                    srcRow.copyTo(dstRow);
                using (Mat srcRow = scoreNx1.row(i))
                using (Mat dstRow = _hfFaceScoreFilteredScore.row(r))
                    srcRow.copyTo(dstRow);
                using (Mat srcRow = decodedNx16.row(i))
                using (Mat dstRow = _hfFaceScoreFilteredDecodedNx16.row(r))
                    srcRow.copyTo(dstRow);
                r++;
            }

            boxOut = _hfFaceScoreFilteredBoxXywh;
            scoreOut = _hfFaceScoreFilteredScore;
            decodedOut = _hfFaceScoreFilteredDecodedNx16;
        }

        MatOfInt HolisticFaceNonMaxSuppressionCalculator(Mat boxXywhTensorNorm, Mat scoreCol, Mat decodedBoxesNx16)
        {
            if (_hfFaceNmsIndices == null)
                _hfFaceNmsIndices = new MatOfInt();
            if (_hfWnmsMergedBoxXywh == null)
                _hfWnmsMergedBoxXywh = new Mat();
            if (_hfWnmsMergedDecodedNx16 == null)
                _hfWnmsMergedDecodedNx16 = new Mat();
            if (_hfWnmsMergedScore == null)
                _hfWnmsMergedScore = new Mat();

            int num = boxXywhTensorNorm.rows();
            int numKpFloats = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumKeypoints
                * MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumValuesPerKeypoint;
            if (num <= 0 || scoreCol == null || scoreCol.rows() < num
                         || decodedBoxesNx16 == null || decodedBoxesNx16.rows() < num)
            {
                _hfWnmsMergedBoxXywh.create(0, 4, CvType.CV_32FC1);
                _hfWnmsMergedDecodedNx16.create(0, MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords, CvType.CV_32FC1);
                _hfWnmsMergedScore.create(0, 1, CvType.CV_32FC1);
                _hfFaceNmsIndices.create(0, 1, CvType.CV_32SC1);
                return _hfFaceNmsIndices;
            }

            _hfWnmsIndexed.Clear();
            for (int i = 0; i < num; i++)
                _hfWnmsIndexed.Add((i, scoreCol.at<float>(i, 0)[0]));
            _hfWnmsIndexed.Sort((a, b) => b.sc.CompareTo(a.sc));

            _hfWnmsRemained.Clear();
            _hfWnmsRemained.AddRange(_hfWnmsIndexed);

            _hfNmsMergedBoxScratch.Clear();
            _hfNmsMergedDecScratch.Clear();
            _hfNmsMergedScScratch.Clear();

            if (_hfWnmsKpAccumulator == null || _hfWnmsKpAccumulator.Length < numKpFloats)
                _hfWnmsKpAccumulator = new float[numKpFloats];

            int nCoordFull = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords;
            float[] decBuf = _hfFaceDetectorDecodeRowDst;

            while (_hfWnmsRemained.Count > 0)
            {
                int originalSize = _hfWnmsRemained.Count;
                var anchor = _hfWnmsRemained[0];

                float ax = boxXywhTensorNorm.at<float>(anchor.idx, 0)[0];
                float ay = boxXywhTensorNorm.at<float>(anchor.idx, 1)[0];
                float aw = boxXywhTensorNorm.at<float>(anchor.idx, 2)[0];
                float ah = boxXywhTensorNorm.at<float>(anchor.idx, 3)[0];

                _hfWnmsNextRemained.Clear();
                for (int t = 0; t < _hfWnmsRemained.Count; t++)
                {
                    var item = _hfWnmsRemained[t];
                    float bx = boxXywhTensorNorm.at<float>(item.idx, 0)[0];
                    float by = boxXywhTensorNorm.at<float>(item.idx, 1)[0];
                    float bw = boxXywhTensorNorm.at<float>(item.idx, 2)[0];
                    float bh = boxXywhTensorNorm.at<float>(item.idx, 3)[0];
                    if (HolisticFaceNonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) > _minFaceSuppressionThreshold)
                        continue;
                    _hfWnmsNextRemained.Add(item);
                }

                float wXmin = 0f, wYmin = 0f, wXmax = 0f, wYmax = 0f;
                float totalScore = 0f;
                float[] kpAcc = _hfWnmsKpAccumulator;
                Array.Clear(kpAcc, 0, numKpFloats);
                for (int t = 0; t < _hfWnmsRemained.Count; t++)
                {
                    var c = _hfWnmsRemained[t];
                    float bx = boxXywhTensorNorm.at<float>(c.idx, 0)[0];
                    float by = boxXywhTensorNorm.at<float>(c.idx, 1)[0];
                    float bw = boxXywhTensorNorm.at<float>(c.idx, 2)[0];
                    float bh = boxXywhTensorNorm.at<float>(c.idx, 3)[0];
                    if (HolisticFaceNonMaxSuppressionCalculator_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) <= _minFaceSuppressionThreshold)
                        continue;

                    float s = c.sc;
                    totalScore += s;
                    wXmin += bx * s;
                    wYmin += by * s;
                    wXmax += (bx + bw) * s;
                    wYmax += (by + bh) * s;
                    decodedBoxesNx16.get(c.idx, 0, decBuf.AsSpan(0, nCoordFull));
                    int kpOff = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                    for (int k = 0; k < numKpFloats; k++)
                        kpAcc[k] += decBuf[kpOff + k] * s;
                }

                if (totalScore <= 0f)
                    break;

                float outXmin = wXmin / totalScore;
                float outYmin = wYmin / totalScore;
                float outW = wXmax / totalScore - outXmin;
                float outH = wYmax / totalScore - outYmin;

                float[] outDec = RentHolisticFaceDetectorNmsDec16();
                outDec[0] = outYmin;
                outDec[1] = outXmin;
                outDec[2] = outYmin + outH;
                outDec[3] = outXmin + outW;
                int kOff = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                for (int k = 0; k < numKpFloats; k++)
                    outDec[kOff + k] = kpAcc[k] / totalScore;

                float[] box4 = RentHolisticFaceDetectorNmsBox4();
                box4[0] = outXmin;
                box4[1] = outYmin;
                box4[2] = outW;
                box4[3] = outH;
                _hfNmsMergedBoxScratch.Add(box4);
                _hfNmsMergedDecScratch.Add(outDec);
                _hfNmsMergedScScratch.Add(anchor.sc);

                if (originalSize == _hfWnmsNextRemained.Count)
                    break;

                (_hfWnmsRemained, _hfWnmsNextRemained) = (_hfWnmsNextRemained, _hfWnmsRemained);
            }

            int kOut = _hfNmsMergedScScratch.Count;
            _hfWnmsMergedBoxXywh.create(kOut, 4, CvType.CV_32FC1);
            _hfWnmsMergedDecodedNx16.create(kOut, MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords, CvType.CV_32FC1);
            _hfWnmsMergedScore.create(kOut, 1, CvType.CV_32FC1);
            Span<float> putScore1 = stackalloc float[1];
            Span<int> putIdx1 = stackalloc int[1];
            for (int r = 0; r < kOut; r++)
            {
                _hfWnmsMergedBoxXywh.put(r, 0, _hfNmsMergedBoxScratch[r].AsSpan(0, 4));
                _hfWnmsMergedDecodedNx16.put(r, 0, _hfNmsMergedDecScratch[r].AsSpan(0, nCoordFull));
                putScore1[0] = _hfNmsMergedScScratch[r];
                _hfWnmsMergedScore.put(r, 0, putScore1);
            }

            _hfFaceNmsIndices.create(kOut, 1, CvType.CV_32SC1);
            for (int r = 0; r < kOut; r++)
            {
                putIdx1[0] = r;
                _hfFaceNmsIndices.put(r, 0, putIdx1);
            }

            ReleaseHolisticFaceDetectorNmsMergedScratchLists();

            return _hfFaceNmsIndices;
        }

        static float HolisticFaceNonMaxSuppressionCalculator_ComputeIouXywh(
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

        List<float[]> HolisticFaceDetectionProjectionCalculator(Mat boxXywhTensorNorm, Mat scoreCol, Mat decodedBoxesNx16,
            MatOfInt indices, int imgW, int imgH)
        {
            var list = new List<float[]>();
            if (indices == null || indices.empty() || _hfFaceDetectorProjectionMatrix16 == null)
                return list;

            ReadOnlySpan<float> m = _hfFaceDetectorProjectionMatrix16;
            int selected = indices.rows();
            Span<float> dst = stackalloc float[kHolisticFaceDetectorProjectedRowLength];
            float[] boxTn = _hfFaceDetectorAnchorRow4;
            float[] allTn = _hfFaceDetectorDecodeRowSrc;

            int nCoordFull = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords;
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
                HolisticFaceDetectionProjection_Project(m, xminTn, yminTn, out float p0x, out float p0y);
                HolisticFaceDetectionProjection_Project(m, xminTn + wTn, yminTn, out float p1x, out float p1y);
                HolisticFaceDetectionProjection_Project(m, xminTn + wTn, yminTn + hTn, out float p2x, out float p2y);
                HolisticFaceDetectionProjection_Project(m, xminTn, yminTn + hTn, out float p3x, out float p3y);
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

                decodedBoxesNx16.get(idx, 0, allTn.AsSpan(0, nCoordFull));
                int kpOff = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsKeypointCoordOffset;
                int nKp2 = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumKeypoints
                    * MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumValuesPerKeypoint;
                for (int j = 0; j < nKp2; j += 2)
                {
                    float kx = allTn[kpOff + j];
                    float ky = allTn[kpOff + j + 1];
                    HolisticFaceDetectionProjection_Project(m, kx, ky, out float nx, out float ny);
                    dst[4 + j] = nx;
                    dst[4 + j + 1] = ny;
                }

                dst[16] = scoreCol.at<float>(idx, 0)[0];

                float[] row = RentHolisticFaceDetectorProjRow17();
                dst.CopyTo(row);
                list.Add(row);
            }

            list = HolisticFaceClipDetectionVectorSizeCalculator(list, kHolisticNumFacesFaceDetector);
            return list;
        }

        static void HolisticFaceDetectionProjection_Project(ReadOnlySpan<float> m, float tx, float ty, out float nx, out float ny)
        {
            nx = tx * m[0] + ty * m[1] + m[3];
            ny = tx * m[4] + ty * m[5] + m[7];
        }

        List<float[]> HolisticFaceClipDetectionVectorSizeCalculator(List<float[]> detections, int maxVecSize)
        {
            if (detections == null)
                return HolisticFaceDetectorGraphEmptyDetections;
            if (detections.Count <= maxVecSize)
                return detections;

            for (int i = maxVecSize; i < detections.Count; i++)
                ReleaseHolisticFaceDetectorProjRow17(detections[i]);

            var clipped = new List<float[]>(maxVecSize);
            for (int i = 0; i < maxVecSize; i++)
                clipped.Add(detections[i]);
            return clipped;
        }

        Mat HolisticFaceDetectorGraph_BuildNmsBoxXywhFromDecoded(int num)
        {
            if (_hfFaceTensorsToDetectionsWorking == null
                || _hfFaceTensorsToDetectionsWorking.rows() != num
                || _hfFaceTensorsToDetectionsWorking.cols() != 4)
            {
                _hfFaceTensorsToDetectionsWorking?.Dispose();
                _hfFaceTensorsToDetectionsWorking = new Mat(num, 4, CvType.CV_32FC1);
            }

            Mat dst = _hfFaceTensorsToDetectionsWorking;
            float[] row = _hfFaceDetectorDecodeRowSrc;
            int nCoord = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords;
            Span<float> put4 = stackalloc float[4];
            for (int i = 0; i < num; i++)
            {
                _hfFaceDetectorDecodedBoxesNx16.get(i, 0, row.AsSpan(0, nCoord));
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

        Mat HolisticFaceDetectorGraph_PrepareBoxMajorRows(Mat output0, int numBoxes)
        {
            int c = MediaPipeFaceLandmarker.kFaceDetectorTensorsToDetectionsNumCoords;
            if (output0.size(1) == numBoxes && output0.size(2) == c)
                return output0.reshape(1, numBoxes);

            if (output0.size(1) == c && output0.size(2) == numBoxes)
            {
                using (Mat m16xN = output0.reshape(1, c))
                {
                    Mat transposed = new Mat(numBoxes, c, CvType.CV_32FC1);
                    Core.transpose(m16xN, transposed);
                    return transposed;
                }
            }

            long total = output0.total();
            if (total == (long)numBoxes * c)
            {
                Mat reshaped = output0.reshape(1, numBoxes);
                if (reshaped.rows() == numBoxes && reshaped.cols() == c)
                    return reshaped;
            }

            throw new InvalidOperationException(
                $"Unsupported face_detector output tensor shape: dims={output0.dims()}");
        }

        Mat HolisticFaceDetectorGraph_PrepareScoreColumn(Mat output1, int numBoxes)
        {
            if (output1.size(1) == numBoxes && (output1.size(2) == 1 || output1.channels() * output1.size(2) == 1))
                return output1.reshape(1, numBoxes);

            if (output1.size(1) == 1 && output1.size(2) == numBoxes)
            {
                Mat col = new Mat(numBoxes, 1, CvType.CV_32FC1);
                Core.transpose(output1.reshape(1, numBoxes), col);
                return col;
            }

            long total = output1.total();
            if (total == numBoxes)
                return output1.reshape(1, numBoxes);

            throw new InvalidOperationException(
                $"Unsupported face_detector score tensor shape: size1={output1.size(1)} size2={output1.size(2)}");
        }

        bool ImagePreprocessingGraph_SingleFaceLandmarks(Mat image, HolisticNormalizedRect faceRect,
            out HolisticSingleFaceLmPreprocessOut pre)
        {
            pre = default;
            int imgW = image.cols();
            int imgH = image.rows();
            int ts = MediaPipeFaceLandmarker.kFaceLandmarksDetectorImageSize;
            if (imgW <= 0 || imgH <= 0 || ts <= 0)
                return false;

            const int lmC = 3;
            const float image01Divisor = 255f;

            if (_hfFaceLmWarpDstPts == null)
            {
                _hfFaceLmWarpDstPts = new Mat(4, 2, CvType.CV_32FC1);
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
                _hfFaceLmWarpDstPts.put(0, 0, dstPtsArr);
                _hfFaceLmWarpSrcPts = new Mat(4, 2, CvType.CV_32FC1);
            }

            if (_hfFaceLmWarpedBgr == null || _hfFaceLmWarpedBgr.rows() != ts || _hfFaceLmWarpedBgr.cols() != ts)
            {
                _hfFaceLmWarpedBgr?.Dispose();
                _hfFaceLmWarpedRgb?.Dispose();
                _hfFaceLandmarksInferenceBlob?.Dispose();
                _hfFaceLmWarpedBgr = new Mat(ts, ts, CvType.CV_8UC3);
                _hfFaceLmWarpedRgb = new Mat(ts, ts, CvType.CV_8UC3);
                _hfFaceLandmarksInferenceBlob = new Mat(new int[] { 1, ts, ts, lmC }, CvType.CV_32FC1);
                _hfFaceLandmarksInferenceBlobHxW =
                    _hfFaceLandmarksInferenceBlob.reshape(lmC, new int[] { ts, ts });
            }

            float cx = faceRect.XCenter * imgW;
            float cy = faceRect.YCenter * imgH;
            float rw = faceRect.Width * imgW;
            float rh = faceRect.Height * imgH;
            if (rw <= 0f || rh <= 0f || float.IsNaN(rw) || float.IsNaN(rh))
                return false;

            HolisticFacePadRoiLikeImageToTensor(ts, ts, true, ref rw, ref rh,
                out float padL, out float padT, out float padR, out float padB);

            double angleDeg = faceRect.Rotation * 180.0 / Math.PI;

            Imgproc.boxPoints((cx, cy, rw, rh, angleDeg), _hfFaceLmWarpSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_hfFaceLmWarpSrcPts, _hfFaceLmWarpDstPts))
            {
                Imgproc.warpPerspective(image, _hfFaceLmWarpedBgr, projMat, (ts, ts),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
            }

            Imgproc.cvtColor(_hfFaceLmWarpedBgr, _hfFaceLmWarpedRgb, Imgproc.COLOR_BGR2RGB);
            _hfFaceLmWarpedRgb.convertTo(_hfFaceLandmarksInferenceBlobHxW, CvType.CV_32F,
                1.0 / image01Divisor);

            pre = new HolisticSingleFaceLmPreprocessOut
            {
                FaceBlob = _hfFaceLandmarksInferenceBlob,
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

        static void HolisticFacePadRoiLikeImageToTensor(int tensorW, int tensorH, bool keepAspectRatio,
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

        List<Mat> InferenceSubgraph_SingleFaceLandmarks(Mat faceBlob)
        {
            if (_faceLandmarksNet == null || _hfFaceLandmarksNetOutLayerNames == null)
            {
                _hfFaceLandmarksForwardOutputList.Clear();
                return _hfFaceLandmarksForwardOutputList;
            }

            _faceLandmarksNet.setInput(faceBlob);
            _hfFaceLandmarksForwardOutputList.Clear();
            _faceLandmarksNet.forward(_hfFaceLandmarksForwardOutputList, _hfFaceLandmarksNetOutLayerNames);
            return _hfFaceLandmarksForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<List<Mat>> InferenceSubgraph_SingleFaceLandmarksAsync(Mat faceBlob, CancellationToken cancellationToken)
        {
            if (_faceLandmarksNet == null || _hfFaceLandmarksNetOutLayerNames == null)
            {
                _hfFaceLandmarksForwardOutputList.Clear();
                return _hfFaceLandmarksForwardOutputList;
            }

            _hfFaceLandmarksForwardOutputList.Clear();
            _faceLandmarksNet.setInput(faceBlob);
            await _faceLandmarksNet.forwardTaskAsync(_hfFaceLandmarksForwardOutputList, _hfFaceLandmarksNetOutLayerNames, cancellationToken);
            return _hfFaceLandmarksForwardOutputList;
        }
#endif

        /// <summary>
        /// Equivalent to <c>SingleFaceLandmarksDetectorGraph</c> in
        /// <c>face_landmarks_detector_graph.cc</c>.
        /// Child path: <see cref="ImagePreprocessingGraph_SingleFaceLandmarks"/> ->
        /// <see cref="InferenceSubgraph_SingleFaceLandmarks"/> ->
        /// <see cref="SplitTensorVectorCalculator_FaceLandmarks"/> → <see cref="TensorsToFaceLandmarksGraph"/> → …
        /// The overall order matches <see cref="MediaPipeFaceLandmarker.SingleFaceLandmarksDetectorGraph"/>.
        /// </summary>
        HolisticSingleFaceGraphFaceResult? SingleFaceLandmarksDetectorGraph(Mat image, HolisticNormalizedRect faceRect, int imgW, int imgH)
        {
            if (_faceLandmarksNet == null || _hfFaceLandmarksNetOutLayerNames == null)
                return null;
            if (!ImagePreprocessingGraph_SingleFaceLandmarks(image, faceRect, out HolisticSingleFaceLmPreprocessOut pre))
                return null;

            List<Mat> inferenceTensors = InferenceSubgraph_SingleFaceLandmarks(pre.FaceBlob);
            if (inferenceTensors == null
                || inferenceTensors.Count < MediaPipeFaceLandmarker.kFaceLandmarksOutputTensorsNum)
                return null;

            if (!SplitTensorVectorCalculator_FaceLandmarks(inferenceTensors, out Mat landmarkTensor, out Mat presenceTensor))
                return null;

            float[] letterboxedNormLm = TensorsToFaceLandmarksGraph(landmarkTensor, pre.ModelW, pre.ModelH);
            float presenceScore = TensorsToFloatsCalculator_FacePresence(presenceTensor);
            bool facePresence = ThresholdingCalculator_FacePresence(presenceScore);

            float[] afterLetterbox = LandmarkLetterboxRemovalCalculator_Face(letterboxedNormLm,
                pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft, pre.LetterboxPaddingBottom,
                pre.LetterboxPaddingRight);

            Vec3f[] projectedRaw =
                LandmarkProjectionCalculator_SingleFaceLandmarks(afterLetterbox, faceRect, pre.ImageW, pre.ImageH);
            Vec3f[] projected = AllowIf_FaceNormLandmarks(facePresence, projectedRaw);

            HolisticNormalizedRect nextFrame;
            if (facePresence)
            {
                HolisticFaceLandmarkPseudoDetection det = LandmarksToDetectionCalculator_Face478(projected);
                HolisticNormalizedRect faceLmRect =
                    DetectionsToRectsCalculator_FaceLandmarksRoi_FromDetection(det, pre.ImageW, pre.ImageH);
                nextFrame = RectTransformationCalculator_FaceLandmarksNextFrame(faceLmRect, pre.ImageW, pre.ImageH);
            }
            else
                nextFrame = default;

            nextFrame = AllowIf_FaceNextFrameRect(facePresence, nextFrame);

            return new HolisticSingleFaceGraphFaceResult
            {
                FacePresence = facePresence,
                FacePresenceScore = presenceScore,
                NormLandmarks = projected,
                NextFrameRect = nextFrame,
            };
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<HolisticSingleFaceGraphFaceResult?> SingleFaceLandmarksDetectorGraphAsync(Mat image, HolisticNormalizedRect faceRect, int imgW, int imgH, CancellationToken cancellationToken)
        {
            if (_faceLandmarksNet == null || _hfFaceLandmarksNetOutLayerNames == null)
                return null;
            if (!ImagePreprocessingGraph_SingleFaceLandmarks(image, faceRect, out HolisticSingleFaceLmPreprocessOut pre))
                return null;

            var inferenceTensors = await InferenceSubgraph_SingleFaceLandmarksAsync(pre.FaceBlob, cancellationToken);
            if (inferenceTensors == null
                || inferenceTensors.Count < MediaPipeFaceLandmarker.kFaceLandmarksOutputTensorsNum)
                return null;

            if (!SplitTensorVectorCalculator_FaceLandmarks(inferenceTensors, out Mat landmarkTensor, out Mat presenceTensor))
                return null;

            float[] letterboxedNormLm = TensorsToFaceLandmarksGraph(landmarkTensor, pre.ModelW, pre.ModelH);
            float presenceScore = TensorsToFloatsCalculator_FacePresence(presenceTensor);
            bool facePresence = ThresholdingCalculator_FacePresence(presenceScore);

            float[] afterLetterbox = LandmarkLetterboxRemovalCalculator_Face(letterboxedNormLm,
                pre.LetterboxPaddingTop, pre.LetterboxPaddingLeft, pre.LetterboxPaddingBottom,
                pre.LetterboxPaddingRight);

            Vec3f[] projectedRaw =
                LandmarkProjectionCalculator_SingleFaceLandmarks(afterLetterbox, faceRect, pre.ImageW, pre.ImageH);
            Vec3f[] projected = AllowIf_FaceNormLandmarks(facePresence, projectedRaw);

            HolisticNormalizedRect nextFrame;
            if (facePresence)
            {
                HolisticFaceLandmarkPseudoDetection det = LandmarksToDetectionCalculator_Face478(projected);
                HolisticNormalizedRect faceLmRect =
                    DetectionsToRectsCalculator_FaceLandmarksRoi_FromDetection(det, pre.ImageW, pre.ImageH);
                nextFrame = RectTransformationCalculator_FaceLandmarksNextFrame(faceLmRect, pre.ImageW, pre.ImageH);
            }
            else
                nextFrame = default;

            nextFrame = AllowIf_FaceNextFrameRect(facePresence, nextFrame);

            return new HolisticSingleFaceGraphFaceResult
            {
                FacePresence = facePresence,
                FacePresenceScore = presenceScore,
                NormLandmarks = projected,
                NextFrameRect = nextFrame,
            };
        }
#endif

        static HolisticNormalizedRect DetectionsToRectsCalculator_FaceLandmarksRoi_FromDetection(
            HolisticFaceLandmarkPseudoDetection det, int imgW, int imgH)
        {
            float xmin = det.Xmin;
            float ymin = det.Ymin;
            float wBox = det.Width;
            float hBox = det.Height;
            float centerX = xmin + wBox * 0.5f;
            float centerY = ymin + hBox * 0.5f;

            int k0 = MediaPipeFaceLandmarker.kFaceLandmarksDetectionsToRectsRotationStartKeypointIndex;
            int k1 = MediaPipeFaceLandmarker.kFaceLandmarksDetectionsToRectsRotationEndKeypointIndex;
            Vec3f[] kp = det.KeypointsNorm;
            if (kp == null || kp.Length <= Mathf.Max(k0, k1))
                return default;

            float x0 = kp[k0].Item1 * imgW;
            float y0 = kp[k0].Item2 * imgH;
            float x1 = kp[k1].Item1 * imgW;
            float y1 = kp[k1].Item2 * imgH;

            float targetRad = MediaPipeFaceLandmarker.kFaceLandmarksDetectionsToRectsTargetAngleDegrees * (Mathf.PI / 180f);
            float rotation = HolisticFaceNormalizeRadians(targetRad - Mathf.Atan2(-(y1 - y0), x1 - x0));

            return new HolisticNormalizedRect
            {
                XCenter = centerX,
                YCenter = centerY,
                Width = wBox,
                Height = hBox,
                Rotation = rotation,
            };
        }

        static HolisticNormalizedRect RectTransformationCalculator_FaceLandmarksNextFrame(HolisticNormalizedRect rect, int imageW, int imageH)
        {
            if (imageW <= 0 || imageH <= 0)
                return default;

            float width = rect.Width;
            float height = rect.Height;
            float rotation = rect.Rotation;
            float xCenter = rect.XCenter;
            float yCenter = rect.YCenter;

            float longSidePx = Mathf.Max(width * imageW, height * imageH);
            width = longSidePx / imageW;
            height = longSidePx / imageH;
            width *= MediaPipeFaceLandmarker.kFaceLandmarksNextFrameRoiScale;
            height *= MediaPipeFaceLandmarker.kFaceLandmarksNextFrameRoiScale;

            return new HolisticNormalizedRect
            {
                XCenter = xCenter,
                YCenter = yCenter,
                Width = width,
                Height = height,
                Rotation = rotation,
            };
        }

        static HolisticNormalizedRect AllowIf_FaceNextFrameRect(bool facePresence, HolisticNormalizedRect rectWhenPresent) =>
            facePresence ? rectWhenPresent : default;

        bool SplitTensorVectorCalculator_FaceLandmarks(List<Mat> inferenceTensors, out Mat landmarkTensor,
            out Mat presenceTensor)
        {
            landmarkTensor = presenceTensor = null;
            if (inferenceTensors == null || inferenceTensors.Count < MediaPipeFaceLandmarker.kFaceLandmarksOutputTensorsNum)
                return false;
            landmarkTensor = inferenceTensors[1];
            presenceTensor = inferenceTensors[0];
            return landmarkTensor != null && presenceTensor != null;
        }

        /// <summary>Equivalent to <c>TensorsToFaceLandmarksGraph</c>, containing only the internal <c>TensorsToLandmarksCalculator</c> path.</summary>
        float[] TensorsToFaceLandmarksGraph(Mat landmarkTensor, int modelW, int modelH) =>
            TensorsToLandmarksCalculator_Face(landmarkTensor, modelW, modelH);

        float[] TensorsToLandmarksCalculator_Face(Mat tensor, int inputW, int inputH)
        {
            const float normalizeZ = 1f;
            int n = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            int need = n * 3;
            long tTotal = tensor.total();
            if (tTotal < need)
            {
                if (_hfFaceTensorsToLmNorm == null || _hfFaceTensorsToLmNorm.Length < need)
                    _hfFaceTensorsToLmNorm = new float[need];
                Array.Clear(_hfFaceTensorsToLmNorm, 0, need);
                return _hfFaceTensorsToLmNorm;
            }

            if (_hfFaceTensorsToLmRaw == null || _hfFaceTensorsToLmRaw.Length < need)
                _hfFaceTensorsToLmRaw = new float[need];
            if (_hfFaceTensorsToLmNorm == null || _hfFaceTensorsToLmNorm.Length < need)
                _hfFaceTensorsToLmNorm = new float[need];

            using (var flat = tensor.reshape(1, (int)tTotal))
            {
                float[] raw = _hfFaceTensorsToLmRaw;
                float[] norm = _hfFaceTensorsToLmNorm;
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

        static float TensorsToFloatsCalculator_FacePresence(Mat presenceTensor)
        {
            float v = presenceTensor.at<float>(0, 0)[0];
            return 1f / (1f + Mathf.Exp(-v));
        }

        bool ThresholdingCalculator_FacePresence(float score) => score >= _minFacePresenceConfidence;

        float[] LandmarkLetterboxRemovalCalculator_Face(float[] normLandmarks, float padTop, float padLeft,
            float padBottom, float padRight)
        {
            int el = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum * 3;
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

            float[] o = _hfFaceLetterboxRemovedNormScratch;
            int n = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            for (int i = 0; i < n; i++)
            {
                int k = i * 3;
                o[k] = (normLandmarks[k] - padLeft) / w;
                o[k + 1] = (normLandmarks[k + 1] - padTop) / h;
                o[k + 2] = normLandmarks[k + 2] / w;
            }

            return o;
        }

        Vec3f[] LandmarkProjectionCalculator_SingleFaceLandmarks(float[] normLandmarksLetterboxRemoved,
            HolisticNormalizedRect faceRect, int imgW, int imgH)
        {
            int n = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            var screen = new Vec3f[n];
            if (normLandmarksLetterboxRemoved == null || normLandmarksLetterboxRemoved.Length < n * 3)
                return screen;

            float cx = faceRect.XCenter * imgW;
            float cy = faceRect.YCenter * imgH;
            float rw = faceRect.Width * imgW;
            float rh = faceRect.Height * imgH;
            float rot = faceRect.Rotation;
            HolisticGetRotatedSubRectToRectTransformMatrix(cx, cy, rw, rh, rot, imgW, imgH, false, _hfFaceLmProjectionMatrix16);
            float zScale = HolisticFaceLandmarkProjection_CalculateZScale(_hfFaceLmProjectionMatrix16);

            for (int i = 0; i < n; i++)
            {
                int o = i * 3;
                float lx = normLandmarksLetterboxRemoved[o];
                float ly = normLandmarksLetterboxRemoved[o + 1];
                float lz = normLandmarksLetterboxRemoved[o + 2];
                HolisticFaceLandmarkProjection_ProjectXY(lx, ly, lz, _hfFaceLmProjectionMatrix16, out float nx, out float ny);
                screen[i] = new Vec3f(nx, ny, zScale * lz);
            }

            return screen;
        }

        static void HolisticFaceLandmarkProjection_ProjectXY(float x, float y, float z, float[] m, out float nx, out float ny)
        {
            nx = x * m[0] + y * m[1] + z * m[2] + m[3];
            ny = x * m[4] + y * m[5] + z * m[6] + m[7];
        }

        static float HolisticFaceLandmarkProjection_CalculateZScale(float[] m)
        {
            HolisticFaceLandmarkProjection_ProjectXY(0f, 0f, 0f, m, out float ax, out float ay);
            HolisticFaceLandmarkProjection_ProjectXY(1f, 0f, 0f, m, out float bx, out float by);
            float dx = bx - ax;
            float dy = by - ay;
            return Mathf.Sqrt(dx * dx + dy * dy);
        }

        static Vec3f[] AllowIf_FaceNormLandmarks(bool facePresence, Vec3f[] landmarksWhenPresent)
        {
            int n = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            if (!facePresence || landmarksWhenPresent == null)
                return new Vec3f[n];
            return landmarksWhenPresent;
        }

        /// <summary>
        /// Equivalent to <c>FaceBlendshapesGraph</c> in <c>face_blendshapes_graph.cc</c>.
        /// Child path: <see cref="SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset"/> ->
        /// <see cref="LandmarksToTensorCalculator_FaceBlendshapes"/> ->
        /// <see cref="InferenceSubgraph_FaceBlendshapes"/> → <see cref="SplitTensorVectorCalculator_FaceBlendshapesOutputTensor"/> →
        /// <see cref="TensorsToClassificationCalculator_FaceBlendshapes"/>.
        /// </summary>
        float[] FaceBlendshapesGraph(Vec3f[] normLandmarks478, int imageWidth, int imageHeight)
        {
            int nLm = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            int nBs = MediaPipeFaceLandmarker.kFaceBlendshapeCoefficientCount;
            if (normLandmarks478 == null || normLandmarks478.Length != nLm || _faceBlendshapesNet == null)
                return new float[nBs];

            Vec3f[] subset = SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset(normLandmarks478);
            LandmarksToTensorCalculator_FaceBlendshapes(subset, imageWidth, imageHeight);
            List<Mat> outs = InferenceSubgraph_FaceBlendshapes();
            Mat tensorVec = outs != null && outs.Count > 0 ? outs[0] : null;
            Mat coeffTensor = SplitTensorVectorCalculator_FaceBlendshapesOutputTensor(tensorVec);
            return TensorsToClassificationCalculator_FaceBlendshapes(coeffTensor);
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<float[]> FaceBlendshapesGraphAsync(Vec3f[] normLandmarks478, int imageWidth, int imageHeight, CancellationToken cancellationToken)
        {
            int nLm = MediaPipeFaceLandmarker.kFaceMeshWithIrisLandmarksNum;
            int nBs = MediaPipeFaceLandmarker.kFaceBlendshapeCoefficientCount;
            if (normLandmarks478 == null || normLandmarks478.Length != nLm || _faceBlendshapesNet == null)
                return new float[nBs];

            Vec3f[] subset = SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset(normLandmarks478);
            LandmarksToTensorCalculator_FaceBlendshapes(subset, imageWidth, imageHeight);
            List<Mat> outs = await InferenceSubgraph_FaceBlendshapesAsync(cancellationToken);
            Mat tensorVec = outs != null && outs.Count > 0 ? outs[0] : null;
            Mat coeffTensor = SplitTensorVectorCalculator_FaceBlendshapesOutputTensor(tensorVec);
            return TensorsToClassificationCalculator_FaceBlendshapes(coeffTensor);
        }
#endif

        Vec3f[] SplitNormalizedLandmarkListCalculator_FaceBlendshapesSubset(Vec3f[] normLandmarks478)
        {
            int[] idxMap = MediaPipeFaceLandmarker.kFaceBlendshapesLandmarkSubsetIndices;
            int len = idxMap.Length;
            if (_hfFaceBlendshapesSubsetScratch == null || _hfFaceBlendshapesSubsetScratch.Length != len)
                _hfFaceBlendshapesSubsetScratch = new Vec3f[len];
            Vec3f[] dst = _hfFaceBlendshapesSubsetScratch;
            for (int i = 0; i < len; i++)
                dst[i] = normLandmarks478[idxMap[i]];
            return dst;
        }

        void LandmarksToTensorCalculator_FaceBlendshapes(Vec3f[] subset146, int imageWidth, int imageHeight)
        {
            int[] idxMap = MediaPipeFaceLandmarker.kFaceBlendshapesLandmarkSubsetIndices;
            int n = idxMap.Length;
            if (_hfFaceBlendshapesInputBlob == null
                || _hfFaceBlendshapesInputBlob.dims() != 3
                || _hfFaceBlendshapesInputBlob.size(0) != 1
                || _hfFaceBlendshapesInputBlob.size(1) != n
                || _hfFaceBlendshapesInputBlob.size(2) != 2)
            {
                _hfFaceBlendshapesInputBlob?.Dispose();
                _hfFaceBlendshapesInputBlob = new Mat(new int[] { 1, n, 2 }, CvType.CV_32FC1);
            }

            int flatLen = n * 2;
            if (_hfFaceBlendshapesLandmarkFlattenBuf == null || _hfFaceBlendshapesLandmarkFlattenBuf.Length < flatLen)
                _hfFaceBlendshapesLandmarkFlattenBuf = new float[flatLen];
            Span<float> buf = _hfFaceBlendshapesLandmarkFlattenBuf.AsSpan(0, flatLen);
            for (int i = 0; i < n; i++)
            {
                buf[i * 2] = subset146[i].Item1 * imageWidth;
                buf[i * 2 + 1] = subset146[i].Item2 * imageHeight;
            }

            OpenCVMatUtils.CopyToMat<float>(buf, _hfFaceBlendshapesInputBlob);
        }

        List<Mat> InferenceSubgraph_FaceBlendshapes()
        {
            if (_faceBlendshapesNet == null || _hfFaceBlendshapesNetOutLayerNames == null)
            {
                _hfFaceBlendshapesForwardOutputList.Clear();
                return _hfFaceBlendshapesForwardOutputList;
            }

            _faceBlendshapesNet.setInput(_hfFaceBlendshapesInputBlob);
            _hfFaceBlendshapesForwardOutputList.Clear();
            _faceBlendshapesNet.forward(_hfFaceBlendshapesForwardOutputList, _hfFaceBlendshapesNetOutLayerNames);
            return _hfFaceBlendshapesForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        async Task<List<Mat>> InferenceSubgraph_FaceBlendshapesAsync(CancellationToken cancellationToken)
        {
            if (_faceBlendshapesNet == null || _hfFaceBlendshapesNetOutLayerNames == null)
            {
                _hfFaceBlendshapesForwardOutputList.Clear();
                return _hfFaceBlendshapesForwardOutputList;
            }

            _hfFaceBlendshapesForwardOutputList.Clear();
            _faceBlendshapesNet.setInput(_hfFaceBlendshapesInputBlob);
            await _faceBlendshapesNet.forwardTaskAsync(_hfFaceBlendshapesForwardOutputList, _hfFaceBlendshapesNetOutLayerNames, cancellationToken);
            return _hfFaceBlendshapesForwardOutputList;
        }
#endif

        /// <summary>Disposes face-tracking reuse <see cref="Mat"/> instances. Called from <see cref="Dispose(bool)"/>.</summary>
        void DisposeHolisticFaceTrackingScratch()
        {
            _hfFaceDetectorLetterboxBgr?.Dispose();
            _hfFaceDetectorInferenceBlob?.Dispose();
            _hfFaceDetectorInferenceRgb8u?.Dispose();
            _hfFaceDetectorInferenceBlobHxW = null;
            _hfFaceDetectorWarpSrcPts?.Dispose();
            _hfFaceDetectorWarpDstPts?.Dispose();
            _hfFaceDetectorDecodedBoxesNx16?.Dispose();
            _hfFaceTensorsToDetectionsWorking?.Dispose();
            _hfFaceNmsIndices?.Dispose();
            _hfWnmsMergedBoxXywh?.Dispose();
            _hfWnmsMergedDecodedNx16?.Dispose();
            _hfWnmsMergedScore?.Dispose();
            _hfFaceLmWarpSrcPts?.Dispose();
            _hfFaceLmWarpDstPts?.Dispose();
            _hfFaceLmWarpedBgr?.Dispose();
            _hfFaceLmWarpedRgb?.Dispose();
            _hfFaceLandmarksInferenceBlob?.Dispose();
            _hfFaceLandmarksInferenceBlobHxW = null;
            _hfFaceBlendshapesInputBlob?.Dispose();
            foreach (var m in _hfFaceDetectorForwardOutputList)
                m?.Dispose();
            _hfFaceDetectorForwardOutputList.Clear();
            foreach (var m in _hfFaceLandmarksForwardOutputList)
                m?.Dispose();
            _hfFaceLandmarksForwardOutputList.Clear();
            foreach (var m in _hfFaceBlendshapesForwardOutputList)
                m?.Dispose();
            _hfFaceBlendshapesForwardOutputList.Clear();
            _hfNumpyClipLo?.Dispose();
            _hfNumpyClipLo = null;
            _hfNumpyClipHi?.Dispose();
            _hfNumpyClipHi = null;
        }

        static Mat SplitTensorVectorCalculator_FaceBlendshapesOutputTensor(Mat tensorsVectorHead) => tensorsVectorHead;

        static float[] TensorsToClassificationCalculator_FaceBlendshapes(Mat coefficients1x52)
        {
            int C = MediaPipeFaceLandmarker.kFaceBlendshapeCoefficientCount;
            var coeffs = new float[C];
            if (coefficients1x52 == null || coefficients1x52.empty())
                return coeffs;

            long t = coefficients1x52.total();
            int n = (int)Math.Min(t, C);
            if (n <= 0)
                return coeffs;

            using (Mat flat = coefficients1x52.reshape(1, (int)t))
            {
                flat.get(0, 0, coeffs.AsSpan(0, n));
            }

            return coeffs;
        }

    }
}
#endif
#endif
