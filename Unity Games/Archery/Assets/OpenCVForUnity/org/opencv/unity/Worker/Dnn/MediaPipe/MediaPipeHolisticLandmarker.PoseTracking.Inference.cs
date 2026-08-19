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
        struct HolisticPoseDetectorGraphResult
        {
            public List<HolisticPoseDetectionData> PoseDetections;
        }

        struct HolisticSinglePoseLandmarksInnerResult
        {
            public bool PosePresence;
            public float PosePresenceScore;
            /// <summary>The 33 points corresponding to upstream [MediaPipe](https://github.com/google-ai-edge/mediapipe) <c>NormalizedLandmark</c>, normalized to the full image.</summary>
            public Vec3f[] NormLandmarks;
            public Vec3f[] WorldLandmarks;
            public float[] LandmarkVisibility;
            public float[] LandmarkVisibilityWorld;
            public float[] LandmarkPresence;
            public Vec3f[] AuxiliaryLandmarksNorm;
            public Mat SegmentationMaskFull;
        }

        static Mat _holisticPoseDetectorSsdAnchors2254Cache;

        /// <summary>
        /// Equivalent to <c>PoseDetectorGraph</c> in <c>pose_detector_graph.cc</c>.
        /// Child calculators are delegated to dedicated helper methods.
        /// </summary>
        HolisticPoseDetectorGraphResult PoseDetectorGraph(Mat image, HolisticNormalizedRect? normRect)
        {
            var empty = new HolisticPoseDetectorGraphResult { PoseDetections = new List<HolisticPoseDetectionData>() };
            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;
            if (normRect.HasValue)
                throw new NotSupportedException("Holistic PoseDetectorGraph supports only the NORM_RECT-disconnected full-image path.");

            Mat inputBlob = null;
            List<Mat> outputBlobs = null;
            try
            {
                ImagePreprocessingGraph_HolisticPoseDetector(image, out _, out inputBlob, out int imageW, out int imageH, out float[] letterboxPadding);
                outputBlobs = InferenceSubgraph_PoseDetection_Holistic(inputBlob);
                if (outputBlobs == null || outputBlobs.Count < 2)
                    return empty;

                Mat output0 = outputBlobs[0];
                Mat output1 = outputBlobs[1];
                int num = output0.size(1);
                if (num <= 0)
                    return empty;

                Mat anchors = SsdAnchorsCalculator_Holistic(out Mat anchorsNx8);
                TensorsToDetectionsCalculator_Holistic(output0, output1, anchors, anchorsNx8, out Mat boxXywh, out Mat scoreForNms, out Mat boxLmForNms);
                var letterboxed = new List<HolisticPoseDetectionData>();
                NonMaxSuppressionCalculator_Holistic(boxXywh, scoreForNms, boxLmForNms, letterboxed);

                var afterLetterbox = DetectionLetterboxRemovalCalculator_Holistic(letterboxed, letterboxPadding);
                List<HolisticPoseDetectionData> clipped = ClipDetectionVectorSizeCalculator_Holistic(afterLetterbox, kHolisticNumPoses);
                return new HolisticPoseDetectorGraphResult { PoseDetections = clipped };
            }
            finally
            {
                inputBlob?.Dispose();
            }
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="PoseDetectorGraph"/> (via <see cref="InferenceSubgraph_PoseDetection_HolisticAsync"/>).
        /// Inference output <see cref="Mat"/> instances are owned by <see cref="MultiBackendNet"/>; only <paramref name="inputBlob"/> is disposed here.
        /// </summary>
        async Task<HolisticPoseDetectorGraphResult> PoseDetectorGraphAsync(Mat image, HolisticNormalizedRect? normRect, CancellationToken cancellationToken)
        {
            var empty = new HolisticPoseDetectorGraphResult { PoseDetections = new List<HolisticPoseDetectionData>() };
            if (image == null || image.empty() || image.cols() <= 0 || image.rows() <= 0)
                return empty;
            if (normRect.HasValue)
                throw new NotSupportedException("Holistic PoseDetectorGraph supports only the NORM_RECT-disconnected full-image path.");

            Mat inputBlob = null;
            try
            {
                ImagePreprocessingGraph_HolisticPoseDetector(image, out _, out inputBlob, out int imageW, out int imageH, out float[] letterboxPadding);
                var outputBlobs = await InferenceSubgraph_PoseDetection_HolisticAsync(inputBlob, cancellationToken);
                if (outputBlobs == null || outputBlobs.Count < 2)
                    return empty;

                Mat output0 = outputBlobs[0];
                Mat output1 = outputBlobs[1];
                int num = output0.size(1);
                if (num <= 0)
                    return empty;

                Mat anchors = SsdAnchorsCalculator_Holistic(out Mat anchorsNx8);
                TensorsToDetectionsCalculator_Holistic(output0, output1, anchors, anchorsNx8, out Mat boxXywh, out Mat scoreForNms, out Mat boxLmForNms);
                var letterboxed = new List<HolisticPoseDetectionData>();
                NonMaxSuppressionCalculator_Holistic(boxXywh, scoreForNms, boxLmForNms, letterboxed);

                var afterLetterbox = DetectionLetterboxRemovalCalculator_Holistic(letterboxed, letterboxPadding);
                List<HolisticPoseDetectionData> clipped = ClipDetectionVectorSizeCalculator_Holistic(afterLetterbox, kHolisticNumPoses);
                return new HolisticPoseDetectorGraphResult { PoseDetections = clipped };
            }
            finally
            {
                inputBlob?.Dispose();
            }
        }
#endif

        /// <summary>
        /// Equivalent to the Holistic pose-detection <c>ImagePreprocessingGraph</c>, producing a
        /// 224 letterboxed <c>[-1,1]</c> blob.
        /// </summary>
        /// <remarks>
        /// The resized integer dimensions are truncated with <c>(int)(width * ratio)</c>, matching the
        /// palm-detection letterbox path in <see cref="MediaPipeHandLandmarker"/>.
        /// </remarks>
        void ImagePreprocessingGraph_HolisticPoseDetector(Mat image, out Mat letterbox224, out Mat inputBlob, out int imageW, out int imageH, out float[] letterboxPadding)
        {
            const int tensorSize = 224;
            imageW = image.cols();
            imageH = image.rows();

            if (_hpPoseDetectorLetterbox224 == null)
                _hpPoseDetectorLetterbox224 = new Mat(tensorSize, tensorSize, image.type());
            letterbox224 = _hpPoseDetectorLetterbox224;

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
                    resized.copyTo(roi);
            }

            float padLeft = padX / (float)tensorSize;
            float padTop = padY / (float)tensorSize;
            float padRight = (tensorSize - padX - newW) / (float)tensorSize;
            float padBottom = (tensorSize - padY - newH) / (float)tensorSize;
            _hpPoseDetectorLetterboxPadding4[0] = padLeft;
            _hpPoseDetectorLetterboxPadding4[1] = padTop;
            _hpPoseDetectorLetterboxPadding4[2] = padRight;
            _hpPoseDetectorLetterboxPadding4[3] = padBottom;
            letterboxPadding = _hpPoseDetectorLetterboxPadding4;

            inputBlob = Dnn.blobFromImage(
                letterbox224,
                1.0 / 127.5,
                (tensorSize, tensorSize),
                (127.5, 127.5, 127.5, 0),
                true,
                false,
                CvType.CV_32F);
        }

        List<Mat> InferenceSubgraph_PoseDetection_Holistic(Mat inputBlob)
        {
            if (_hpPoseDetectorOutLayerNames == null || _hpPoseDetectorOutLayerNames.Count == 0)
                _hpPoseDetectorOutLayerNames = _poseDetectorNet.getUnconnectedOutLayersNames();
            _hpPoseDetectorForwardOutputList.Clear();
            _poseDetectorNet.setInput(inputBlob);
            _poseDetectorNet.forward(_hpPoseDetectorForwardOutputList, _hpPoseDetectorOutLayerNames);
            return _hpPoseDetectorForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Sentis-backed <see cref="InferenceSubgraph_PoseDetection_Holistic"/> (via <see cref="OpenCVForUnity.UnityIntegration.Worker.DnnModule.MultiBackendNet.forwardTaskAsync"/>).
        /// Invoked only from <see cref="MediaPipeHolisticLandmarker.RunCoreProcessingTaskAsync"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_PoseDetection_HolisticAsync(Mat inputBlob, CancellationToken cancellationToken)
        {
            if (_hpPoseDetectorOutLayerNames == null || _hpPoseDetectorOutLayerNames.Count == 0)
                _hpPoseDetectorOutLayerNames = _poseDetectorNet.getUnconnectedOutLayersNames();
            _hpPoseDetectorForwardOutputList.Clear();
            _poseDetectorNet.setInput(inputBlob);
            await _poseDetectorNet.forwardTaskAsync(_hpPoseDetectorForwardOutputList, _hpPoseDetectorOutLayerNames, cancellationToken);
            return _hpPoseDetectorForwardOutputList;
        }
#endif

        Mat SsdAnchorsCalculator_Holistic(out Mat anchorsNx8)
        {
            Mat anchors = GetHolisticPoseDetectorSsdAnchors2254Shared();
            if (_hpPoseDetectorAnchorsNx8 == null)
            {
                _hpPoseDetectorAnchorsNx8 = new Mat();
                Core.repeat(anchors, 1, 4, _hpPoseDetectorAnchorsNx8);
            }
            anchorsNx8 = _hpPoseDetectorAnchorsNx8;
            return anchors;
        }

        void TensorsToDetectionsCalculator_Holistic(Mat output0, Mat output1, Mat anchors, Mat anchorsNx8, out Mat boxXywh, out Mat scoreForNms, out Mat boxLmForNms)
        {
            const int inputSize = 224;
            int num = output0.size(1);
            if (_hpPoseTensorsToDetectionsBoxXywh == null)
                _hpPoseTensorsToDetectionsBoxXywh = new Mat();
            _hpPoseTensorsToDetectionsBoxXywh.create(num, 4, CvType.CV_32FC1);

            using (Mat score = output1.reshape(1, num))
            using (var boxAndLandmark = output0.reshape(1, num))
            {
                NumpyClip_Holistic(score, -100.0, 100.0);
                Core.multiply(score, (-1.0, 0, 0, 0), score);
                Core.exp(score, score);
                Core.add(score, (1.0, 0, 0, 0), score);
                Core.divide(1.0, score, score);

                using (var boxAndLandmarkNx1c2 = boxAndLandmark.reshape(2, num))
                    Core.divide(boxAndLandmarkNx1c2, (inputSize, inputSize, 0, 0), boxAndLandmarkNx1c2);

                using (var cxy = boxAndLandmark.colRange(0, 2))
                    Core.add(cxy, anchors, cxy);

                using (var lm = boxAndLandmark.colRange(4, 12))
                    Core.add(lm, anchorsNx8, lm);

                using (var cxy2 = boxAndLandmark.colRange(0, 2))
                using (var wh2 = boxAndLandmark.colRange(2, 4))
                using (var dstXy = _hpPoseTensorsToDetectionsBoxXywh.colRange(0, 2))
                using (var dstWh = _hpPoseTensorsToDetectionsBoxXywh.colRange(2, 4))
                {
                    cxy2.copyTo(dstWh);
                    Core.divide(wh2, (2.0, 0, 0, 0), dstXy);
                    Core.subtract(dstWh, dstXy, cxy2);
                    Core.add(dstWh, dstXy, wh2);
                    cxy2.copyTo(dstXy);
                    Core.subtract(wh2, cxy2, dstWh);
                }

                if (_hpPoseTensorsToDetectionsNmsBoxXywh == null)
                    _hpPoseTensorsToDetectionsNmsBoxXywh = new Mat();
                if (_hpPoseTensorsToDetectionsNmsScore == null)
                    _hpPoseTensorsToDetectionsNmsScore = new Mat();
                if (_hpPoseTensorsToDetectionsNmsBoxLm == null)
                    _hpPoseTensorsToDetectionsNmsBoxLm = new Mat();

                int k = 0;
                for (int src = 0; src < num; src++)
                {
                    float sc = score.at<float>(src, 0)[0];
                    if (sc < _minPoseDetectionConfidence)
                        continue;
                    k++;
                }

                _hpPoseTensorsToDetectionsNmsBoxXywh.create(k, 4, CvType.CV_32FC1);
                _hpPoseTensorsToDetectionsNmsScore.create(k, 1, CvType.CV_32FC1);
                _hpPoseTensorsToDetectionsNmsBoxLm.create(k, 12, CvType.CV_32FC1);

                int dst = 0;
                for (int src = 0; src < num; src++)
                {
                    float sc = score.at<float>(src, 0)[0];
                    if (sc < _minPoseDetectionConfidence)
                        continue;
                    _hpPoseTensorsToDetectionsBoxXywh.row(src).copyTo(_hpPoseTensorsToDetectionsNmsBoxXywh.row(dst));
                    _hpPoseTensorsToDetectionsNmsScore.at<float>(dst, 0)[0] = sc;
                    boxAndLandmark.row(src).copyTo(_hpPoseTensorsToDetectionsNmsBoxLm.row(dst));
                    dst++;
                }

                boxXywh = _hpPoseTensorsToDetectionsNmsBoxXywh;
                scoreForNms = _hpPoseTensorsToDetectionsNmsScore;
                boxLmForNms = _hpPoseTensorsToDetectionsNmsBoxLm;
            }
        }

        /// <summary>
        /// Equivalent to <c>NonMaxSuppressionCalculator</c> using upstream
        /// <c>WeightedNonMaxSuppression</c>, following the same procedure as
        /// <see cref="MediaPipePoseLandmarker"/>.
        /// </summary>
        void NonMaxSuppressionCalculator_Holistic(Mat boxXywh, Mat score, Mat boxAndLandmarkNx12, List<HolisticPoseDetectionData> outLetterboxed)
        {
            outLetterboxed.Clear();
            int num = boxXywh.rows();
            if (num <= 0 || score == null || score.rows() < num)
                return;

            _hpPoseWnmsIndexed.Clear();
            for (int i = 0; i < num; i++)
                _hpPoseWnmsIndexed.Add((i, score.at<float>(i, 0)[0]));
            _hpPoseWnmsIndexed.Sort((a, b) => b.sc.CompareTo(a.sc));

            _hpPoseWnmsRemained.Clear();
            _hpPoseWnmsRemained.AddRange(_hpPoseWnmsIndexed);

            while (_hpPoseWnmsRemained.Count > 0)
            {
                int originalSize = _hpPoseWnmsRemained.Count;
                var anchor = _hpPoseWnmsRemained[0];

                float ax = boxXywh.at<float>(anchor.idx, 0)[0];
                float ay = boxXywh.at<float>(anchor.idx, 1)[0];
                float aw = boxXywh.at<float>(anchor.idx, 2)[0];
                float ah = boxXywh.at<float>(anchor.idx, 3)[0];

                _hpPoseWnmsNextRemained.Clear();
                for (int t = 0; t < _hpPoseWnmsRemained.Count; t++)
                {
                    var item = _hpPoseWnmsRemained[t];
                    float bx = boxXywh.at<float>(item.idx, 0)[0];
                    float by = boxXywh.at<float>(item.idx, 1)[0];
                    float bw = boxXywh.at<float>(item.idx, 2)[0];
                    float bh = boxXywh.at<float>(item.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_Holistic_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) > _minPoseSuppressionThreshold)
                        continue;
                    _hpPoseWnmsNextRemained.Add(item);
                }

                float wXmin = 0f, wYmin = 0f, wXmax = 0f, wYmax = 0f;
                float totalScore = 0f;
                Span<float> kpAcc = _hpPoseWnmsKpAcc8.AsSpan(0, 8);
                kpAcc.Clear();
                for (int t = 0; t < _hpPoseWnmsRemained.Count; t++)
                {
                    var c = _hpPoseWnmsRemained[t];
                    float bx = boxXywh.at<float>(c.idx, 0)[0];
                    float by = boxXywh.at<float>(c.idx, 1)[0];
                    float bw = boxXywh.at<float>(c.idx, 2)[0];
                    float bh = boxXywh.at<float>(c.idx, 3)[0];
                    if (NonMaxSuppressionCalculator_Holistic_ComputeIouXywh(ax, ay, aw, ah, bx, by, bw, bh) <= _minPoseSuppressionThreshold)
                        continue;

                    float s = c.sc;
                    totalScore += s;
                    wXmin += bx * s;
                    wYmin += by * s;
                    wXmax += (bx + bw) * s;
                    wYmax += (by + bh) * s;
                    boxAndLandmarkNx12.get(c.idx, 0, _hpPoseWnmsRowBuf12);
                    for (int k = 0; k < 8; k++)
                        kpAcc[k] += _hpPoseWnmsRowBuf12[4 + k] * s;
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

                outLetterboxed.Add(new HolisticPoseDetectionData
                {
                    RelXmin = outXmin,
                    RelYmin = outYmin,
                    RelWidth = outW,
                    RelHeight = outH,
                    Score = anchor.sc,
                    RelKeypointsXy = relKp,
                });

                if (originalSize == _hpPoseWnmsNextRemained.Count)
                    break;

                (_hpPoseWnmsRemained, _hpPoseWnmsNextRemained) = (_hpPoseWnmsNextRemained, _hpPoseWnmsRemained);
            }
        }

        static float NonMaxSuppressionCalculator_Holistic_ComputeIouXywh(
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

        List<HolisticPoseDetectionData> DetectionLetterboxRemovalCalculator_Holistic(List<HolisticPoseDetectionData> detections, float[] letterboxPadding)
        {
            float left = letterboxPadding[0];
            float top = letterboxPadding[1];
            float lr = letterboxPadding[0] + letterboxPadding[2];
            float tb = letterboxPadding[1] + letterboxPadding[3];
            float invW = 1.0f / (1.0f - lr);
            float invH = 1.0f / (1.0f - tb);
            var result = new List<HolisticPoseDetectionData>(detections.Count);
            for (int i = 0; i < detections.Count; i++)
            {
                var d = detections[i];
                var o = new HolisticPoseDetectionData
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
        /// Equivalent to <c>AlignmentPointsRectsCalculator</c> with
        /// <c>ConvertAlignmentPointsDetectionsToRect</c>.
        /// Used only for the Holistic pose-detection stream inside <c>CalculateRoiFromDetections</c> in
        /// <c>holistic_pose_tracking.cc</c>.
        /// </summary>
        List<HolisticNormalizedRect> AlignmentPointsRectsCalculator_ConvertAlignmentPointsDetectionsToRect_HolisticPose(
            List<HolisticPoseDetectionData> detections, int imageWidth, int imageHeight)
        {
            var rects = new List<HolisticNormalizedRect>(detections.Count);
            for (int i = 0; i < detections.Count; i++)
            {
                var d = detections[i];
                float kx0 = d.RelKeypointsXy[0] * imageWidth;
                float ky0 = d.RelKeypointsXy[1] * imageHeight;
                float kx1 = d.RelKeypointsXy[2] * imageWidth;
                float ky1 = d.RelKeypointsXy[3] * imageHeight;
                float boxSize = Mathf.Sqrt((kx1 - kx0) * (kx1 - kx0) + (ky1 - ky0) * (ky1 - ky0)) * 2.0f;
                float rot = NormalizePoseRadians_Holistic(Mathf.PI * 0.5f - Mathf.Atan2(-(ky1 - ky0), kx1 - kx0));
                rects.Add(new HolisticNormalizedRect
                {
                    XCenter = d.RelKeypointsXy[0],
                    YCenter = d.RelKeypointsXy[1],
                    Width = boxSize / imageWidth,
                    Height = boxSize / imageHeight,
                    Rotation = rot,
                });
            }
            return rects;
        }

        /// <summary>
        /// List-form equivalent of <c>RectTransformationCalculator</c> with
        /// <c>ScaleAndMakeSquare</c> at scale 1.25.
        /// </summary>
        List<HolisticNormalizedRect> RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromDetectionsList(
            List<HolisticNormalizedRect> poseRects, int imageWidth, int imageHeight)
        {
            var result = new List<HolisticNormalizedRect>(poseRects.Count);
            for (int i = 0; i < poseRects.Count; i++)
                result.Add(RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromDetections(poseRects[i], imageWidth, imageHeight));
            return result;
        }

        /// <summary>
        /// Single-rectangle equivalent of <c>RectTransformationCalculator</c> with
        /// <c>ScaleAndMakeSquare</c> at scale 1.25.
        /// Follows the same order as upstream <c>TransformNormalizedRect</c> in
        /// <c>rect_transformation_calculator.cc</c>.
        /// </summary>
        /// <remarks>
        /// Effective options are <c>shift_x/y=0</c>, <c>square_long=true</c>, and
        /// <c>scale_x/y=1.25</c>.
        /// </remarks>
        HolisticNormalizedRect RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromDetections(
            HolisticNormalizedRect rect, int imageWidth, int imageHeight)
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

            return new HolisticNormalizedRect
            {
                XCenter = cx,
                YCenter = cy,
                Width = width * scaleX,
                Height = height * scaleY,
                Rotation = rotation,
            };
        }

        /// <summary>
        /// Equivalent to <c>RectTransformationCalculator</c> with <c>ScaleAndMakeSquare</c> at 1.25 for
        /// <c>roi_from_auxiliary_landmarks</c> derived from auxiliary landmarks.
        /// Uses the same transform and options as the detection-ROI path (§2-A-7) inside upstream
        /// <c>CalculateRoiFromAuxiliaryLandmarks</c>.
        /// </summary>
        HolisticNormalizedRect RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromAuxiliaryLandmarks(
            HolisticNormalizedRect rect, int imageWidth, int imageHeight) =>
            RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromDetections(rect, imageWidth, imageHeight);

        List<HolisticPoseDetectionData> ClipDetectionVectorSizeCalculator_Holistic(List<HolisticPoseDetectionData> detections, int maxVecSize)
        {
            if (detections == null || detections.Count <= maxVecSize)
                return detections != null ? new List<HolisticPoseDetectionData>(detections) : new List<HolisticPoseDetectionData>();
            var clipped = new List<HolisticPoseDetectionData>(detections);
            clipped.RemoveRange(maxVecSize, clipped.Count - maxVecSize);
            return clipped;
        }

        static float NormalizePoseRadians_Holistic(float angleRadians)
        {
            return angleRadians - 2f * Mathf.PI * Mathf.Floor((angleRadians - (-Mathf.PI)) / (2f * Mathf.PI));
        }

        static void NumpyClip_Holistic(Mat a, double aMin, double aMax)
        {
            Core.min(a, (aMax, 0, 0, 0), a);
            Core.max(a, (aMin, 0, 0, 0), a);
        }

        static Mat BuildHolisticPoseDetectorSsdAnchors2254()
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
                while (lastSameStrideLayer < stridesLen && strides[lastSameStrideLayer] == strides[layerId])
                {
                    float sc = HolisticSsdCalculateScale(minScale, maxScale, lastSameStrideLayer, stridesLen);
                    if (lastSameStrideLayer == 0 && reduceBoxesInLowestLayer)
                    {
                        aspectRatios.Add(1.0f);
                        aspectRatios.Add(2.0f);
                        aspectRatios.Add(0.5f);
                        scales.Add(0.1f);
                        scales.Add(sc);
                        scales.Add(sc);
                    }
                    else
                    {
                        for (int arId = 0; arId < aspectRatiosOptions.Length; arId++)
                        {
                            aspectRatios.Add(aspectRatiosOptions[arId]);
                            scales.Add(sc);
                        }
                        if (interpolatedScaleAspectRatio > 0f)
                        {
                            float scaleNext = lastSameStrideLayer == stridesLen - 1
                                ? 1.0f
                                : HolisticSsdCalculateScale(minScale, maxScale, lastSameStrideLayer + 1, stridesLen);
                            scales.Add(Mathf.Sqrt(sc * scaleNext));
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
                throw new InvalidOperationException($"SSD anchor count mismatch: expected {expectedRows}, actual {outIx / 2}.");
            Mat anchors = new Mat(expectedRows, 2, CvType.CV_32FC1);
            anchors.put(0, 0, xy.AsSpan(0, expectedRows * 2));
            return anchors;
        }

        static float HolisticSsdCalculateScale(float minScale, float maxScale, int strideIndex, int numStrides)
        {
            if (numStrides == 1)
                return (minScale + maxScale) * 0.5f;
            return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1.0f);
        }

        static Mat GetHolisticPoseDetectorSsdAnchors2254Shared()
        {
            if (_holisticPoseDetectorSsdAnchors2254Cache != null)
                return _holisticPoseDetectorSsdAnchors2254Cache;
            lock (typeof(MediaPipeHolisticLandmarker))
            {
                if (_holisticPoseDetectorSsdAnchors2254Cache != null)
                    return _holisticPoseDetectorSsdAnchors2254Cache;
                _holisticPoseDetectorSsdAnchors2254Cache = BuildHolisticPoseDetectorSsdAnchors2254();
                return _holisticPoseDetectorSsdAnchors2254Cache;
            }
        }
    }
}
#endif
#endif
