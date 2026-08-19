#if !UNITY_WSA_10_0
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
using System;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityIntegration;
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe
{
    public partial class MediaPipeHolisticLandmarker
    {

        HolisticPoseDetectionData LandmarksToDetectionCalculator_ConvertLandmarksToDetection_HolisticPoseAuxiliary(
            Vec3f[] auxNorm, int imgW, int imgH)
        {
            _ = imgW;
            _ = imgH;
            float xmin = float.MaxValue, ymin = float.MaxValue, xmax = float.MinValue, ymax = float.MinValue;
            for (int i = 0; i < 2 && i < auxNorm.Length; i++)
            {
                float rx = auxNorm[i].Item1;
                float ry = auxNorm[i].Item2;
                xmin = Mathf.Min(xmin, rx);
                xmax = Mathf.Max(xmax, rx);
                ymin = Mathf.Min(ymin, ry);
                ymax = Mathf.Max(ymax, ry);
                _hpHolisticAuxLandmarksToDetKp8[i * 2] = rx;
                _hpHolisticAuxLandmarksToDetKp8[i * 2 + 1] = ry;
            }
            return new HolisticPoseDetectionData
            {
                RelXmin = xmin,
                RelYmin = ymin,
                RelWidth = xmax - xmin,
                RelHeight = ymax - ymin,
                RelKeypointsXy = _hpHolisticAuxLandmarksToDetKp8,
                Score = 1f,
            };
        }

        /// <summary>
        /// Equivalent to upstream <c>CalculateScaleRoiFromAuxiliaryLandmarks</c> in
        /// <c>holistic_pose_tracking.cc</c>: <c>LandmarksToDetection</c> ->
        /// <c>AlignmentPointsRectsCalculator</c> only, without the 1.25 scale step.
        /// </summary>
        HolisticNormalizedRect CalculateScaleRoiFromAuxiliaryLandmarks_HolisticPose(Vec3f[] auxPixels, int iw, int ih)
        {
            var det = LandmarksToDetectionCalculator_ConvertLandmarksToDetection_HolisticPoseAuxiliary(auxPixels, iw, ih);
            var rects = AlignmentPointsRectsCalculator_ConvertAlignmentPointsDetectionsToRect_HolisticPose(
                new System.Collections.Generic.List<HolisticPoseDetectionData> { det }, iw, ih);
            return rects.Count > 0 ? rects[0] : HolisticFullImageNormalizedRect();
        }

        /// <summary>
        /// Equivalent to upstream <c>CalculateRoiFromAuxiliaryLandmarks</c>:
        /// <c>LandmarksToDetection</c> -> <c>AlignmentPointsRectsCalculator</c> ->
        /// <c>RectTransformationCalculator</c> with scale 1.25.
        /// </summary>
        HolisticNormalizedRect CalculateRoiFromAuxiliaryLandmarks_HolisticPose(Vec3f[] auxSmoothedPixels, int iw, int ih)
        {
            var det = LandmarksToDetectionCalculator_ConvertLandmarksToDetection_HolisticPoseAuxiliary(auxSmoothedPixels, iw, ih);
            var rects = AlignmentPointsRectsCalculator_ConvertAlignmentPointsDetectionsToRect_HolisticPose(
                new System.Collections.Generic.List<HolisticPoseDetectionData> { det }, iw, ih);
            if (rects.Count == 0)
                return HolisticFullImageNormalizedRect();
            return RectTransformationCalculator_ScaleAndMakeSquare_HolisticPoseRoiFromAuxiliaryLandmarks(rects[0], iw, ih);
        }

        /// <summary>
        /// Equivalent to <c>LandmarksSmoothingCalculator</c> for the two auxiliary points using
        /// One Euro parameters <c>min_cutoff=0.01</c>, <c>beta=10</c>, and
        /// <c>derivate_cutoff=1</c> (§2-C-4).
        /// </summary>
        Vec3f[] LandmarksSmoothingCalculator_HolisticPoseAuxiliaryLandmarks(
            Vec3f[] auxPixels, int imageWidth, int imageHeight, HolisticNormalizedRect scaleRoi)
        {
            return _holisticAuxiliarySmoothing.Apply(auxPixels, imageWidth, imageHeight, scaleRoi,
                _runningMode == MediaPipeHolisticRunningMode.VIDEO);
        }

        /// <summary>Equivalent to <c>VisibilitySmoothingCalculator</c> for Holistic pose normalized 2D main output with <c>alpha=0.1</c>.</summary>
        float[] VisibilitySmoothingCalculator_HolisticPoseNormalizedLandmarks2D(float[] rawVisibility)
        {
            return _holisticPoseOutputSmoothing.ApplyVisibility(rawVisibility,
                _runningMode == MediaPipeHolisticRunningMode.VIDEO);
        }

        /// <summary>Equivalent to <c>LandmarksSmoothingCalculator</c> for Holistic pose normalized 2D main landmarks with parameters <c>0.05 / 80 / 1</c>.</summary>
        Vec3f[] LandmarksSmoothingCalculator_HolisticPoseNormalizedLandmarks2D(
            Vec3f[] normLmPixels, int imageWidth, int imageHeight, HolisticNormalizedRect scaleRoi)
        {
            return _holisticPoseOutputSmoothing.ApplyNormalized(normLmPixels, imageWidth, imageHeight, scaleRoi,
                _runningMode == MediaPipeHolisticRunningMode.VIDEO);
        }

        /// <summary>Equivalent to <c>SplitLandmarkListCalculator</c> with <c>SplitToRanges</c> {0,33}.</summary>
        static Vec3f[] SplitLandmarkListCalculator_SplitToRanges_0_33(Vec3f[] world)
        {
            var o = new Vec3f[33];
            for (int i = 0; i < 33 && world != null && i < world.Length; i++)
                o[i] = world[i];
            return o;
        }

        /// <summary>Equivalent to <c>VisibilitySmoothingCalculator</c> for Holistic pose world visibility with <c>alpha=0.1</c>.</summary>
        float[] VisibilitySmoothingCalculator_HolisticPoseWorldLandmarks(float[] vis)
        {
            return _holisticWorldSmoothing.ApplyVisibility(vis, _runningMode == MediaPipeHolisticRunningMode.VIDEO);
        }

        /// <summary>Equivalent to <c>LandmarksSmoothingCalculator</c> for the 33 Holistic pose world landmarks, without <c>scale_roi</c>, using parameters <c>0.1 / 40 / 1</c>.</summary>
        Vec3f[] LandmarksSmoothingCalculator_HolisticPoseWorldLandmarks(Vec3f[] world33)
        {
            return _holisticWorldSmoothing.ApplyWorld(world33, _runningMode == MediaPipeHolisticRunningMode.VIDEO);
        }

        /// <summary>
        /// Equivalent to <c>SegmentationSmoothingCalculator</c> for the Holistic pose path, following the
        /// CPU path in <c>segmentation_smoothing_calculator.cc</c>.
        /// For non-empty input, writes into <c>_hpSegmentationSmoothedReuse</c> and does not reallocate
        /// the <see cref="Mat"/> until the resolution changes.
        /// The mask is assumed to use <see cref="CvType.CV_32FC1"/>. Without a previous frame it uses
        /// <see cref="Mat.copyTo"/>; otherwise it scans the <see cref="float"/> data through
        /// <see cref="Mat.AsSpan{T}"/> (<c>Mat_Ex</c> in <c>OpenCVForUnity</c>), using the no-argument
        /// form for continuous buffers and the row-indexed form for non-contiguous buffers.
        /// </summary>
        Mat SegmentationSmoothingCalculator_HolisticPose(Mat maskCurrent, Mat maskPrevious, float combineWithPreviousRatio)
        {
            if (maskCurrent == null || maskCurrent.empty())
                return new Mat();
            int rows = maskCurrent.rows();
            int cols = maskCurrent.cols();
            EnsureHolisticSegmentationSmoothedReuse(rows, cols);
            Mat smoothed = _hpSegmentationSmoothedReuse;

            bool hasPrev = maskPrevious != null && !maskPrevious.empty()
                && maskPrevious.rows() == rows && maskPrevious.cols() == cols;

            if (!hasPrev)
            {
                maskCurrent.copyTo(smoothed);
                return smoothed;
            }

            if (maskCurrent.isContinuous() && maskPrevious.isContinuous() && smoothed.isContinuous())
            {
                int n = rows * cols;
                Span<float> spanCur = maskCurrent.AsSpan<float>();
                Span<float> spanPrev = maskPrevious.AsSpan<float>();
                Span<float> spanOut = smoothed.AsSpan<float>();
                for (int k = 0; k < n; k++)
                    spanOut[k] = HolisticSegmentationBlend(spanPrev[k], spanCur[k], combineWithPreviousRatio);
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    Span<float> spanCur = maskCurrent.AsSpan<float>(i);
                    Span<float> spanPrev = maskPrevious.AsSpan<float>(i);
                    Span<float> spanDst = smoothed.AsSpan<float>(i);
                    for (int j = 0; j < cols; j++)
                        spanDst[j] = HolisticSegmentationBlend(spanPrev[j], spanCur[j], combineWithPreviousRatio);
                }
            }

            return smoothed;
        }

        static float HolisticSegmentationBlend(float prevMaskValue, float newMaskValue, float combineWithPreviousRatio)
        {
            const float c1 = 5.68842f;
            const float c2 = -0.748699f;
            const float c3 = -57.8051f;
            const float c4 = 291.309f;
            const float c5 = -624.717f;
            float t = newMaskValue - 0.5f;
            float x = t * t;
            float uncertainty = 1.0f - Mathf.Min(1.0f, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))));
            return newMaskValue + (prevMaskValue - newMaskValue) * (uncertainty * combineWithPreviousRatio);
        }

        sealed class HolisticAuxiliaryLandmarkSmoothingPipeline
        {
            const double kFreq = 30.0;
            readonly HolisticOneEuroFilter[] _fx = new HolisticOneEuroFilter[2];
            readonly HolisticOneEuroFilter[] _fy = new HolisticOneEuroFilter[2];
            readonly HolisticOneEuroFilter[] _fz = new HolisticOneEuroFilter[2];

            public HolisticAuxiliaryLandmarkSmoothingPipeline()
            {
                for (int i = 0; i < 2; i++)
                {
                    _fx[i] = HolisticOneEuroFilter.Create(kFreq, 0.01, 10.0, 1.0);
                    _fy[i] = HolisticOneEuroFilter.Create(kFreq, 0.01, 10.0, 1.0);
                    _fz[i] = HolisticOneEuroFilter.Create(kFreq, 0.01, 10.0, 1.0);
                }
            }

            public void Reset()
            {
                for (int i = 0; i < 2; i++)
                {
                    _fx[i].Reset();
                    _fy[i].Reset();
                    _fz[i].Reset();
                }
            }

            public Vec3f[] Apply(Vec3f[] auxNorm, int iw, int ih, HolisticNormalizedRect scaleRoi, bool useStream)
            {
                if (auxNorm == null || auxNorm.Length < 2)
                    return auxNorm ?? new Vec3f[2];
                long ts = (long)Environment.TickCount * 1_000_000L;
                float objectScale = HolisticGetObjectScaleRoi(scaleRoi, iw, ih);
                if (objectScale < 1e-6f)
                    return auxNorm;
                double valueScale = 1.0 / objectScale;
                var o = new Vec3f[2];
                for (int i = 0; i < 2; i++)
                {
                    double xPx = auxNorm[i].Item1 * iw;
                    double yPx = auxNorm[i].Item2 * ih;
                    double zPx = auxNorm[i].Item3 * iw;
                    xPx = _fx[i].Apply(ts, xPx, valueScale, 1.0);
                    yPx = _fy[i].Apply(ts, yPx, valueScale, 1.0);
                    zPx = _fz[i].Apply(ts, zPx, valueScale, 1.0);
                    o[i] = new Vec3f((float)(xPx / iw), (float)(yPx / ih), (float)(zPx / iw));
                }
                return o;
            }
        }

        sealed class HolisticPoseLandmarkOutputSmoothingPipeline
        {
            const double kFreq = 30.0;
            readonly HolisticLowPassFilter[] _vis;
            readonly HolisticOneEuroFilter[] _fx;
            readonly HolisticOneEuroFilter[] _fy;
            readonly HolisticOneEuroFilter[] _fz;

            public HolisticPoseLandmarkOutputSmoothingPipeline()
            {
                int n = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
                _vis = new HolisticLowPassFilter[n];
                _fx = new HolisticOneEuroFilter[n];
                _fy = new HolisticOneEuroFilter[n];
                _fz = new HolisticOneEuroFilter[n];
                for (int i = 0; i < n; i++)
                {
                    _vis[i] = HolisticLowPassFilter.Create(0.1f);
                    _fx[i] = HolisticOneEuroFilter.Create(kFreq, 0.05, 80.0, 1.0);
                    _fy[i] = HolisticOneEuroFilter.Create(kFreq, 0.05, 80.0, 1.0);
                    _fz[i] = HolisticOneEuroFilter.Create(kFreq, 0.05, 80.0, 1.0);
                }
            }

            public void Reset()
            {
                foreach (var f in _vis) f.Reset();
                foreach (var f in _fx) f.Reset();
                foreach (var f in _fy) f.Reset();
                foreach (var f in _fz) f.Reset();
            }

            public float[] ApplyVisibility(float[] raw, bool useStream)
            {
                if (raw == null || raw.Length != _vis.Length)
                    return raw;
                var o = new float[raw.Length];
                for (int i = 0; i < raw.Length; i++)
                    o[i] = _vis[i].Apply(raw[i]);
                return o;
            }

            public Vec3f[] ApplyNormalized(Vec3f[] lm, int iw, int ih, HolisticNormalizedRect scaleRoi, bool useStream)
            {
                if (lm == null || lm.Length != _fx.Length)
                    return lm;
                long ts = (long)Environment.TickCount * 1_000_000L;
                float objectScale = HolisticGetObjectScaleRoi(scaleRoi, iw, ih);
                if (objectScale < 1e-6f)
                    return lm;
                double valueScale = 1.0 / objectScale;
                var o = new Vec3f[lm.Length];
                for (int i = 0; i < lm.Length; i++)
                {
                    double xPx = lm[i].Item1 * iw;
                    double yPx = lm[i].Item2 * ih;
                    double zPx = lm[i].Item3 * iw;
                    xPx = _fx[i].Apply(ts, xPx, valueScale, 1.0);
                    yPx = _fy[i].Apply(ts, yPx, valueScale, 1.0);
                    zPx = _fz[i].Apply(ts, zPx, valueScale, 1.0);
                    o[i] = new Vec3f((float)(xPx / iw), (float)(yPx / ih), (float)(zPx / iw));
                }
                return o;
            }
        }

        sealed class HolisticWorldLandmarkSmoothingPipeline
        {
            const double kFreq = 30.0;
            readonly HolisticLowPassFilter[] _vis;
            readonly HolisticOneEuroFilter[] _fx;
            readonly HolisticOneEuroFilter[] _fy;
            readonly HolisticOneEuroFilter[] _fz;

            public HolisticWorldLandmarkSmoothingPipeline()
            {
                const int n = 33;
                _vis = new HolisticLowPassFilter[n];
                _fx = new HolisticOneEuroFilter[n];
                _fy = new HolisticOneEuroFilter[n];
                _fz = new HolisticOneEuroFilter[n];
                for (int i = 0; i < n; i++)
                {
                    _vis[i] = HolisticLowPassFilter.Create(0.1f);
                    _fx[i] = HolisticOneEuroFilter.Create(kFreq, 0.1, 40.0, 1.0);
                    _fy[i] = HolisticOneEuroFilter.Create(kFreq, 0.1, 40.0, 1.0);
                    _fz[i] = HolisticOneEuroFilter.Create(kFreq, 0.1, 40.0, 1.0);
                }
            }

            public void Reset()
            {
                foreach (var f in _vis) f.Reset();
                foreach (var f in _fx) f.Reset();
                foreach (var f in _fy) f.Reset();
                foreach (var f in _fz) f.Reset();
            }

            public float[] ApplyVisibility(float[] vis, bool useStream)
            {
                if (vis == null)
                    return vis;
                int n = Mathf.Min(vis.Length, _vis.Length);
                var o = (float[])vis.Clone();
                for (int i = 0; i < n; i++)
                    o[i] = _vis[i].Apply(vis[i]);
                return o;
            }

            public Vec3f[] ApplyWorld(Vec3f[] world33, bool useStream)
            {
                if (world33 == null || world33.Length < 33)
                    return world33;
                long ts = (long)Environment.TickCount * 1_000_000L;
                var o = new Vec3f[MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT];
                for (int i = 0; i < 33; i++)
                {
                    double x = _fx[i].Apply(ts, world33[i].Item1, 1.0, 1.0);
                    double y = _fy[i].Apply(ts, world33[i].Item2, 1.0, 1.0);
                    double z = _fz[i].Apply(ts, world33[i].Item3, 1.0, 1.0);
                    o[i] = new Vec3f((float)x, (float)y, (float)z);
                }
                return o;
            }
        }

        static float HolisticGetObjectScaleRoi(HolisticNormalizedRect roi, int imageWidth, int imageHeight)
        {
            float w = roi.Width * imageWidth;
            float h = roi.Height * imageHeight;
            return (w + h) * 0.5f;
        }

        sealed class HolisticLowPassFilter
        {
            float _alpha;
            bool _initialized;
            float _storedValue;
            float _rawValue;

            public static HolisticLowPassFilter Create(float alpha)
            {
                return new HolisticLowPassFilter { _alpha = Mathf.Clamp01(alpha), _initialized = false };
            }

            public void Reset() => _initialized = false;

            public float Apply(float value)
            {
                float result = _initialized ? _alpha * value + (1f - _alpha) * _storedValue : value;
                if (!_initialized) _initialized = true;
                _rawValue = value;
                _storedValue = result;
                return result;
            }

            public float ApplyWithAlpha(float value, float alpha)
            {
                _alpha = Mathf.Clamp01(alpha);
                return Apply(value);
            }

            public bool HasLastRaw => _initialized;
            public float LastRawValue() => _rawValue;
        }

        sealed class HolisticOneEuroFilter
        {
            const long kUninit = -1;
            const double kEps = 1e-6;
            double _frequency;
            readonly double _minCutoff;
            readonly double _beta;
            readonly double _derivateCutoff;
            long _lastTimeNs;
            HolisticLowPassFilter _x;
            HolisticLowPassFilter _dx;

            HolisticOneEuroFilter(double frequency, double minCutoff, double beta, double derivateCutoff)
            {
                _frequency = frequency;
                _minCutoff = minCutoff;
                _beta = beta;
                _derivateCutoff = derivateCutoff;
                _lastTimeNs = kUninit;
                _x = HolisticLowPassFilter.Create((float)HolisticGetAlpha(minCutoff, frequency));
                _dx = HolisticLowPassFilter.Create((float)HolisticGetAlpha(derivateCutoff, frequency));
            }

            public static HolisticOneEuroFilter Create(double frequency, double minCutoff, double beta, double derivateCutoff)
            {
                if (frequency <= kEps || minCutoff <= kEps || derivateCutoff <= kEps)
                    throw new ArgumentException("The OneEuroFilter parameters are invalid.");
                return new HolisticOneEuroFilter(frequency, minCutoff, beta, derivateCutoff);
            }

            public void Reset()
            {
                _lastTimeNs = kUninit;
                _x = HolisticLowPassFilter.Create((float)HolisticGetAlpha(_minCutoff, _frequency));
                _dx = HolisticLowPassFilter.Create((float)HolisticGetAlpha(_derivateCutoff, _frequency));
            }

            public double Apply(long timestampNs, double value, double valueScale, double betaScale)
            {
                if (_lastTimeNs >= timestampNs)
                    return value;
                if (_lastTimeNs != 0 && timestampNs != 0)
                    _frequency = 1.0 / ((timestampNs - _lastTimeNs) * 1e-9);
                _lastTimeNs = timestampNs;
                double dvalue = _x.HasLastRaw
                    ? (value - _x.LastRawValue()) * valueScale * _frequency
                    : 0.0;
                double edvalue = _dx.ApplyWithAlpha((float)dvalue, (float)HolisticGetAlpha(_derivateCutoff, _frequency));
                double cutoff = _minCutoff + betaScale * _beta * Math.Abs(edvalue);
                return _x.ApplyWithAlpha((float)value, (float)HolisticGetAlpha(cutoff, _frequency));
            }

            static double HolisticGetAlpha(double cutoff, double frequency)
            {
                double te = 1.0 / frequency;
                double tau = 1.0 / (2.0 * Math.PI * cutoff);
                return 1.0 / (1.0 + tau / te);
            }
        }
    }
}
#endif
#endif
