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

        HolisticSinglePoseLandmarksInnerResult? SinglePoseLandmarksDetectorGraph(Mat image, HolisticNormalizedRect poseRect, Mat segmentationFullPlane)
        {
            void ZeroSeg()
            {
                segmentationFullPlane?.setTo((0d, 0d, 0d, 0d));
            }

            if (!ImagePreprocessingGraph_SinglePoseLandmarks(image, poseRect, out HolisticSinglePoseLandmarkPreprocessOut pre))
            {
                ZeroSeg();
                return null;
            }

            List<Mat> inferenceTensors = InferenceSubgraph_PoseLandmarks(pre.PoseBlob);
            if (inferenceTensors == null || inferenceTensors.Count < kHpPoseLandmarkTensorSplitCount)
            {
                ZeroSeg();
                return null;
            }

            if (!SplitTensorVectorCalculator_PoseLandmarks(inferenceTensors,
                    out Mat landmarkTensors, out Mat poseFlagTensors, out Mat segmentationTensors,
                    out Mat heatmapTensors, out Mat worldLandmarkTensors))
            {
                ZeroSeg();
                return null;
            }

            float posePresenceScore = TensorsToFloatsCalculator_PosePresence(poseFlagTensors);
            bool posePresence = ThresholdingCalculator_PosePresence(posePresenceScore);
            if (!GateCalculator_PoseLandmarkTensors(posePresence))
            {
                ZeroSeg();
                int L = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
                return new HolisticSinglePoseLandmarksInnerResult
                {
                    PosePresence = false,
                    PosePresenceScore = posePresenceScore,
                    NormLandmarks = new Vec3f[L],
                    WorldLandmarks = new Vec3f[L],
                    AuxiliaryLandmarksNorm = new Vec3f[2],
                    LandmarkVisibility = new float[L],
                    LandmarkVisibilityWorld = new float[L],
                    LandmarkPresence = new float[L],
                    SegmentationMaskFull = null,
                };
            }

            HolisticPoseLandmarkDecoded[] decoded = TensorsToLandmarksCalculator_PoseImage(landmarkTensors, pre.ModelW, pre.ModelH);
            HolisticPoseLandmarkDecoded[] refined = RefineLandmarksFromHeatmapCalculator(decoded, heatmapTensors);
            SplitNormalizedLandmarkListCalculator(refined, out HolisticPoseLandmarkDecoded[] mainLm, out HolisticPoseLandmarkDecoded[] auxLm);

            HolisticPoseLandmarkDecoded[] world33;
            if (_poseTrackingRequest.NeedWorldLandmarks)
            {
                HolisticPoseLandmarkDecoded[] worldRaw = TensorsToLandmarksCalculator_PoseWorld(worldLandmarkTensors);
                SplitLandmarkListCalculator_PoseWorld(worldRaw, out world33);
                VisibilityCopyCalculator_PoseWorld(mainLm, world33);
            }
            else
                world33 = new HolisticPoseLandmarkDecoded[33];

            HolisticPoseLandmarkDecoded[] mainAfterLb = LandmarkLetterboxRemovalCalculator_Pose(mainLm, pre);
            HolisticPoseLandmarkDecoded[] auxAfterLb = LandmarkLetterboxRemovalCalculator_Pose(auxLm, pre);

            Vec3f[] projected = LandmarkProjectionCalculator_Pose(mainAfterLb, poseRect);
            Vec3f[] auxProjected = LandmarkProjectionCalculator_PoseAux(auxAfterLb, poseRect);
            Vec3f[] worldProj = _poseTrackingRequest.NeedWorldLandmarks
                ? WorldLandmarkProjectionCalculator_Pose(world33, poseRect)
                : new Vec3f[MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT];

            int Lm = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            var visibility = new float[Lm];
            var visibilityWorld = new float[Lm];
            var presenceLm = new float[Lm];
            for (int i = 0; i < Lm; i++)
            {
                visibility[i] = mainAfterLb[i].Visibility;
                visibilityWorld[i] = world33[i].Visibility;
                presenceLm[i] = mainAfterLb[i].Presence;
            }

            Mat segFull = null;
            if (_poseTrackingRequest.NeedSegmentationMask && segmentationFullPlane != null)
                SegmentationMaskFromTensorToFullImage(segmentationTensors, pre, segmentationFullPlane);

            if (_poseTrackingRequest.NeedSegmentationMask && segmentationFullPlane != null)
            {
                segFull = segmentationFullPlane;
                segmentationFullPlane = null;
            }

            return new HolisticSinglePoseLandmarksInnerResult
            {
                PosePresence = true,
                PosePresenceScore = posePresenceScore,
                NormLandmarks = projected,
                WorldLandmarks = worldProj,
                LandmarkVisibility = visibility,
                LandmarkVisibilityWorld = visibilityWorld,
                LandmarkPresence = presenceLm,
                AuxiliaryLandmarksNorm = auxProjected,
                SegmentationMaskFull = segFull,
            };
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Async Sentis variant of <see cref="SinglePoseLandmarksDetectorGraph"/>.
        /// </summary>
        async Task<HolisticSinglePoseLandmarksInnerResult?> SinglePoseLandmarksDetectorGraphAsync(Mat image, HolisticNormalizedRect poseRect, Mat segmentationFullPlane, CancellationToken cancellationToken)
        {
            void ZeroSeg()
            {
                segmentationFullPlane?.setTo((0d, 0d, 0d, 0d));
            }

            if (!ImagePreprocessingGraph_SinglePoseLandmarks(image, poseRect, out HolisticSinglePoseLandmarkPreprocessOut pre))
            {
                ZeroSeg();
                return null;
            }

            var inferenceTensors = await InferenceSubgraph_PoseLandmarksAsync(pre.PoseBlob, cancellationToken);
            if (inferenceTensors == null || inferenceTensors.Count < kHpPoseLandmarkTensorSplitCount)
            {
                ZeroSeg();
                return null;
            }

            if (!SplitTensorVectorCalculator_PoseLandmarks(inferenceTensors,
                    out Mat landmarkTensors, out Mat poseFlagTensors, out Mat segmentationTensors,
                    out Mat heatmapTensors, out Mat worldLandmarkTensors))
            {
                ZeroSeg();
                return null;
            }

            float posePresenceScore = TensorsToFloatsCalculator_PosePresence(poseFlagTensors);
            bool posePresence = ThresholdingCalculator_PosePresence(posePresenceScore);
            if (!GateCalculator_PoseLandmarkTensors(posePresence))
            {
                ZeroSeg();
                int L = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
                return new HolisticSinglePoseLandmarksInnerResult
                {
                    PosePresence = false,
                    PosePresenceScore = posePresenceScore,
                    NormLandmarks = new Vec3f[L],
                    WorldLandmarks = new Vec3f[L],
                    AuxiliaryLandmarksNorm = new Vec3f[2],
                    LandmarkVisibility = new float[L],
                    LandmarkVisibilityWorld = new float[L],
                    LandmarkPresence = new float[L],
                    SegmentationMaskFull = null,
                };
            }

            HolisticPoseLandmarkDecoded[] decoded = TensorsToLandmarksCalculator_PoseImage(landmarkTensors, pre.ModelW, pre.ModelH);
            HolisticPoseLandmarkDecoded[] refined = RefineLandmarksFromHeatmapCalculator(decoded, heatmapTensors);
            SplitNormalizedLandmarkListCalculator(refined, out HolisticPoseLandmarkDecoded[] mainLm, out HolisticPoseLandmarkDecoded[] auxLm);

            HolisticPoseLandmarkDecoded[] world33;
            if (_poseTrackingRequest.NeedWorldLandmarks)
            {
                HolisticPoseLandmarkDecoded[] worldRaw = TensorsToLandmarksCalculator_PoseWorld(worldLandmarkTensors);
                SplitLandmarkListCalculator_PoseWorld(worldRaw, out world33);
                VisibilityCopyCalculator_PoseWorld(mainLm, world33);
            }
            else
                world33 = new HolisticPoseLandmarkDecoded[33];

            HolisticPoseLandmarkDecoded[] mainAfterLb = LandmarkLetterboxRemovalCalculator_Pose(mainLm, pre);
            HolisticPoseLandmarkDecoded[] auxAfterLb = LandmarkLetterboxRemovalCalculator_Pose(auxLm, pre);

            Vec3f[] projected = LandmarkProjectionCalculator_Pose(mainAfterLb, poseRect);
            Vec3f[] auxProjected = LandmarkProjectionCalculator_PoseAux(auxAfterLb, poseRect);
            Vec3f[] worldProj = _poseTrackingRequest.NeedWorldLandmarks
                ? WorldLandmarkProjectionCalculator_Pose(world33, poseRect)
                : new Vec3f[MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT];

            int Lm = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            var visibility = new float[Lm];
            var visibilityWorld = new float[Lm];
            var presenceLm = new float[Lm];
            for (int i = 0; i < Lm; i++)
            {
                visibility[i] = mainAfterLb[i].Visibility;
                visibilityWorld[i] = world33[i].Visibility;
                presenceLm[i] = mainAfterLb[i].Presence;
            }

            Mat segFull = null;
            if (_poseTrackingRequest.NeedSegmentationMask && segmentationFullPlane != null)
                SegmentationMaskFromTensorToFullImage(segmentationTensors, pre, segmentationFullPlane);

            if (_poseTrackingRequest.NeedSegmentationMask && segmentationFullPlane != null)
            {
                segFull = segmentationFullPlane;
                segmentationFullPlane = null;
            }

            return new HolisticSinglePoseLandmarksInnerResult
            {
                PosePresence = true,
                PosePresenceScore = posePresenceScore,
                NormLandmarks = projected,
                WorldLandmarks = worldProj,
                LandmarkVisibility = visibility,
                LandmarkVisibilityWorld = visibilityWorld,
                LandmarkPresence = presenceLm,
                AuxiliaryLandmarksNorm = auxProjected,
                SegmentationMaskFull = segFull,
            };
        }
#endif

        /// <summary>Equivalent to <c>ImagePreprocessingGraph</c> for a single 256x256 pose ROI.</summary>
        bool ImagePreprocessingGraph_SinglePoseLandmarks(Mat image, HolisticNormalizedRect poseRect, out HolisticSinglePoseLandmarkPreprocessOut pre)
        {
            pre = default;
            int imgW = image.cols();
            int imgH = image.rows();
            if (imgW <= 0 || imgH <= 0)
                return false;

            const int inputSize = kHpPoseLandmarkInputSize;
            if (_hpSinglePoseLandmarkBlob == null)
            {
                _hpSinglePoseLandmarkSrcPts = new Mat(4, 2, CvType.CV_32FC1);
                _hpSinglePoseLandmarkDstPts = new Mat(4, 2, CvType.CV_32FC1);
                Span<float> dstPtsArr = stackalloc float[8];
                float dw = inputSize, dh = inputSize;
                dstPtsArr[0] = 0f; dstPtsArr[1] = dh;
                dstPtsArr[2] = 0f; dstPtsArr[3] = 0f;
                dstPtsArr[4] = dw; dstPtsArr[5] = 0f;
                dstPtsArr[6] = dw; dstPtsArr[7] = dh;
                _hpSinglePoseLandmarkDstPts.put(0, 0, dstPtsArr);
                _hpSinglePoseLandmarkWarpedBgr = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _hpSinglePoseLandmarkWarpedRgb = new Mat(inputSize, inputSize, CvType.CV_8UC3);
                _hpSinglePoseLandmarkBlob = new Mat(new int[] { 1, inputSize, inputSize, 3 }, CvType.CV_32FC1);
                _hpSinglePoseLandmarkBlobHxW = _hpSinglePoseLandmarkBlob.reshape(3, new int[] { inputSize, inputSize });
            }

            float cx = poseRect.XCenter * imgW;
            float cy = poseRect.YCenter * imgH;
            float rw = poseRect.Width * imgW;
            float rh = poseRect.Height * imgH;
            if (rw <= 0f || rh <= 0f || float.IsNaN(rw) || float.IsNaN(rh))
                return false;

            PadRoiLikeImageToTensorCalculator(inputSize, inputSize, true, ref rw, ref rh,
                out float padL, out float padT, out float padR, out float padB);

            double angleDeg = poseRect.Rotation * 180.0 / Math.PI;

            Imgproc.boxPoints((cx, cy, rw, rh, angleDeg), _hpSinglePoseLandmarkSrcPts);
            using (Mat projMat = Imgproc.getPerspectiveTransform(_hpSinglePoseLandmarkSrcPts, _hpSinglePoseLandmarkDstPts))
            {
                if (_hpSinglePoseLandmarkProjMat3x3 == null)
                    _hpSinglePoseLandmarkProjMat3x3 = new Mat(3, 3, CvType.CV_32FC1);
                projMat.copyTo(_hpSinglePoseLandmarkProjMat3x3);
                Imgproc.warpPerspective(image, _hpSinglePoseLandmarkWarpedBgr, projMat, (inputSize, inputSize),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
            }
            Imgproc.cvtColor(_hpSinglePoseLandmarkWarpedBgr, _hpSinglePoseLandmarkWarpedRgb, Imgproc.COLOR_BGR2RGB);
            _hpSinglePoseLandmarkWarpedRgb.convertTo(_hpSinglePoseLandmarkBlobHxW, CvType.CV_32F, 1.0 / 255.0);

            pre = new HolisticSinglePoseLandmarkPreprocessOut
            {
                PoseBlob = _hpSinglePoseLandmarkBlob,
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

        List<Mat> InferenceSubgraph_PoseLandmarks(Mat poseBlob)
        {
            if (_hpPoseLandmarksNetOutLayerNames == null || _hpPoseLandmarksNetOutLayerNames.Count == 0)
                _hpPoseLandmarksNetOutLayerNames = _poseLandmarksNet.getUnconnectedOutLayersNames();
            _hpPoseLandmarksForwardOutputList.Clear();
            _poseLandmarksNet.setInput(poseBlob);
            _poseLandmarksNet.forward(_hpPoseLandmarksForwardOutputList, _hpPoseLandmarksNetOutLayerNames);
            return _hpPoseLandmarksForwardOutputList;
        }

#if OPENCV_SENTIS_AVAILABLE
        /// <summary>
        /// Sentis-backed <see cref="InferenceSubgraph_PoseLandmarks"/> (via <see cref="OpenCVForUnity.UnityIntegration.Worker.DnnModule.MultiBackendNet.forwardTaskAsync"/>).
        /// Invoked only from <see cref="MediaPipeHolisticLandmarker.RunCoreProcessingTaskAsync"/>.
        /// </summary>
        async Task<List<Mat>> InferenceSubgraph_PoseLandmarksAsync(Mat poseBlob, CancellationToken cancellationToken)
        {
            if (_hpPoseLandmarksNetOutLayerNames == null || _hpPoseLandmarksNetOutLayerNames.Count == 0)
                _hpPoseLandmarksNetOutLayerNames = _poseLandmarksNet.getUnconnectedOutLayersNames();
            _hpPoseLandmarksForwardOutputList.Clear();
            _poseLandmarksNet.setInput(poseBlob);
            await _poseLandmarksNet.forwardTaskAsync(_hpPoseLandmarksForwardOutputList, _hpPoseLandmarksNetOutLayerNames, cancellationToken);
            return _hpPoseLandmarksForwardOutputList;
        }
#endif

        /// <summary>Equivalent to <c>SplitTensorVectorCalculator</c>, splitting into five tensors.</summary>
        static bool SplitTensorVectorCalculator_PoseLandmarks(List<Mat> tensors,
            out Mat landmarkTensors, out Mat poseFlagTensors, out Mat segmentationTensors,
            out Mat heatmapTensors, out Mat worldLandmarkTensors)
        {
            landmarkTensors = poseFlagTensors = segmentationTensors = heatmapTensors = worldLandmarkTensors = null;
            if (tensors == null || tensors.Count < kHpPoseLandmarkTensorSplitCount)
                return false;
            landmarkTensors = tensors[0];
            poseFlagTensors = tensors[1];
            segmentationTensors = tensors[2];
            heatmapTensors = tensors[3];
            worldLandmarkTensors = tensors[4];
            return true;
        }

        /// <summary>Equivalent to <c>TensorsToFloatsCalculator</c> for presence output.</summary>
        static float TensorsToFloatsCalculator_PosePresence(Mat poseFlagTensors)
        {
            return poseFlagTensors.at<float>(0, 0)[0];
        }

        /// <summary>
        /// Equivalent to <c>ThresholdingCalculator</c>, using <c>score &gt; threshold</c> exactly as in
        /// upstream <c>thresholding_calculator.cc</c>.
        /// </summary>
        bool ThresholdingCalculator_PosePresence(float score)
        {
            return score > _minPosePresenceConfidence;
        }

        /// <summary>Equivalent to <c>GateCalculator</c> in ALLOW mode.</summary>
        static bool GateCalculator_PoseLandmarkTensors(bool allow) => allow;

        /// <summary>
        /// Ensures <see cref="_hpLandmarksTensorFlatScratch"/> is large enough to read the landmark tensor
        /// after <c>reshape(1, total)</c>.
        /// </summary>
        void EnsureHolisticLandmarksTensorFlatScratch(int need)
        {
            if (_hpLandmarksTensorFlatScratch == null || _hpLandmarksTensorFlatScratch.Length < need)
                _hpLandmarksTensorFlatScratch = new float[need];
        }

        /// <summary>
        /// Equivalent to upstream <c>TensorsToLandmarksCalculator</c> for image-normalized output.
        /// Z is computed as <c>z / input_image_width</c> because
        /// <c>tensors_to_pose_landmarks_and_segmentation.pbtxt</c> leaves <c>normalize_z</c> unspecified,
        /// so the proto default 1.0 applies.
        /// </summary>
        HolisticPoseLandmarkDecoded[] TensorsToLandmarksCalculator_PoseImage(Mat tensor, int inputW, int inputH)
        {
            int total = (int)tensor.total();
            int numDims = total / kHpPoseLandmarkModelLandmarkCount;
            if (numDims < 3)
                numDims = 3;
            if (_hpDecodedLandmarkScratch == null)
                _hpDecodedLandmarkScratch = new HolisticPoseLandmarkDecoded[kHpPoseLandmarkModelLandmarkCount];
            var arr = _hpDecodedLandmarkScratch;
            using (var flat = tensor.reshape(1, total))
            {
                EnsureHolisticLandmarksTensorFlatScratch(total);
                flat.get(0, 0, _hpLandmarksTensorFlatScratch.AsSpan(0, total));
                ReadOnlySpan<float> buf = _hpLandmarksTensorFlatScratch.AsSpan(0, total);
                for (int i = 0; i < kHpPoseLandmarkModelLandmarkCount; i++)
                {
                    int o = i * numDims;
                    float vis = numDims > 3 && o + 3 < total ? HolisticSigmoid(buf[o + 3]) : 0f;
                    float pres = numDims > 4 && o + 4 < total ? HolisticSigmoid(buf[o + 4]) : 0f;
                    arr[i] = new HolisticPoseLandmarkDecoded
                    {
                        X = buf[o] / inputW,
                        Y = o + 1 < total ? buf[o + 1] / inputH : 0f,
                        Z = o + 2 < total ? buf[o + 2] / inputW : 0f,
                        Visibility = vis,
                        Presence = pres,
                    };
                }
            }
            return arr;
        }

        HolisticPoseLandmarkDecoded[] TensorsToLandmarksCalculator_PoseWorld(Mat tensor)
        {
            int total = (int)tensor.total();
            int numDims = total / kHpPoseLandmarkModelLandmarkCount;
            if (numDims < 3) numDims = 3;
            var arr = _hpWorldDecodedLandmarkScratch;
            using (var flat = tensor.reshape(1, total))
            {
                EnsureHolisticLandmarksTensorFlatScratch(total);
                flat.get(0, 0, _hpLandmarksTensorFlatScratch.AsSpan(0, total));
                ReadOnlySpan<float> buf = _hpLandmarksTensorFlatScratch.AsSpan(0, total);
                for (int i = 0; i < kHpPoseLandmarkModelLandmarkCount; i++)
                {
                    int o = i * numDims;
                    arr[i] = new HolisticPoseLandmarkDecoded
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

        static void SplitNormalizedLandmarkListCalculator(HolisticPoseLandmarkDecoded[] all, out HolisticPoseLandmarkDecoded[] main33, out HolisticPoseLandmarkDecoded[] aux2)
        {
            main33 = new HolisticPoseLandmarkDecoded[33];
            aux2 = new HolisticPoseLandmarkDecoded[2];
            for (int i = 0; i < 33; i++)
                main33[i] = all[i];
            aux2[0] = all[33];
            aux2[1] = all[34];
        }

        static void SplitLandmarkListCalculator_PoseWorld(HolisticPoseLandmarkDecoded[] all, out HolisticPoseLandmarkDecoded[] world33)
        {
            world33 = new HolisticPoseLandmarkDecoded[33];
            for (int i = 0; i < 33; i++)
                world33[i] = all[i];
        }

        static void VisibilityCopyCalculator_PoseWorld(HolisticPoseLandmarkDecoded[] fromNorm, HolisticPoseLandmarkDecoded[] toWorld)
        {
            for (int i = 0; i < 33; i++)
            {
                var w = toWorld[i];
                w.Visibility = fromNorm[i].Visibility;
                w.Presence = fromNorm[i].Presence;
                toWorld[i] = w;
            }
        }

        static HolisticPoseLandmarkDecoded[] LandmarkLetterboxRemovalCalculator_Pose(HolisticPoseLandmarkDecoded[] lm, HolisticSinglePoseLandmarkPreprocessOut pre)
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
            var o = new HolisticPoseLandmarkDecoded[lm.Length];
            for (int i = 0; i < lm.Length; i++)
            {
                o[i] = new HolisticPoseLandmarkDecoded
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

        /// <summary>Identical to <see cref="MediaPipePoseLandmarker"/>'s <c>LandmarkProjectionCalculator_Pose</c>, producing full-image normalized output.</summary>
        static Vec3f[] LandmarkProjectionCalculator_Pose(HolisticPoseLandmarkDecoded[] lm, HolisticNormalizedRect roi)
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

        static Vec3f[] LandmarkProjectionCalculator_PoseAux(HolisticPoseLandmarkDecoded[] lm, HolisticNormalizedRect roi)
        {
            return LandmarkProjectionCalculator_Pose(lm, roi);
        }

        static Vec3f[] WorldLandmarkProjectionCalculator_Pose(HolisticPoseLandmarkDecoded[] world, HolisticNormalizedRect roi)
        {
            int n = MediaPipePoseLandmarker.PoseLandmarkerEstimationData.LANDMARK_VEC5F_COUNT;
            var v = new Vec3f[n];
            float ca = (float)Math.Cos(roi.Rotation);
            float sa = (float)Math.Sin(roi.Rotation);
            for (int i = 0; i < 33 && i < world.Length; i++)
            {
                float x = world[i].X;
                float y = world[i].Y;
                float z = world[i].Z;
                v[i] = new Vec3f(ca * x - sa * y, sa * x + ca * y, z);
            }
            return v;
        }

        HolisticPoseLandmarkDecoded[] RefineLandmarksFromHeatmapCalculator(HolisticPoseLandmarkDecoded[] landmarks, Mat heatmapTensor)
        {
            if (heatmapTensor == null || heatmapTensor.empty())
                return landmarks;
            if (!TryGetHeatmapHwc(heatmapTensor, out int hmH, out int hmW, out int hmC))
                return landmarks;
            if (hmC != kHpPoseLandmarkModelLandmarkCount)
                return landmarks;

            int kernel = kHpPoseLandmarkHeatmapKernelSize;
            int offset = (kernel - 1) / 2;
            float minConf = 0.5f;
            var outLm = _hpHeatmapRefineDecodedScratch;
            int nCopy = Math.Min(landmarks.Length, kHpPoseLandmarkModelLandmarkCount);
            for (int i = 0; i < nCopy; i++)
                outLm[i] = landmarks[i];
            for (int i = nCopy; i < kHpPoseLandmarkModelLandmarkCount; i++)
                outLm[i] = default;

            int hmRowSize = hmW * hmC;
            int hmPixelSize = hmC;
            int hmTotal = (int)heatmapTensor.total();
            using (var hmFlat = heatmapTensor.reshape(1, hmTotal))
            {
                if (_hpHeatmapReadScratch == null || _hpHeatmapReadScratch.Length < hmTotal)
                    _hpHeatmapReadScratch = new float[hmTotal];
                hmFlat.get(0, 0, _hpHeatmapReadScratch.AsSpan(0, hmTotal));
                ReadOnlySpan<float> hm = _hpHeatmapReadScratch.AsSpan(0, hmTotal);
                for (int lmIndex = 0; lmIndex < kHpPoseLandmarkModelLandmarkCount; lmIndex++)
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
                            float conf = HolisticSigmoid(hm[idx]);
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

        void SegmentationMaskFromTensorToFullImage(Mat segmentationTensor, in HolisticSinglePoseLandmarkPreprocessOut pre, Mat dstFullImageFloat01)
        {
            if (dstFullImageFloat01 == null || dstFullImageFloat01.empty())
                return;
            if (segmentationTensor == null || segmentationTensor.empty()
                || _hpSinglePoseLandmarkProjMat3x3 == null || _hpSinglePoseLandmarkProjMat3x3.empty())
            {
                dstFullImageFloat01.setTo((0d, 0d, 0d, 0d));
                return;
            }
            if (!TensorsToSegmentationCalculator_Pose(segmentationTensor, ref _hpSegmentationScratchSmall))
            {
                dstFullImageFloat01.setTo((0d, 0d, 0d, 0d));
                return;
            }
            if (_hpSegmentationFullWarpInvMat3x3 == null || _hpSegmentationFullWarpInvMat3x3.rows() != 3 || _hpSegmentationFullWarpInvMat3x3.cols() != 3
                || _hpSegmentationFullWarpInvMat3x3.type() != CvType.CV_32FC1)
            {
                _hpSegmentationFullWarpInvMat3x3?.Dispose();
                _hpSegmentationFullWarpInvMat3x3 = new Mat(3, 3, CvType.CV_32FC1);
            }
            if (Core.invert(_hpSinglePoseLandmarkProjMat3x3, _hpSegmentationFullWarpInvMat3x3, Core.DECOMP_LU) == 0)
            {
                dstFullImageFloat01.setTo((0d, 0d, 0d, 0d));
                return;
            }
            Imgproc.warpPerspective(_hpSegmentationScratchSmall, dstFullImageFloat01, _hpSegmentationFullWarpInvMat3x3,
                (pre.ImageW, pre.ImageH), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, (0d, 0d, 0d, 0d));
        }

        /// <summary>
        /// Equivalent to <c>TensorsToSegmentationCalculator</c> for a single-channel SIGMOID mask.
        /// Writes the tensor-space <c>HxW</c> <c>CV_32FC1</c> mask into <paramref name="dstSmallMask"/>.
        /// </summary>
        /// <remarks>
        /// The <paramref name="tensor"/> reshaped to <c>1xflatLen</c> and <paramref name="dstSmallMask"/>
        /// are assumed to be contiguous 32-bit float buffers suitable for <see cref="Mat.AsSpan{T}"/>
        /// (<c>elemSize()==4</c>), which is normally true for OpenCV DNN outputs.
        /// If those assumptions are not met, <see cref="Mat.AsSpan{T}"/> may throw an exception.
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
                    dst[i] = HolisticSigmoid(src[i]);
            }

            return true;
        }

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
                if (d0 == 1 && d1 == 1 && d2 > 1 && d3 > 1)
                {
                    h = d2;
                    w = d3;
                    return true;
                }
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

        static float HolisticSigmoid(float v)
        {
            return 1f / (1f + Mathf.Exp(-v));
        }
    }
}
#endif
#endif
