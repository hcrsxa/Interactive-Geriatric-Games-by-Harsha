#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using OpenCVForUnity.CoreModule;
using UnityEngine;
using KeyPoint = OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe.MediaPipeHandLandmarker.KeyPoint;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe.SkeletonVisualizer
{
    /// <summary>
    /// Visualizes a 3D hand skeleton with <see cref="LineRenderer"/> from MediaPipe hand pose estimation (world landmarks).
    /// Topology matches <see cref="MediaPipeHandLandmarker"/> <c>HAND_LANDMARK_CONNECTIONS</c> (21-landmark hand mesh).
    /// </summary>
    public class MediaPipeHandPoseSkeletonVisualizer : MediaPipeSkeletonVisualizerBase
    {
        /// <summary>
        /// Same edge set as <see cref="MediaPipeHandLandmarker"/> <c>HAND_LANDMARK_CONNECTIONS</c> (21-landmark hand mesh).
        /// </summary>
        protected static readonly (int, int)[] HandEdges =
        {
            ((int)KeyPoint.Wrist, (int)KeyPoint.Thumb1),
            ((int)KeyPoint.Thumb1, (int)KeyPoint.Thumb2),
            ((int)KeyPoint.Thumb2, (int)KeyPoint.Thumb3),
            ((int)KeyPoint.Thumb3, (int)KeyPoint.Thumb4),

            ((int)KeyPoint.Wrist, (int)KeyPoint.Index1),
            ((int)KeyPoint.Index1, (int)KeyPoint.Index2),
            ((int)KeyPoint.Index2, (int)KeyPoint.Index3),
            ((int)KeyPoint.Index3, (int)KeyPoint.Index4),

            ((int)KeyPoint.Wrist, (int)KeyPoint.Middle1),
            ((int)KeyPoint.Middle1, (int)KeyPoint.Middle2),
            ((int)KeyPoint.Middle2, (int)KeyPoint.Middle3),
            ((int)KeyPoint.Middle3, (int)KeyPoint.Middle4),

            ((int)KeyPoint.Wrist, (int)KeyPoint.Ring1),
            ((int)KeyPoint.Ring1, (int)KeyPoint.Ring2),
            ((int)KeyPoint.Ring2, (int)KeyPoint.Ring3),
            ((int)KeyPoint.Ring3, (int)KeyPoint.Ring4),

            ((int)KeyPoint.Wrist, (int)KeyPoint.Pinky1),
            ((int)KeyPoint.Pinky1, (int)KeyPoint.Pinky2),
            ((int)KeyPoint.Pinky2, (int)KeyPoint.Pinky3),
            ((int)KeyPoint.Pinky3, (int)KeyPoint.Pinky4),

            ((int)KeyPoint.Index1, (int)KeyPoint.Middle1),
            ((int)KeyPoint.Middle1, (int)KeyPoint.Ring1),
            ((int)KeyPoint.Ring1, (int)KeyPoint.Pinky1),
        };

        protected override (int, int)[] SkeletonEdges => HandEdges;
        protected override string SkeletonLineObjectName => "HandLine";

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
        protected Vec3f[] _landmarksWorldBuffer;
#endif

        protected void Reset()
        {
            _skeletonStartWidth = 0.004f;
            _skeletonEndWidth = 0.001f;
        }

        protected override void OnDestroy()
        {
            base.OnDestroy();

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            _landmarksWorldBuffer = null;
#endif
        }

#if NET_STANDARD_2_1
        /// <summary>
        /// Updates the hand pose skeleton visualization with the provided world landmarks.
        /// </summary>
        /// <param name="landmarksWorld">ReadOnlySpan of 21 world landmarks representing hand keypoints.</param>
        public virtual void UpdatePose(ReadOnlySpan<Vec3f> landmarksWorld)
        {
            UpdatePoseCore(landmarksWorld);
        }

        protected virtual void UpdatePoseCore(ReadOnlySpan<Vec3f> landmarksWorld)
        {
            if (landmarksWorld.Length < 21)
                throw new ArgumentException("Invalid landmarksWorld array. It must have at least 21 elements.");

            UpdateAllSkeletonLines(landmarksWorld);
        }
#endif

        /// <summary>
        /// Updates the hand pose skeleton visualization with the provided world landmarks.
        /// </summary>
        /// <param name="landmarksWorld">Array of 21 world landmarks representing hand keypoints.</param>
        public virtual void UpdatePose(Vec3f[] landmarksWorld)
        {
            if (landmarksWorld == null)
                throw new ArgumentNullException(nameof(landmarksWorld));

            UpdatePoseCore(landmarksWorld);
        }

        protected virtual void UpdatePoseCore(Vec3f[] landmarksWorld)
        {
            if (landmarksWorld.Length < 21)
                throw new ArgumentException("Invalid landmarksWorld array. It must have at least 21 elements.");

            UpdateAllSkeletonLines(landmarksWorld);
        }

        /// <summary>
        /// Updates the hand pose skeleton visualization with the provided result matrix.
        /// </summary>
        /// <param name="result">Hand pose estimation results matrix from MediaPipeHandPoseEstimator.Infer method.</param>
        public virtual void UpdatePose(Mat result)
        {
            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return;
            if (result.rows() < 132)
                throw new ArgumentException("Invalid results matrix. It must have at least 132 rows.");

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<Vec3f> landmarksWorld = result.AsSpanRowRange<Vec3f>(67, 67 + 63);
            UpdatePose(landmarksWorld);
#else
            if (_landmarksWorldBuffer == null)
                _landmarksWorldBuffer = new Vec3f[21];

            // Copy only world landmarks data from pose data.
            OpenCVMatUtils.CopyFromMat<Vec3f>(result.rowRange(67, 67 + 63), _landmarksWorldBuffer);

            UpdatePose(_landmarksWorldBuffer);
#endif
        }
    }
}
#endif
