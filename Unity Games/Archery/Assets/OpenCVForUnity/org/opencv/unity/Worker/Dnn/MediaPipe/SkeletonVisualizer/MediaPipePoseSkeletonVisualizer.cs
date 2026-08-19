#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using OpenCVForUnity.CoreModule;
using UnityEngine;
using KeyPoint = OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe.MediaPipePoseLandmarker.KeyPoint;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe.SkeletonVisualizer
{
    /// <summary>
    /// Visualizes a 3D body pose skeleton with <see cref="LineRenderer"/> from MediaPipe pose estimation results.
    /// Topology follows the 33-keypoint BlazePose-style layout used by <see cref="MediaPipePoseEstimator"/>.
    /// </summary>
    public class MediaPipePoseSkeletonVisualizer : MediaPipeSkeletonVisualizerBase
    {
        /// <summary>
        /// Body pose edge list (vertex index pairs) aligned with <see cref="MediaPipePoseEstimator.KeyPoint"/>.
        /// </summary>
        protected static readonly (int, int)[] PoseEdges =
        {
            ((int)KeyPoint.Nose, (int)KeyPoint.LeftEyeInner),
            ((int)KeyPoint.LeftEyeInner, (int)KeyPoint.LeftEye),
            ((int)KeyPoint.LeftEye, (int)KeyPoint.LeftEyeOuter),
            ((int)KeyPoint.LeftEyeOuter, (int)KeyPoint.LeftEar),
            ((int)KeyPoint.Nose, (int)KeyPoint.RightEyeInner),
            ((int)KeyPoint.RightEyeInner, (int)KeyPoint.RightEye),
            ((int)KeyPoint.RightEye, (int)KeyPoint.RightEyeOuter),
            ((int)KeyPoint.RightEyeOuter, (int)KeyPoint.RightEar),
            ((int)KeyPoint.MouthLeft, (int)KeyPoint.MouthRight),
            ((int)KeyPoint.RightShoulder, (int)KeyPoint.RightElbow),
            ((int)KeyPoint.RightElbow, (int)KeyPoint.RightWrist),
            ((int)KeyPoint.RightWrist, (int)KeyPoint.RightThumb),
            ((int)KeyPoint.RightWrist, (int)KeyPoint.RightPinky),
            ((int)KeyPoint.RightWrist, (int)KeyPoint.RightIndex),
            ((int)KeyPoint.RightPinky, (int)KeyPoint.RightIndex),
            ((int)KeyPoint.LeftShoulder, (int)KeyPoint.LeftElbow),
            ((int)KeyPoint.LeftElbow, (int)KeyPoint.LeftWrist),
            ((int)KeyPoint.LeftWrist, (int)KeyPoint.LeftThumb),
            ((int)KeyPoint.LeftWrist, (int)KeyPoint.LeftIndex),
            ((int)KeyPoint.LeftWrist, (int)KeyPoint.LeftPinky),
            ((int)KeyPoint.LeftPinky, (int)KeyPoint.LeftIndex),
            ((int)KeyPoint.LeftShoulder, (int)KeyPoint.RightShoulder),
            ((int)KeyPoint.LeftShoulder, (int)KeyPoint.LeftHip),
            ((int)KeyPoint.LeftHip, (int)KeyPoint.RightHip),
            ((int)KeyPoint.RightHip, (int)KeyPoint.RightShoulder),
            ((int)KeyPoint.RightHip, (int)KeyPoint.RightKnee),
            ((int)KeyPoint.RightKnee, (int)KeyPoint.RightAnkle),
            ((int)KeyPoint.RightAnkle, (int)KeyPoint.RightHeel),
            ((int)KeyPoint.RightAnkle, (int)KeyPoint.RightFootIndex),
            ((int)KeyPoint.RightHeel, (int)KeyPoint.RightFootIndex),
            ((int)KeyPoint.LeftHip, (int)KeyPoint.LeftKnee),
            ((int)KeyPoint.LeftKnee, (int)KeyPoint.LeftAnkle),
            ((int)KeyPoint.LeftAnkle, (int)KeyPoint.LeftFootIndex),
            ((int)KeyPoint.LeftAnkle, (int)KeyPoint.LeftHeel),
            ((int)KeyPoint.LeftHeel, (int)KeyPoint.LeftFootIndex),
        };

        protected override (int, int)[] SkeletonEdges => PoseEdges;
        protected override string SkeletonLineObjectName => "PoseLine";

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
        protected Vec3f[] _landmarksWorldBuffer;
#endif

        protected void Reset()
        {
            _skeletonStartWidth = 0.032f;
            _skeletonEndWidth = 0.008f;
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
        /// Updates the pose skeleton visualization with the provided world landmarks.
        /// </summary>
        /// <param name="landmarksWorld">ReadOnlySpan of 33 world landmarks representing pose keypoints.</param>
        public virtual void UpdatePose(ReadOnlySpan<Vec3f> landmarksWorld)
        {
            UpdatePoseCore(landmarksWorld);
        }

        protected virtual void UpdatePoseCore(ReadOnlySpan<Vec3f> landmarksWorld)
        {
            if (landmarksWorld.Length < 33)
                throw new ArgumentException("Invalid landmarks_world array. It must have at least 33 elements.");

            UpdateAllSkeletonLines(landmarksWorld);
        }
#endif

        /// <summary>
        /// Updates the pose skeleton visualization with the provided world landmarks.
        /// </summary>
        /// <param name="landmarksWorld">Array of 33 world landmarks representing pose keypoints.</param>
        public virtual void UpdatePose(Vec3f[] landmarksWorld)
        {
            if (landmarksWorld == null)
                throw new ArgumentNullException(nameof(landmarksWorld));

            UpdatePoseCore(landmarksWorld);
        }

        protected virtual void UpdatePoseCore(Vec3f[] landmarksWorld)
        {
            if (landmarksWorld.Length < 33)
                throw new ArgumentException("Invalid landmarks_world array. It must have at least 33 elements.");

            UpdateAllSkeletonLines(landmarksWorld);
        }

        /// <summary>
        /// Updates the pose skeleton visualization with the provided pose estimation results.
        /// </summary>
        /// <param name="result">Pose estimation results matrix from MediaPipePoseEstimator.Infer method.</param>
        public virtual void UpdatePose(Mat result)
        {
            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return;
            if (result.rows() < 317)
                throw new ArgumentException("Invalid results matrix. It must have at least 317 rows.");

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<Vec3f> landmarksWorld = result.AsSpanRowRange<Vec3f>(199, 199 + 99);
            UpdatePose(landmarksWorld);
#else

            if (_landmarksWorldBuffer == null)
                _landmarksWorldBuffer = new Vec3f[33];

            // Copy only world landmarks data from pose data.
            OpenCVMatUtils.CopyFromMat<Vec3f>(result.rowRange(199, 199 + 99), _landmarksWorldBuffer);

            UpdatePose(_landmarksWorldBuffer);
#endif
        }
    }
}
#endif
