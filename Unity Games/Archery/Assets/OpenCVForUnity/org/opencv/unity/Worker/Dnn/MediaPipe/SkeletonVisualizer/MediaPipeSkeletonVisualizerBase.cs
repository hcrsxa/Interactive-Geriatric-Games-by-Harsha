#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using System.Collections.Generic;
using OpenCVForUnity.CoreModule;
using UnityEngine;

namespace OpenCVForUnity.UnityIntegration.Worker.DnnModule.MediaPipe.SkeletonVisualizer
{
    /// <summary>
    /// Base class for visualizing MediaPipe-style landmark estimation as a 3D skeleton using <see cref="LineRenderer"/> instances.
    /// </summary>
    public abstract class MediaPipeSkeletonVisualizerBase : MonoBehaviour
    {
        /// <summary>
        /// Holds one skeleton segment: a <see cref="GameObject"/> and its <see cref="LineRenderer"/>.
        /// </summary>
        public class Skeleton
        {
            public GameObject LineObject;
            public LineRenderer Line;
        }

        /// <summary>
        /// Edge list as pairs of landmark vertex indices. Derived classes return a static array defining the topology.
        /// </summary>
        protected abstract (int, int)[] SkeletonEdges { get; }

        /// <summary>
        /// Base name for child <see cref="GameObject"/> instances created for each skeleton line.
        /// </summary>
        protected virtual string SkeletonLineObjectName => "Line";

        [SerializeField]
        protected Material _skeletonMaterial;
        public virtual Material SkeletonMaterial
        {
            get => _skeletonMaterial;
            set => _skeletonMaterial = value;
        }

        [SerializeField]
        protected float _skeletonX;
        public virtual float SkeletonX
        {
            get => _skeletonX;
            set => _skeletonX = value;
        }

        [SerializeField]
        protected float _skeletonY;
        public virtual float SkeletonY
        {
            get => _skeletonY;
            set => _skeletonY = value;
        }

        [SerializeField]
        protected float _skeletonZ;
        public virtual float SkeletonZ
        {
            get => _skeletonZ;
            set => _skeletonZ = value;
        }

        [SerializeField]
        protected float _skeletonScale = 1f;
        public virtual float SkeletonScale
        {
            get => _skeletonScale;
            set => _skeletonScale = value;
        }

        [SerializeField]
        protected float _skeletonStartWidth = 0.005f;
        public virtual float SkeletonStartWidth
        {
            get => _skeletonStartWidth;
            set => _skeletonStartWidth = value;
        }

        [SerializeField]
        protected float _skeletonEndWidth = 0.005f;
        public virtual float SkeletonEndWidth
        {
            get => _skeletonEndWidth;
            set => _skeletonEndWidth = value;
        }

        [SerializeField]
        protected bool _showSkeleton = true;
        public virtual bool ShowSkeleton
        {
            get => _showSkeleton;
            set
            {
                _showSkeleton = value;
                ClearLine();
            }
        }

        /// <summary>
        /// When true, treats input as Unity left-handed and does not negate Y. When false, converts from MediaPipe-style coordinates to Unity by negating Y.
        /// </summary>
        [SerializeField]
        [Tooltip("ON: Input is already Unity left-handed; do not negate Y. OFF: Convert from MediaPipe-style right-handed to Unity (negate Y).")]
        protected bool _inputCoordinatesAreUnityLeftHanded;
        public virtual bool InputCoordinatesAreUnityLeftHanded
        {
            get => _inputCoordinatesAreUnityLeftHanded;
            set => _inputCoordinatesAreUnityLeftHanded = value;
        }

        protected readonly List<Skeleton> _skeletons = new List<Skeleton>();

        protected virtual void OnDestroy()
        {
            if (_skeletons != null)
            {
                foreach (var skeleton in _skeletons)
                {
                    if (skeleton.LineObject != null)
                    {
                        UnityEngine.Object.Destroy(skeleton.LineObject);
                    }
                }
                _skeletons.Clear();
            }
        }

#if NET_STANDARD_2_1
        /// <summary>
        /// Updates every <see cref="LineRenderer"/> segment from the landmark span.
        /// </summary>
        /// <param name="skipOutOfRangeIndices">
        /// When true, skips edges whose endpoint indices are outside the landmark span (e.g. optional face mesh vertices).
        /// </param>
        protected virtual void UpdateAllSkeletonLines(ReadOnlySpan<Vec3f> landmarksWorld, bool skipOutOfRangeIndices = false)
        {
            EnsureSkeletonsCreated();
            (int, int)[] edges = SkeletonEdges;
            for (int i = 0; i < edges.Length; i++)
            {
                var (idx1, idx2) = edges[i];
                if (skipOutOfRangeIndices && ((uint)idx1 >= (uint)landmarksWorld.Length || (uint)idx2 >= (uint)landmarksWorld.Length))
                    continue;

                SetLinePosition(i, idx1, idx2, landmarksWorld);
            }
        }
#endif

        /// <summary>
        /// Updates every <see cref="LineRenderer"/> segment from the landmark array.
        /// </summary>
        /// <param name="skipOutOfRangeIndices">
        /// When true, skips edges whose endpoint indices are outside the landmark array length.
        /// </param>
        protected virtual void UpdateAllSkeletonLines(Vec3f[] landmarksWorld, bool skipOutOfRangeIndices = false)
        {
            EnsureSkeletonsCreated();
            (int, int)[] edges = SkeletonEdges;
            for (int i = 0; i < edges.Length; i++)
            {
                var (idx1, idx2) = edges[i];
                if (skipOutOfRangeIndices && ((uint)idx1 >= (uint)landmarksWorld.Length || (uint)idx2 >= (uint)landmarksWorld.Length))
                    continue;

                SetLinePosition(i, idx1, idx2, landmarksWorld);
            }
        }

#if NET_STANDARD_2_1
        protected virtual void SetLinePosition(int index, int idx1, int idx2, ReadOnlySpan<Vec3f> landmarksWorld)
        {
            SetLinePositionFromLandmarks(index, landmarksWorld[idx1], landmarksWorld[idx2]);
        }
#endif

        protected virtual void SetLinePosition(int index, int idx1, int idx2, Vec3f[] landmarksWorld)
        {
            SetLinePositionFromLandmarks(index, landmarksWorld[idx1], landmarksWorld[idx2]);
        }

        protected virtual void SetLinePositionFromLandmarks(int index, Vec3f landmark1, Vec3f landmark2)
        {
            float yMul = _inputCoordinatesAreUnityLeftHanded ? 1f : -1f;

            _skeletons[index].Line.SetPosition(0, new Vector3(
                landmark1.Item1 * SkeletonScale * 1 + SkeletonX,
                landmark1.Item2 * SkeletonScale * yMul + SkeletonY,
                landmark1.Item3 * SkeletonScale * 1 + SkeletonZ
                ));

            _skeletons[index].Line.SetPosition(1, new Vector3(
                landmark2.Item1 * SkeletonScale * 1 + SkeletonX,
                landmark2.Item2 * SkeletonScale * yMul + SkeletonY,
                landmark2.Item3 * SkeletonScale * 1 + SkeletonZ
                ));
        }

        protected virtual void EnsureSkeletonsCreated()
        {
            if (_skeletons.Count == 0)
            {
                int n = SkeletonEdges.Length;
                for (int i = 0; i < n; i++)
                {
                    AddSkeleton();
                }
            }
        }

        protected virtual void AddSkeleton()
        {
            var index = _skeletons.Count;
            var lineObject = new GameObject($"{SkeletonLineObjectName}_{index}");
            lineObject.transform.parent = gameObject.transform;

            lineObject.layer = gameObject.layer;

            var sk = new Skeleton
            {
                LineObject = lineObject
            };

            sk.Line = sk.LineObject.AddComponent<LineRenderer>();
            sk.Line.startWidth = SkeletonStartWidth * SkeletonScale;
            sk.Line.endWidth = SkeletonEndWidth * SkeletonScale;

            sk.Line.positionCount = 2;
            sk.Line.material = SkeletonMaterial;

            _skeletons.Add(sk);
        }

        /// <summary>
        /// Clears each <see cref="LineRenderer"/> so stale segments are not left visible when detection drops out.
        /// </summary>
        public virtual void ClearLine()
        {
            int n = SkeletonEdges.Length;
            if (_skeletons.Count != n)
                return;

            for (int i = 0; i < n; ++i)
            {
                var skeleton = _skeletons[i];
                skeleton.Line.positionCount = 0;
                skeleton.Line.positionCount = 2;
            }
        }
    }
}

#endif
