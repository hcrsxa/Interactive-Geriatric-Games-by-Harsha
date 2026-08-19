using UnityEngine;

public class FixedFrameRate : MonoBehaviour
{
    public int FPS = 60;

    void Awake()

    {
        // ref: https://www.youtube.com/watch?v=6Fonc1ND4qI
        // https://www.reddit.com/r/Unity3D/comments/qfo6py/how_to_limit_frame_rate_in_unity/

        // QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = FPS;
    }

}

