using System;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using UnityEngine;

/// <summary>
/// Utility to display OpenCV For Unity Mats on a set of MeshRenderer panels (quads) named Display0..DisplayN
/// grouped under a parent GameObject (default name "DisplayPanels").
/// Optimized for MeshRenderer (quads). Call InitDisplayPanels(width,height) once real size is known,
/// then call ShowMatOnDisplay(index, mat) to update a panel texture. Simpler; logs on unexpected cases.
///
/// NOTE: This implementation expects Mats in RGBA channel order for 4-channel images (CV_8UC4 / CV_32FC4).
/// If you have BGRA Mats from other OpenCV code, convert them to RGBA with Imgproc.cvtColor(..., COLOR_BGRA2RGBA)
/// before calling ShowMatOnDisplay.
/// </summary>
public class DisplayPanels : MonoBehaviour
{
    [Tooltip("Name of the parent GameObject that contains child panels named Display0..DisplayN")]
    public string panelsParentName = "DisplayPanels";

    // number of panels (default to 5 for Display0..Display4)
    public int panelCount = 5;

    // internal arrays sized to panelCount
    Renderer[] panels;              // cached renderers (we expect MeshRenderer for quads)
    Texture2D[] textures;           // backing textures per panel (preallocated by InitDisplayPanels)
    Material[] instancedMaterials;  // per-panel material instances (so we don't modify shared materials)

    bool panelsInitialized = false;

    void Awake()
    {
        InitArrays();
        FindAndCachePanels();
    }

    void OnDestroy()
    {
        // cleanup Unity objects we created
        for (int i = 0; i < panelCount; i++)
        {
            if (textures[i] != null)
            {
                Destroy(textures[i]);
                textures[i] = null;
            }

            if (instancedMaterials[i] != null)
            {
                Destroy(instancedMaterials[i]);
                instancedMaterials[i] = null;
            }
        }
    }

    void InitArrays()
    {
        panels = new Renderer[panelCount];
        textures = new Texture2D[panelCount];
        instancedMaterials = new Material[panelCount];
    }

    void FindAndCachePanels()
    {
        var parentGO = GameObject.Find(panelsParentName);
        if (parentGO == null)
        {
            Debug.LogWarning($"DisplayPanels: Parent GameObject named \"{panelsParentName}\" not found. Looking for root-named children.");
            // continue — fallback to searching scene root for DisplayX
        }

        for (int i = 0; i < panelCount; i++)
        {
            string childName = $"Display{i}";
            Transform childT = parentGO != null ? parentGO.transform.Find(childName) : null;
            GameObject childGO = childT != null ? childT.gameObject : GameObject.Find(childName);

            if (childGO == null)
            {
                Debug.LogWarning($"DisplayPanels: Panel GameObject \"{childName}\" not found.");
                panels[i] = null;
                continue;
            }

            var r = childGO.GetComponent<Renderer>();
            if (r == null)
            {
                Debug.LogWarning($"DisplayPanels: GameObject \"{childName}\" does not have a Renderer component.");
            }
            else if (!(r is MeshRenderer))
            {
                Debug.LogWarning($"DisplayPanels: Panel \"{childName}\" is not a MeshRenderer. This implementation expects quads with MeshRenderer.");
            }

            panels[i] = r;
        }
    }

    /// <summary>
    /// Initialize per-panel textures and instanced materials. Call once the real image width/height is known.
    /// This prepares MeshRenderer panels (quads) so ShowMatOnDisplay only needs to upload Mat -> Texture2D.
    /// </summary>
    public void InitDisplayPanels(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            Debug.LogError("DisplayPanels.InitDisplayPanels: invalid width/height.");
            panelsInitialized = false;
            return;
        }

        for (int i = 0; i < panelCount; i++)
        {
            var r = panels[i] as MeshRenderer;
            if (r == null)
            {
                // skip missing or non-mesh panels
                textures[i] = null;
                if (instancedMaterials[i] != null)
                {
                    Destroy(instancedMaterials[i]);
                    instancedMaterials[i] = null;
                }
                continue;
            }

            // destroy previous texture/material if any
            if (textures[i] != null)
            {
                Destroy(textures[i]);
                textures[i] = null;
            }
            if (instancedMaterials[i] != null)
            {
                Destroy(instancedMaterials[i]);
                instancedMaterials[i] = null;
            }

            // create texture and material instance
            textures[i] = new Texture2D(width, height, TextureFormat.RGBA32, false);
            textures[i].wrapMode = TextureWrapMode.Clamp;

            Material baseMat = r.sharedMaterial;
            if (baseMat != null)
                instancedMaterials[i] = new Material(baseMat);
            else
            {
                var shader = Shader.Find("Unlit/Texture") ?? Shader.Find("Sprites/Default");
                instancedMaterials[i] = new Material(shader);
            }

            instancedMaterials[i].mainTexture = textures[i];
            r.material = instancedMaterials[i]; // assign instance to renderer
            r.enabled = true;
        }

        panelsInitialized = true;
        Debug.Log($"DisplayPanels: initialized {panelCount} panels with texture size {width}x{height}.");
    }

    bool ValidateIndexAndReady(int index)
    {
        if (index < 0 || index >= panelCount)
        {
            Debug.LogError($"DisplayPanels: index out of range: {index}");
            return false;
        }

        if (!panelsInitialized)
        {
            Debug.LogWarning("DisplayPanels: panels not initialized. Call InitDisplayPanels(width,height) first.");
            return false;
        }

        if (panels[index] == null)
        {
            Debug.LogWarning($"DisplayPanels: panel {index} not available (missing GameObject or MeshRenderer).");
            return false;
        }

        if (!(panels[index] is MeshRenderer))
        {
            Debug.LogWarning($"DisplayPanels: panel {index} renderer is not MeshRenderer. This implementation expects quads.");
            return false;
        }

        if (textures[index] == null)
        {
            Debug.LogWarning($"DisplayPanels: backing texture for panel {index} is null.");
            return false;
        }

        return true;
    }

    /// <summary>
    /// Upload srcMat to the panel's preallocated texture. Supports:
    ///  - CV_8UC1 (GRAY)
    ///  - CV_8UC4 (RGBA)
    ///  - CV_32FC1 (GRAY floats, assumed 0..255)
    ///  - CV_32FC4 (RGBA floats, assumed 0..255)
    /// The method expects 4-channel Mats to be RGBA. If you have BGRA mats, convert them to RGBA before calling.
    /// If srcMat size differs from backing texture size or unsupported type, logs and returns false.
    /// </summary>
    public bool ShowMatOnDisplay(int index, Mat srcMat)
    {
        if (!ValidateIndexAndReady(index) || srcMat == null)
            return false;

        int w = srcMat.cols();
        int h = srcMat.rows();
        if (w <= 0 || h <= 0)
        {
            Debug.LogError("DisplayPanels: invalid mat size.");
            return false;
        }

        if (textures[index] == null || textures[index].width != w || textures[index].height != h)
        {
            Debug.LogWarning($"DisplayPanels: srcMat size {w}x{h} does not match preallocated texture {textures[index]?.width}x{textures[index]?.height}. Upload skipped.");
            return false;
        }

        Mat rgba = new Mat();
        Mat tmp = null;
        try
        {
            int type = srcMat.type();
            int ch = srcMat.channels();

            if (type == CvType.CV_8UC1 && ch == 1)
            {
                Imgproc.cvtColor(srcMat, rgba, Imgproc.COLOR_GRAY2RGBA);
            }
            else if (type == CvType.CV_8UC4 && ch == 4)
            {
                // Source is already RGBA  — copy directly.
                srcMat.copyTo(rgba);
            }
            else if (type == CvType.CV_32FC1 && ch == 1)
            {
                tmp = new Mat();
                double maxVal = Core.minMaxLoc(srcMat).maxVal;
                double scale = 1.0; // (maxVal > 1.0) ? 1.0 : 255.0;
                srcMat.convertTo(tmp, CvType.CV_8UC1, scale);
                Imgproc.cvtColor(tmp, rgba, Imgproc.COLOR_GRAY2RGBA);
            }
            else if (type == CvType.CV_32FC4 && ch == 4)
            {
                tmp = new Mat();
                double maxVal = Core.minMaxLoc(srcMat).maxVal;
                double scale = 1.0; // (maxVal > 1.0) ? 1.0 : 255.0;
                srcMat.convertTo(tmp, CvType.CV_8UC4, scale);
                tmp.copyTo(rgba); // tmp is expected RGBA
            }
            else
            {
                Debug.LogWarning($"DisplayPanels: unsupported Mat type/channel ({type}, ch={ch}) for panel {index}.");
                return false;
            }

            // upload RGBA Mat into Texture2D
            OpenCVMatUtils.MatToTexture2D(rgba, textures[index]);

            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError("DisplayPanels: ShowMatOnDisplay failed: " + ex.Message);
            return false;
        }
        finally
        {
            rgba?.Dispose();
            tmp?.Dispose();
        }
    }

    /// <summary>
    /// Hide the panel (disables the MeshRenderer).
    /// </summary>
    public void HideDisplay(int index)
    {
        if (index < 0 || index >= panelCount) return;
        var r = panels[index];
        if (r != null) r.enabled = false;
    }

    /// <summary>
    /// Unhide the panel (enables the MeshRenderer).
    /// </summary>
    public void UnhidePanel(int index)
    {
        if (index < 0 || index >= panelCount) return;
        var r = panels[index];
        if (r != null) r.enabled = true;
    }

    /// <summary>
    /// Clear panel contents (remove texture from material and destroy backing texture/material instance).
    /// </summary>
    public void ClearPanel(int index)
    {
        if (index < 0 || index >= panelCount) return;
        var r = panels[index] as MeshRenderer;
        if (r != null)
        {
            if (instancedMaterials[index] != null)
            {
                instancedMaterials[index].mainTexture = null;
                Destroy(instancedMaterials[index]);
                instancedMaterials[index] = null;
            }
            else
            {
                var mat = r.material;
                if (mat != null)
                    mat.mainTexture = null;
            }
        }

        if (textures[index] != null)
        {
            Destroy(textures[index]);
            textures[index] = null;
        }
    }

    /// <summary>
    /// Convenience: hide or unhide all panels.
    /// </summary>
    public void SetAllPanelsVisible(bool visible)
    {
        for (int i = 0; i < panelCount; i++)
        {
            if (panels[i] != null)
                panels[i].enabled = visible;
        }
    }
}