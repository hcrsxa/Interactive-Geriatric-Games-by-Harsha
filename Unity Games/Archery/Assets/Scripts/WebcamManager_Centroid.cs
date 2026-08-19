using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityUtils;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Captures webcam frames, converts them to OpenCV Mats, and tracks a torchlight centroid.
/// Applies Red Channel extraction, Y-Axis ceiling cropping, Aspect Ratio bounding box filtering,
/// and Largest Area selection to isolate the torchlight from overhead room lighting.
/// </summary>
public class WebcamManager : MonoBehaviour
{
    [Header("Webcam Settings")]
    public int requestedWidth = 320;   // 16:9 aspect ratio
    public int requestedHeight = 180;
    public int requestedFPS = 30;
    public string deviceName = ""; // empty = default webcam
    public double threshold = 28.0;
    public float K = 0.05f;

    [Header("Contour & Centroid Filtering")]
    [Tooltip("Threshold applied to the Red channel (0..255) to create the binary mask")]
    public int contourThreshold = 40;
    [Tooltip("Ignore contours with area smaller than this (pixels)")]
    public double minContourArea = 5.0;
    [Tooltip("Ignore contours with area larger than this (pixels)")]
    public double maxContourArea = 5000.0;
    [Tooltip("Centroid rectangle half-size for debug drawing (px)")]
    public int centroidHalfSize = 3;

    [Header("Ceiling & Aspect Ratio Noise Filters")]
    [Tooltip("Percentage of the top frame height to ignore (e.g., 0.25 = top 25% ignored for ceiling lights)")]
    [Range(0f, 0.5f)]
    public float topCeilingCropPercentage = 0.25f;
    [Tooltip("Minimum allowable aspect ratio (width/height) for circular torchlight blob")]
    public float minAspectRatio = 0.5f;
    [Tooltip("Maximum allowable aspect ratio (width/height) for circular torchlight blob")]
    public float maxAspectRatio = 2.0f;

    [Header("Image Mirroring")]
    [Tooltip("Mirror incoming webcam image horizontally for intuitive physical interaction.")]
    public bool mirrorInput = true;

    private const string previewPlane = "PreviewPlane";
    private DisplayPanels displayPanels;

    WebCamTexture webCamTexture;
    Texture2D previewTexture;
    Texture2D imageInTexture;
    Texture2D imageOutTexture;

    // Reused OpenCV Mats to avoid per-frame garbage collection
    Mat imageIn, imageOut, imageOut1, imageOut2, imageOut3, imageOut4;
    Mat prevImageIn, prevImageOut;
    Mat fullMat;
    Mat rMat, gMat, bMat;
    Mat maskMat;
    Mat hierarchy;
    List<MatOfPoint> contours;
    private Mat _tmpSingle;

    // Runtime state
    private int camWidth = 0;
    private int camHeight = 0;
    private bool cameraReady = false;
    private Coroutine webcamInitCoroutine;

    private int procWidth = 320;
    private int procHeight = 180;

    private bool isFirstFrame = true;
    private int skipFramesCounter = 5;

    private Scalar[] contourColors;
    Renderer previewRenderer;

    void Start()
    {
        var previewObj = GameObject.Find(previewPlane);
        if (previewObj == null)
        {
            Debug.LogWarning($"WebcamManager: GameObject named \"{previewPlane}\" not found.");
            enabled = false;
            return;
        }

        previewRenderer = previewObj.GetComponent<Renderer>();
        if (previewRenderer == null)
        {
            Debug.LogWarning("WebcamManager: PreviewPlane does not have a Renderer component.");
            enabled = false;
            return;
        }

        displayPanels = GameObject.Find("DisplayPanels")?.GetComponent<DisplayPanels>();
        if (displayPanels == null)
            Debug.LogWarning("WebcamManager: DisplayPanels component not found. Visual debug panels disabled.");

        contourColors = new Scalar[]
        {
            new Scalar(255, 0, 0, 255),   // Red
            new Scalar(0, 255, 0, 255),   // Green
            new Scalar(0, 0, 255, 255),   // Blue
            new Scalar(255, 255, 0, 255), // Yellow
            new Scalar(255, 0, 255, 255), // Magenta
            new Scalar(0, 255, 255, 255), // Cyan
        };

        webcamInitCoroutine = StartCoroutine(InitializeWebcamAndBuffersCoroutine(3.0f));
    }

    IEnumerator InitializeWebcamAndBuffersCoroutine(float timeoutSeconds)
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        string usedDevice = deviceName;
        if (string.IsNullOrEmpty(usedDevice) && devices.Length > 0)
            usedDevice = devices[0].name;

        webCamTexture = new WebCamTexture(usedDevice, requestedWidth, requestedHeight, requestedFPS);
        webCamTexture.Play();

        float elapsed = 0f;
        while ((webCamTexture == null || webCamTexture.width <= 16 || webCamTexture.height <= 16) && elapsed < timeoutSeconds)
        {
            yield return null;
            elapsed += Time.deltaTime;
        }

        if (webCamTexture == null)
        {
            Debug.LogError("WebcamManager: Webcam failed to initialize.");
            cameraReady = false;
            yield break;
        }

        camWidth = webCamTexture.width > 16 ? webCamTexture.width : requestedWidth;
        camHeight = webCamTexture.height > 16 ? webCamTexture.height : requestedHeight;

        Debug.Log($"WebcamManager initialized successfully: {camWidth}x{camHeight}");

        if (previewTexture != null) Destroy(previewTexture);
        previewTexture = new Texture2D(camWidth, camHeight, TextureFormat.RGBA32, false);
        previewRenderer.material.mainTexture = previewTexture;

        if (mirrorInput)
        {
            previewRenderer.material.mainTextureScale = new Vector2(-1f, 1f);
            previewRenderer.material.mainTextureOffset = new Vector2(1f, 0f);
        }
        else
        {
            previewRenderer.material.mainTextureScale = new Vector2(1f, 1f);
            previewRenderer.material.mainTextureOffset = new Vector2(0f, 0f);
        }

        setupMatsAndDisplayPanels();
        cameraReady = true;
        webcamInitCoroutine = null;
    }

    void setupMatsAndDisplayPanels()
    {
        if (imageInTexture != null) Destroy(imageInTexture);
        if (imageOutTexture != null) Destroy(imageOutTexture);
        if (prevImageIn != null) { prevImageIn.Dispose(); prevImageIn = null; }
        if (prevImageOut != null) { prevImageOut.Dispose(); prevImageOut = null; }

        imageInTexture = new Texture2D(procWidth, procHeight, TextureFormat.RGBA32, false);
        imageOutTexture = new Texture2D(procWidth, procHeight, TextureFormat.RGBA32, false);

        imageIn?.Dispose(); imageOut?.Dispose(); imageOut1?.Dispose(); imageOut2?.Dispose();
        imageOut3?.Dispose(); imageOut4?.Dispose(); prevImageIn?.Dispose(); prevImageOut?.Dispose();
        _tmpSingle?.Dispose(); fullMat?.Dispose(); fullMat = null;

        rMat?.Dispose(); rMat = null;
        gMat?.Dispose(); gMat = null;
        bMat?.Dispose(); bMat = null;
        maskMat?.Dispose(); maskMat = null;
        hierarchy?.Dispose(); hierarchy = null;

        if (contours != null)
        {
            foreach (var c in contours) c?.Dispose();
            contours = null;
        }

        imageIn = new Mat(); imageOut = new Mat(); imageOut1 = new Mat();
        imageOut2 = new Mat(); imageOut3 = new Mat(); imageOut4 = new Mat();
        prevImageIn = new Mat(); prevImageOut = new Mat();

        imageIn.create(procHeight, procWidth, CvType.CV_8UC4);
        imageOut.create(procHeight, procWidth, CvType.CV_8UC4);
        imageOut1.create(procHeight, procWidth, CvType.CV_8UC4);
        imageOut2.create(procHeight, procWidth, CvType.CV_8UC4);
        imageOut3.create(procHeight, procWidth, CvType.CV_8UC1);
        imageOut4.create(procHeight, procWidth, CvType.CV_32FC4);
        prevImageIn.create(procHeight, procWidth, CvType.CV_8UC4);
        prevImageOut.create(procHeight, procWidth, CvType.CV_8UC4);

        rMat = new Mat(); gMat = new Mat(); bMat = new Mat(); maskMat = new Mat();
        rMat.create(procHeight, procWidth, CvType.CV_8UC1);
        gMat.create(procHeight, procWidth, CvType.CV_8UC1);
        bMat.create(procHeight, procWidth, CvType.CV_8UC1);
        maskMat.create(procHeight, procWidth, CvType.CV_8UC1);

        hierarchy = new Mat();
        contours = new List<MatOfPoint>();

        _tmpSingle = new Mat();
        _tmpSingle.create(procHeight, procWidth, CvType.CV_8UC1);
        _tmpSingle.setTo(new Scalar(0));

        fullMat = new Mat();
        fullMat.create(camHeight, camWidth, CvType.CV_8UC4);

        imageIn.setTo(new Scalar(0, 0, 0, 255));
        imageOut.setTo(new Scalar(0, 0, 0, 255));
        imageOut1.setTo(new Scalar(0, 0, 0, 255));
        imageOut2.setTo(new Scalar(0, 0, 0, 255));
        imageOut3.setTo(new Scalar(0));
        imageOut4.setTo(new Scalar(0, 0, 0, 255));
        prevImageIn.setTo(new Scalar(0, 0, 0, 255));
        prevImageOut.setTo(new Scalar(0, 0, 0, 255));

        if (displayPanels != null)
        {
            displayPanels.InitDisplayPanels(procWidth, procHeight);
            displayPanels.ShowMatOnDisplay(0, imageIn);
            displayPanels.ShowMatOnDisplay(1, imageOut1);
            displayPanels.ShowMatOnDisplay(2, imageOut2);
            displayPanels.ShowMatOnDisplay(3, imageOut3);
            displayPanels.ShowMatOnDisplay(4, imageOut4);
        }
    }

    /// <summary>
    /// Processes current webcam frame, applies filters (Red Channel, Ceiling Crop, Aspect Ratio),
    /// and returns the single largest valid torchlight centroid normalized between -1.0 and 1.0.
    /// </summary>
    public bool GetCentroid(out Vector2 camPos)
    {
        camPos = Vector2.zero;

        if (imageIn == null || imageIn.empty())
            return false;

        try
        {
            // 1. EXTRACT RED CHANNEL (Channel 0 in RGBA) for optimal contrast through red optical filter
            Mat redChannel = new Mat();
            Core.extractChannel(imageIn, redChannel, 0);

            // 2. Binary thresholding on Red Channel
            Mat mask = new Mat();
            Imgproc.threshold(redChannel, mask, contourThreshold, 255, Imgproc.THRESH_BINARY);

            Mat maskClone = new Mat();
            mask.copyTo(maskClone);

            List<MatOfPoint> localContours = new List<MatOfPoint>();
            Mat localHierarchy = new Mat();
            Imgproc.findContours(maskClone, localContours, localHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            double maxArea = -1.0;
            Point bestCentroid = null;

            // 3. FILTER CONTOURS (Ceiling Crop, Aspect Ratio, Largest Area Candidate)
            for (int i = 0; i < localContours.Count; i++)
            {
                double area = Imgproc.contourArea(localContours[i]);
                if (area < minContourArea || area > maxContourArea) continue;

                Moments mu = Imgproc.moments(localContours[i]);
                double m00 = mu.get_m00();
                if (Math.Abs(m00) < Double.Epsilon) continue;

                double cx = mu.get_m10() / m00;
                double cy = mu.get_m01() / m00;

                // FILTER 1: Ceiling Crop (Ignore contours in the top portion of frame)
                float ceilingCutoffY = procHeight * topCeilingCropPercentage;
                if (cy < ceilingCutoffY) continue;

                // FILTER 2: Aspect Ratio Filter (Reject rectangular ceiling light panels)
                OpenCVForUnity.CoreModule.Rect boundingRect = Imgproc.boundingRect(localContours[i]);
                float aspectRatio = (float)boundingRect.width / boundingRect.height;
                if (aspectRatio < minAspectRatio || aspectRatio > maxAspectRatio) continue;

                // FILTER 4: Select Candidate with the Largest Surface Area
                if (area > maxArea)
                {
                    maxArea = area;
                    bestCentroid = new Point(cx, cy);
                }
            }

            // Cleanup local Mats
            foreach (var c in localContours) c?.Dispose();
            localContours.Clear();
            localHierarchy.Dispose();
            maskClone.Dispose();
            mask.Dispose();
            redChannel.Dispose();

            // 4. NORMALIZE BEST CENTROID POSITION
            if (bestCentroid != null)
            {
                float normX = ((float)bestCentroid.x - procWidth / 2f) / (procWidth / 2f);
                float normY = (-(float)bestCentroid.y + procHeight / 2f) / (procWidth / 2f); // inverted Y for Unity coordinates

                camPos = new Vector2(normX, normY);
                return true;
            }

            return false; // No valid torchlight contour passed filters
        }
        catch (Exception ex)
        {
            Debug.LogError("WebcamManager.GetCentroid failed: " + ex.Message);
            return false;
        }
    }

    void Update()
    {
        if (!cameraReady || webCamTexture == null || !webCamTexture.isPlaying || previewRenderer == null)
            return;

        if (!webCamTexture.didUpdateThisFrame)
            return;

        try
        {
            Color32[] p = webCamTexture.GetPixels32();
            previewTexture.SetPixels32(p);
            previewTexture.Apply(false);

            OpenCVMatUtils.Texture2DToMat(previewTexture, fullMat);

            if (mirrorInput)
            {
                Core.flip(fullMat, fullMat, 1);
            }

            Imgproc.resize(fullMat, imageIn, new Size(procWidth, procHeight), 0, 0, Imgproc.INTER_AREA);

            if (skipFramesCounter > 0)
            {
                skipFramesCounter--;
                return;
            }

            if (isFirstFrame)
            {
                imageIn.copyTo(prevImageIn);
                imageIn.copyTo(prevImageOut);
                isFirstFrame = false;
            }

            // Update debug visualization panels if assigned
            if (displayPanels != null)
            {
                // Channel 0 (Red) extraction for debug display
                Core.extractChannel(imageIn, rMat, 0);
                Imgproc.threshold(rMat, maskMat, contourThreshold, 255, Imgproc.THRESH_BINARY);
                Imgproc.cvtColor(maskMat, imageOut1, Imgproc.COLOR_GRAY2RGBA);

                displayPanels.HideDisplay(0);
                displayPanels.ShowMatOnDisplay(1, imageIn);
                displayPanels.ShowMatOnDisplay(4, imageOut1);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("WebcamManager Update frame processing error: " + ex.Message);
        }
    }

    void OnDisable()
    {
        StopWebcam();
    }

    void OnDestroy()
    {
        StopWebcam();
    }

    void StopWebcam()
    {
        cameraReady = false;

        if (webcamInitCoroutine != null)
        {
            StopCoroutine(webcamInitCoroutine);
            webcamInitCoroutine = null;
        }

        if (webCamTexture != null)
        {
            if (webCamTexture.isPlaying)
                webCamTexture.Stop();
            Destroy(webCamTexture);
            webCamTexture = null;
        }

        imageIn?.Dispose(); imageIn = null;
        imageOut?.Dispose(); imageOut = null;
        imageOut1?.Dispose(); imageOut1 = null;
        imageOut2?.Dispose(); imageOut2 = null;
        imageOut3?.Dispose(); imageOut3 = null;
        imageOut4?.Dispose(); imageOut4 = null;
        prevImageIn?.Dispose(); prevImageIn = null;
        prevImageOut?.Dispose(); prevImageOut = null;

        rMat?.Dispose(); rMat = null;
        gMat?.Dispose(); gMat = null;
        bMat?.Dispose(); bMat = null;
        maskMat?.Dispose(); maskMat = null;
        hierarchy?.Dispose(); hierarchy = null;

        if (contours != null)
        {
            foreach (var c in contours) c?.Dispose();
            contours = null;
        }

        _tmpSingle?.Dispose(); _tmpSingle = null;
        fullMat?.Dispose(); fullMat = null;

        if (previewTexture != null) Destroy(previewTexture);
        if (imageOutTexture != null) Destroy(imageOutTexture);
    }
}