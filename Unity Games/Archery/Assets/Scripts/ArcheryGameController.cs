using UnityEngine;
using UnityEngine.UI; // Needed for Slider and Button references
using UnityEngine.Rendering; // Needed for Post-Processing Volume control
using UnityEngine.Rendering.Universal; // Needed for URP Depth of Field
using TMPro; // TextMeshPro library for high-clarity UI
using System.Collections.Generic;
using System.IO.Ports; // Serial Communication Library
using System.Threading; // Background Threading for Hardware

public class ArcheryGameController : MonoBehaviour
{
    // Enums for Control Mode Switching
    public enum ControlInputMode { HardwareLoadCell, KeyboardFallback }

    // Internal Bow Draw State Machine
    private enum DrawState { Rest, Drawing, Anchored }

    [Header("1. Control Input Mode (Press 'Y' to Toggle Live)")]
    [Tooltip("Current input mode. Toggle at runtime by pressing the 'Y' key.")]
    public ControlInputMode currentInputMode = ControlInputMode.HardwareLoadCell;
    [Tooltip("Optional HUD Text element displaying current input mode (e.g. 'INPUT: LOAD CELL').")]
    public TextMeshProUGUI inputModeHUDText;

    [Header("2. Hardware Configuration (M5StickC Load Cell)")]
    [Tooltip("Assigned USB COM Port for your M5StickC (e.g. COM3).")]
    public string portName = "COM3";
    public int baudRate = 115200;

    [Header("24-Bit Load Cell Calibration Settings (Geriatric Low-Force Range)")]
    [Tooltip("Ignore idle baseline noise and mounting pre-load below this threshold (8000).")]
    public float rawIdleDeadzone = 8000f;
    [Tooltip("Raw 24-bit ADC reading corresponding to full maximum physical pull (30,000 max target for reduced exertion).")]
    public float rawMaxPullForce = 30000f;

    // Internal runtime tare offset captured when pressing 'C'
    private float dynamicTareOffset = 0f;

    [Header("3. Anchor & Lock Archery Mechanics (Rehab Pacing)")]
    [Tooltip("Time in seconds a patient must hold pull force steady to trigger Anchor Lock.")]
    public float anchorHoldDuration = 1.0f;
    [Tooltip("Minimum force threshold (% of max pull) required to initiate drawing string.")]
    public float minDrawForceThreshold = 0.05f;
    [Tooltip("Force drop percentage required while Anchored to trigger deliberate release (e.g. 0.50 = 50% drop).")]
    public float deliberateReleaseDropThreshold = 0.50f;
    [Tooltip("Smoothing factor for raw sensor telemetry (filters ADC noise spikes).")]
    public float forceSmoothingFactor = 8.0f;

    // Background Threading Variables for Non-Blocking Serial Read
    private SerialPort serialPort;
    private Thread readThread;
    private bool isThreadRunning = false;
    private string lastReceivedSerialData = "";
    private readonly object lockObject = new object();

    [Header("4. Camera System (Wii Sports Style Focus)")]
    [Tooltip("Assign your independent Main Camera transform here.")]
    public Camera mainCamera;
    public Vector3 cameraOffsetFromBow = new Vector3(0f, 0.2f, -1.2f);
    public float defaultFOV = 60f;
    public float zoomFOV = 35f;
    public float cameraZoomSpeed = 3f;
    [Tooltip("Offset relative to the arrow when camera follows it in flight.")]
    public Vector3 arrowFollowOffset = new Vector3(0.5f, 0.3f, -1.5f);

    [Header("Post-Processing Focus Adjustment")]
    public Volume globalVolume;
    private DepthOfField depthOfField;
    [Tooltip("Distance from camera to target board (keeps target sharp).")]
    public float targetFocusDistance = 20f;
    public float maxFocalLength = 35f;
    public float minFocalLength = 1f;

    [Header("5. Bow & Mechanical Aiming Setup")]
    [Tooltip("Assign your 'Recursive Bow' object here.")]
    public Transform bowTransform;
    [Tooltip("Assign 'StringPosition' child transform from Hierarchy here.")]
    public Transform bowStringTransform;
    public Vector3 stringRestLocalPos = Vector3.zero;
    public Vector3 stringDrawnLocalPos = new Vector3(0f, 0f, -0.4f);
    public float mouseSensitivity = 2f;
    private float rotationX = 0f;
    private float rotationY = 0f;

    [Header("6. Active Target Setup")]
    [Tooltip("Drag your active Archery Target object (e.g., ArcheryTarget3) here.")]
    public Transform activeTarget;

    [Header("7. Arrow Physics & Flight Timeout Fail-Safe")]
    public GameObject arrowPrefab;
    public Transform arrowSpawnPoint;
    public float maxLaunchForce = 45f;
    [Tooltip("Maximum flight time in seconds before a missed arrow is automatically reset.")]
    public float maxFlightDuration = 3.5f;
    private float flightTimer = 0f;

    [Header("8. 3D World-Space Focus Ring Lens")]
    public Transform focusRing3D;
    public float distanceFromCamera = 3.5f;
    public float maxRingScale = 2.0f;
    public float minRingScale = 0.5f;
    public float ringShrinkSpeed = 1.2f;
    [Tooltip("Time in seconds before muscle fatigue causes aim instability (geriatric rehab safety delay).")]
    public float fatigueDelay = 2.5f;
    public float fatigueExpandSpeed = 1.8f;
    public float shakeIntensity = 0.08f;

    [Header("9. Tension System & Visual Feedback")]
    [Tooltip("How fast tension charges when using fallback Spacebar input.")]
    public float tensionChargeSpeed = 0.8f;
    private float currentTension = 0f;
    private float holdTimer = 0f;
    private float peakTensionReached = 0f;
    private float anchoredPeakForce = 0f; // Stores exact peak force when locked
    private DrawState currentDrawState = DrawState.Rest;

    [Header("10. Target Sticking & Ring-Based Scoring")]
    public float visualOffsetDistance = 0.02f;
    public float targetOuterRadius = 0.5f;

    [Header("11. Game Menu UI References")]
    public GameObject mainMenuPanel;
    public Button menuStartButton;
    public Button menuQuitButton;

    [Header("12. Pause Panel UI References")]
    public GameObject pausePanel;
    public Button pauseResumeButton;
    public Button pauseRestartButton;
    public Button pauseQuitButton;

    [Header("13. HUD & Summary Panel UI References")]
    public TextMeshProUGUI scoreText;
    public Slider tensionSlider;
    public GameObject summaryPanel;
    public TextMeshProUGUI finalScoreText;
    public Button summaryRestartButton;
    public Button summaryQuitButton;

    [Header("14. Rehabilitation Session Settings")]
    public int totalArrowsPerSession = 5;

    [Header("15. Webcam Torchlight Tracking Interface")]
    [Tooltip("Assign your WebcamManager GameObject here.")]
    public GameObject webcamManager;
    private WebcamManager webcamScript; // Cached reference
    [Tooltip("Enable Centroid Tracking (Press 'T' during gameplay to toggle).")]
    public bool isWebcamTracking = false;
    public float camSensitivity = 5.0f;
    [Tooltip("Lower values (e.g. 3.0-5.0) give heavy smoothing for 1-meter tracking distance.")]
    public float trackingSmoothing = 4.0f;
    [Tooltip("Ignore sub-pixel jitter smaller than this deadzone threshold.")]
    public float positionDeadzone = 0.005f;

    // Internal Memory for Smooth Centroid Filtering
    private Vector2 smoothedCentroid = Vector2.zero;
    private Vector2 lastValidCentroid = Vector2.zero;

    // Internal Load Cell Hardware Telemetry State
    private float rawIncomingHardwareForce = 0f;
    private float smoothedHardwareForce = 0f;

    // Internal State Enums
    private enum GameState { MainMenu, Playing, ArrowInFlight, Paused, Summary }
    private GameState currentState;

    // Internal metrics & Tracking
    private int score = 0;
    private int arrowsFired = 0;
    private Transform activeInFlightArrow;
    private List<GameObject> activeSpawnedArrows = new List<GameObject>();

    void Start()
    {
        if (mainCamera == null) mainCamera = Camera.main;

        if (globalVolume != null && globalVolume.profile != null)
        {
            globalVolume.profile.TryGet(out depthOfField);
        }

        BindUIButtons();

        if (tensionSlider != null)
        {
            tensionSlider.minValue = 0f;
            tensionSlider.maxValue = 1f;
            tensionSlider.value = 0f;
        }

        if (focusRing3D != null) focusRing3D.gameObject.SetActive(false);

        ShowMainMenu();

        if (bowTransform != null)
        {
            rotationX = bowTransform.localEulerAngles.y;
            rotationY = -bowTransform.localEulerAngles.x;
        }

        if (webcamManager != null)
        {
            webcamScript = webcamManager.GetComponent<WebcamManager>();
            if (webcamScript == null)
            {
                Debug.LogWarning("ArcheryGameController: Assigned webcamManager lacks a WebcamManager component!");
            }
        }

        UpdateInputModeUI();
        StartSerialThread();
    }

    void Update()
    {
        HandleKeyboardGlobalInputs();

        if (currentInputMode == ControlInputMode.HardwareLoadCell)
        {
            ProcessHardwareSerialData();
        }

        // Press 'C' at any time to calibrate/zero out baseline resting load cell noise on the mounted bow
        if (Input.GetKeyDown(KeyCode.C))
        {
            CalibrateZeroBaseline();
        }

        switch (currentState)
        {
            case GameState.Playing:
                HandleAiming();
                HandleDrawingTensionLogic();
                UpdateCameraPositionAndZoom();
                Update3DFocusRing();
                break;
            case GameState.ArrowInFlight:
                HandleCameraArrowFollow();
                CheckFlightTimeout();
                break;
        }
    }

    #region Input Control Mode Switching ('Y' Key)

    private void ToggleInputControlMode()
    {
        if (currentInputMode == ControlInputMode.HardwareLoadCell)
        {
            currentInputMode = ControlInputMode.KeyboardFallback;
            Debug.Log("⌨️ Control Mode Switched: KEYBOARD (SPACEBAR)");
        }
        else
        {
            currentInputMode = ControlInputMode.HardwareLoadCell;
            Debug.Log("⚖️ Control Mode Switched: HARDWARE LOAD CELL (ESP32)");
        }

        ResetDrawingState();
        UpdateInputModeUI();
    }

    private void UpdateInputModeUI()
    {
        if (inputModeHUDText != null)
        {
            inputModeHUDText.text = (currentInputMode == ControlInputMode.HardwareLoadCell)
                ? "INPUT: LOAD CELL (ESP32)"
                : "INPUT: KEYBOARD (SPACEBAR)";
        }
    }

    #endregion

    #region Serial Hardware Threading (M5StickC USB Serial Read)

    private void StartSerialThread()
    {
        try
        {
            serialPort = new SerialPort(portName, baudRate);
            serialPort.ReadTimeout = 100;
            serialPort.Open();

            isThreadRunning = true;
            readThread = new Thread(ReadSerialPort);
            readThread.Start();
            Debug.Log($"🔌 Connected to M5StickC USB Telemetry on {portName}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ Failed to open Serial Port {portName}: {e.Message}");
        }
    }

    private void ReadSerialPort()
    {
        while (isThreadRunning && serialPort != null && serialPort.IsOpen)
        {
            try
            {
                string data = serialPort.ReadLine().Trim();
                lock (lockObject)
                {
                    lastReceivedSerialData = data;
                }
            }
            catch (System.TimeoutException) { }
            catch (System.Exception e)
            {
                Debug.LogWarning($"Serial Thread Warning: {e.Message}");
            }
        }
    }

    private void ProcessHardwareSerialData()
    {
        string data = "";
        lock (lockObject)
        {
            data = lastReceivedSerialData;
            lastReceivedSerialData = ""; // Clear buffer
        }

        if (string.IsNullOrEmpty(data)) return;

        if (data.StartsWith("FORCE:"))
        {
            string forceValueStr = data.Replace("FORCE:", "").Trim();
            if (float.TryParse(forceValueStr, out float parsedForce))
            {
                rawIncomingHardwareForce = parsedForce;
            }
        }
    }

    void OnApplicationQuit()
    {
        isThreadRunning = false;
        if (readThread != null && readThread.IsAlive) readThread.Join();
        if (serialPort != null && serialPort.IsOpen) serialPort.Close();
    }

    #endregion

    #region State & Navigation System

    private void BindUIButtons()
    {
        if (menuStartButton != null) { menuStartButton.onClick.RemoveAllListeners(); menuStartButton.onClick.AddListener(StartNewGameSession); }
        if (menuQuitButton != null) { menuQuitButton.onClick.RemoveAllListeners(); menuQuitButton.onClick.AddListener(QuitGameApplication); }

        if (pauseResumeButton != null) { pauseResumeButton.onClick.RemoveAllListeners(); pauseResumeButton.onClick.AddListener(ResumeGame); }
        if (pauseRestartButton != null) { pauseRestartButton.onClick.RemoveAllListeners(); pauseRestartButton.onClick.AddListener(ReturnToMainMenu); }
        if (pauseQuitButton != null) { pauseQuitButton.onClick.RemoveAllListeners(); pauseQuitButton.onClick.AddListener(QuitGameApplication); }

        if (summaryRestartButton != null) { summaryRestartButton.onClick.RemoveAllListeners(); summaryRestartButton.onClick.AddListener(StartNewGameSession); }
        if (summaryQuitButton != null) { summaryQuitButton.onClick.RemoveAllListeners(); summaryQuitButton.onClick.AddListener(QuitGameApplication); }
    }

    private void HandleKeyboardGlobalInputs()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            switch (currentState)
            {
                case GameState.MainMenu:
                    QuitGameApplication();
                    break;
                case GameState.Playing:
                case GameState.ArrowInFlight:
                    PauseGame();
                    break;
                case GameState.Paused:
                    QuitGameApplication();
                    break;
            }
        }

        if (Input.GetKeyDown(KeyCode.Y))
        {
            ToggleInputControlMode();
        }

        if (Input.GetKeyDown(KeyCode.T))
        {
            isWebcamTracking = !isWebcamTracking;
            Debug.Log($"🎥 Webcam Torchlight Tracking Toggled: {(isWebcamTracking ? "ENABLED" : "DISABLED")}");
        }
    }

    public void ShowMainMenu()
    {
        currentState = GameState.MainMenu;
        Time.timeScale = 1f;

        ClearEmbeddedArrows();

        if (mainMenuPanel != null) mainMenuPanel.SetActive(true);
        if (pausePanel != null) pausePanel.SetActive(false);
        if (summaryPanel != null) summaryPanel.SetActive(false);
        if (focusRing3D != null) focusRing3D.gameObject.SetActive(false);

        ResetCameraAndBlur();
        UpdateCursorState();
    }

    public void StartNewGameSession()
    {
        ClearEmbeddedArrows();

        score = 0;
        arrowsFired = 0;
        currentState = GameState.Playing;
        Time.timeScale = 1f;

        if (mainMenuPanel != null) mainMenuPanel.SetActive(false);
        if (pausePanel != null) pausePanel.SetActive(false);
        if (summaryPanel != null) summaryPanel.SetActive(false);

        if (tensionSlider != null) tensionSlider.value = 0f;
        UpdateHUD();
        UpdateCursorState();
        ResetCameraAndBlur();
    }

    public void PauseGame()
    {
        currentState = GameState.Paused;
        Time.timeScale = 0f;

        if (pausePanel != null) pausePanel.SetActive(true);
        UpdateCursorState();
    }

    public void ResumeGame()
    {
        currentState = (activeInFlightArrow != null) ? GameState.ArrowInFlight : GameState.Playing;
        Time.timeScale = 1f;

        if (pausePanel != null) pausePanel.SetActive(false);
        UpdateCursorState();
    }

    public void ReturnToMainMenu()
    {
        ShowMainMenu();
    }

    public void QuitGameApplication()
    {
        Debug.Log("Exiting Archery Rehabilitation Application...");
        Application.Quit();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    private void UpdateCursorState()
    {
        if (currentState == GameState.Playing)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
        else
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
    }

    #endregion

    #region Archery Gameplay Core & Hardware Interaction

    private void HandleAiming()
    {
        if (bowTransform == null) return;

        Vector3 tremorOffset = Vector3.zero;
        if (currentDrawState != DrawState.Rest && holdTimer > fatigueDelay)
        {
            float shakeAmount = (holdTimer - fatigueDelay) * shakeIntensity;
            tremorOffset = new Vector3(Random.Range(-shakeAmount, shakeAmount), Random.Range(-shakeAmount, shakeAmount), 0f);
        }

        if (isWebcamTracking && webcamScript != null)
        {
            Vector2 targetRawCentroid = lastValidCentroid;

            if (webcamScript.GetCentroid(out Vector2 newCentroid))
            {
                if (Vector2.Distance(newCentroid, lastValidCentroid) > positionDeadzone)
                {
                    lastValidCentroid = newCentroid;
                }
                targetRawCentroid = lastValidCentroid;
            }

            smoothedCentroid = Vector2.Lerp(smoothedCentroid, targetRawCentroid, Time.deltaTime * trackingSmoothing);

            if (activeTarget != null)
            {
                Vector3 bowToTarget = activeTarget.position - bowTransform.position;
                Quaternion baseTargetRotation = Quaternion.LookRotation(bowToTarget);

                float basePitch = baseTargetRotation.eulerAngles.x;
                if (basePitch > 180f) basePitch -= 360f;

                float baseYaw = baseTargetRotation.eulerAngles.y;
                if (baseYaw > 180f) baseYaw -= 360f;

                float targetRotX = baseYaw + (smoothedCentroid.x * camSensitivity);
                float targetRotY = -basePitch - (smoothedCentroid.y * camSensitivity);

                rotationX = Mathf.Clamp(targetRotX, -40f, 40f);
                rotationY = Mathf.Clamp(targetRotY, -25f, 25f);
            }
        }
        else
        {
            float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
            float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

            rotationX += mouseX;
            rotationY -= mouseY;

            rotationY = Mathf.Clamp(rotationY, -25f, 25f);
            rotationX = Mathf.Clamp(rotationX, -40f, 40f);
        }

        bowTransform.localRotation = Quaternion.Euler(rotationY + tremorOffset.y, rotationX + tremorOffset.x, tremorOffset.z);
    }

    /// <summary>
    /// Anchor & Lock State Machine implementation for realistic bow drawing and deliberate releasing.
    /// Incorporates dynamic tare offset and mounted bow threshold normalization (8k rest to 30k max pull).
    /// </summary>
    private void HandleDrawingTensionLogic()
    {
        if (currentInputMode == ControlInputMode.HardwareLoadCell)
        {
            // 1. Subtract runtime dynamic tare offset (captured via 'C' key)
            float netRawForce = rawIncomingHardwareForce - dynamicTareOffset;

            // 2. Smooth incoming 24-bit ADC force values
            smoothedHardwareForce = Mathf.Lerp(smoothedHardwareForce, netRawForce, Time.deltaTime * forceSmoothingFactor);

            // 3. Apply idle deadzone to filter out structural pre-load noise (8000 baseline threshold)
            float adjustedForce = Mathf.Max(0f, smoothedHardwareForce - rawIdleDeadzone);

            // 4. Normalize force based on user's test pull range (8k - 30k mapped to 0.0 - 1.0)
            float rawTension = Mathf.Clamp01(adjustedForce / (rawMaxPullForce - rawIdleDeadzone));

            switch (currentDrawState)
            {
                case DrawState.Rest:
                    // Initiate drawing when force exceeds minimum threshold
                    if (rawTension >= minDrawForceThreshold)
                    {
                        currentDrawState = DrawState.Drawing;
                        holdTimer = 0f;
                        peakTensionReached = rawTension;
                        if (focusRing3D != null) focusRing3D.gameObject.SetActive(true);
                    }
                    break;

                case DrawState.Drawing:
                    holdTimer += Time.deltaTime;
                    currentTension = rawTension;

                    if (currentTension > peakTensionReached)
                    {
                        peakTensionReached = currentTension;
                    }

                    // Update visual UI slider and bow string position during draw phase
                    if (tensionSlider != null) tensionSlider.value = currentTension;
                    if (bowStringTransform != null)
                    {
                        bowStringTransform.localPosition = Vector3.Lerp(stringRestLocalPos, stringDrawnLocalPos, currentTension);
                    }

                    // ANCHOR LOCK CONDITION: Patient holds pull steady for anchorHoldDuration (e.g. 1.0 second)
                    if (holdTimer >= anchorHoldDuration)
                    {
                        currentDrawState = DrawState.Anchored;
                        anchoredPeakForce = peakTensionReached; // Lock in the peak potential energy achieved
                        Debug.Log($"⚓ ANCHOR LOCKED! Stored Peak Potential Energy: {anchoredPeakForce * 100:F0}%");
                    }
                    break;

                case DrawState.Anchored:
                    holdTimer += Time.deltaTime;

                    // Maintain maximum locked tension visual feedback so aiming doesn't cause visual jitter
                    currentTension = anchoredPeakForce;
                    if (tensionSlider != null) tensionSlider.value = currentTension;
                    if (bowStringTransform != null)
                    {
                        bowStringTransform.localPosition = Vector3.Lerp(stringRestLocalPos, stringDrawnLocalPos, currentTension);
                    }

                    // DELIBERATE RELEASE CONDITION: Patient relaxes arm force by deliberateReleaseDropThreshold (e.g. 50%)
                    if (rawTension < (anchoredPeakForce * (1f - deliberateReleaseDropThreshold)))
                    {
                        LaunchArrowWithPower(anchoredPeakForce); // Launch using locked peak power!
                        ResetDrawingState();
                    }
                    break;
            }
        }
        else // KEYBOARD FALLBACK MODE (SPACEBAR)
        {
            bool isKeyboardDrawing = Input.GetKey(KeyCode.Space);
            bool isKeyboardRelease = Input.GetKeyUp(KeyCode.Space);

            if (Input.GetKeyDown(KeyCode.Space) && currentDrawState == DrawState.Rest)
            {
                currentDrawState = DrawState.Drawing;
                currentTension = 0f;
                holdTimer = 0f;
                if (focusRing3D != null) focusRing3D.gameObject.SetActive(true);
            }

            if (currentDrawState == DrawState.Drawing && isKeyboardDrawing)
            {
                holdTimer += Time.deltaTime;
                currentTension += Time.deltaTime * tensionChargeSpeed;
                currentTension = Mathf.Clamp01(currentTension);

                if (tensionSlider != null) tensionSlider.value = currentTension;
                if (bowStringTransform != null)
                {
                    bowStringTransform.localPosition = Vector3.Lerp(stringRestLocalPos, stringDrawnLocalPos, currentTension);
                }
            }

            if (currentDrawState == DrawState.Drawing && isKeyboardRelease)
            {
                LaunchArrowWithPower(currentTension);
                ResetDrawingState();
            }
        }
    }

    /// <summary>
    /// Captures the current resting baseline reading to zero out bow pre-load at runtime.
    /// </summary>
    public void CalibrateZeroBaseline()
    {
        dynamicTareOffset = rawIncomingHardwareForce;
        Debug.Log($"⚖️ Bow Zero Calibrated Successfully! Baseline Offset set to: {dynamicTareOffset}");
    }

    private void ResetDrawingState()
    {
        currentDrawState = DrawState.Rest;
        currentTension = 0f;
        peakTensionReached = 0f;
        anchoredPeakForce = 0f;
        holdTimer = 0f;
        smoothedHardwareForce = 0f;

        if (tensionSlider != null) tensionSlider.value = 0f;
        if (focusRing3D != null) focusRing3D.gameObject.SetActive(false);
        if (bowStringTransform != null) bowStringTransform.localPosition = stringRestLocalPos;
    }

    private void Update3DFocusRing()
    {
        if (focusRing3D == null || currentDrawState == DrawState.Rest || mainCamera == null) return;

        Ray aimRay = mainCamera.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0f));
        focusRing3D.position = aimRay.origin + (aimRay.direction * Mathf.Max(2.0f, distanceFromCamera));
        focusRing3D.rotation = Quaternion.LookRotation(aimRay.direction);

        float currentScale = maxRingScale;
        if (holdTimer <= fatigueDelay)
        {
            currentScale = Mathf.Lerp(maxRingScale, minRingScale, holdTimer * ringShrinkSpeed);
        }
        else
        {
            float fatigueTime = holdTimer - fatigueDelay;
            currentScale = Mathf.Min(maxRingScale, minRingScale + (fatigueTime * fatigueExpandSpeed));
        }

        focusRing3D.localScale = new Vector3(currentScale, currentScale, currentScale);
    }

    private void UpdateCameraPositionAndZoom()
    {
        if (mainCamera == null || bowTransform == null) return;

        Vector3 targetCamPos = bowTransform.position + (bowTransform.rotation * cameraOffsetFromBow);
        mainCamera.transform.position = Vector3.Lerp(mainCamera.transform.position, targetCamPos, Time.deltaTime * 10f);
        mainCamera.transform.rotation = Quaternion.Slerp(mainCamera.transform.rotation, bowTransform.rotation, Time.deltaTime * 10f);

        float targetFOV = (currentDrawState != DrawState.Rest) ? zoomFOV : defaultFOV;
        mainCamera.fieldOfView = Mathf.Lerp(mainCamera.fieldOfView, targetFOV, Time.deltaTime * cameraZoomSpeed);

        if (depthOfField != null)
        {
            depthOfField.focusDistance.value = targetFocusDistance;
            float targetFocalLength = (currentDrawState != DrawState.Rest) ? maxFocalLength : minFocalLength;
            depthOfField.focalLength.value = Mathf.Lerp(depthOfField.focalLength.value, targetFocalLength, Time.deltaTime * cameraZoomSpeed * 2f);
        }
    }

    /// <summary>
    /// Launches the active arrow object using a non-linear power curve based on stored peak energy.
    /// </summary>
    private void LaunchArrowWithPower(float launchPower)
    {
        if (arrowPrefab == null || arrowSpawnPoint == null) return;

        GameObject spawnedArrow = Instantiate(arrowPrefab, arrowSpawnPoint.position, arrowSpawnPoint.rotation);
        activeInFlightArrow = spawnedArrow.transform;
        activeSpawnedArrows.Add(spawnedArrow);

        Rigidbody rb = spawnedArrow.GetComponent<Rigidbody>();
        if (rb != null)
        {
            // Apply non-linear power curve (Power^1.5) to give full, punchy release speed
            float effectivePower = Mathf.Pow(launchPower, 1.5f);
            float fireVelocity = effectivePower * maxLaunchForce;

            rb.linearVelocity = arrowSpawnPoint.forward * fireVelocity;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            Debug.Log($"🏹 Arrow Launched! Power Level: {effectivePower * 100:F0}% | Velocity: {fireVelocity:F1} m/s");
        }

        ArrowCollisionProxy proxy = spawnedArrow.AddComponent<ArrowCollisionProxy>();
        proxy.Initialize(this);

        currentState = GameState.ArrowInFlight;
        flightTimer = 0f;
        arrowsFired++;
    }

    private void HandleCameraArrowFollow()
    {
        if (mainCamera == null || activeInFlightArrow == null) return;

        Vector3 targetCameraPosition = activeInFlightArrow.position + (activeInFlightArrow.rotation * arrowFollowOffset);
        mainCamera.transform.position = Vector3.Lerp(mainCamera.transform.position, targetCameraPosition, Time.deltaTime * 10f);
        mainCamera.transform.LookAt(activeInFlightArrow.position);
    }

    private void CheckFlightTimeout()
    {
        flightTimer += Time.deltaTime;
        if (flightTimer >= maxFlightDuration)
        {
            Debug.LogWarning("⚠️ Arrow missed target / flight timeout reached! Resetting camera for next shot.");
            if (activeInFlightArrow != null)
            {
                Destroy(activeInFlightArrow.gameObject);
                activeInFlightArrow = null;
            }
            FinishArrowFlightSequence();
        }
    }

    private void ResetCameraAndBlur()
    {
        if (mainCamera == null) return;

        if (bowTransform != null)
        {
            mainCamera.transform.position = bowTransform.position + (bowTransform.rotation * cameraOffsetFromBow);
            mainCamera.transform.rotation = bowTransform.rotation;
        }

        mainCamera.fieldOfView = defaultFOV;

        if (depthOfField != null)
        {
            depthOfField.focalLength.value = minFocalLength;
        }
    }

    public void ProcessTargetImpact(GameObject arrow, Collision collision)
    {
        Rigidbody rb = arrow.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.isKinematic = true;
        }

        Collider arrowCollider = arrow.GetComponent<Collider>();
        if (arrowCollider != null) arrowCollider.enabled = false;

        ContactPoint contact = collision.contacts[0];
        Vector3 surfaceNormal = contact.normal;

        if (Vector3.Dot(surfaceNormal, arrow.transform.forward) > 0)
        {
            surfaceNormal = -arrow.transform.forward;
        }

        arrow.transform.position = contact.point + (surfaceNormal * visualOffsetDistance);
        arrow.transform.rotation = Quaternion.LookRotation(-surfaceNormal);
        arrow.transform.SetParent(collision.transform);

        int calculatedPoints = CalculateRingScore(collision.transform, contact.point);
        score += calculatedPoints;
        UpdateHUD();

        Invoke("FinishArrowFlightSequence", 1.2f);
    }

    private int CalculateRingScore(Transform targetTransform, Vector3 impactPoint)
    {
        Vector3 localImpact = targetTransform.InverseTransformPoint(impactPoint);
        float localRadius = new Vector2(localImpact.x, localImpact.y).magnitude;

        Debug.Log($"🎯 Target Struck! Unscaled Local Radius: {localRadius:F3} | Assigned Outer Radius: {targetOuterRadius}");

        if (localRadius > targetOuterRadius)
        {
            Debug.Log("🎯 Impact outside scoring rings! 0 Points awarded.");
            return 0;
        }

        float normalizedRadius = Mathf.Clamp01(localRadius / targetOuterRadius);
        int ringScore = 10 - Mathf.FloorToInt(normalizedRadius * 10f);
        ringScore = Mathf.Clamp(ringScore, 1, 10);

        Debug.Log($"🎯 Ring Score Calculated: {ringScore} Points. Total Score: {score + ringScore}");
        return ringScore;
    }

    private void FinishArrowFlightSequence()
    {
        activeInFlightArrow = null;
        ResetCameraAndBlur();

        if (arrowsFired >= totalArrowsPerSession)
        {
            EndRehabSession();
        }
        else
        {
            currentState = GameState.Playing;
            UpdateCursorState();
        }
    }

    private void UpdateHUD()
    {
        if (scoreText != null)
        {
            scoreText.text = $"SCORE: {score}";
        }
    }

    private void EndRehabSession()
    {
        currentState = GameState.Summary;
        UpdateCursorState();

        if (summaryPanel != null) summaryPanel.SetActive(true);

        if (finalScoreText != null)
        {
            finalScoreText.text = $"GREAT WORK!\nTOTAL SCORE: {score} POINTS";
        }
    }

    private void ClearEmbeddedArrows()
    {
        for (int i = activeSpawnedArrows.Count - 1; i >= 0; i--)
        {
            if (activeSpawnedArrows[i] != null)
            {
                Destroy(activeSpawnedArrows[i]);
            }
        }
        activeSpawnedArrows.Clear();
    }

    #region Rehabilitation Metrics & Data Logging Extensions

    public void CalibrateLoadCellBounds(float zeroReading, float maxPullReading)
    {
        rawIdleDeadzone = zeroReading;
        rawMaxPullForce = maxPullReading;
        Debug.Log($"⚖️ Load Cell Calibrated in Unity: Deadzone Offset = {rawIdleDeadzone}, Max Pull = {rawMaxPullForce}");
    }

    #endregion

    #endregion
}

public class ArrowCollisionProxy : MonoBehaviour
{
    private ArcheryGameController masterController;

    public void Initialize(ArcheryGameController controller)
    {
        masterController = controller;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (masterController == null) return;

        if (collision.gameObject.name == "Target" || collision.transform.name.Contains("Target"))
        {
            masterController.ProcessTargetImpact(gameObject, collision);
            Destroy(this);
        }
    }
}