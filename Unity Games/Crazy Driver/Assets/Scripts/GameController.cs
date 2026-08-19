using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO.Ports; // Serial Port API for Bluetooth Classic (SPP)
using System.Threading;  // Background thread to prevent Unity main thread lag

public class GameController : MonoBehaviour
{
    private enum GameState { TitleScreen, Countdown, Gameplay, Paused, GameOver }
    private GameState currentState = GameState.TitleScreen;

    [Header("Manual UI Panel Assignments")]
    public GameObject titleScreenPanel;
    public GameObject pausePanel;
    public GameObject gameOverPanel;

    [Header("Manual TextMeshPro Assignments")]
    public TMP_Text countdownText;
    public TMP_Text gameplayScoreText;
    public TMP_Text finalScoreText;
    public TMP_Text highScoreText;

    [Header("Direct Button Drag & Drop Slots")]
    public Button titleQuitButton;
    [Space(5)]
    public Button pauseResumeButton;
    public Button pauseRestartButton;
    public Button pauseQuitButton;
    [Space(5)]
    public Button gameOverRestartButton;
    public Button gameOverQuitButton;

    [Header("Hardware Integration (M5Stick SPP)")]
    [Tooltip("Enable via Inspector or press 'Y' during runtime")]
    public bool isHardwareModeActive = false;
    [Tooltip("Outgoing COM port assigned by Windows Bluetooth settings (e.g., COM3, COM4)")]
    public string bluetoothComPort = "COM3";
    [Tooltip("Set to 115200 to eliminate serial latency for real-time rehabilitation feedback")]
    public int baudRate = 115200;

    [Header("Clinical Evaluation & Calibration")]
    [Tooltip("Check this or press 'I' at runtime if tilting right moves the car left due to how the patient holds/mounts the M5Stick.")]
    public bool invertTiltDirection = false; // Clinical toggle for therapist/evaluator
    [Tooltip("G-Force threshold to trigger a lane shift (e.g., 0.35)")]
    public float tiltThreshold = 0.35f;
    [Tooltip("Cooldown delay (in seconds) between lane shifts to prevent accidental multi-lane jumps")]
    public float tiltDebounceDelay = 0.35f;

    // Serial & Threading internal variables
    private SerialPort sp;
    private Thread readThread;
    private bool isThreadRunning = false;
    private string pendingTelemetryData = "";
    private readonly object lockObject = new object();

    // Hardware tilt tracking
    private float lastLaneShiftTime = 0f;
    private bool isTiltLatched = false;

    [Header("Background Scrolling")]
    public Transform road1;
    public Transform road2;
    public float baseScrollSpeed = 5f;
    private float currentScrollSpeed;
    private float roadHeight;

    [Header("Player & Traffic Scale Configurations")]
    public float playerCarScale = 0.3f;
    public float trafficCarScale = 0.3f;

    [Header("Unified Distance Collision Settings")]
    public float trafficHitboxRadius = 1.3f;

    [Header("Visual Crash Assets")]
    public GameObject explosionPrefab;

    [Header("6-Lane X Coordinates")]
    public float[] laneXPositions = new float[] { -7.0f, -4.4f, -1.4f, 1.4f, 4.4f, 7.0f };

    [Header("Player Settings")]
    public Transform playerCar;
    public float laneMoveSpeed = 15f;
    private int currentLane = 3;
    private Vector3 targetPosition;

    [Header("Traffic Car Sprites (Direct Drop)")]
    public Sprite[] trafficSprites;
    public float baseSpawnInterval = 2.5f;
    private float currentSpawnInterval;
    private float spawnTimer;
    private List<GameObject> activeTraffic = new List<GameObject>();

    [Header("Scoring & Progression")]
    private float score = 0f;
    private float survivalTimer = 0f;
    private float difficultyTimer = 0f;
    private int difficultyLevel = 1;

    void Start()
    {
        if (playerCar == null) playerCar = GameObject.Find("Player_Car")?.transform;
        if (road1 == null) road1 = GameObject.Find("Road1")?.transform;
        if (road2 == null) road2 = GameObject.Find("Road2")?.transform;

        if (road1 != null && road2 != null)
        {
            SpriteRenderer sr = road1.GetComponent<SpriteRenderer>();
            roadHeight = sr != null ? sr.bounds.size.y : 10f;
        }

        currentScrollSpeed = baseScrollSpeed;
        currentSpawnInterval = baseSpawnInterval;

        if (playerCar != null)
        {
            playerCar.localScale = new Vector3(playerCarScale, playerCarScale, 1f);
            playerCar.rotation = Quaternion.identity;
            SpriteRenderer playerSr = playerCar.GetComponent<SpriteRenderer>();
            if (playerSr != null) playerSr.sortingOrder = 20;

            targetPosition = new Vector3(laneXPositions[currentLane], playerCar.position.y, playerCar.position.z);
            playerCar.position = targetPosition;
        }

        AssignButtonListener(titleQuitButton, ButtonActionQuit);
        AssignButtonListener(pauseResumeButton, ButtonActionResume);
        AssignButtonListener(pauseRestartButton, ButtonActionRestart);
        AssignButtonListener(pauseQuitButton, ButtonActionQuit);
        AssignButtonListener(gameOverRestartButton, ButtonActionRestart);
        AssignButtonListener(gameOverQuitButton, ButtonActionQuit);

        if (isHardwareModeActive) OpenBluetoothPort();

        SwitchState(GameState.TitleScreen);
    }

    void AssignButtonListener(Button targetButton, UnityEngine.Events.UnityAction action)
    {
        if (targetButton != null)
        {
            targetButton.onClick.RemoveAllListeners();
            targetButton.onClick.AddListener(action);
        }
    }

    void Update()
    {
        // TOGGLE HARDWARE MODE AT RUNTIME WITH 'Y' KEY
        if (Input.GetKeyDown(KeyCode.Y))
        {
            ToggleHardwareMode(!isHardwareModeActive);
        }

        // CLINICAL CALIBRATION: Press 'I' at runtime to invert tilt direction on the fly
        if (Input.GetKeyDown(KeyCode.I))
        {
            invertTiltDirection = !invertTiltDirection;
            Debug.Log("<color=orange>[CLINICAL CALIBRATION] Tilt Inversion set to: " + invertTiltDirection + "</color>");
        }

        switch (currentState)
        {
            case GameState.TitleScreen:
                if (Input.anyKeyDown && !Input.GetMouseButtonDown(0))
                {
                    SwitchState(GameState.Countdown);
                }
                else if (Input.GetMouseButtonDown(0))
                {
                    if (!UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject())
                        SwitchState(GameState.Countdown);
                }
                break;

            case GameState.Gameplay:
                HandleBackgroundScroll();
                HandlePlayerInput();
                HandlePlayerMovement();
                HandleTrafficSpawning();
                HandleTrafficMovementAndCollision();
                HandleScoringAndProgression();

                if (Input.GetKeyDown(KeyCode.Escape))
                {
                    SwitchState(GameState.Paused);
                }
                break;

            case GameState.Paused:
                if (Input.GetKeyDown(KeyCode.Escape))
                {
                    SwitchState(GameState.Gameplay);
                }
                break;
        }
    }

    void ToggleHardwareMode(bool enable)
    {
        isHardwareModeActive = enable;
        if (isHardwareModeActive)
        {
            Debug.Log("<color=cyan>[REHAB SYSTEM] Hardware Mode ACTIVATED. Connecting to " + bluetoothComPort + "</color>");
            OpenBluetoothPort();
        }
        else
        {
            Debug.Log("<color=yellow>[REHAB SYSTEM] Hardware Mode DEACTIVATED. Reverting to Keyboard control.</color>");
            CloseBluetoothPort();
        }
    }

    void HandlePlayerInput()
    {
        bool moveLeft = false;
        bool moveRight = false;

        // -------------------------------------------------------------
        // 1. HARDWARE INPUT PROCESSING (M5Stick Sensor Stream)
        // -------------------------------------------------------------
        if (isHardwareModeActive)
        {
            string rawData = "";
            lock (lockObject)
            {
                rawData = pendingTelemetryData;
            }

            if (!string.IsNullOrEmpty(rawData))
            {
                // Parse CSV telemetry string format "accX,btnState"
                string[] parts = rawData.Split(',');
                if (parts.Length >= 1 && float.TryParse(parts[0], out float accX))
                {
                    // CLINICAL INVERSION ENGINE: Flips the sign vector if the clinical toggle is active
                    if (invertTiltDirection)
                    {
                        accX *= -1f;
                    }

                    float currentTime = Time.time;

                    // Evaluate tilt threshold and debounce cooldown
                    if (!isTiltLatched && (currentTime - lastLaneShiftTime >= tiltDebounceDelay))
                    {
                        if (accX < -tiltThreshold)
                        {
                            moveLeft = true;
                            isTiltLatched = true;
                            lastLaneShiftTime = currentTime;
                        }
                        else if (accX > tiltThreshold)
                        {
                            moveRight = true;
                            isTiltLatched = true;
                            lastLaneShiftTime = currentTime;
                        }
                    }

                    // Reset tilt latch when patient returns wrist toward neutral center
                    if (isTiltLatched && Mathf.Abs(accX) < (tiltThreshold * 0.6f))
                    {
                        isTiltLatched = false;
                    }
                }
            }
        }
        // -------------------------------------------------------------
        // 2. KEYBOARD FALLBACK INPUT
        // -------------------------------------------------------------
        else
        {
            if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A)) moveLeft = true;
            if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D)) moveRight = true;
        }

        // Execute 6-Lane Position Logic
        if (moveLeft && currentLane > 0)
        {
            currentLane--;
            targetPosition = new Vector3(laneXPositions[currentLane], playerCar.position.y, playerCar.position.z);
        }
        if (moveRight && currentLane < laneXPositions.Length - 1)
        {
            currentLane++;
            targetPosition = new Vector3(laneXPositions[currentLane], playerCar.position.y, playerCar.position.z);
        }
    }

    // --- BLUETOOTH SERIAL THREADING SYSTEM ---
    void OpenBluetoothPort()
    {
        CloseBluetoothPort(); // Clear existing port handles
        try
        {
            sp = new SerialPort(bluetoothComPort, baudRate);
            sp.ReadTimeout = 1000;
            sp.Open();

            isThreadRunning = true;
            readThread = new Thread(ReadBluetoothThread);
            readThread.Start();
            Debug.Log("<color=green>[BLUETOOTH] Successfully connected to " + bluetoothComPort + "</color>");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("[BLUETOOTH] Unable to open " + bluetoothComPort + ". Error: " + e.Message);
        }
    }

    void ReadBluetoothThread()
    {
        while (isThreadRunning && sp != null && sp.IsOpen)
        {
            try
            {
                string line = sp.ReadLine().Trim();
                if (!string.IsNullOrEmpty(line))
                {
                    lock (lockObject)
                    {
                        pendingTelemetryData = line;
                    }
                }
            }
            catch (System.TimeoutException) { }
            catch (System.Exception e)
            {
                Debug.LogWarning("[THREAD] Serial Read Exception: " + e.Message);
            }
        }
    }

    void CloseBluetoothPort()
    {
        isThreadRunning = false;
        if (readThread != null && readThread.IsAlive)
        {
            readThread.Join(500);
        }
        if (sp != null && sp.IsOpen)
        {
            sp.Close();
            Debug.Log("[BLUETOOTH] Port closed cleanly.");
        }
    }

    void OnApplicationQuit()
    {
        CloseBluetoothPort();
    }

    void HandleBackgroundScroll()
    {
        if (road1 == null || road2 == null) return;

        road1.Translate(Vector3.down * currentScrollSpeed * Time.deltaTime);
        road2.Translate(Vector3.down * currentScrollSpeed * Time.deltaTime);

        if (road1.position.y <= -roadHeight)
        {
            road1.position = new Vector3(road1.position.x, road2.position.y + roadHeight, road1.position.z);
        }
        if (road2.position.y <= -roadHeight)
        {
            road2.position = new Vector3(road2.position.x, road1.position.y + roadHeight, road2.position.z);
        }
    }

    void HandlePlayerMovement()
    {
        if (playerCar == null) return;

        playerCar.localScale = new Vector3(playerCarScale, playerCarScale, 1f);
        targetPosition = new Vector3(laneXPositions[currentLane], playerCar.position.y, playerCar.position.z);
        playerCar.position = Vector3.MoveTowards(playerCar.position, targetPosition, laneMoveSpeed * Time.deltaTime);
    }

    void HandleTrafficSpawning()
    {
        if (currentState != GameState.Gameplay) return;
        if (trafficSprites == null || trafficSprites.Length == 0) return;

        spawnTimer += Time.deltaTime;
        if (spawnTimer >= currentSpawnInterval)
        {
            spawnTimer = 0f;
            int randomLane = Random.Range(0, laneXPositions.Length);
            int randomSpriteIndex = Random.Range(0, trafficSprites.Length);

            Vector3 spawnPos = new Vector3(laneXPositions[randomLane], 6f, 0f);

            GameObject newTraffic = new GameObject("Oncoming_Traffic_Car");
            newTraffic.transform.position = spawnPos;
            newTraffic.transform.rotation = Quaternion.Euler(0, 0, 180);

            newTraffic.transform.localScale = new Vector3(trafficCarScale, trafficCarScale, 1f);

            SpriteRenderer sr = newTraffic.AddComponent<SpriteRenderer>();
            sr.sprite = trafficSprites[randomSpriteIndex];
            sr.sortingOrder = 5;

            activeTraffic.Add(newTraffic);
        }
    }

    void HandleTrafficMovementAndCollision()
    {
        if (playerCar == null) return;

        for (int i = activeTraffic.Count - 1; i >= 0; i--)
        {
            GameObject traffic = activeTraffic[i];
            if (traffic == null) { activeTraffic.RemoveAt(i); continue; }

            traffic.transform.Translate(Vector3.up * (currentScrollSpeed * 0.5f) * Time.deltaTime);

            float dist = Vector3.Distance(playerCar.position, traffic.transform.position);

            if (dist < trafficHitboxRadius)
            {
                Vector3 crashPoint = (playerCar.position + traffic.transform.position) / 2f;

                if (explosionPrefab != null)
                {
                    GameObject fx = Instantiate(explosionPrefab, crashPoint, Quaternion.identity);
                    Destroy(fx, 1.5f);
                }

                // Randomized spin-out angles
                float randomPlayerAngle = Random.Range(15f, 45f) * (Random.value > 0.5f ? 1f : -1f);
                float randomTrafficAngle = Random.Range(15f, 45f) * (Random.value > 0.5f ? 1f : -1f);

                playerCar.transform.rotation = Quaternion.Euler(0, 0, randomPlayerAngle);
                traffic.transform.rotation = Quaternion.Euler(0, 0, 180f + randomTrafficAngle);

                SwitchState(GameState.GameOver);
                return;
            }

            if (traffic.transform.position.y < -6f)
            {
                activeTraffic.RemoveAt(i);
                Destroy(traffic);
            }
        }
    }

    void HandleScoringAndProgression()
    {
        survivalTimer += Time.deltaTime;
        score = survivalTimer * 10f;
        if (gameplayScoreText != null)
            gameplayScoreText.text = "Score: " + Mathf.FloorToInt(score).ToString();

        difficultyTimer += Time.deltaTime;
        if (difficultyTimer >= 60f)
        {
            difficultyTimer = 0f;
            difficultyLevel++;
            currentScrollSpeed = baseScrollSpeed + (difficultyLevel * 1.5f);
            currentSpawnInterval = Mathf.Max(0.8f, baseSpawnInterval - (difficultyLevel * 0.3f));
        }
    }

    void SwitchState(GameState newState)
    {
        currentState = newState;

        if (titleScreenPanel != null) titleScreenPanel.SetActive(newState == GameState.TitleScreen);
        if (pausePanel != null) pausePanel.SetActive(newState == GameState.Paused);
        if (gameOverPanel != null) gameOverPanel.SetActive(newState == GameState.GameOver);

        if (countdownText != null) countdownText.gameObject.SetActive(newState == GameState.Countdown);
        if (gameplayScoreText != null) gameplayScoreText.gameObject.SetActive(newState == GameState.Gameplay);

        Time.timeScale = (newState == GameState.Paused) ? 0f : 1f;

        switch (newState)
        {
            case GameState.TitleScreen:
                ResetGameplayVariables();
                break;

            case GameState.Countdown:
                ResetGameplayVariables();
                StartCoroutine(RunCountdownSequence());
                break;

            case GameState.GameOver:
                int finalScore = Mathf.FloorToInt(score);
                int savedHighScore = PlayerPrefs.GetInt("HighScore", 0);
                if (finalScore > savedHighScore)
                {
                    savedHighScore = finalScore;
                    PlayerPrefs.SetInt("HighScore", savedHighScore);
                    PlayerPrefs.Save();
                }
                if (finalScoreText != null) finalScoreText.text = "Final Score: " + finalScore.ToString();
                if (highScoreText != null) highScoreText.text = "High Score: " + savedHighScore.ToString();
                break;
        }
    }

    IEnumerator RunCountdownSequence()
    {
        if (countdownText == null)
        {
            SwitchState(GameState.Gameplay);
            yield break;
        }

        countdownText.gameObject.SetActive(true);
        countdownText.text = "READY";
        yield return new WaitForSeconds(1.0f);

        countdownText.text = "SET";
        yield return new WaitForSeconds(1.0f);

        countdownText.text = "GO!";
        yield return new WaitForSeconds(1.0f);

        SwitchState(GameState.Gameplay);
    }

    void ResetGameplayVariables()
    {
        score = 0f;
        survivalTimer = 0f;
        difficultyTimer = 0f;
        difficultyLevel = 1;
        currentScrollSpeed = baseScrollSpeed;
        currentSpawnInterval = baseSpawnInterval;
        currentLane = 3;
        spawnTimer = 0f;

        if (playerCar != null)
        {
            playerCar.localScale = new Vector3(playerCarScale, playerCarScale, 1f);
            playerCar.rotation = Quaternion.identity;
            targetPosition = new Vector3(laneXPositions[currentLane], playerCar.position.y, playerCar.position.z);
            playerCar.position = targetPosition;
        }

        foreach (GameObject t in activeTraffic)
        {
            if (t != null) Destroy(t);
        }
        activeTraffic.Clear();

        GameObject[] objectsInScene = GameObject.FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        foreach (GameObject obj in objectsInScene)
        {
            if (obj.name == "Oncoming_Traffic_Car")
            {
                Destroy(obj);
            }
        }
    }

    void ButtonActionResume() => SwitchState(GameState.Gameplay);
    void ButtonActionRestart() => SwitchState(GameState.Countdown);
    void ButtonActionQuit()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }
}