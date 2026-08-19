using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using TMPro;
using UnityEngine.UI;
using System.IO.Ports;
using System.Threading;

public class GameController : MonoBehaviour
{
    // --------------------------------------------------------
    // VARIABLES & SETTINGS
    // --------------------------------------------------------

    [Header("Hardware Controller (M5Stick)")]
    public bool useHardwareController = false;
    public string comPort = "COM3";
    public float tiltSensitivity = 10f;

    [Header("Main UI Elements")]
    public TextMeshProUGUI scoreTextDisplay;
    public TextMeshProUGUI timerTextDisplay;

    [Header("Menu & Game Over UI")]
    public GameObject menuPanel;
    public GameObject instructionsPanel;
    public GameObject gameOverPanel;
    public TextMeshProUGUI gameOverTextDisplay;
    public TextMeshProUGUI countdownTextDisplay;

    [Header("Menu Buttons")]
    public Button startButton;
    public Button instructionsButton;
    public Button backButton;

    [Header("Player Settings")]
    public float playerSpeed = 10f;
    public float playerScale = 1f;
    public int health = 5;
    public float playerFixedY = -4f;

    [Header("Shooting Settings")]
    public float shootInterval = 0.3f;
    public float bulletSpeed = 15f;
    public float bulletHitboxSize = 1f;
    public float bulletScale = 0.5f;
    public float bulletRotation = 90f;

    [Header("Game Settings")]
    public float gameTimer = 30f;
    public float spawnInterval = 1f;
    public float dropSpeed = 5f;
    public float hitboxSize = 1.5f;

    [Header("Background Scrolling")]
    public Transform background1;
    public Transform background2;
    public float scrollSpeed = 3f;
    public float backgroundHeight = 10.5f;

    [Header("Assets")]
    public Sprite playerSprite;
    public Sprite goodDropSprite;
    public Sprite badDropSprite;
    public Sprite heartSprite;
    public Sprite emptyHeartSprite;
    public Sprite bulletSprite;

    // --------------------------------------------------------
    // INTERNAL TRACKING
    // --------------------------------------------------------
    private GameObject playerObject;
    private List<GameObject> activeDrops = new List<GameObject>();
    private List<GameObject> activeHearts = new List<GameObject>();
    private List<GameObject> activeBullets = new List<GameObject>();

    private float spawnTimer = 0f;
    private float shootTimer = 0f;
    private int score = 0;
    private bool isGameOver = false;
    private bool inMenu = true;
    private bool isCountingDown = false;

    // --- HARDWARE TRACKING ---
    private SerialPort serialPort;
    private Thread serialThread;
    private float hwTilt = 0f;
    private bool hwShoot = false;
    private bool keepReading = false;

    // --------------------------------------------------------
    // CORE GAME LOOP
    // --------------------------------------------------------

    void Start()
    {
        if (startButton != null) startButton.onClick.AddListener(OnStartButtonClicked);
        if (instructionsButton != null) instructionsButton.onClick.AddListener(ShowInstructions);
        if (backButton != null) backButton.onClick.AddListener(HideInstructions);

        SetupPlayer();
        SetupHearts();
        UpdateUI();

        // Initial connection attempt if box is checked before play
        if (useHardwareController) ConnectToHardware();

        ShowMenu();
    }

    void Update()
    {
        if (inMenu || isCountingDown) return;

        if (isGameOver)
        {
            // CRITICAL FIX: Only uses the New Input System now!
            if (Keyboard.current != null && Keyboard.current.rKey.wasPressedThisFrame)
            {
                RestartGame();
            }
            return;
        }

        // --- IN-GAME HARDWARE TOGGLE ---
        // CRITICAL FIX: Only uses the New Input System now!
        if (Keyboard.current != null && Keyboard.current.yKey.wasPressedThisFrame)
        {
            useHardwareController = !useHardwareController;
            Debug.Log("Hardware Mode Toggled: " + useHardwareController);

            // If we just turned hardware mode ON, and the port isn't open yet, open it!
            if (useHardwareController && (serialPort == null || !serialPort.IsOpen))
            {
                ConnectToHardware();
            }
        }

        HandleTimer();
        HandleBackgroundScroll();
        HandlePlayerMovement();
        HandleSpawning();
        HandleShooting();
        HandleBulletsAndAsteroids();
        HandleDropsAndCollisions();
    }

    // --------------------------------------------------------
    // HARDWARE LOGIC (M5STICK COMMUNICATION)
    // --------------------------------------------------------

    void ConnectToHardware()
    {
        try
        {
            serialPort = new SerialPort(comPort, 115200);

            // CRITICAL FIX 1: Never wait more than 50 milliseconds for data.
            serialPort.ReadTimeout = 50;

            serialPort.Open();
            keepReading = true;

            serialThread = new Thread(ReadSerialLoop);
            // CRITICAL FIX 2: Tells Unity it's safe to force-kill this thread if you hit Stop.
            serialThread.IsBackground = true;
            serialThread.Start();

            Debug.Log("Connected to M5Stick on " + comPort);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to connect to hardware: " + e.Message);
            useHardwareController = false;
        }
    }

    void ReadSerialLoop()
    {
        while (keepReading && serialPort != null && serialPort.IsOpen)
        {
            try
            {
                string data = serialPort.ReadLine();
                string[] parts = data.Split(',');

                if (parts.Length == 2)
                {
                    float.TryParse(parts[0], out hwTilt);

                    int btn;
                    int.TryParse(parts[1], out btn);
                    hwShoot = (btn == 1);
                }
            }
            catch (System.TimeoutException)
            {
                // CRITICAL FIX 3: This is normal! It just means no data arrived in 50ms. 
                // We catch it silently so the loop can keep spinning without crashing.
            }
            catch
            {
                // Prevents a total CPU meltdown if the Bluetooth abruptly disconnects
                Thread.Sleep(10);
            }
        }
    }

    void OnDestroy()
    {
        keepReading = false;

        // CRITICAL FIX 4: We MUST close the port BEFORE we try to join the thread. 
        // This violently interrupts any stuck ReadLine() operations so Unity can shut down safely.
        if (serialPort != null && serialPort.IsOpen)
        {
            try { serialPort.Close(); } catch { }
        }

        if (serialThread != null && serialThread.IsAlive)
        {
            serialThread.Join(500);
        }
    }

    // --------------------------------------------------------
    // MENU & COUNTDOWN LOGIC
    // --------------------------------------------------------

    void ShowMenu()
    {
        inMenu = true;
        if (menuPanel != null) menuPanel.SetActive(true);
        if (instructionsPanel != null) instructionsPanel.SetActive(false);
        if (gameOverPanel != null) gameOverPanel.SetActive(false);
        if (countdownTextDisplay != null) countdownTextDisplay.gameObject.SetActive(false);

        if (playerObject != null) playerObject.SetActive(false);
    }

    void ShowInstructions()
    {
        if (menuPanel != null) menuPanel.SetActive(false);
        if (instructionsPanel != null) instructionsPanel.SetActive(true);
    }

    void HideInstructions()
    {
        if (instructionsPanel != null) instructionsPanel.SetActive(false);
        if (menuPanel != null) menuPanel.SetActive(true);
    }

    void OnStartButtonClicked()
    {
        if (menuPanel != null) menuPanel.SetActive(false);
        StartCoroutine(CountdownSequence());
    }

    private IEnumerator CountdownSequence()
    {
        inMenu = false;
        isCountingDown = true;

        if (countdownTextDisplay != null)
        {
            countdownTextDisplay.gameObject.SetActive(true);

            countdownTextDisplay.text = "READY";
            yield return new WaitForSeconds(1f);

            countdownTextDisplay.text = "SET";
            yield return new WaitForSeconds(1f);

            countdownTextDisplay.text = "GO!";
            yield return new WaitForSeconds(1f);

            countdownTextDisplay.gameObject.SetActive(false);
        }

        StartGame();
    }

    void StartGame()
    {
        isCountingDown = false;
        if (playerObject != null) playerObject.SetActive(true);
    }

    // --------------------------------------------------------
    // GAME LOGIC METHODS
    // --------------------------------------------------------

    void HandleBackgroundScroll()
    {
        if (background1 == null || background2 == null) return;
        background1.position += Vector3.down * scrollSpeed * Time.deltaTime;
        background2.position += Vector3.down * scrollSpeed * Time.deltaTime;

        if (background1.position.y <= -backgroundHeight)
            background1.position = new Vector3(0, background2.position.y + backgroundHeight, 0);

        if (background2.position.y <= -backgroundHeight)
            background2.position = new Vector3(0, background1.position.y + backgroundHeight, 0);
    }

    void SetupHearts()
    {
        float startX = -2f;
        float spacingX = 1f;
        float positionY = -4.5f;

        for (int i = 0; i < health; i++)
        {
            GameObject heart = new GameObject("Heart_" + i);
            SpriteRenderer sr = heart.AddComponent<SpriteRenderer>();
            sr.sprite = heartSprite;
            sr.sortingOrder = 10;

            if (sr.sprite == null)
            {
                sr.color = Color.red;
                heart.transform.localScale = new Vector3(0.4f, 0.4f, 1f);
            }
            else heart.transform.localScale = new Vector3(1f, 1f, 1f);

            heart.transform.position = new Vector3(startX + (i * spacingX), positionY, 0f);
            activeHearts.Add(heart);
        }
    }

    void HandleTimer()
    {
        gameTimer -= Time.deltaTime;
        if (gameTimer <= 0)
        {
            gameTimer = 0;
            EndGame("TIME UP!");
        }
        UpdateUI();
    }

    void HandleShooting()
    {
        if (playerObject == null || !playerObject.activeSelf) return;

        bool wantsToShoot = useHardwareController ? hwShoot : true;

        if (wantsToShoot)
        {
            shootTimer += Time.deltaTime;
            if (shootTimer >= shootInterval)
            {
                shootTimer = 0f;

                GameObject bullet = new GameObject("Bullet");
                bullet.transform.position = playerObject.transform.position + new Vector3(0, 0.5f, 0);

                SpriteRenderer sr = bullet.AddComponent<SpriteRenderer>();
                sr.sprite = bulletSprite;
                sr.sortingOrder = 4;

                if (sr.sprite == null)
                {
                    sr.color = Color.yellow;
                    bullet.transform.localScale = new Vector3(0.2f, 0.6f, 1f);
                }
                else
                {
                    bullet.transform.localScale = new Vector3(bulletScale, bulletScale, 1f);
                    bullet.transform.rotation = Quaternion.Euler(0, 0, bulletRotation);
                }

                activeBullets.Add(bullet);
            }
        }
        else
        {
            shootTimer = shootInterval;
        }
    }

    void HandleBulletsAndAsteroids()
    {
        for (int i = activeBullets.Count - 1; i >= 0; i--)
        {
            GameObject bullet = activeBullets[i];
            if (bullet == null) { activeBullets.RemoveAt(i); continue; }

            bullet.transform.position += Vector3.up * bulletSpeed * Time.deltaTime;

            if (bullet.transform.position.y >= 8f)
            {
                Destroy(bullet);
                activeBullets.RemoveAt(i);
                continue;
            }

            bool bulletDestroyed = false;

            for (int j = activeDrops.Count - 1; j >= 0; j--)
            {
                GameObject drop = activeDrops[j];
                if (drop == null) continue;

                if (drop.CompareTag("BadDrop"))
                {
                    float dist = Vector3.Distance(bullet.transform.position, drop.transform.position);

                    if (dist < bulletHitboxSize)
                    {
                        score += 2;
                        UpdateUI();

                        Destroy(drop);
                        activeDrops.RemoveAt(j);

                        Destroy(bullet);
                        activeBullets.RemoveAt(i);

                        bulletDestroyed = true;
                        break;
                    }
                }
            }

            if (bulletDestroyed) continue;
        }
    }

    void HandleSpawning()
    {
        spawnTimer += Time.deltaTime;
        if (spawnTimer >= spawnInterval)
        {
            spawnTimer = 0f;
            CreateDrop();
        }
    }

    void CreateDrop()
    {
        bool isGood = Random.value > 0.4f;
        GameObject drop = new GameObject(isGood ? "GoodDrop" : "BadDrop");
        drop.transform.position = new Vector3(Random.Range(-8f, 8f), 6f, 0f);

        SpriteRenderer sr = drop.AddComponent<SpriteRenderer>();
        sr.sprite = isGood ? goodDropSprite : badDropSprite;
        sr.sortingOrder = 5;

        if (sr.sprite == null)
        {
            sr.color = isGood ? Color.blue : Color.red;
            drop.transform.localScale = Vector3.one * 0.5f;
        }

        drop.tag = isGood ? "GoodDrop" : "BadDrop";
        activeDrops.Add(drop);
    }

    void HandleDropsAndCollisions()
    {
        for (int i = activeDrops.Count - 1; i >= 0; i--)
        {
            GameObject drop = activeDrops[i];
            if (drop == null) { activeDrops.RemoveAt(i); continue; }

            drop.transform.position += Vector3.down * dropSpeed * Time.deltaTime;

            if (drop.transform.position.y <= -6f)
            {
                Destroy(drop);
                activeDrops.RemoveAt(i);
                continue;
            }

            if (playerObject != null)
            {
                float dist = Vector3.Distance(playerObject.transform.position, drop.transform.position);

                if (dist < hitboxSize)
                {
                    if (drop.CompareTag("GoodDrop"))
                    {
                        score++;
                        UpdateUI();
                    }
                    else if (drop.CompareTag("BadDrop"))
                    {
                        health--;
                        if (health >= 0 && health < activeHearts.Count)
                        {
                            SpriteRenderer heartRenderer = activeHearts[health].GetComponent<SpriteRenderer>();
                            if (emptyHeartSprite != null) heartRenderer.sprite = emptyHeartSprite;
                            else heartRenderer.color = Color.gray;
                        }
                        if (health <= 0) EndGame("HULL CRITICAL!");
                    }

                    Destroy(drop);
                    activeDrops.RemoveAt(i);
                }
            }
        }
    }

    void UpdateUI()
    {
        if (scoreTextDisplay != null) scoreTextDisplay.text = "Score: " + score;
        if (timerTextDisplay != null) timerTextDisplay.text = "Time: " + Mathf.CeilToInt(gameTimer) + "s";
    }

    void EndGame(string reason)
    {
        isGameOver = true;
        if (gameOverTextDisplay != null) gameOverTextDisplay.text = reason + "\nFinal Score: " + score + "\n\nPress R to Restart";
        if (gameOverPanel != null) gameOverPanel.SetActive(true);
        if (playerObject != null) playerObject.SetActive(false);
    }

    void RestartGame()
    {
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    void SetupPlayer()
    {
        playerObject = new GameObject("PlayerShip");
        SpriteRenderer sr = playerObject.AddComponent<SpriteRenderer>();
        sr.sprite = playerSprite;
        sr.sortingOrder = 10;
        playerObject.transform.localScale = new Vector3(playerScale, playerScale, 1f);
        if (sr.sprite == null) sr.color = Color.green;
    }

    void HandlePlayerMovement()
    {
        if (playerObject == null || Camera.main == null) return;

        Vector3 targetPos = playerObject.transform.position;

        if (useHardwareController)
        {
            float mappedX = hwTilt * tiltSensitivity;
            mappedX = Mathf.Clamp(mappedX, -8f, 8f);

            targetPos = new Vector3(mappedX, playerFixedY, 0f);
        }
        else
        {
            Vector2 screenMousePos = Mouse.current.position.ReadValue();
            float cameraDepth = Mathf.Abs(Camera.main.transform.position.z);
            targetPos = Camera.main.ScreenToWorldPoint(new Vector3(screenMousePos.x, screenMousePos.y, cameraDepth));

            targetPos.z = 0f;
        }

        playerObject.transform.position = Vector3.Lerp(playerObject.transform.position, targetPos, playerSpeed * Time.deltaTime);
    }
}