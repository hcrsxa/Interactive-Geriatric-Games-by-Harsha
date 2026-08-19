using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;
using UnityEngine.UI;

public class GameController : MonoBehaviour
{
    // --- SINGLETON PATTERN ---
    // This allows any other script to easily find the GameController by typing GameController.Instance
    public static GameController Instance { get; private set; }

    [Header("Game Stats & Settings")]
    public int scoreToWin = 10;
    public float playerSpeed = 7f;
    public float playerRadius = 0.5f;   // Used for our custom math collision
    public float dropRadius = 0.5f;     // Used for our custom math collision

    [Header("Animation Settings")]
    public float baseDropSize = 1f;       // The normal size of the prefab
    public float goodPulseSpeed = 3f;     // How fast it breathes
    public float goodPulseAmount = 0.15f; // How much it grows/shrinks
    public float badPulseSpeed = 15f;     // Fast, erratic heartbeat
    public float badPulseAmount = 0.25f;  // Aggressive size change

    // --- GAME STATE VARIABLES ---
    private int currentScore = 0;
    private bool isGameOver = false;
    private bool isPaused = false;
    private bool inMenu = true;           // Keeps track of whether we are in the main menu

    [Header("Game Objects & Prefabs")]
    public GameObject playerObject;
    public GameObject goodDropPrefab;
    public GameObject badDropPrefab;
    public GameObject explosionPrefab;
    public GameObject goodEffectPrefab;

    [Header("Spawn & Boundary Settings")]
    public float spawnInterval = 1.5f;    // Time between drop waves
    public float spawnWidth = 8f;         // How far left/right drops can spawn
    public float spawnY = 6f;             // Base height for spawning
    public float spawnYRange = 4f;        // Adds randomness to the height so they don't fall in a flat line
    public float bottomY = -6f;           // The invisible floor where drops self-destruct

    [Header("UI Panels")]
    public GameObject menuPanel;
    public GameObject instructionsPanel;
    public GameObject pausePanel;
    public GameObject gameplayUI;

    [Header("UI Text")]
    public TextMeshProUGUI scoreText;
    public TextMeshProUGUI messageText;

    [Header("UI Buttons")]
    public Button startButton;
    public Button instructionsButton;
    public Button backButton;
    public Button resumeButton;
    public Button restartButton;

    // --- TRACKING LISTS ---
    // We store every drop that spawns in this list so we can manually move them and check collisions
    private List<GameObject> activeDrops = new List<GameObject>();
    private float spawnTimer = 0f;

    // --- INPUT VARIABLES ---
    [HideInInspector] public bool KeyPressedA;
    [HideInInspector] public bool KeyPressedW;
    [HideInInspector] public bool KeyPressedS;
    [HideInInspector] public bool KeyPressedD;
    [HideInInspector] public bool KeyPressedSpace;
    [HideInInspector] public bool KeyPressedESC;
    [HideInInspector] public bool KeyPressedR;
    [HideInInspector] public bool MouseButtonDown;
    [HideInInspector] public bool MouseButtonUp;
    [HideInInspector] public bool MouseClicked;

    // Awake runs before Start(). It sets up our Singleton.
    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    // Start runs once when the game loads.
    void Start()
    {
        // 1. Hook up all the buttons through code instead of the Inspector
        if (startButton != null) startButton.onClick.AddListener(StartGame);
        if (instructionsButton != null) instructionsButton.onClick.AddListener(ShowInstructions);
        if (backButton != null) backButton.onClick.AddListener(HideInstructions);
        if (resumeButton != null) resumeButton.onClick.AddListener(ResumeGame);
        if (restartButton != null) restartButton.onClick.AddListener(RestartGame);

        // 2. Make sure the game starts on the Menu screen
        ShowMenu();
    }

    // Update runs every single frame (approx 60 times per second)
    void Update()
    {
        // 1. Always grab the latest player input first
        GetInputs();

        // 2. If the game is over, only listen for the Restart key
        if (isGameOver)
        {
            if (KeyPressedR) RestartGame();
            return; // Stops the rest of the Update loop from running
        }

        // 3. Handle Pausing with the ESC key (Only if we are actually playing!)
        if (!inMenu && !isGameOver && KeyPressedESC)
        {
            if (isPaused) ResumeGame();
            else PauseGame();
        }

        // 4. Run the core gameplay logic (Only if playing and not paused)
        if (!inMenu && !isPaused && !isGameOver)
        {
            HandlePlayerMovement();
            HandleSpawning();
            HandleDropsAndCollisions();
        }
    }

    // Grabs raw input data from Unity's old input system and stores it in our variables
    void GetInputs()
    {
        KeyPressedA = Input.GetKey(KeyCode.A);
        KeyPressedW = Input.GetKey(KeyCode.W);
        KeyPressedS = Input.GetKey(KeyCode.S);
        KeyPressedD = Input.GetKey(KeyCode.D);
        KeyPressedSpace = Input.GetKeyDown(KeyCode.Space);
        KeyPressedESC = Input.GetKeyDown(KeyCode.Escape);
        KeyPressedR = Input.GetKeyDown(KeyCode.R);

        MouseButtonDown = Input.GetMouseButtonDown(0);
        MouseButtonUp = Input.GetMouseButtonUp(0);
        MouseClicked = MouseButtonDown;
    }

    // --- UI CONTROLS ---

    // Turns on the Menu UI and freezes the game world
    public void ShowMenu()
    {
        inMenu = true;
        menuPanel.SetActive(true);
        instructionsPanel.SetActive(false);
        pausePanel.SetActive(false);
        gameplayUI.SetActive(false);
        Time.timeScale = 0f; // timeScale 0 stops all physics and animations
    }

    // Hides the menus, resets the score, and unfreezes the game
    public void StartGame()
    {
        inMenu = false;
        isPaused = false;
        isGameOver = false;
        currentScore = 0;

        menuPanel.SetActive(false);
        instructionsPanel.SetActive(false);
        pausePanel.SetActive(false);
        gameplayUI.SetActive(true);

        if (messageText != null) messageText.text = "";
        UpdateScoreText();

        Time.timeScale = 1f; // timeScale 1 restores normal speed
    }

    public void ShowInstructions()
    {
        menuPanel.SetActive(false);
        instructionsPanel.SetActive(true);
    }

    public void HideInstructions()
    {
        instructionsPanel.SetActive(false);
        menuPanel.SetActive(true);
    }

    // Freezes the game and shows the Pause panel
    public void PauseGame()
    {
        isPaused = true;
        pausePanel.SetActive(true);
        Time.timeScale = 0f;
    }

    // Unfreezes the game and hides the Pause panel
    public void ResumeGame()
    {
        isPaused = false;
        pausePanel.SetActive(false);
        Time.timeScale = 1f;
    }

    // Reloads the entire scene from scratch
    public void RestartGame()
    {
        Time.timeScale = 1f;
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    // --- CORE GAMEPLAY LOGIC ---

    // Translates WASD key presses into actual movement for the player ball
    void HandlePlayerMovement()
    {
        if (playerObject == null) return;

        Vector3 move = Vector3.zero; // Start with no movement (0,0,0)

        // Add directions based on keys held down
        if (KeyPressedW) move.y += 1;
        if (KeyPressedS) move.y -= 1;
        if (KeyPressedA) move.x -= 1;
        if (KeyPressedD) move.x += 1;

        // .normalized ensures diagonal movement isn't faster than straight movement
        // Time.deltaTime ensures movement is smooth regardless of framerate
        move = move.normalized * playerSpeed * Time.deltaTime;

        // Apply the movement to the player's position
        playerObject.transform.position += move;
    }

    // Handles generating new drops
    void HandleSpawning()
    {
        spawnTimer += Time.deltaTime; // Count up the timer

        // If the timer reaches the interval, it's time to spawn!
        if (spawnTimer >= spawnInterval)
        {
            spawnTimer = 0f; // Reset timer

            // Decide how many drops to spawn in this wave (1 to 3)
            int dropsToSpawn = Random.Range(1, 4);

            for (int i = 0; i < dropsToSpawn; i++)
            {
                // Calculate a random X and staggered Y position
                float randomX = Random.Range(-spawnWidth, spawnWidth);
                float randomYOffset = Random.Range(0f, spawnYRange);
                float staggeredY = spawnY + randomYOffset;

                Vector3 spawnPos = new Vector3(randomX, staggeredY, 0f);

                // 80% chance to be a Good Drop, 20% chance to be a Bad Drop
                bool isGoodDrop = Random.value > 0.2f;
                GameObject dropToSpawn = isGoodDrop ? goodDropPrefab : badDropPrefab;

                // Create the drop in the game world
                GameObject newDrop = Instantiate(dropToSpawn, spawnPos, Quaternion.identity);
                newDrop.tag = isGoodDrop ? "GoodDrop" : "BadDrop"; // Tag it for collision checks

                // Add it to our master list so the game knows it exists
                activeDrops.Add(newDrop);
            }
        }
    }

    // Manages the animations, boundaries, and collisions for all active drops
    void HandleDropsAndCollisions()
    {
        if (playerObject == null) return;

        // CRITICAL: We loop through the list backwards. 
        // If we loop forwards and delete an item, the list shifts and causes errors.
        for (int i = activeDrops.Count - 1; i >= 0; i--)
        {
            GameObject drop = activeDrops[i];

            // Safety check: if the drop was somehow destroyed elsewhere, remove it from list
            if (drop == null) { activeDrops.RemoveAt(i); continue; }

            // --- ANIMATION (THE HEARTBEAT PULSE) ---
            // We use Mathf.Sin to create a smooth wave pattern based on time
            if (drop.CompareTag("GoodDrop"))
            {
                float newScale = baseDropSize + Mathf.Sin(Time.time * goodPulseSpeed) * goodPulseAmount;
                drop.transform.localScale = new Vector3(newScale, newScale, 1f);
            }
            else if (drop.CompareTag("BadDrop"))
            {
                float newScale = baseDropSize + Mathf.Sin(Time.time * badPulseSpeed) * badPulseAmount;
                drop.transform.localScale = new Vector3(newScale, newScale, 1f);
            }

            // --- BOUNDARY CHECK (Hitting the floor) ---
            if (drop.transform.position.y <= bottomY)
            {
                // Spawn an explosion/effect just before it deletes
                if (drop.CompareTag("BadDrop") && explosionPrefab != null)
                {
                    Instantiate(explosionPrefab, drop.transform.position, Quaternion.identity);
                }
                else if (drop.CompareTag("GoodDrop") && goodEffectPrefab != null)
                {
                    Instantiate(goodEffectPrefab, drop.transform.position, Quaternion.identity);
                }

                Destroy(drop);            // Remove object from game
                activeDrops.RemoveAt(i);  // Remove object from list
                continue;                 // Skip to the next drop in the loop
            }

            // --- MANUAL COLLISION CHECK (Custom Hitbox Math) ---
            // Calculate the distance between the center of the player and the center of the drop
            float distance = Vector3.Distance(playerObject.transform.position, drop.transform.position);

            // If the distance is smaller than their two radii combined, they are touching!
            if (distance <= (playerRadius + dropRadius))
            {
                if (drop.CompareTag("GoodDrop"))
                {
                    AddScore(1);
                    if (goodEffectPrefab != null)
                    {
                        Instantiate(goodEffectPrefab, drop.transform.position, Quaternion.identity);
                    }
                }
                else if (drop.CompareTag("BadDrop"))
                {
                    LoseGame(); // Hit a bad drop, trigger game over
                }

                Destroy(drop);
                activeDrops.RemoveAt(i);
            }
        }
    }

    // --- GAME RULES ---

    public void AddScore(int amount)
    {
        currentScore += amount;
        UpdateScoreText();
        if (currentScore >= scoreToWin) WinGame();
    }

    public void LoseGame()
    {
        isGameOver = true;
        if (messageText != null) messageText.text = "GAME OVER!\nHit a Red Drop.\nPress 'R' to Restart";

        // Spawn a massive explosion on the player and hide them
        if (explosionPrefab != null && playerObject != null)
        {
            Instantiate(explosionPrefab, playerObject.transform.position, Quaternion.identity);
            playerObject.SetActive(false);
        }

        Time.timeScale = 0f; // Freeze game
    }

    private void WinGame()
    {
        isGameOver = true;
        if (messageText != null) messageText.text = "YOU WIN!\nGot 10 Drops!\nPress 'R' to Restart";
        Time.timeScale = 0f; // Freeze game
    }

    private void UpdateScoreText()
    {
        if (scoreText != null) scoreText.text = "Score: " + currentScore;
    }
}