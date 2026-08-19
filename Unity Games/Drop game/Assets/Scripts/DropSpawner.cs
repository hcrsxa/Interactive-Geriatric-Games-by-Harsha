using UnityEngine;

public class DropSpawner : MonoBehaviour
{
    [Header("Drop Prefabs")]
    public GameObject goodDropPrefab;
    public GameObject badDropPrefab;

    [Header("Spawn Settings")]
    public float spawnInterval = 1.5f;
    public float spawnWidth = 8f;      // Left/Right range for all drops
    public float spawnHeight = 4f;     // Up/Down range for Good Drops

    private float timer = 0f;

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= spawnInterval)
        {
            SpawnDrop();
            timer = 0f;
        }
    }

    void SpawnDrop()
    {
        // 80% chance for a Good Drop, 20% chance for a Bad Drop
        bool isGoodDrop = Random.value > 0.2f;

        if (isGoodDrop)
        {
            // Good Drops spawn floating anywhere in the designated area
            float randomX = Random.Range(-spawnWidth, spawnWidth);
            float randomY = Random.Range(-spawnHeight, spawnHeight);
            Vector3 spawnPosition = new Vector3(randomX, randomY, 0f);

            Instantiate(goodDropPrefab, spawnPosition, Quaternion.identity);
        }
        else
        {
            // Bad Drops still spawn up high at the Spawner's exact Y position
            float randomX = Random.Range(-spawnWidth, spawnWidth);
            Vector3 spawnPosition = new Vector3(randomX, transform.position.y, 0f);

            Instantiate(badDropPrefab, spawnPosition, Quaternion.identity);
        }
    }
}