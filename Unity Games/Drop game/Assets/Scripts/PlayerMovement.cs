using UnityEngine;
using UnityEngine.InputSystem; // Added this to use the new system

[RequireComponent(typeof(Rigidbody2D))]
public class PlayerMovement : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 7f;

    private Rigidbody2D rb;
    private Vector2 movement;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        movement = Vector2.zero;

        // Check for WASD using the new Keyboard system
        if (Keyboard.current != null)
        {
            if (Keyboard.current.wKey.isPressed) movement.y += 1;
            if (Keyboard.current.sKey.isPressed) movement.y -= 1;
            if (Keyboard.current.dKey.isPressed) movement.x += 1;
            if (Keyboard.current.aKey.isPressed) movement.x -= 1;
        }

        movement = movement.normalized;
    }

    void FixedUpdate()
    {
        // Read the speed directly from the GameController!
        float currentSpeed = GameController.Instance.playerSpeed;
        rb.MovePosition(rb.position + movement * currentSpeed * Time.fixedDeltaTime);
    }
}