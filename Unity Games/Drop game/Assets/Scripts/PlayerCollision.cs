using UnityEngine;

public class PlayerCollision : MonoBehaviour
{
    private void OnTriggerEnter2D(Collider2D collision)
    {
        // Did we hit a Good Drop?
        if (collision.gameObject.CompareTag("GoodDrop"))
        {
            // Tell the Game Controller to add 1 point
            GameController.Instance.AddScore(1);

            // Destroy the blue ball
            Destroy(collision.gameObject);
        }
        // Did we hit a Bad Drop?
        else if (collision.gameObject.CompareTag("BadDrop"))
        {
            // Tell the Game Controller we lost
            GameController.Instance.LoseGame();
        }
    }
}