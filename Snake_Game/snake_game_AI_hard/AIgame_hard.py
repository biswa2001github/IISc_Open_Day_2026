import torch
from agent import Agent  # Agent contains get_state() and the model
from game import SnakeGameAI  # The game used during training

def run_ai_game(num_episodes=3):
    # Create an agent and load the saved model.
    agent = Agent()
    try:
        agent.model.load_state_dict(torch.load('./model/model.pth'))
        print("Model loaded successfully.")
    except Exception as e:
        print("Could not load model:", e)
        return
    agent.model.eval()  # Set the model to evaluation mode.

    # Run for a fixed number of episodes.
    for episode in range(1, num_episodes + 1):
        game = SnakeGameAI()  # Create a new game instance for each episode.
        print(f"Starting episode {episode}...")
        while True:
            # Get the current state using the agent's method.
            state = agent.get_state(game)
            state_tensor = torch.tensor(state, dtype=torch.float)
            
            with torch.no_grad():
                prediction = agent.model(state_tensor)
            # Choose the action with the highest Q-value.
            move = torch.argmax(prediction).item()
            # Convert the move (0,1,2) into a one-hot action vector.
            action = [0, 0, 0]
            action[move] = 1
            
            # Perform the game step using the predicted action.
            reward, game_over, score = game.play_step(action)
            
            if game_over:
                print(f"Episode {episode} finished. Final Score: {score}")
                break

if __name__ == '__main__':
    run_ai_game(num_episodes=3)