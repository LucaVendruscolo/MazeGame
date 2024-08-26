from gymnasium.envs.registration import register

register(
    id='BallGame',  # Use an env naming convention similar to others in Gym
    entry_point='sheeprl.envs.BallGame.BallEnv',  # Replace `path.to.your` with the path to the module containing your class
)