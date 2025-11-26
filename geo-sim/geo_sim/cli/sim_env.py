from geo_sim.cli.env import CSSMovementsEnv

env = CSSMovementsEnv()

observations, infos = env.reset()

# Example: define one policy per agent
def group_policy(obs, info, group):
    # Here we just move randomly as a placeholder
    return env.action_space(group).sample()

done = False
while True:
    # 1. Get actions from each agent’s policy
    actions = {}
    for agent in env.agents:              # usually ["prisoner", "guard"] until game over
        obs = observations[agent]
        info = infos[agent]
        actions[agent] = group_policy(obs, info, agent)

    # 2. Step the environment with those actions
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 3. Optional: render / log / train
    env.render()

    # 4. Check if episode is over
    if any(terminations.values()) or all(truncations.values()):
        break
