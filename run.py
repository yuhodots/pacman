import argparse
from tqdm import tqdm
from utils import get_env, get_agent, visualize_matrix, display_q_value


def get_arguments():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('-env', type=str, default='SmallGridEnv',
                        choices=['SmallGridEnv', 'BigGridEnv', 'UnistEnv'])

    # Agent
    parser.add_argument('-agent', type=str, default='MCAgent',
                        choices=['MCAgent'])
    parser.add_argument('-epsilon', type=float, default=1.0)
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-gamma', type=float, default=0.995)

    # Experiment option
    parser.add_argument('-n_episode', type=int, default=10000)
    parser.add_argument('-n_tick', type=int, default=1000)
    parser.add_argument('-seed', type=int, default=42)

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()

    env = get_env(args)
    agent = get_agent(args, n_state=env.observation_space.n, n_action=env.action_space.n)

    for e_idx in tqdm(range(args.n_episode), desc="episode"):
        state = env.reset()     # reset environment, select initial
        action = agent.get_action(state)
        done = False
        while not done:
            next_state, reward, done, info = env.step(action)   # step
            next_action = agent.get_action(next_state)          # Get next action
            agent.save_sample(state, action, reward, done)      # Store samples
            state = next_state
            action = next_action
        # End of the episode
        agent.update_q()    # Update Q value using sampled episode
        agent.update_epsilon(100 / (e_idx + 1))     # Decaying epsilon

    env.reset()
    visualize_matrix(env.world, strs='', cmap='Pastel1', title='SmallGridEnv')
    display_q_value(agent.Q, env, title="Monte Carlo Policy Iteration",
                    fig_size=8, text_fs=8, title_fs=15)


if __name__ == "__main__":
    main()
