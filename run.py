import argparse
from utils import get_env, get_agent, visualize_matrix, display_q_value, save_q_value
from utils import run_algorithm, test_algorithm


def get_arguments():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('-env', type=str, default='SmallGridEnv',
                        choices=['SmallGridEnv', 'BigGridEnv', 'UnistEnv'])

    # Agent
    parser.add_argument('-agent', type=str, default='MCAgent',
                        choices=['MCAgent', 'SARSAAgent', 'QlearningAgent'])
    parser.add_argument('-epsilon', type=float, default=1.0)
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-gamma', type=float, default=0.995)

    # Experiment option
    parser.add_argument('-n_episode', type=int, default=10000)
    parser.add_argument('-n_tick', type=int, default=1000)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-save_dir_plot', type=str, default='./results/plot/')
    parser.add_argument('-save_dir_value', type=str, default='./results/q_value/')
    parser.add_argument('-memo', type=str, default='')

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()

    env = get_env(args)
    agent = get_agent(args, n_state=env.observation_space.n, n_action=env.action_space.n)
    visualize_matrix(env.world, title=args.env, save_path=args.save_dir_plot + args.env + '.png')

    env, agent = run_algorithm(args, env, agent)
    filename = args.env + '_' + args.agent + args.memo
    save_q_value(agent.Q, save_path=args.save_dir_value + filename)
    display_q_value(agent.Q, env, title=args.agent, save_path=args.save_dir_plot + filename + '.png')
    # test_algorithm(args, env, agent)


if __name__ == "__main__":
    main()
