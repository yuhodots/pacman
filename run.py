import argparse
from utils import get_env, get_agent
from utils import visualize_matrix, display_q_value, save_q_value, plot_rewards, make_animation, str2bool
from utils import run_algorithm, test_algorithm


def get_arguments():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('-env', type=str, default='SmallGridEnv',
                        choices=['SmallGridEnv', 'BigGridEnv', 'UnistEnv'])

    # Agent
    parser.add_argument('-agent', type=str, default='MCAgent',
                        choices=['MCAgent', 'SARSAAgent', 'QlearningAgent', 'DoubleQlearningAgent',
                                 'LinearApprox'])
    parser.add_argument('-epsilon', type=float, default=1.0)
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-gamma', type=float, default=0.995)

    # Experiment option
    parser.add_argument('-step_ghost', type=str2bool, default=False)
    parser.add_argument('-n_episode', type=int, default=10000)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-save_dir_plot', type=str, default='./results/grid_plot/')
    parser.add_argument('-save_dir_value', type=str, default='./results/q_value/')
    parser.add_argument('-save_dir_reward', type=str, default='./results/reward/')
    parser.add_argument('-save_dir_animate', type=str, default='./results/animation/')
    parser.add_argument('-memo', type=str, default='')

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()

    env = get_env(args)
    agent = get_agent(args, n_state=env.observation_space.n, n_action=env.action_space.n)
    visualize_matrix(env.world, title=args.env, save_path=args.save_dir_plot + args.env + '.png')

    env, agent, rewards = run_algorithm(args, env, agent)
    filename = args.env + '_' + args.agent + args.memo
    plot_rewards(args.n_episode, rewards, save_path=args.save_dir_reward + filename + '_rewards.png')
    make_animation(args, env, agent, save_path=args.save_dir_animate + filename)
    # test_algorithm(args, env, agent)

    if not ("Approx" in args.agent):
        save_q_value(agent, save_path=args.save_dir_value + filename)
        display_q_value(agent, env, title=args.agent, save_path=args.save_dir_plot + filename + '_q-value.png')
    print("Code execution is complete.")


if __name__ == "__main__":
    main()
