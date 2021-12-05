import argparse
from env import GridEnv

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = argparse.ArgumentParser()
    return parser

def main():
    parser = get_arguments()
    args = parser.parse_args()

    env = GridEnv()

    for i_episode in range(50):
        env.reset()
        print("Reset env.")
        env.render()

        for t in range(200):
            action = env.action_space.sample()
            print("action: {}".format(action))
            obs, reward, done, info = env.step(action)
            print("obs: {}, reward:{}, done:{}".format(obs, reward, done))
            env.render()

            env.step_ghost()

            if done:
                print("Episode finished after {} timesteps\n".format(t + 1))
                break

    env.close()


if __name__ == "__main__":
    main()
