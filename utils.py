import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

from envs import SmallGridEnv, BigGridEnv, UnistEnv
from agents import MCAgent, SARSAAgent, QlearningAgent, DoubleQlearningAgent, \
    LinearApprox, REINFORCE, ActorCritic


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_env(args):
    env_name = args.env
    if env_name == "SmallGridEnv":
        return SmallGridEnv()
    elif env_name == "BigGridEnv":
        return BigGridEnv()
    elif env_name == "UnistEnv":
        return UnistEnv()
    else:
        raise Exception("There is no env '{}'".format(env_name))


def get_agent(args,
              n_state,
              n_action):
    agent_name = args.agent
    if agent_name == "MCAgent":
        return MCAgent(n_state, n_action, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    elif agent_name == "SARSAAgent":
        return SARSAAgent(n_state, n_action, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    elif agent_name == "QlearningAgent":
        return QlearningAgent(n_state, n_action, epsilon=args.epsilon, alpha=args.alpha,
                              gamma=args.gamma, seed=args.seed)
    elif agent_name == "DoubleQlearningAgent":
        return DoubleQlearningAgent(n_state, n_action, epsilon=args.epsilon, alpha=args.alpha,
                                    gamma=args.gamma, seed=args.seed)
    elif agent_name == "LinearApprox":
        return LinearApprox(n_state, n_action, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    elif agent_name == "REINFORCE":
        return REINFORCE(n_state, n_action, lr=args.lr, gamma=args.gamma)
    elif agent_name == "ActorCritic":
        return ActorCritic(n_state, n_action, lr=args.lr, gamma=args.gamma)
    else:
        raise Exception("There is no agent '{}'".format(agent_name))


def run_algorithm(args,
                  env,
                  agent):
    agent_name = args.agent
    episode = tqdm(range(args.n_episode), desc="episode")
    episode_rewards = np.zeros(args.n_episode)

    if agent_name == "MCAgent":
        for e_idx in episode:
            state = env.reset()
            action = agent.get_action(state)
            done = False
            while not done:
                state_prime, reward, done, info = env.step(action)   # step
                episode_rewards[e_idx] += reward
                next_action = agent.get_action(state_prime)          # Get next action
                agent.save_sample(state, action, reward, done)       # Store samples
                state = state_prime
                action = next_action
                if args.step_ghost:
                    env.step_ghost()
            # End of the episode
            agent.update_q()    # Update Q value using sampled episode
            agent.update_epsilon(100 / (e_idx + 1))     # Decaying epsilon
    elif (agent_name == "SARSAAgent") or (agent_name == "QlearningAgent") or (agent_name == "DoubleQlearningAgent"):
        for e_idx in episode:
            state = env.reset()
            action = agent.get_action(state)
            done = False
            while not done:
                state_prime, reward, done, info = env.step(action)   # step
                action_prime = agent.get_action(state_prime)         # Get next action
                episode_rewards[e_idx] += reward
                if agent_name == "SARSAAgent":
                    agent.update_q(state, action, reward, state_prime, action_prime, done)
                else:   # (agent_name == "QlearningAgent") or (agent_name == "DoubleQlearningAgent")
                    agent.update_q(state, action, reward, state_prime, done)
                state = state_prime
                action = action_prime
                if args.step_ghost:
                    env.step_ghost()
            agent.update_epsilon(100 / (e_idx + 1))     # Decaying epsilon
    elif agent_name == "LinearApprox":
        for e_idx in episode:
            state = env.reset()
            state = agent.featurize_state(state)
            action = agent.get_action(state)
            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_state = agent.featurize_state(next_state)
                next_action = agent.get_action(next_state)
                episode_rewards[e_idx] += reward

                # Update weight
                td_target = reward + agent.gamma * agent.Q(next_state, next_action)
                td_error = agent.Q(state, action) - td_target
                dw = td_error.dot(state)
                agent.w[action] -= agent.alpha * dw

                state = next_state
                action = next_action
                if args.step_ghost:
                    env.step_ghost()

            agent.update_alpha((1 - (e_idx + 1) / args.n_episode) * 0.5)
            agent.update_epsilon((1 - (e_idx + 1) / args.n_episode) * 0.5)
    elif agent_name == "REINFORCE":
        for n_epi in episode:
            state = env.reset()
            state = agent.featurize_state(state)
            done = False
            while not done:
                action, prob = agent.get_action(state)
                prob = prob.squeeze()
                next_state, reward, done, _ = env.step(action)
                next_state = agent.featurize_state(next_state)
                agent.save((reward, prob[action]))
                state = next_state
                episode_rewards[n_epi] += reward
            agent.update()
    elif agent_name == "ActorCritic":
        for n_epi in episode:
            state = env.reset()
            state = agent.featurize_state(state)
            done = False
            while not done:
                for t in range(args.update_step):
                    action = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = agent.featurize_state(next_state)
                    agent.save((state, action, reward, next_state, done))
                    state = next_state
                    episode_rewards[n_epi] += reward
                    if done:
                        break
                agent.update()
    else:
        raise Exception("There is no agent '{}'".format(agent_name))

    return env, agent, episode_rewards


def test_algorithm(args,
                   env,
                   agent):

    return env, agent


def visualize_matrix(M,
                     cmap='Pastel1',
                     title='Title',
                     title_fs=15,
                     save_path='',
                     REMOVE_TICK_LABELS=True):
    n_row, n_col = M.shape[0], M.shape[1]
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    im = ax.imshow(M, cmap=plt.get_cmap(cmap), extent=(0, n_col, n_row, 0),
                   interpolation='nearest', aspect='equal')
    ax.set_xticks(np.arange(0, n_col, 1))
    ax.set_yticks(np.arange(0, n_row, 1))
    ax.grid(color='w', linewidth=2)
    ax.set_frame_on(False)
    cax = plt.colorbar(im, cax=divider.append_axes('right', size='5%', pad=0.05), orientation='vertical')
    cax.set_ticks([-1, 0, 1, 2, 3])
    cax.set_ticklabels(['Agent', 'Empty', 'Star', 'Ghost', 'Wall'])
    fig.suptitle(title, size=title_fs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if REMOVE_TICK_LABELS:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    plt.show()
    if save_path != '':
        plt.savefig(save_path)
    cax = plt.gca()
    cax.set_xticklabels([])
    cax.set_yticklabels([])
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def plot_pi_v(Pi,
              V,
              title='',
              cmap='viridis',
              title_fs=15,
              REMOVE_TICK_LABELS=True):
    n_row, n_col = V.shape[0], V.shape[1]
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    im = ax.imshow(V, cmap=plt.get_cmap(cmap), extent=(0, n_col, n_row, 0))
    ax.set_xticks(np.arange(0, n_col, 1))
    ax.set_yticks(np.arange(0, n_row, 1))
    ax.grid(color='w', linewidth=2)

    # arrow plot
    arr_len = 0.2
    for i in range(4):
        for j in range(4):
            s = i * 4 + j
            if Pi[s][0] > 0: plt.arrow(j + 0.5, i + 0.5, -arr_len, 0,
                                       color="r", alpha=Pi[s][0], width=0.01,
                                       head_width=0.5, head_length=0.2, overhang=1)
            if Pi[s][1] > 0: plt.arrow(j + 0.5, i + 0.5, 0, arr_len,
                                       color="r", alpha=Pi[s][1], width=0.01,
                                       head_width=0.5, head_length=0.2, overhang=1)
            if Pi[s][2] > 0: plt.arrow(j + 0.5, i + 0.5, arr_len, 0,
                                       color="r", alpha=Pi[s][2], width=0.01,
                                       head_width=0.5, head_length=0.2, overhang=1)
            if Pi[s][3] > 0: plt.arrow(j + 0.5, i + 0.5, 0, -arr_len,
                                       color="r", alpha=Pi[s][3], width=0.01,
                                       head_width=0.5, head_length=0.2, overhang=1)

    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.suptitle(title, size=title_fs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if REMOVE_TICK_LABELS:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    plt.show()


def plot_rewards(n_episodes,
                 episode_rewards,
                 save_path):
    plt.figure()
    plt.plot(np.arange(n_episodes), episode_rewards)
    plt.savefig(save_path)


def display_q_value(agent,
                    env,
                    title='',
                    fig_size=8,
                    text_fs=8,
                    title_fs=15,
                    save_path=''):
    try:
        Q = agent.Q
    except:
        print("There is no Q value in `DoubleQlearningAgent`. "
              "So, use `agent.Q_A` instead of `agent.Q`.")
        Q = agent.Q_A
    n_state, n_action = Q.shape

    # Triangle patches for each action
    lft_tri = np.array([[0, 0], [-0.5, -0.5], [-0.5, 0.5]])
    dw_tri = np.array([[0, 0], [-0.5, 0.5], [0.5, 0.5]])
    up_tri = np.array([[0, 0], [0.5, -0.5], [-0.5, -0.5]])
    rgh_tri = np.array([[0, 0], [0.5, 0.5], [0.5, -0.5]])

    high_color = np.array([1.0, 0.0, 0.0, 0.8])
    low_color = np.array([1.0, 1.0, 1.0, 0.8])

    plt.figure(figsize=(fig_size, fig_size))
    plt.title(title, fontsize=title_fs)

    for i in range(env.world_shape[0]):
        for j in range(env.world_shape[1]):
            if not [i, j] in env.wall_pos:
                s = env.get_selected_obs([i, j])
                min_q = np.min(Q[s])
                max_q = np.max(Q[s])
                for a in range(n_action):
                    q_value = Q[s, a]
                    ratio = (q_value - min_q) / (max_q - min_q + 1e-10)
                    if ratio > 1:
                        clr = high_color
                    elif ratio < 0:
                        clr = low_color
                    else:
                        clr = high_color * ratio + low_color * (1 - ratio)
                    if a == 0:  # Up arrow
                        plt.gca().add_patch(plt.Polygon([j, i] + up_tri, color=clr, ec='k'))
                        plt.text(j - 0.0, i - 0.25, "%.2f" % (q_value), fontsize=text_fs, va='center', ha='center')
                    if a == 1:  # Down arrow
                        plt.gca().add_patch(plt.Polygon([j, i] + dw_tri, color=clr, ec='k'))
                        plt.text(j - 0.0, i + 0.25, "%.2f" % (q_value), fontsize=text_fs, va='center', ha='center')
                    if a == 2:  # Left arrow
                        plt.gca().add_patch(plt.Polygon([j, i] + lft_tri, color=clr, ec='k'))
                        plt.text(j - 0.25, i + 0.0, "%.2f" % (q_value), fontsize=text_fs, va='center', ha='center')
                    if a == 3:  # Right arrow
                        plt.gca().add_patch(plt.Polygon([j, i] + rgh_tri, color=clr, ec='k'))
                        plt.text(j + 0.25, i + 0.0, "%.2f" % (q_value), fontsize=text_fs, va='center', ha='center')

    plt.xlim([-0.5, env.world_shape[0] - 0.5])
    plt.xticks(range(env.world_shape[0]))
    plt.ylim([-0.5, env.world_shape[1] - 0.5])
    plt.yticks(range(env.world_shape[1]))
    plt.gca().invert_yaxis()
    plt.show()
    if save_path != '':
        plt.savefig(save_path)


def save_q_value(agent,
                 save_path=''):
    if save_path != '':
        try:
            np.save(file=save_path, arr=agent.Q)
        except:
            np.savez(save_path, Q_A=agent.Q_A, Q_B=agent.Q_B)


def display_frames_as_gif(frames,
                          save_path):
    patch = plt.imshow(frames[0])

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames))
    anim.save(save_path + '.gif', writer='imagemagick', fps=5, dpi=100)
    plt.close(anim._fig)


def make_animation(args,
                   eval_env,
                   agent,
                   save_path):
    state, done, ep_ret, ep_len = eval_env.reset(), False, 0, 0
    frames = []
    while not done:
        frame = visualize_matrix(eval_env.world, title=args.env + '_' + args.agent)
        frames.append(frame)
        if "Agent" in args.agent:
            action = agent.get_action(state)
        else:
            if args.agent == "LinearApprox":
                action = agent.get_action(agent.featurize_state(state), test=True)
            elif args.agent == "REINFORCE":
                action, prob = agent.get_action(agent.featurize_state(state))
            elif args.agent == "ActorCritic":
                action = agent.get_action(agent.featurize_state(state))
        state, reward, done, _ = eval_env.step(action)
        ep_ret += reward
        ep_len += 1
        if args.step_ghost:
            eval_env.step_ghost()
        if ep_len > 1000:
            print("Too many frames (maybe infinite loop)")
            break
    display_frames_as_gif(frames, save_path)
    print("animation creation is completed.")
