from typing import Any, Tuple
from functools import partial
import jax
from jax import jit, grad
import jax.numpy as jnp
from jax.experimental import optix
import haiku as hk 
from numpyro.distributions.discrete import Categorical
import gym
from gym.spaces import Discrete, Box

OptState = Any

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    def reward_to_go(rews):
        n = len(rews)
        rtgs = jnp.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    # make core of policy network
    def mlp(obs_dim, hidden_sizes, n_acts, x):
      net = hk.nets.MLP(output_sizes=[obs_dim]+hidden_sizes+[n_acts], activation=jnp.tanh)
      return net(x)
    
    logits_net = hk.transform(partial(mlp, obs_dim, hidden_sizes, n_acts))
    
    # make function to compute action distribution
    def get_policy(params:hk.Params, 
                    obs:Any):
        logits = logits_net.apply(params, obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    @jit
    def get_action(key, params, obs):
        return get_policy(params, obs).sample(key)

    # make loss function whose gradient, for the right data, is policy gradient
    @jit
    def compute_loss(params, obs, act, returns):
        logp = get_policy(params, obs).log_prob(act)
        return jnp.mean(-(logp * returns))

    # make optimizer
    opt_init, opt_update = optix.adam(lr)

    #step update
    @jit
    def update(params: hk.Params,
          opt_state: OptState,
          batch_obs: jnp.DeviceArray,
          batch_acts: jnp.DeviceArray,
          batch_returns:jnp.DeviceArray) -> Tuple[hk.Params, OptState, jnp.DeviceArray]:
                  
      batch_loss, g = jax.value_and_grad(compute_loss)(params,
                               batch_obs,
                               batch_acts,
                               batch_returns)
      updates, opt_state = opt_update(g, opt_state)
      new_params = optix.apply_updates(params, updates)
      return new_params, opt_state, batch_loss
        

    # for training policy
    def train_one_epoch(params, opt_state, key):
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_returns = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
          
            key, _ = jax.random.split(key, 2)
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(key, params, jnp.asarray(obs, dtype=jnp.float32))
            act = int(act)
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_returns += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        params, opt_state, batch_loss = update(params,
                                              opt_state,
                                              jnp.asarray(batch_obs, dtype=jnp.float32),
                                              jnp.asarray(batch_acts, dtype=jnp.int32),
                                              jnp.asarray(batch_returns, dtype=jnp.float32))
        return batch_loss, batch_rets, batch_lens, params, opt_state

    key = jax.random.PRNGKey(42)
    keychain = jax.random.split(key, epochs)
    sample_ts = env.reset()
    ts_with_batch = jax.tree_map(lambda t: jnp.expand_dims(t, 0), sample_ts)
    params = logits_net.init(jax.random.PRNGKey(3), sample_ts)
    opt_state = opt_init(params)
    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens, params, opt_state = train_one_epoch(params, opt_state, keychain[i])
        print(f'epoch: {i:3}\t loss: {batch_loss:.4}\t return: {mean(batch_rets):.4} \t ep_len: {mean(batch_lens):.4}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)