# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create a posterior log-density surface visualization on a cloud TPU."""

import os
from jax import numpy as jnp
import numpy as onp
import jax
import argparse
import functools
import tqdm
import sys
import ast

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import precision_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import tree_utils
from bnn_hmc.utils import data_utils
from bnn_hmc.utils import models
from bnn_hmc.utils import losses
from bnn_hmc.utils import script_utils  # pytype: disable=import-error

from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Compute posterior surface plots")
    cmd_args_utils.add_common_flags(parser)
    parser.add_argument(
        "--limit_bottom",
        type=float,
        default=-0.75,
        help="Limit of the loss surface visualization along the"
             "vertical direction at the bottom")
    parser.add_argument(
        "--limit_top",
        type=float,
        default=1.75,
        help="Limit of the loss surface visualization along the"
             "vertical direction at the top")
    parser.add_argument(
        "--limit_left",
        type=float,
        default=-0.75,
        help="Limit of the loss surface visualization along the"
             "horizontal direction on the left")
    parser.add_argument(
        "--limit_right",
        type=float,
        default=1.75,
        help="Limit of the loss surface visualization along the"
             "horizontal direction on the right")
    parser.add_argument(
        "--grid_size",
        type=int,
        default=50,
        help="Number of grid points in each direction")
    parser.add_argument(
        "--checkpoint1",
        type=str,
        help="Path to the first checkpoint")
    parser.add_argument(
        "--checkpoint2",
        type=str,
        help="Path to the second checkpoint")
    parser.add_argument(
        "--checkpoint3",
        type=str,
        help="Path to the third checkpoint")
    parser.add_argument(
        "--plot",
        action="store_true")
    return parser.parse_args()


def get_u_v_o(params1, params2, params3):
    u_params = tree_utils.tree_diff(params2, params1)
    u_norm = tree_utils.tree_norm(u_params)
    u_params = tree_utils.tree_scalarmul(u_params, 1 / u_norm)
    v_params = tree_utils.tree_diff(params3, params1)
    uv_dot = tree_utils.tree_dot(u_params, v_params)
    v_params = jax.tree_multimap(lambda v, u: v - uv_dot * u, v_params, u_params)
    v_norm = tree_utils.tree_norm(v_params)
    v_params = tree_utils.tree_scalarmul(v_params, 1 / v_norm)

    return u_params, u_norm, v_params, v_norm, params1


def load_params(path):
    checkpoint_dict = checkpoint_utils.load_checkpoint(path)
    return checkpoint_dict["params"]


def run_visualization(args):
    print(args)
    subdirname = "posterior_visualization"
    dirname = os.path.join(args.dir, subdirname)
    os.makedirs(dirname, exist_ok=True)
    cmd_args_utils.save_cmd(dirname, None)

    # Initialize data, model, losses and metrics
    (train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn,
     log_prior_fn, _, predict_fn, ensemble_upd_fn, metrics_fns,
     tabulate_metrics) = script_utils.get_data_model_fns(args)

    def eval(params, net_state, dataset):
        likelihood, _ = log_likelihood_fn(net_apply, params, net_state, dataset,
                                          True)
        prior = log_prior_fn(params)
        likelihood = jax.lax.psum(likelihood, axis_name="i")
        log_prob = likelihood + prior

        return log_prob, likelihood, prior

    params1 = load_params(args.checkpoint1)
    params2 = load_params(args.checkpoint2)
    params3 = load_params(args.checkpoint3)

    for params in [params1, params2, params3]:
        net_state, test_predictions = onp.asarray(
            predict_fn(net_apply, params, net_state, train_set))
        stats = train_utils.evaluate_metrics(test_predictions, train_set[1],
                                                  metrics_fns)
        print(stats)
        net_state, test_predictions = onp.asarray(
            predict_fn(net_apply, params, net_state, test_set))
        stats = train_utils.evaluate_metrics(test_predictions, test_set[1],
                                             metrics_fns)
        print(stats)

    print('Log likes:')
    for params in [params1, params2, params3]:
        print(jax.pmap(eval, axis_name='i', in_axes=(None, None, 0))
              (params, net_state, train_set))

    u_vec, u_norm, v_vec, v_norm, origin = get_u_v_o(params1, params2, params3)

    u_ts = onp.linspace(args.limit_left, args.limit_right, args.grid_size)
    v_ts = onp.linspace(args.limit_bottom, args.limit_top, args.grid_size)
    n_u, n_v = len(u_ts), len(v_ts)
    log_probs = onp.zeros((n_u, n_v))
    log_likelihoods = onp.zeros((n_u, n_v))
    log_priors = onp.zeros((n_u, n_v))
    grid = onp.zeros((n_u, n_v, 2))

    @functools.partial(jax.pmap, axis_name="i", in_axes=(None, 0))
    def eval_row_of_plot(u_t_, dataset):
        def loop_body(_, v_t_):
            params = jax.tree_multimap(
                lambda u, v, o: o + u * u_t_ * u_norm + v * v_t_ * v_norm, u_vec,
                v_vec, origin)
            logprob, likelihood, prior = eval(params, net_state, dataset)
            arr = jnp.array([logprob, likelihood, prior])
            return None, arr

        _, vals = jax.lax.scan(loop_body, None, v_ts)
        row_logprobs, row_likelihoods, row_priors = jnp.split(vals, [1, 2], axis=1)
        return row_logprobs, row_likelihoods, row_priors

    for u_i, u_t in enumerate(tqdm.tqdm(u_ts)):
        log_probs_i, likelihoods_i, priors_i = eval_row_of_plot(u_t, train_set)
        log_probs_i, likelihoods_i, priors_i = map(
            lambda arr: arr[0], [log_probs_i, likelihoods_i, priors_i])
        log_probs[u_i] = log_probs_i[:, 0]
        log_likelihoods[u_i] = likelihoods_i[:, 0]
        log_priors[u_i] = priors_i[:, 0]
        grid[u_i, :, 0] = onp.array([u_t] * n_v)
        grid[u_i, :, 1] = v_ts

    onp.savez(
        os.path.join(dirname, "surface_plot.npz"),
        log_probs=log_probs,
        log_priors=log_priors,
        log_likelihoods=log_likelihoods,
        grid=grid,
        u_norm=u_norm,
        v_norm=v_norm)

    if args.plot:
        plt.contour(grid[:, :, 0], grid[:, :, 1], log_probs, zorder=1)
        plt.contourf(grid[:, :, 0], grid[:, :, 1], log_probs, zorder=0, alpha=0.55)
        plt.plot([0., 1., 0.5], [0., 0., 1.], "ro", ms=20, markeredgecolor="k")
        plt.colorbar()
        plt.savefig(os.path.join(dirname, "log_prob.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    cmd_args = get_args()
    train_utils.set_up_jax(cmd_args.tpu_ip, cmd_args.use_float64)
    script_utils.print_visible_devices()
    run_visualization(cmd_args)
