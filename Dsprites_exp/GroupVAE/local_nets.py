#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: local_nets.py
# --- Creation Date: 21-09-2020
# --- Last Modified: Fri 02 Oct 2020 02:46:38 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Networks.
"""
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import math
import tensorflow as tf
from utils_fn import split_latents
slim = tf.contrib.slim


def group_decoder1_64(z,
                      scope="DEC",
                      n_act_points=10,
                      output_channel=1,
                      group_feats_size=400,
                      nconti=6,
                      ncat=3,
                      reuse=False,
                      is_train=False):
    nets_dict = dict()
    nets_dict['input'] = z
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d_transpose],
                                weights_initializer=tf.contrib.slim.
                                variance_scaling_initializer(),
                                stride=2,
                                padding='SAME',
                                activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.fully_connected],
                                    biases_initializer=tf.zeros_initializer()):

                    lie_alg_basis_ls = []
                    latent_dim = nconti
                    mat_dim = int(math.sqrt(group_feats_size))
                    for i in range(latent_dim):
                        init = tf.initializers.random_normal(0, 0.01)
                        lie_alg_tmp = tf.get_variable(
                            'lie_alg_' + str(i),
                            shape=[1, mat_dim, mat_dim],
                            initializer=init)
                        lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
                        lie_alg_tmp = lie_alg_tmp - tf.transpose(
                            lie_alg_tmp, perm=[0, 2, 1])
                        lie_alg_basis_ls.append(lie_alg_tmp)
                    lie_alg_basis = tf.concat(
                        lie_alg_basis_ls,
                        axis=0)[tf.newaxis,
                                ...]  # [1, lat_dim, mat_dim, mat_dim]
                    nets_dict['lie_alg_basis'] = lie_alg_basis

                    input_conti = nets_dict['input'][:, :nconti]
                    input_cat = nets_dict['input'][:, nconti:]
                    lie_alg_mul = input_conti[
                        ..., tf.newaxis, tf.newaxis] * nets_dict[
                            'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]

                    nets_dict['lie_alg'] = tf.reduce_sum(
                        lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
                    nets_dict['lie_group_mat'] = tf.linalg.expm(
                        nets_dict['lie_alg'])  # [b, mat_dim, mat_dim]

                    # lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
                    scale_cat = slim.fully_connected(
                        input_cat,
                        mat_dim,
                        activation_fn=None,
                        scope='scale_cat')  # [b, mat_dim]
                    scale_cat_alg = tf.eye(mat_dim, dtype=scale_cat.dtype)[tf.newaxis, ...] * \
                        scale_cat[..., tf.newaxis] # [b, mat_dim, mat_dim]
                    nets_dict['scale_group'] = tf.linalg.expm(
                        scale_cat_alg)  # [b, mat_dim, mat_dim]
                    act_init = tf.initializers.random_normal(0, 0.01)
                    act_points = tf.get_variable(
                        'act_points',
                        shape=[1, mat_dim, n_act_points],
                        initializer=act_init)
                    # nets_dict['act_points'] = tf.matmul(nets_dict['scale_group'],
                    # act_points) # [b, mat_dim, n_act_points]
                    nets_dict['act_points'] = act_points
                    # transed_act_points = tf.matmul(nets_dict['lie_group_mat'], nets_dict['act_points'])
                    # transed_act_points_tensor = tf.reshape(transed_act_points,
                    # [-1, mat_dim * n_act_points])
                    # nets_dict['act_points_transed'] = transed_act_points_tensor
                    nets_dict['act_points'] = tf.matmul(
                        nets_dict['lie_group_mat'], nets_dict['scale_group'])
                    nets_dict['act_points_transed'] = tf.reshape(
                        nets_dict['act_points'], [-1, mat_dim * mat_dim])

                    nets_dict['fc0'] = slim.fully_connected(
                        nets_dict['act_points_transed'],
                        256,
                        activation_fn=tf.nn.relu,
                        scope="fc0")
                    nets_dict['fc1'] = slim.fully_connected(
                        nets_dict['fc0'],
                        4 * 4 * 64,
                        activation_fn=tf.nn.relu,
                        scope="fc1")
                    n = tf.reshape(nets_dict['fc1'], [-1, 4, 4, 64])
                    nets_dict['deconv2d0'] = slim.conv2d_transpose(
                        n, 64, [4, 4], scope='deconv2d_0')
                    nets_dict['deconv2d1'] = slim.conv2d_transpose(
                        nets_dict['deconv2d0'], 32, [4, 4], scope='deconv2d_1')
                    nets_dict['deconv2d2'] = slim.conv2d_transpose(
                        nets_dict['deconv2d1'], 32, [4, 4], scope='deconv2d_2')
                    nets_dict['output'] = slim.conv2d_transpose(
                        nets_dict['deconv2d2'],
                        output_channel, [4, 4],
                        activation_fn=None,
                        scope='deconv2d_3')
                    return nets_dict

def group_spl_decoder1_64(z,
                          scope="DEC",
                          n_act_points=10,
                          output_channel=1,
                          group_feats_size=400,
                          nconti=6,
                          ncat=3,
                          reuse=False,
                          is_train=False):
    nets_dict = dict()
    nets_dict['input'] = z
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d_transpose],
                                weights_initializer=tf.contrib.slim.
                                variance_scaling_initializer(),
                                stride=2,
                                padding='SAME',
                                activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.fully_connected],
                                    biases_initializer=tf.zeros_initializer()):

                    lie_alg_basis_ls = []
                    latent_dim = nconti

                    mat_dim = int(math.sqrt(group_feats_size))
                    for i in range(latent_dim):
                        init = tf.initializers.random_normal(0, 0.01)
                        lie_alg_tmp = tf.get_variable(
                            'lie_alg_' + str(i),
                            shape=[1, mat_dim, mat_dim],
                            initializer=init)
                        lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
                        lie_alg_tmp = lie_alg_tmp - tf.transpose(
                            lie_alg_tmp, perm=[0, 2, 1])
                        lie_alg_basis_ls.append(lie_alg_tmp)
                    lie_alg_basis = tf.concat(
                        lie_alg_basis_ls,
                        axis=0)[tf.newaxis,
                                ...]  # [1, lat_dim, mat_dim, mat_dim]
                    nets_dict['lie_alg_basis'] = lie_alg_basis

                    input_conti = nets_dict['input'][:, :nconti]
                    input_cat = nets_dict['input'][:, nconti:]
                    latents_in_cut_ls = split_latents(input_conti,
                                                      hy_ncut=1)  # [x0, x1]

                    def train_fn(latents_in_cut_ls, nets_dict):
                        lie_alg_mul_0 = latents_in_cut_ls[0][
                            ..., tf.newaxis, tf.newaxis] * nets_dict[
                                'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]
                        lie_alg_mul_1 = latents_in_cut_ls[1][
                            ..., tf.newaxis, tf.newaxis] * nets_dict[
                                'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]
                        lie_alg_0 = tf.reduce_sum(lie_alg_mul_0, axis=1)  # [b, mat_dim, mat_dim]
                        lie_alg_1 = tf.reduce_sum(lie_alg_mul_1, axis=1)  # [b, mat_dim, mat_dim]
                        lie_alg = lie_alg_0 + lie_alg_1
                        lie_group_0 = tf.linalg.expm(lie_alg_0)  # [b, mat_dim, mat_dim]
                        lie_group_1 = tf.linalg.expm(lie_alg_1)  # [b, mat_dim, mat_dim]
                        lie_group_mat = tf.matmul(lie_group_0, lie_group_1)
                        return lie_alg, lie_group_mat


                    def val_fn(input_conti, nets_dict):
                        lie_alg_mul = input_conti[..., tf.newaxis, tf.newaxis] * nets_dict[
                            'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]
                        lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
                        lie_group_mat = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
                        return lie_alg, lie_group_mat

                    nets_dict['lie_alg'], nets_dict['lie_group_mat'] = \
                        tf.cond(is_train, train_fn, val_fn)

                    # if not is_train:
                    # lie_alg_mul = input_conti[
                    # ..., tf.newaxis, tf.newaxis] * nets_dict[
                    # 'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]
                    # nets_dict['lie_alg'] = tf.reduce_sum(
                    # lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
                    # nets_dict['lie_group_mat'] = tf.linalg.expm(
                    # nets_dict['lie_alg'])  # [b, mat_dim, mat_dim]
                    # else:
                    # lie_alg_mul_0 = latents_in_cut_ls[0][
                    # ..., tf.newaxis, tf.newaxis] * nets_dict[
                    # 'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]
                    # lie_alg_mul_1 = latents_in_cut_ls[1][
                    # ..., tf.newaxis, tf.newaxis] * nets_dict[
                    # 'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]
                    # lie_alg_0 = tf.reduce_sum(
                    # lie_alg_mul_0, axis=1)  # [b, mat_dim, mat_dim]
                    # lie_alg_1 = tf.reduce_sum(
                    # lie_alg_mul_1, axis=1)  # [b, mat_dim, mat_dim]
                    # nets_dict['lie_alg'] = lie_alg_0 + lie_alg_1
                    # lie_group_0 = tf.linalg.expm(
                    # lie_alg_0)  # [b, mat_dim, mat_dim]
                    # lie_group_1 = tf.linalg.expm(
                    # lie_alg_1)  # [b, mat_dim, mat_dim]
                    # nets_dict['lie_group_mat'] = tf.matmul(
                    # lie_group_0, lie_group_1)

                    # lie_alg_mul = input_conti[
                    # ..., tf.newaxis, tf.newaxis] * nets_dict[
                    # 'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]

                    # nets_dict['lie_alg'] = tf.reduce_sum(
                    # lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
                    # nets_dict['lie_group_mat'] = tf.linalg.expm(
                    # nets_dict['lie_alg'])  # [b, mat_dim, mat_dim]

                    # lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
                    scale_cat = slim.fully_connected(
                        input_cat,
                        mat_dim,
                        activation_fn=None,
                        scope='scale_cat')  # [b, mat_dim]
                    scale_cat_alg = tf.eye(mat_dim, dtype=scale_cat.dtype)[tf.newaxis, ...] * \
                        scale_cat[..., tf.newaxis] # [b, mat_dim, mat_dim]
                    nets_dict['scale_group'] = tf.linalg.expm(
                        scale_cat_alg)  # [b, mat_dim, mat_dim]
                    # act_init = tf.initializers.random_normal(0, 0.01)
                    # act_points = tf.get_variable(
                    # 'act_points',
                    # shape=[1, mat_dim, n_act_points],
                    # initializer=act_init)
                    # nets_dict['act_points'] = tf.matmul(nets_dict['scale_group'],
                    # act_points) # [b, mat_dim, n_act_points]
                    # nets_dict['act_points'] = act_points
                    # transed_act_points = tf.matmul(nets_dict['lie_group_mat'], nets_dict['act_points'])
                    # transed_act_points_tensor = tf.reshape(transed_act_points,
                    # [-1, mat_dim * n_act_points])
                    # nets_dict['act_points_transed'] = transed_act_points_tensor
                    nets_dict['act_points'] = tf.matmul(
                        nets_dict['lie_group_mat'], nets_dict['scale_group'])
                    nets_dict['act_points_transed'] = tf.reshape(
                        nets_dict['act_points'], [-1, mat_dim * mat_dim])

                    nets_dict['fc0'] = slim.fully_connected(
                        nets_dict['act_points_transed'],
                        256,
                        activation_fn=tf.nn.relu,
                        scope="fc0")
                    nets_dict['fc1'] = slim.fully_connected(
                        nets_dict['fc0'],
                        4 * 4 * 64,
                        activation_fn=tf.nn.relu,
                        scope="fc1")
                    n = tf.reshape(nets_dict['fc1'], [-1, 4, 4, 64])
                    nets_dict['deconv2d0'] = slim.conv2d_transpose(
                        n, 64, [4, 4], scope='deconv2d_0')
                    nets_dict['deconv2d1'] = slim.conv2d_transpose(
                        nets_dict['deconv2d0'], 32, [4, 4], scope='deconv2d_1')
                    nets_dict['deconv2d2'] = slim.conv2d_transpose(
                        nets_dict['deconv2d1'], 32, [4, 4], scope='deconv2d_2')
                    nets_dict['output'] = slim.conv2d_transpose(
                        nets_dict['deconv2d2'],
                        output_channel, [4, 4],
                        activation_fn=None,
                        scope='deconv2d_3')
                    return nets_dict
