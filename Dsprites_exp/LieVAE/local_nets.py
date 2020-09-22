#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: local_nets.py
# --- Creation Date: 21-09-2020
# --- Last Modified: Tue 22 Sep 2020 16:52:39 AEST
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
slim = tf.contrib.slim


def lie_encoder1_64(x,
                    output_dim,
                    output_nonlinearity=None,
                    scope="ENC",
                    group_feats_size=400,
                    reuse=False):
    nets_dict = dict()
    nets_dict['input'] = x
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.contrib.slim.
                                variance_scaling_initializer(),
                                stride=2,
                                padding='SAME',
                                activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.fully_connected],
                                    biases_initializer=tf.zeros_initializer()):
                    nets_dict['conv2d0'] = slim.conv2d(nets_dict['input'],
                                                       32, [4, 4],
                                                       scope='conv2d_0')
                    nets_dict['conv2d1'] = slim.conv2d(nets_dict['conv2d0'],
                                                       32, [4, 4],
                                                       scope='conv2d_1')
                    nets_dict['conv2d2'] = slim.conv2d(nets_dict['conv2d1'],
                                                       64, [4, 4],
                                                       scope='conv2d_2')
                    nets_dict['conv2d3'] = slim.conv2d(nets_dict['conv2d2'],
                                                       64, [4, 4],
                                                       scope='conv2d_3')
                    n = tf.reshape(nets_dict['conv2d3'], [-1, 4 * 4 * 64])
                    nets_dict['fc0'] = slim.fully_connected(
                        n, 256, activation_fn=tf.nn.relu, scope="output_fc0")

                    nets_dict['gfeats_flat'] = slim.fully_connected(
                        nets_dict['fc0'],
                        group_feats_size,
                        activation_fn=None,
                        scope='output_gfeats')
                    mat_dim = int(math.sqrt(group_feats_size))
                    nets_dict['gfeats_mat'] = tf.reshape(
                        nets_dict['gfeats_flat'], [-1, mat_dim, mat_dim])

                    nets_dict['output'] = slim.fully_connected(
                        nets_dict['gfeats_flat'],
                        output_dim,
                        activation_fn=output_nonlinearity,
                        scope="output_fc1")
                    return nets_dict


def lie_decoder1_64(z,
                    scope="DEC",
                    output_channel=1,
                    group_feats_size=400,
                    nconti=6,
                    ncat=3,
                    reuse=False):
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
                        init = tf.initializers.random_normal(0, 0.1)
                        lie_alg_tmp = tf.get_variable(
                            'lie_alg_' + str(i),
                            shape=[1, mat_dim, mat_dim],
                            initializer=init)
                        lie_alg_basis_ls.append(lie_alg_tmp)
                    nets_dict['lie_alg_basis'] = tf.concat(
                        lie_alg_basis_ls,
                        axis=0)[tf.newaxis,
                                ...]  # [1, lat_dim, mat_dim, mat_dim]

                    input_conti = nets_dict['input'][:, :nconti]
                    input_cat = nets_dict['input'][:, nconti:]
                    lie_alg_mul = input_conti[
                        ..., tf.newaxis, tf.newaxis] * nets_dict[
                            'lie_alg_basis']  # [b, lat_dim, mat_dim, mat_dim]

                    nets_dict['lie_alg'] = tf.reduce_sum(
                        lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
                    nets_dict['lie_group_mat'] = tf.linalg.expm(
                        nets_dict['lie_alg'])  # [b, mat_dim, mat_dim]
                    nets_dict['lie_group_flat'] = tf.reshape(
                        nets_dict['lie_group_mat'], [-1, mat_dim * mat_dim])

                    nets_dict['fc0_conti'] = slim.fully_connected(
                        nets_dict['lie_group_flat'],
                        256,
                        activation_fn=tf.nn.relu,
                        scope="fc0")
                    nets_dict['fc0_cat'] = slim.fully_connected(
                        input_cat,
                        256,
                        activation_fn=tf.nn.relu,
                        scope="fc0_cat")
                    nets_dict[
                        'fc0'] = nets_dict['fc0_conti'] + nets_dict['fc0_cat']
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
