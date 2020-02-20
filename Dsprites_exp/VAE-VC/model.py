import os
import sys
import datetime
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from utils.general_class import ModelPlugin
from utils.ortools_op import SolveMaxMatching
from utils.visual_op import matrix_image2big_image
from utils.writer_op import write_pkl, write_gif
from utils.tqdm_op import tqdm_range
from utils.eval_op import DisentanglemetricFactorMask, DisentanglemetricFactorJointMask
from utils.np_op import np_softmax

from tfops.transform_op import apply_tf_op, apply_tf_op_multi_output, apply_tf_op_multi_input
from tfops.train_op import get_train_op_v2
from tfops.lr_op import DECAY_DICT, DECAY_PARAMS_DICT
from tfops.nets import encoder1_64
from tfops.nets import decoder1_64
from local_nets import disc_net_64
from tfops.loss import sigmoid_cross_entropy_without_mean, vae_kl_cost_weight

import tensorflow as tf
import numpy as np

class Model(ModelPlugin):
    def __init__(self, dataset, logfilepath, args):
        super().__init__(dataset, logfilepath, args)
        self.build()

    def build(self):
        self.logger.info("Model building starts")
        tf.reset_default_graph()
        tf.set_random_seed(self.args.rseed)

        self.input1 = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.height, self.width, self.nchannel])
        self.epsilon_input = tf.placeholder(tf.float32, shape=[self.args.nbatch, self.args.nconti])
        self.objective = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.args.ncat])
        self.istrain = tf.placeholder(tf.bool, shape= [])
        self.I_weight = tf.placeholder(tf.float32, shape = [])
        self.F_weight = tf.placeholder(tf.float32, shape = [])

        # For VC-Loss
        self.delta_dim = tf.placeholder(tf.int32, shape=[self.args.nbatch])
        if self.args.use_discrete:
            self.objective_2_idx = tf.placeholder(tf.int32, shape = [self.args.nbatch])
        else:
            self.objective_2 = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.args.ncat])

        self.generate_sess()

        self.mcf = SolveMaxMatching(nworkers=self.args.nbatch, ntasks=self.args.ncat, k=1, pairwise_lamb=self.args.plamb)
        # Encoding
        self.encoder_net = encoder1_64
        self.decoder_net = decoder1_64
        self.disc_net = disc_net_64

        # Continuous rep
        self.mean_total, self.stddev_total = tf.split(self.encoder_net(self.input1, output_dim=2*self.args.nconti, scope='encoder', reuse=False)['output'], num_or_size_splits=2, axis=1)
        self.stddev_total = tf.nn.softplus(self.stddev_total)
        self.z_sample = tf.add(self.mean_total, tf.multiply(self.stddev_total, self.epsilon_input))

        # For VC-Loss
        if self.args.delta_type == 'onedim':
            # C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
            # C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), latents.dtype)
            self.z_delta = tf.cast(tf.one_hot(self.delta_dim, self.args.nconti), self.z_sample.dtype)
            rand_eps = tf.random.normal([self.args.nbatch, 1], mean=0.0, stddev=2.0)
            self.delta_target = self.z_delta * rand_eps
            self.z_added = self.delta_target
            self.z_added = self.z_added + self.z_sample
        elif self.args.delta_type == 'fulldim':
            # C_delta_latents = tf.random.uniform([minibatch_size, C_global_size], minval=0, maxval=1.0, dtype=latents.dtype)
            self.delta_target = tf.random.uniform([self.args.nbatch, self.args.nconti], minval=0, maxval=1.0, dtype=self.z_sample.dtype)
            self.z_added = (self.delta_target - 0.5) * self.args.vc_epsilon
            self.z_added = self.z_added + self.z_sample

        self.dec_output_dict = self.decoder_net(z=tf.concat([self.z_sample, self.objective], axis=-1), output_channel=self.nchannel, scope="decoder", reuse=False)
        self.dec_output = self.dec_output_dict['output']
        self.feat_output = self.dec_output_dict['deconv2d2']
        self.F_loss = tf.reduce_mean(self.feat_output * self.feat_output)
        self.F_loss = self.args.F_beta * self.F_loss

        if self.args.use_discrete:
            self.objective_2 = tf.cast(tf.one_hot(self.objective_2_idx, self.args.ncat), self.z_added.dtype)
        self.dec_output_2 = self.decoder_net(z=tf.concat([self.z_added, self.objective_2], axis=-1), output_channel=self.nchannel, scope="decoder", reuse=True)['output']
        self.disc_output = self.disc_net(img1=self.dec_output, img2=self.dec_output_2, target_dim=self.args.nconti, scope='discriminator', reuse=False)['output']

        if self.args.delta_type == 'onedim':
            # Loss VC CEloss
            self.disc_prob = tf.nn.softmax(self.disc_output, axis=1)
            self.I_loss = tf.reduce_mean(tf.reduce_sum(self.z_delta * tf.log(self.disc_prob + 1e-12), axis=1))
            self.I_loss = - self.args.C_lambda * self.I_loss
        elif self.args.delta_type == 'fulldim':
            # Loss VC MSEloss
            self.I_loss = tf.reduce_mean(tf.reduce_sum((tf.nn.sigmoid(self.disc_output) - self.delta_target) ** 2, axis=1))
            self.I_loss = self.args.C_lambda * self.I_loss

        # Unary vector
        self.rec_cost_vector = sigmoid_cross_entropy_without_mean(labels=self.input1, logits=self.dec_output)

        # Loss
        self.rec_cost = tf.reduce_mean(self.rec_cost_vector)

        weight = tf.constant(np.array(self.args.nconti*[self.args.beta_max]), dtype=tf.float32)
        kl_cost = vae_kl_cost_weight(mean=self.mean_total, stddev=self.stddev_total, weight=weight)
        self.loss = self.rec_cost+kl_cost+tf.losses.get_regularization_loss()+\
                self.I_loss*self.I_weight+self.F_loss*self.F_weight

        tf.summary.scalar('rec_loss', self.rec_cost)
        tf.summary.scalar('I_loss', self.I_loss)
        tf.summary.scalar('F_loss', self.F_loss)
        self.merged = tf.summary.merge_all()

        # Decode
        self.latent_ph = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.args.nconti+self.args.ncat])
        self.dec_output_ph = tf.nn.sigmoid(self.decoder_net(z=self.latent_ph, output_channel=self.nchannel, scope="decoder", reuse=True)['output'])

        self.logger.info("Model building ends")

    def decode(self, latent_input):
        return apply_tf_op(inputs=latent_input, session=self.sess, input_gate=self.latent_ph, output_gate=self.dec_output_ph, batch_size=self.args.nbatch)

    def set_up_train(self):
        self.logger.info("Model setting up train starts")

        if not hasattr(self, 'start_iter'): self.start_iter = 0
        self.logger.info("Start iter: {}".format(self.start_iter))

        decay_func = DECAY_DICT[self.args.dtype]
        decay_params = DECAY_PARAMS_DICT[self.args.dtype][self.args.nbatch][self.args.dptype].copy() 
        decay_params['initial_step'] = self.start_iter

        self.lr, update_step_op = decay_func(**decay_params)
        self.update_step_op = [update_step_op]

        var_list = [v for v in tf.trainable_variables() if 'encoder' in v.name] + \
                [v for v in tf.trainable_variables() if 'decoder' in v.name] + \
                [v for v in tf.trainable_variables() if 'discriminator' in v.name]

        with tf.control_dependencies(tf.get_collection("update_ops")):
            self.train_op = get_train_op_v2(tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999), loss=self.loss, var_list=var_list)

        self.logger.info("Model setting up train ends")

    def run_batch(self, train_idx):
        feed_dict = dict()
        feed_dict[self.input1] = self.dataset.next_batch(batch_size=self.args.nbatch)[0]
        feed_dict[self.istrain] = True
        feed_dict[self.epsilon_input] = np.random.normal(size=[self.args.nbatch, self.args.nconti])

        # For VC-Loss
        feed_dict[self.delta_dim] = np.random.randint(0, self.args.nconti, size=[self.args.nbatch])
        # feed_dict[self.objective_2_idx] = np.random.randint(0, self.args.ncat, size=[self.args.nbatch])
        feed_dict[self.objective_2] = np.zeros([self.args.nbatch, self.args.ncat])

        if self.args.use_discrete:
            # with discrete
            if train_idx<self.args.ntime:
                feed_dict[self.objective] = np.zeros([self.args.nbatch, self.args.ncat])
                feed_dict[self.I_weight] = 1.
                feed_dict[self.F_weight] = 1.
            else:
                unary = np.zeros([self.args.nbatch, self.args.ncat])
                for idx in range(self.args.ncat):
                    feed_dict[self.objective] = np.tile(np.reshape(np.eye(self.args.ncat)[idx], [1,-1]), [self.args.nbatch, 1])
                    unary[:,idx] = self.sess.run(self.rec_cost_vector, feed_dict=feed_dict)
                feed_dict[self.objective] = self.mcf.solve(-unary)[1]
                feed_dict[self.I_weight] = 1.
                feed_dict[self.F_weight] = 1.
        else:
            # no discrete
            feed_dict[self.objective] = np.zeros([self.args.nbatch, self.args.ncat])
            if train_idx<self.args.ntime:
                feed_dict[self.I_weight] = 1.
                feed_dict[self.F_weight] = 1.
            else:
                feed_dict[self.I_weight] = 1.
                feed_dict[self.F_weight] = 1.

        summary, _ = self.sess.run([self.merged, self.train_op], feed_dict=feed_dict)
        return summary

    def train(self, niter, piter, siter, save_dir=None, asset_dir=None):
        self.logger.info("Model training starts")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(asset_dir, current_time, 'train')
        # test_log_dir = os.path.join(asset_dir, current_time, '/test')
        train_summary_writer = tf.summary.FileWriter(train_log_dir, self.sess.graph)

        final_iter = self.start_iter+niter
        max_accuracy = -1
        max_acc_iter = -1

        for iter_ in tqdm_range(self.start_iter, final_iter):
            train_idx = (iter_ - self.start_iter)//piter
            summary = self.run_batch(train_idx)
            train_summary_writer.add_summary(summary, iter_)

            if (iter_+1)%siter==0 or iter_+1==final_iter:
                if self.args.use_discrete:
                    include_discrete = False if train_idx < self.args.ntime else True
                    accuracy = self.evaluate(include_discrete=include_discrete)
                else:
                    accuracy = self.evaluate(include_discrete=False)

                self.latent_traversal_gif(path=asset_dir+'{}.gif'.format(iter_+1), include_discrete=include_discrete)
                if max_accuracy==-1 or max_accuracy<accuracy:
                    self.save(iter_, save_dir)
                    self.logger.info("Save process")
                    max_accuracy = accuracy
                    max_acc_iter = iter_
                self.logger.info('max_accuracy: '+str(max_accuracy))
                self.logger.info('max_acc_iter: '+str(max_acc_iter))

                with open(os.path.join(asset_dir, 'acc.txt'), 'a') as f:
                    f.write('iter: '+str(iter_) + ', acc: ' + str(accuracy) + \
                            '; max_iter:' + str(max_acc_iter) + \
                            ', max_acc:' + str(max_accuracy))
        self.logger.info("Model training ends")

    def evaluate(self, print_option=False, include_discrete=False, eps=1e-8, nsample=1024):
        if include_discrete:
            total_mean, total_std, latent_cat = self.get_latent_total()
            return DisentanglemetricFactorJointMask(mean=total_mean, std=total_std, latent_cat=latent_cat, nclasses=self.dataset.latents_sizes, sampler=self.dataset.next_batch_latent_fix_idx, print_option=print_option, ignore_discrete=False)
        else:
            total_mean, total_std = self.get_mean_std()
            return DisentanglemetricFactorMask(mean=total_mean, std=total_std, nclasses=self.dataset.latents_sizes, sampler=self.dataset.next_batch_latent_fix_idx, print_option=print_option)

    def get_mean_std(self):
        total_mean, total_std = apply_tf_op_multi_output(inputs=self.image, session=self.sess, input_gate=self.input1, output_gate_list=[self.mean_total, self.stddev_total], batch_size=self.args.nbatch, train_gate=self.istrain)
        return total_mean, total_std

    def get_latent_total(self):
        total_mean, total_std = self.get_mean_std()
        unary = np.zeros([self.ndata, self.args.ncat])
        for idx in range(self.args.ncat):
            unary[:,idx] = apply_tf_op_multi_input(inputs_list=[self.image, np.zeros([self.ndata, self.args.nconti]), np.tile(np.reshape(np.eye(self.args.ncat)[idx], [1,-1]), [self.ndata, 1])], session=self.sess, input_gate_list=[self.input1, self.epsilon_input, self.objective], output_gate=self.rec_cost_vector, batch_size=self.args.nbatch, train_gate=self.istrain)
        latent_cat = np_softmax(-unary)
        return total_mean, total_std, latent_cat

    def latent_traversal_gif(self, path, include_discrete=False, nimage=50, nmin=-1.0, nmax=1.0):
        gif = list()
        for i in range(nimage):
            value = nmin + (nmax - nmin)*i/nimage
            latent_conti = value*np.eye(self.args.nconti)
            if include_discrete:
                latent_cat = np.eye(self.args.ncat)
                gif.append(matrix_image2big_image(np.concatenate([np.expand_dims(self.decode(latent_input=np.concatenate([latent_conti, np.tile(np.expand_dims(latent_cat[j], axis=0), [self.args.nconti,1])], axis=1)), axis=0) for j in range(self.args.ncat)], axis=0)))
            else:
                latent_cat = np.zeros([self.args.ncat])
                gif.append(matrix_image2big_image(np.expand_dims(self.decode(latent_input=np.concatenate([latent_conti, np.tile(np.expand_dims(latent_cat, axis=0), [self.args.nconti,1])], axis=1)), axis=0)))
        write_gif(content=gif, path=path)


