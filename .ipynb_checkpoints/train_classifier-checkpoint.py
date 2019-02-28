#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import utils
import argparse
from nets.model import get_model_fn
from dataloader import DataLoader
def parse_args():
    parser = argparse.ArgumentParser("Training Classifier")

    #Dataset Configuration
    parser.add_argument("--image_size", help = "Image Size", type = int, default = 224)
    parser.add_argument("--gray2rgb", help = "Convert Image between RGB and Grayscale:  0: keep; 1: gray2rgb; -1: rgb2gray", type = int, default = 0)
    parser.add_argument("--csv_file", help = "csv input file [label,img_name]", type = str, default = "")
    parser.add_argument("--dataset", help = "Dataset Name", type = str, default = "usps")
    parser.add_argument("--split", help = "Split Name", type = str, default = "train")
    parser.add_argument("--dataset_dir", help = "Dataset Dir", type = str, default = "data/usps")

    #Learning Configuration
    parser.add_argument("--model", help = "Model Name", type = str, default = "lenet")
    parser.add_argument("--num_iters", help = "Number of Iterations", type = int, default = 10000)
    parser.add_argument("--batch_size", help = "Batch Size", type = int, default = 32)
    parser.add_argument("--learning_rate", help = "Learning Rate", type = float, default = 1e-4)
    parser.add_argument("--weight_decay", help = "Regularization Rate", type = float, default = 2e5-5)
    parser.add_argument("--model_path", help = "Model Path", type = str, default = "./model/pretrained")
    parser.add_argument("--checkpoint_steps", help = "Checkpoint Step", type = int, default = 100)
    parser.add_argument("--lr_decay_steps", help = "Learning Rate Decay Steps", type = int, default = None)
    parser.add_argument("--lr_decay_rate", help = "Learning Rate Decay Rate", type = float, default = 0.1)
    parser.add_argument("--solver", help = "Optimizer", type = str, default = "adam")
    parser.add_argument("--num_readers", help = "Number of Readers", type = int, default = 4)
    parser.add_argument("--num_preprocessing_threads", help = "Number of Preprocessing Threads", type = int, default = 4)
    parser.add_argument("--gpu_id", help = "Config GPU Id for training", type = str, default = "0")
    parser.add_argument("--checkpoint", help="Config pre-trained model" , type = str, default = None)
    return parser.parse_args()

def main():
    #####################
    ## Parse Arguments ##
    #####################
    options = parse_args()
    
    #####################
    ## Config GPU      ##
    #####################
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]=options.gpu_id

    print("Configurations")
    for var in vars(options):
        print("\t>> {}: {} ".format(var, getattr(options, var)))
    #print("Enter to Continue or Ctrl+C to Break")

    ##################
    ## Load dataset ##
    ##################
    images, labels, no_samples, no_classes = DataLoader.get_dataset_from_folder(
                            dataset_name = options.dataset,
                            dataset_root = options.dataset_dir, 
                            csv_file = options.csv_file,
                            split = options.split,
                            batch_size = options.batch_size,
                            shuffle = True,
                            is_training=True)

    
    ################
    ## Load Model ##
    ################
    model_fn = get_model_fn(options.model)
    net, layers = model_fn(images, image_size = options.image_size, num_classes = no_classes,
                            is_training = True, scope = options.model,
                            weight_decay = options.weight_decay)
    
    variables_to_restore = [var for var in slim.get_variables() if "logits" not in var.name]
    class_loss = tf.losses.sparse_softmax_cross_entropy(labels, net)
    total_loss = tf.losses.get_total_loss()
    accuracy = slim.metrics.accuracy(tf.argmax(net, axis = 1), tf.cast(labels, tf.int64))

    ############################
    ## Learning Configuration ##
    ############################
    learning_rate_op = tf.Variable(options.learning_rate, name='learning_rate', trainable = False)
    if options.solver == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate_op)
    elif options.solver == "sgd":
        optimizer = tf.train.MomentumOptimizer(learning_rate_op, 0.99)
    else:
        raise ValueError("{} optimizer was not support".format(options.solver))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss)
    else:
        train_op = optimizer.minimize(total_loss)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    model_path = tf.train.latest_checkpoint(options.model_path)
    
    if (options.checkpoint is not None):
        if os.path.exists(options.checkpoint):
            tf.train.Saver(variables_to_restore).restore(sess,options.checkpoint)
        
    if model_path is not None:
        saver.restore(sess, model_path)
        print("Restore model succcessfully from {}".format(model_path))

    summaries = [tf.summary.scalar("Accuracy", accuracy),
                    tf.summary.scalar("Loss", total_loss) ]

    summary_op = tf.summary.merge(summaries)
    summary_writer = tf.summary.FileWriter(options.model_path, graph=tf.get_default_graph())
    model_path = os.path.join(options.model_path, "{}_{}.ckpt".format(options.dataset, options.model))
    ##############
    ## Training ##
    ##############
    timer = utils.Timer()
    current_learning_rate = options.learning_rate
    for iter in range(1, options.num_iters + 1):
        timer.tic()
        _, loss, acc, summary, lr = sess.run([train_op, total_loss, accuracy, summary_op, learning_rate_op])
        summary_writer.add_summary(summary, iter)
        print("Iteration [{}/{}]:".format(iter, options.num_iters))
        print("\t>> Total Loss:\t{}".format(loss))
        print("\t>> Accuracy:\t{}".format(acc))
        print("\t>> Learning Rate:\t{}".format(lr))
        print("\t>> Executed Time:\t{} sec/iter".format(timer.toc()))

        if (iter % options.checkpoint_steps) == 0:
            saver.save(sess, model_path)

        if options.lr_decay_steps is not None and (iter % options.lr_decay_steps) == 0:
            current_learning_rate = sess.run(learning_rate_op.assign(current_learning_rate * options.lr_decay_rate))

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
    main()



