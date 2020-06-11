import os
from argparse import ArgumentParser, ArgumentTypeError

import tensorflow as tf
import numpy as np

import nn.hyperparameters as hp
import util.sys as sys
import util.visualize as viz
from nn.models import Generator, Discriminator
from util.datasets import Datasets
from util.lightroom.editor import PhotoEditor, PSNR
from skimage.metrics import structural_similarity as ssim
from skimage import io
from PIL import Image

# Killing optional CPU driver warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx('float32')

gpu_available = tf.test.is_gpu_available()
print(gpu_available)


def parse_args():
    def valid_file(filepath):
        if os.path.exists(filepath):
            return os.path.normpath(filepath)
        else:
            raise ArgumentTypeError("Invalid file: {}".format(filepath))

    def valid_dir(directory):
        if os.path.isdir(directory):
            return os.path.normpath(directory)
        else:
            raise ArgumentTypeError("Invalid directory: {}".format(directory))

    parser = ArgumentParser(
        prog="FilterNet",
        description="A deep learning program for photo-realistic image editing")

    parser.add_argument(
        "--checkpoint-dir",
        default=os.getcwd() + "/model_weights",
        help="Directory to store checkpoint model weights")

    parser.add_argument(
        "--device",
        type=str,
        default="GPU:0" if gpu_available else "CPU:0",
        help="Specify the device of computation eg. CPU:0, GPU:0, GPU:1, ... ")

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"

    # Subparser for train command
    tn = subparsers.add_parser(
        "train",
        description="Train a new model on the given untouched and edited images")
    tn.set_defaults(command="train")

    tn.add_argument(
        "--epochs",
        type=int, default=hp.num_epochs,
        help="Number of epochs to train for")

    tn.add_argument(
        "--restore-checkpoint",
        action="store_true",
        help="Use this flag to resuming training from a previous checkpoint")

    tn.add_argument(
        "--untouched-dir",
        type=valid_dir,
        default=os.getcwd() + "/data/train/untouched",
        help="Directory of untouched images for training")

    tn.add_argument(
        "--edited-dir",
        type=valid_dir,
        default=os.getcwd() + "/data/train/edited",
        help="Directory of expert edited images for training")

    tn.add_argument(
        "--editor",
        choices=['A', 'B', 'C', 'D', 'E'],
        default='C',
        help="Which editor images to use for training")

    tn.add_argument(
        "--print-every-x-batches",
        type=int, default=hp.print_every_x_batches,
        help="After how many batches you want to print")

    tn.add_argument(
        "--save-every-x-batches",
        type=int, default=hp.save_every_x_batches,
        help="After how many batches you want to save")

    # Subparser for test command
    ts = subparsers.add_parser(
        "test", description="Test the model on the given test data")
    ts.set_defaults(command="test")

    ts.add_argument(
        "--untouched-dir",
        type=valid_dir,
        default=os.getcwd() + "/data/test/untouched",
        help="Directory of untouched images for testing")

    ts.add_argument(
        "--edited-dir",
        type=valid_dir,
        default=os.getcwd() + "/data/test/edited",
        help="Directory of expert edited images for testing")

    ts.add_argument(
        "--editor",
        choices=['C'],
        default='C',
        help="Which editor images to use for testing")

    ts.add_argument(
        "--display",
        type=bool,
        default=True,
        help="True if you want to display the test output, false to save to file.")

    ts.add_argument(
        "--num-display",
        type=int,
        default=hp.test_images,
        help="Number of test images to display, must be <= batch size")

    # Subparser for evaluate command
    ev = subparsers.add_parser(
        "evaluate", description="Run the model on the given image")
    ev.set_defaults(command="evaluate")

    ev.add_argument(
        "--image-path",
        type=valid_file,
        help="Path to image to edit")

    ev.add_argument(
        "--output-dir",
        default=os.getcwd() + "/output/",
        help="Directory of output edited images for testing")

    # Subparser for performance command
    pf = subparsers.add_parser(
        "performance",
        description="Evaluate the model by getting PSNR and SSIM metrics")
    pf.set_defaults(command="performance")

    pf.add_argument(
        "--untouched-dir",
        type=valid_dir,
        default=os.getcwd() + "/data/test/untouched",
        help="Directory of untouched images for training")

    pf.add_argument(
        "--edited-dir",
        type=valid_dir,
        default=os.getcwd() + "/data/test/edited",
        help="Directory of expert edited images for training")

    pf.add_argument(
        "--editor",
        choices=['A', 'B', 'C', 'D', 'E'],
        default='C',
        help="Which editor images to use for training")

    return parser.parse_args()


# Make arguments global
ARGS = parse_args()


def save_model_weights(gen, disc):
    gen.save_weights(ARGS.checkpoint_dir + "/generator.h5", save_format='h5')
    disc.save_weights(ARGS.checkpoint_dir + "/discriminator.h5", save_format='h5')


def train(dataset, manager, generator, discriminator):
    for e in range(ARGS.epochs):
        print("========== Epoch {} ==========".format(e))
        for b, batch in enumerate(dataset.data):
            # Update Generator
            for _ in range(hp.gen_update_freq):
                with tf.GradientTape() as gen_tape:
                    x_model = batch[:, 0]
                    prob, value = generator(x_model)
                    act_scaled, act = generator.convert_prob_act(prob.numpy())
                    act = tf.convert_to_tensor(act)

                    y_model = PhotoEditor.edit(x_model.numpy(), act_scaled)
                    y_model = tf.convert_to_tensor(y_model, dtype=tf.float32)
                    d_model = discriminator(x_model)

                    gen_loss = generator.loss_function(x_model, y_model,
                                                       d_model, prob, act,
                                                       value)

                gen_grad = gen_tape.gradient(gen_loss,
                                             generator.trainable_variables)
                generator.optimizer.apply_gradients(
                    zip(gen_grad, generator.trainable_variables))

            # Update Discriminator
            for i in range(hp.disc_update_freq):
                with tf.GradientTape() as disc_tape:
                    x_model, y_expert = batch[:, 0], batch[:, 1]
                    prob, value = generator(x_model)
                    act_scaled, _ = generator.convert_prob_act(prob.numpy())

                    y_model = PhotoEditor.edit(x_model.numpy(), act_scaled)
                    y_model = tf.convert_to_tensor(y_model, dtype=tf.float32)
                    d_expert = discriminator(y_expert)
                    d_model = discriminator(y_model)

                    disc_loss = discriminator.loss_function(y_model, y_expert,
                                                            d_model, d_expert)

                disc_grad = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
                discriminator.optimizer.apply_gradients(
                    zip(disc_grad, discriminator.trainable_variables))

            if b % ARGS.print_every_x_batches == 0:
                print(
                    "Epoch: {} Batch: {} Generator Loss: {} Discriminator Loss: {}".format(
                        e, b, gen_loss, disc_loss))

            if b % ARGS.save_every_x_batches == 0:
                manager.save()
                save_model_weights(generator, discriminator)


def test(dataset, generator):
    for batch in dataset.data:
        x_model = batch[:, 0]
        prob, value = generator(x_model)
        act_scaled, _ = generator.convert_prob_act(prob.numpy())

        y_model = PhotoEditor.edit(x_model.numpy(), act_scaled)
        # Call visualizer to visualize images
        viz.visualize_batch(batch, y_model, ARGS.display, ARGS.num_display)

        prob, value = generator(x_model)
        act_scaled, _ = generator.convert_prob_act(prob.numpy(), det=True,
                                                   det_avg=hp.det_avg)

        y_model = PhotoEditor.edit(x_model.numpy(), act_scaled)
        # Call visualizer to visualize images
        viz.visualize_batch(batch, y_model, ARGS.display, ARGS.num_display)
        break


def performance(dataset, generator):
    print("====== Calculating performance metrics =====")
    batch_count = 0
    psnr = 0
    psnr_rand = 0
    ssim_val = 0
    ssim_rand = 0
    psnr_baseline = 0
    ssim_baseline = 0
    for batch in dataset.data:
        x_model = batch[:, 0]
        prob, value = generator(x_model)
        act_scaled, _ = generator.convert_prob_act(prob.numpy(), det=True,
                                                   det_avg=hp.det_avg)
        y_model = PhotoEditor.edit(x_model.numpy(), act_scaled)
        rand_act = np.random.rand(batch.shape[0], hp.K)
        rand_act = (rand_act * (hp.ak_max * 2)) - hp.ak_max
        rand_edits = PhotoEditor.edit(x_model.numpy(), rand_act)

        # Call to get metrics
        psnr += PSNR(y_model, batch[:, 1])
        ssim_val += ssim(y_model, batch[:, 1].numpy(), data_range=1,
                         multichannel=True)

        psnr_rand += PSNR(rand_edits, batch[:, 1])
        ssim_rand += ssim(rand_edits, batch[:, 1].numpy(), data_range=1,
                          multichannel=True)

        psnr_baseline += PSNR(x_model.numpy(), batch[:, 1])
        ssim_baseline += ssim(x_model.numpy(), batch[:, 1].numpy(),
                              data_range=1,
                              multichannel=True)

        batch_count += 1

    print("=== MODEL ===")
    print("PSNR: {} | SSIM: {}".format(psnr / batch_count,
                                       ssim_val / batch_count))

    print("=== RANDOM ===")
    print("PSNR: {} | SSIM: {}".format(psnr_rand / batch_count,
                                       ssim_rand / batch_count))

    print("=== BASELINE ===")
    print("PSNR: {} | SSIM: {}".format(psnr_baseline / batch_count,
                                       ssim_baseline / batch_count))


def main():
    # Initialize generator and discriminator models
    generator = Generator()
    discriminator = Discriminator()

    # Ensure the checkpoint directory exists
    sys.enforce_dir(ARGS.checkpoint_dir)

    # Set up tf checkpoint manager
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator)

    manager = tf.train.CheckpointManager(
        checkpoint, ARGS.checkpoint_dir,
        max_to_keep=3)

    if ARGS.command != 'train' or ARGS.restore_checkpoint:
        # Restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        save_model_weights(generator, discriminator)
        print("Restored checkpoint")

    try:
        with tf.device("/device:" + ARGS.device):
            if ARGS.command == "train":
                # train here!
                dataset = Datasets(
                    ARGS.untouched_dir, ARGS.edited_dir, "train", ARGS.editor)

                train(dataset, manager, generator, discriminator)

            if ARGS.command == 'test':
                # test here!
                dataset = Datasets(
                    ARGS.untouched_dir, ARGS.edited_dir, "test", ARGS.editor)
                test(dataset, generator)

            if ARGS.command == 'evaluate':
                # Ensure the output directory exists
                sys.enforce_dir(ARGS.output_dir)
                img_name = ARGS.image_path.split("/")[-1].split(".")[0]
                # evaluate here!
                img = io.imread(ARGS.image_path)
                if img.shape[-1] == 4:
                    rgba_img = Image.fromarray(img)
                    img = np.array(rgba_img.convert('RGB'))

                edited_img, _ = edit_original(img, generator)

                io.imshow(edited_img)
                io.show()
                io.imsave(ARGS.output_dir + "/" + img_name + "-edited.png",
                          edited_img)
                pass

            if ARGS.command == 'performance':
                # get performance metrics here!
                dataset = Datasets(
                    ARGS.untouched_dir, ARGS.edited_dir, "test", ARGS.editor)
                performance(dataset, generator)

    except RuntimeError as e:
        # something went wrong should not get here
        print(e)


if __name__ == "__main__":
    main()
