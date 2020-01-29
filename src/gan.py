import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.layers as layers
from tqdm import trange
from subprocess import PIPE, run
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adagrad

BATCH_SIZE = 512
PLOT_EVERY = 10
TOTAL_STEPS = 5000
GRID_RESOLUTION = 400
GENERATOR_DIM = 1


def uniform_to_normal(z):
    '''
    Map a value from ~U(-1, 1) to ~N(0, 1)
    '''
    norm = stats.norm(0, 1)
    return norm.ppf((z+1)/2)


def generate_noise(samples, dimensions=2):
    '''
    Generate a matrix of random noise in [-1, 1] with shape (samples, dimensions)
    '''
    return np.random.uniform(-1, 1, (samples, dimensions))


def build_generator(GENERATOR_DIM, output_dim):
    '''
    Build a generator mapping (R, R) to ([-1,1], [-1,1])
    '''
    input_layer = layers.Input((GENERATOR_DIM,))
    X = input_layer
    for i in range(4):
        X = layers.Dense(16)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(output_dim)(X)
    G = Model(input_layer, output_layer)
    return G


def build_discriminator(dim):
    '''
    Build a discriminator mapping (R, R) to [0, 1]
    '''
    input_layer = layers.Input((dim,))
    X = input_layer
    for i in range(2):
        X = layers.Dense(64)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(1, activation='sigmoid')(X)
    D = Model(input_layer, output_layer)
    D.compile(Adam(learning_rate=0.002, beta_1=0.5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return D


def build_GAN(G, D, GENERATOR_DIM):
    '''
    Given a generator and a discriminator, build a GAN
    '''
    D.trainable = False
    input_layer = layers.Input((GENERATOR_DIM,))
    X = G(input_layer)
    output_layer = D(X)
    GAN = Model(input_layer, output_layer)
    GAN.compile(Adagrad(learning_rate=0.001, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    # GAN.compile(Adam(learning_rate=0.001, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    return GAN


def plot_gan(G, D, step, step_count, D_accuracy, D_loss, G_accuracy, G_loss, filename):
    '''
    Plots for the GAN training video
    '''

    f, ax = plt.subplots(2, 2, figsize=(8, 8))
    f.suptitle(f'\n       {step:05d}', fontsize=10)

    # [0, 0]: plot loss and accuracy
    ax[0, 0].plot(step_count, G_loss, label='G loss', color='darkred', zorder=50, alpha=0.8,)
    ax[0, 0].plot(step_count, G_accuracy, label='G accuracy', color='lightcoral', zorder=40, alpha=0.8,)
    ax[0, 0].plot(step_count, D_loss, label='D loss', color='darkblue', zorder=55, alpha=0.8,)
    ax[0, 0].plot(step_count, D_accuracy, label='D accuracy', color='cornflowerblue', zorder=45, alpha=0.8,)
    ax[0, 0].set_xlim(-5, step + 5)
    ax[0, 0].set_ylim(-0.05, 1.55)
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].legend(loc=1)

    # [0, 1]: Plot actual samples and fake samples

    fake_samples = G.predict(test_noise, batch_size=len(test_noise))
    # sns.kdeplot(test_samples.flatten(), color='blue', alpha=0.6, label='Real', ax=ax[0, 1], shade=True)
    x_vals = np.linspace(-3, 3, 301)
    y_vals = stats.norm(0, 1).pdf(x_vals)
    ax[0, 1].plot(x_vals, y_vals, color='blue', label='real')
    ax[0, 1].fill_between(x_vals, np.zeros(len(x_vals)), y_vals, color='blue', alpha=0.6)
    # sns.kdeplot(test_samples.flatten(), color='blue', alpha=0.6, label='Real', ax=ax[0, 1], shade=True)
    sns.kdeplot(fake_samples.flatten(), color='red', alpha=0.6, label='GAN', ax=ax[0, 1], shade=True)
    ax[0, 1].set_xlim(-3, 3)
    ax[0, 1].set_ylim(0, 0.82)
    ax[0, 1].legend(loc=1)
    ax[0, 1].set_xlabel('Sample Space')
    ax[0, 1].set_ylabel('Probability Density')

    # [1, 0]: Confident real input
    grid_latent = np.linspace(-1, 1, 103)[1:-1].reshape((-1, 1))
    true_mappings = uniform_to_normal(grid_latent)
    GAN_mapping = G.predict(grid_latent)
    ax[1, 0].scatter(grid_latent.flatten(), true_mappings.flatten(),
                     edgecolor='blue', facecolor='None', s=5, alpha=1,
                     linewidth=1, label='Real Mapping')
    ax[1, 0].scatter(grid_latent.flatten(), GAN_mapping.flatten(),
                     edgecolor='red', facecolor='None', s=5, alpha=1,
                     linewidth=1, label='GAN Mapping')
    ax[1, 0].legend(loc=8)
    ax[1, 0].set_xlim(-1, 1)
    ax[1, 0].set_ylim(-3, 3)
    ax[1, 0].set_xlabel('Latent Space')
    ax[1, 0].set_ylabel('Sample Space')

    # [1, 1]: Confident real ouput
    grid_sample = np.linspace(-3, 3, 603)[1:-1].reshape((-1, 1))
    confidences = D.predict(grid_sample, batch_size=BATCH_SIZE).flatten()
    ax[1, 1].plot(grid_sample.flatten(), confidences, color='k')
    lower, upper = -3, 3
    for i in range(0, len(confidences), 50):
        if i == 0:
            continue
        ax[1, 1].plot([i / len(confidences) * (upper - lower) + lower, ]*2,
                      [0, confidences[i]], color='k')
    ax[1, 1].plot([lower, lower, upper, upper], [confidences[0], 0, 0, confidences[-1]], color='k')
    ax[1, 1].fill_between(grid_sample.flatten(), np.zeros(len(confidences)), confidences, color='k', alpha=0.6)
    ax[1, 1].set_xlabel('Sample Space Value')
    ax[1, 1].set_ylabel('Discriminator Confidence')
    ax[1, 1].set_xlim(lower, upper)
    ax[1, 1].set_ylim(-0.00, 1.00)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def train_gan(data, mu, sigma, k, plot):

    G = build_generator(GENERATOR_DIM, 1)
    D = build_discriminator(1)
    GAN = build_GAN(G, D, GENERATOR_DIM)

    step_count = []
    D_accuracy = []
    G_accuracy = []
    D_loss = []
    G_loss = []

    for step in trange(TOTAL_STEPS):

        # Train discriminator
        D.trainable = True
        real_data = uniform_to_normal(generate_noise(BATCH_SIZE // 2, GENERATOR_DIM))
        fake_data = G.predict(generate_noise(BATCH_SIZE // 2, GENERATOR_DIM), batch_size=BATCH_SIZE // 2)
        data = np.concatenate((real_data, fake_data), axis=0)
        real_labels = np.ones((BATCH_SIZE // 2, 1))
        fake_labels = np.zeros((BATCH_SIZE // 2, 1))
        labels = np.concatenate((real_labels, fake_labels), axis=0)
        _D_loss, _D_accuracy = D.train_on_batch(data, labels)

        # Train generator
        D.trainable = False
        noise = generate_noise(BATCH_SIZE, GENERATOR_DIM)
        labels = np.ones((BATCH_SIZE, 1))
        _G_loss, _G_accuracy = GAN.train_on_batch(noise, labels)

        if step % PLOT_EVERY == 0 or (step < 1500 and step % np.floor(PLOT_EVERY / 5) == 0):
            step_count.append(step)
            D_loss.append(_D_loss)
            D_accuracy.append(_D_accuracy)
            G_loss.append(_G_loss)
            G_accuracy.append(_G_accuracy)

    if plot:
        for step in trange(TOTAL_STEPS):
            if step % PLOT_EVERY == 0 or (step < 1500 and step % np.floor(PLOT_EVERY / 5) == 0):
                plot_gan(G=G,
                         D=D,
                         step=step+1,
                         step_count=step_count[:step],
                         D_accuracy=D_accuracy[:step],
                         D_loss=D_loss[:step],
                         G_accuracy=G_accuracy[:step],
                         G_loss=G_loss[:step],
                         filename=f'ims/1_1D_normal/a/1b.{step:05d}.png')

        run('ffmpeg -r 60 -pattern_type glob -i "ims/1_1D_normal/a/*.png" -vcodec mpeg4 -vb 50M ims/1_1D_normal/a/fulltrain.mp4',
            stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)

    G.save(f'gan/{mu}_{sigma}_{k}G.h5')
    D.save(f'gan/{mu}_{sigma}_{k}D.h5')
