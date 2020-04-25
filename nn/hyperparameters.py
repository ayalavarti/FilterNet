from util.lightroom import editor as E
# Image size
img_size = 256

# Sample size to be read into memory temporarily.
preprocess_sample_size = 400

# Training parameters
num_epochs = 50
batch_size = 10
learning_rate = 1e-4
momentum = 0.01

#should we select 10 of the filters because there are more than 10
parameters = {
            E.clarity: .2,
            E.contrast: .3,
            E.exposure: .2,
            E.temp: .2,
            E.tint: .2,
            E.whites: .2,
            E.blacks: .2,
            E.highlights: .2,
            E.shadows: .2,
            E.vibrance: .2,
           # E.saturation: .2
        }

