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
            E.clarity: .125,
            E.contrast: .375,
            E.exposure: .123,
            E.temp: .125,
            E.tint: .125,
            E.whites: .125,
            E.blacks: .125,
            E.highlights: .125,
            E.shadows: .125,
            E.vibrance: .125,
            E.saturation: .125
        }

