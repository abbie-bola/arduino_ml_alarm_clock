class MFCCLayer(layers.Layer):
    def __init__(self, num_mfcc=10, **kwargs):
        super(MFCCLayer, self).__init__(**kwargs)
        self.num_mfcc = num_mfcc

    def call(self, inputs):
        sample_rate = 16000  # Audio sample rate
        input_length = 16000  # 1-second audio clip

        # Remove last dimension if present (ensuring input is [batch, samples])
        inputs = tf.squeeze(inputs, axis=-1)

        # Recalculate frame step to ensure exactly 49 frames
        desired_time_steps = 49
        frame_step = (input_length - 652) // (desired_time_steps - 1)  # ~319 samples
        frame_length = 652  # Keep frame length at 2 * frame step

        # Compute STFT
        stfts = tf.signal.stft(inputs, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
        spectrograms = tf.abs(stfts)

        # Compute Mel spectrogram
        num_spectrogram_bins = tf.shape(stfts)[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=80, num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sample_rate, lower_edge_hertz=80.0, upper_edge_hertz=7600.0
        )
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, axes=1)

        # Use dynamic shape inference to avoid shape mismatch
        mel_spectrograms = tf.ensure_shape(mel_spectrograms, (None, None, 80))

        # Compute log Mel spectrogram
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :self.num_mfcc]
        mfccs = tf.expand_dims(mfccs, axis=-1)  # Shape: (batch, time, num_mfcc, 1)

        return mfccs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 49, self.num_mfcc, 1)  # Ensure expected output shape
