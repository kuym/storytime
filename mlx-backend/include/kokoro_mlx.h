#ifndef KOKORO_MLX_H
#define KOKORO_MLX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Kokoro model from a directory containing config.json
// and *.safetensors weight files.
// Returns 0 on success, -1 on failure.
int32_t kokoro_init(const char *weights_dir);

// Generate audio from IPA phonemes.
// phonemes:  null-terminated IPA string
// voice_path: path to a .safetensors voice file
// speed:     speaking rate multiplier
// n_tokens:  number of phoneme tokens (for style row selection)
// audio_out: receives pointer to float32 audio (caller frees with kokoro_free_audio)
// audio_len: receives number of samples
// Returns 0 on success, -1 on failure.
int32_t kokoro_generate(const char *phonemes, const char *voice_path,
                        float speed, int32_t n_tokens,
                        float **audio_out, int32_t *audio_len);

// Free audio buffer allocated by kokoro_generate.
void kokoro_free_audio(float *audio);

// Release model resources.
void kokoro_cleanup(void);

// Get the model's native sample rate.
int32_t kokoro_sample_rate(void);

#ifdef __cplusplus
}
#endif

#endif // KOKORO_MLX_H
