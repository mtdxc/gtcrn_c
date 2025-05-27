#include "GTCRNImpl.h"
#include <iostream>
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " in_wav out_wav" << std::endl;
        return 1;
    }
    unsigned int channels, sampleRate;
    drwav_uint64 samples;
    short* pcm = drwav_open_file_and_read_pcm_frames_s16(argv[1], &channels, &sampleRate, &samples, nullptr);
    if (!pcm) {
        std::cerr << "Error: Could not read audio file." << std::endl;
        return 1;
    }
    if (channels!=1 && sampleRate!=SAMEPLERATE) {
        std::cerr << "Error: Audio file must be mono and sampled at " << SAMEPLERATE << " Hz." << std::endl;
        return 1;
    }
    drwav out;
    drwav_data_format fmt;
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_PCM;
    fmt.bitsPerSample = 16;
    fmt.channels = channels;
    fmt.sampleRate = sampleRate;
    drwav_init_file_write(&out, argv[2], &fmt, nullptr);

    GTCRNImpl gtcrn("gtcrn_simple.onnx");
    // Example usage of Enhance method
    short output_data[FRAME_LEN]; // Buffer for output data
    for (size_t i = 0; i < samples; i+= FRAME_LEN)
    {
        if (samples - i < FRAME_LEN) {
            std::cerr << "Warning: Not enough samples left for a full block." << std::endl;
            break; // Not enough samples for a full block
        }
        int processed_length = gtcrn.Process(pcm+i, output_data, FRAME_LEN);
        if (processed_length < 0) {
            std::cerr << "Error processing audio data." << std::endl;
            return 1;
        }
        // Here you can do something with output_data, like saving it to a file
        drwav_write_pcm_frames(&out, processed_length, output_data);
    }
    
    drwav_free(pcm, nullptr);
    drwav_uninit(&out);
    return 0;
}