
#ifndef INFO_DATA_H_
#define INFO_DATA_H_
// third party utilities
#include <vector>
#include <iostream>

namespace melo {

    struct WaveHeader {
        bool Validate() const {
            //                 F F I R
            if (chunk_id != 0x46464952) {
                std::cerr << "Expected chunk_id RIFF. Given: 0x" << std::hex
                    << chunk_id;
                return false;
            }
            //               E V A W
            if (format != 0x45564157) {
                std::cerr << "Expected format WAVE. Given: 0x" << std::hex
                    << format;
                return false;
            }

            if (subchunk1_id != 0x20746d66) {
                std::cerr << "Expected subchunk1_id 0x20746d66. Given: 0x"
                    << std::hex << subchunk1_id;
                return false;
            }

            if (subchunk1_size != 16) {  // 16 for PCM
                std::cerr << "Expected subchunk1_size 16. Given: "
                    << subchunk1_size;
                return false;
            }

            if (audio_format != 1) {  // 1 for PCM
                std::cerr << "Expected audio_format 1. Given: "
                    << audio_format;
                return false;
            }

            if (num_channels != 1) {  // we support only single channel for now
                std::cerr << "Expected single channel. Given: "
                    << num_channels;
                return false;
            }
            if (byte_rate != (sample_rate * num_channels * bits_per_sample / 8)) {
                return false;
            }

            if (block_align != (num_channels * bits_per_sample / 8)) {
                return false;
            }

            if (bits_per_sample != 16) {  // we support only 16 bits per sample
                std::cerr << "Expected bits_per_sample 16. Given: "
                    << bits_per_sample;
                return false;
            }

            return true;
        }

        // See
        // https://en.wikipedia.org/wiki/WAV#Metadata
        // and
        // https://www.robotplanet.dk/audio/wav_meta_data/riff_mci.pdf
        void SeekToDataChunk(std::istream& is) {
            //                              a t a d
            while (is && subchunk2_id != 0x61746164) {
                // const char *p = reinterpret_cast<const char *>(&subchunk2_id);
                // printf("Skip chunk (%x): %c%c%c%c of size: %d\n", subchunk2_id, p[0],
                //        p[1], p[2], p[3], subchunk2_size);
                is.seekg(subchunk2_size, std::istream::cur);
                is.read(reinterpret_cast<char*>(&subchunk2_id), sizeof(int32_t));
                is.read(reinterpret_cast<char*>(&subchunk2_size), sizeof(int32_t));
            }
        }

        int32_t chunk_id;
        int32_t chunk_size;
        int32_t format;
        int32_t subchunk1_id;
        int32_t subchunk1_size;
        int16_t audio_format;
        int16_t num_channels;
        int32_t sample_rate;
        int32_t byte_rate;
        int16_t block_align;
        int16_t bits_per_sample;
        int32_t subchunk2_id;    // a tag of this chunk
        int32_t subchunk2_size;  // size of subchunk2
    };
    static_assert(sizeof(WaveHeader) == 44, "");

}  // namespace melo
#endif  //INFO_DATA_H_
